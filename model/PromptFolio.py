""" Federated Text-driven Prompt Generation for Vision-Language Models (ICLR 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import copy
import torch
import torch.nn as nn
from model.prompt_net import PromptTranslator
from clip import clip
from clip.model_fedotp import build_model
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'PromptFL',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    # model = clip.build_model(state_dict or model.state_dict(), design_details)
    model = build_model(state_dict or model.state_dict(), design_details)


    return model



class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.conv1 = clip_model.conv1
        self.class_embedding = clip_model.class_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_pre = clip_model.ln_pre
        self.transformer = clip_model.transformer
        self.ln_post = clip_model.ln_post
        self.proj = clip_model.proj

    def forward(self, x, vis_ctx=[]):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]forwad
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, vis_ctx, False)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = cfg.TRAINER.PLOT.CTX_INIT
        # ctx_init = None
        # ctx_suf_init = "texture"
        ctx_suf_init = None
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        self.class_specific_context = cfg.TRAINER.PLOT.CSC
        self.N = cfg.TRAINER.PLOT.N

        classnames = [name.replace("_", " ") for name in classnames]

        # Calculate the length of classname prompt
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init
        else:
            n_ctx = cfg.TRAINER.PLOT.N_CTX
            prompt_prefix = " ".join(["X"] * n_ctx)

        if ctx_suf_init:
            prompt_suffix = " " + ctx_suf_init
        else:
            prompt_suffix = ""

        prompts = [prompt_prefix + " " + name + prompt_suffix + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        if self.class_specific_context:
            if ctx_init:
                ctx_vectors = embedding[:, 1: 1 + n_ctx, :]  # (n_cls * N) * n_ctx * ctx_dim
            else:
                ctx_vectors = torch.empty(n_cls * self.N, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
            ctx = nn.Parameter(ctx_vectors)  # n_ctx * ctx_dim
        else:
            if ctx_init:
                ctx_vectors = embedding[:self.N, 1: 1 + n_ctx, :]
            else:
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
            ctx = nn.Parameter(ctx_vectors)

        self.ctx = ctx
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # This tokenized_prompt only used to locate the end of the sentence.
        self.prompt_prefix = prompt_prefix
        self.class_token_position = cfg.TRAINER.PLOT.CLASS_TOKEN_POSITION
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

    def init_embedding(self, classnames, clip_model):
        classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(classnames) # update n_cls when classnames change
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        self.prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        self.nc_prompts = [ self.prompt_prefix + '.' ]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in self.prompts])   
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1) 
        with torch.no_grad():
            self.tokenized_prompts = tokenized_prompts
            self.embedding = clip_model.token_embedding(self.tokenized_prompts).type(clip_model.dtype)
        self.register_buffer("token_prefix", self.embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", self.embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS

    def forward(self):
        ctx = self.ctx
        if not self.class_specific_context:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            ctx = ctx.permute(1, 0, 2, 3)
            ctx = ctx.contiguous().view(self.N*self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class PromptFolio(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device="cuda"):
        super().__init__()

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # self.device = torch.device("cuda:0")
        # self.device1 = torch.device("cuda")
        self.N = cfg.TRAINER.PLOT.N
        # self.dataset = cfg.DATASET.NAME_SPACE[0]
        self.frac = cfg.TRAINER.PLOT.FRAC
        self.n_cls = len(classnames)

    def forward(self, image):

        image_features = self.get_img_features(image)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.n_cls = self.tokenized_prompts.shape[0] // self.N
        prompts = self.prompt_learner()
        # feature0 is global, feature1 is local
        text_features0 = self.text_encoder(prompts[:self.n_cls], tokenized_prompts[:self.n_cls])
        text_features1 = self.text_encoder(prompts[self.n_cls:2 * self.n_cls], tokenized_prompts[self.n_cls:2 * self.n_cls])

        text_features0 = text_features0 / text_features0.norm(dim=-1, keepdim=True)
        text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # frac = 0 means fully global
        # frac = 1 means fully local
        text_features = (1 - self.frac) * text_features0 + self.frac * text_features1
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward_global(self, image):

        image_features = self.get_img_features(image)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.n_cls = self.tokenized_prompts.shape[0] // self.N
        prompts = self.prompt_learner()

        # feature0 is global, feature1 is local
        text_features0 = self.text_encoder(prompts[:self.n_cls], tokenized_prompts[:self.n_cls])
        text_features0 = text_features0 / text_features0.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        # frac = 0 means fully global
        # frac = 1 means fully local
        text_features = text_features0
        logits = logit_scale * image_features @ text_features.t()

        return logits


    def get_img_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        if image_features.shape[0] != image.shape[0]:
            image_features = image_features.permute(1, 0, 2)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features