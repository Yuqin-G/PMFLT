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

import torch
import torch.nn as nn
from model.prompt_net import PromptTranslator
from clip import clip
from clip.model_fedpgp import build_model
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

    design_details = {"trainer": 'FedPGP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    # model = clip.build_model(state_dict or model.state_dict())
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
        print("n_cls: ", n_cls)
        n_ctx = cfg.TRAINER.FEDPGP.N_CTX
        ctx_init = cfg.TRAINER.FEDPGP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        bottleneck = cfg.TRAINER.FEDPGP.BOTTLENECK
        self.N = cfg.TRAINER.FEDPGP.N
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.FEDPGP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                U = torch.empty(self.N, n_ctx, bottleneck, dtype=dtype)
                V = torch.empty(self.N, bottleneck, ctx_dim, dtype=dtype)
                sigma = torch.empty(self.N, n_ctx, ctx_dim, dtype = dtype)
                # ctx_vectors = torch.matmul(U, V)

            # nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(U, std=0.02)
            nn.init.normal_(V, std=0.02)
            nn.init.normal_(sigma, std=0.02)# define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        self.sigma = nn.Parameter(sigma)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)
        # tokenized_prompts3.view(3, 100, 77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("embedding", embedding)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.FEDPGP.CLASS_TOKEN_POSITION


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
        # ctx = self.ctx
        U = self.U
        V = self.V
        UV = torch.matmul(U, V)
        sigma = self.sigma
        ctx = UV +self.sigma
        embedding = self.embedding

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])

        if UV.dim() == 3:
            UV = UV.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        UV = UV.permute(1, 0, 2, 3)
        UV = UV.contiguous().view(self.N * self.n_cls, self.n_ctx, UV.shape[3])

        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        sigma = sigma.permute(1, 0, 2, 3)
        sigma = sigma.contiguous().view(self.N * self.n_cls, self.n_ctx, sigma.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        # print(U.shape)
        # print(V.shape)
        # print(sigma.shape)

        # print(prefix.shape)
        # print(ctx.shape)
        # print(suffix.shape)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_sigma = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    sigma,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_UV = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    UV,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return embedding, prompts_sigma, prompts_UV, prompts


class FedPGP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device="cuda"):
    # def __init__(self, cfg, clip_model, device="cuda"):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        
    def forward(self, image):

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tokenized_prompts = self.prompt_learner.tokenized_prompts
        embedding, prompts_sigma, prompts_UV, prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.training == True:
            text_features_0 = self.text_encoder(embedding, tokenized_prompts)
            text_features_0 = text_features_0 / text_features_0.norm(dim=-1, keepdim=True)

            text_features_sigma = self.text_encoder(prompts_sigma, tokenized_prompts)
            text_features_sigma = text_features_sigma / text_features_sigma.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return text_features_0, text_features_sigma, None, text_features, logits
            # return text_features_0, text_features_sigma, text_features_UV, text_features, logits

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward_global(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tokenized_prompts = self.prompt_learner.tokenized_prompts
        _, prompts_sigma, _, _ = self.prompt_learner()

        text_features_sigma = self.text_encoder(prompts_sigma, tokenized_prompts)
        text_features_sigma = text_features_sigma / text_features_sigma.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_global = logit_scale * image_features @ text_features_sigma.t()
        return logits_global
