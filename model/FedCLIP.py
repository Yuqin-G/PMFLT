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
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

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

    model = clip.build_model(state_dict or model.state_dict())

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

    def forward(self, prompts, tokenized_prompts, text_ctx):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, text_ctx, True)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        n_ctx, ctx_depth = cfg.MODEL.N_CTX, cfg.MODEL.D_CTX
        self.meta_net = PromptTranslator(n_ctx, ctx_depth, depth=cfg.MODEL.DEPTH)
        self.meta_net.half()

        self.ctx_depth = ctx_depth
        self.n_ctx = n_ctx

    def forward(self, context_emb):
        text_ctx, vis_ctx = self.meta_net(context_emb.unsqueeze(0))  # (n_ctx, ctx_dim) # self.ctx

        return text_ctx, vis_ctx


class FedCLIP(nn.Module):
    # def __init__(self, cfg, clip_model,  device='cuda'):
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        # self.clip, pre = clip.load("ViT-B/16", device=device)
        self.clip, pre = clip.load(cfg.MODEL.BACKBONE, device='cpu')
        if (cfg.MODEL.BACKBONE == "ViT-B/16"):
            feature_dim = 512
        elif (cfg.MODEL.BACKBONE == "RN50"):
            feature_dim = 1024

        self.device = device
        self.dtype = self.clip.dtype
        self.img_adap = nn.Sequential(
                         nn.Linear(feature_dim, feature_dim), 
                         nn.Tanh(), 
                         nn.Linear(feature_dim, feature_dim), 
                         nn.Softmax(dim=1)).to(self.device)

        self.img_adap.requires_grad = True

        # self.prompt_prefix = 'a picture of a'
        self.prompt_prefix = 'a photo of a'

    def get_tokenized_classnames(self, classnames):
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = self.encode_text(tokenized_prompts.to(self.device))
        # token_prefix = embedding[:, :1, :]  # SOS
        # token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
        self.text_features = text_features
        return text_features, tokenized_prompts


    def forward(self, image, classnames):
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = self.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        image_features = self.encode_image(image)
        image_features_att = self.img_adap(image_features)
        image_features = torch.mul(image_features_att, image_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, image_features, text_features


    def encode_image(self, image, vis_ctx=None):
        self.clip = self.clip.to(self.device)
        out = self.clip.encode_image(image.type(self.dtype))
        self.clip = self.clip.to("cpu")
        return out


    def encode_text(self, classnames):
        self.clip = self.clip.to(self.device)
        out = self.clip.encode_text(classnames)
        self.clip = self.clip.to("cpu")
        return out

