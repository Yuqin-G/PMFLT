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

    model = clip.build_model(state_dict or model.state_dict())

    return model

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, scaling=1.0, dtype=torch.float16):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # self.scaling = self.embed_dim ** -0.5
        self.scaling = scaling
        self.dtype = dtype
        
        self.W_q = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_k = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_v = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_o = nn.Linear(d_model, d_model, dtype=self.dtype)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output, torch.mean(attn_probs, dim=1)


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
        n_ctx = cfg.TRAINER.PFEDMOAP.N_CTX
        ctx_init = cfg.TRAINER.PFEDMOAP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.N = cfg.TRAINER.PFEDMOAP.N

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
            if cfg.TRAINER.PFEDMOAP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        print(f"tokenized_prompts device: {tokenized_prompts.device}")
        print(f"clip_model device: {next(clip_model.parameters()).device}")

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PFEDMOAP.CLASS_TOKEN_POSITION

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
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
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

        return prompts


class pFedMoAP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device="cuda"):
        super().__init__()
        self.n_class = len(classnames)
        self.cfg = cfg
        self.classnames = classnames
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # pFedMoAP specifics
        self.nonlocal_ctx = None # nonlocal prompt learner context
        self.nonlocal_text_features = None
        # self.num_experts = cfg.TRAINER.PFEDMOAP.NUM_EXPERTS
        self.lmbda = cfg.TRAINER.PFEDMOAP.LMBDA

        # initialize mixture of experts gating network
        num_heads = cfg.TRAINER.PFEDMOAP.GATING_HEADS
        gating_embed_dim = cfg.TRAINER.PFEDMOAP.GATING_EMBED_DIM
        reduce_times = self.image_encoder.output_dim // gating_embed_dim
        self.reduce_times = reduce_times
        
        self.gating = MultiheadAttention(gating_embed_dim, num_heads, dropout=0.1, scaling=cfg.TRAINER.PFEDMOAP.SCALING, dtype=self.dtype)
        
    def pool(self, t):
        if len(t.shape) == 4:
            return t[:, :, :, ::self.reduce_times]
        if len(t.shape) == 3:
            return t[:, :, ::self.reduce_times]
        if len(t.shape) == 2:
            return t[:, ::self.reduce_times]
        return None
    
    def _compute_nonlocal_text_features(self):
        if not self.nonlocal_ctx:
            return
        
        # store local state dict
        temp_local_state_dict = copy.deepcopy(self.prompt_learner.state_dict())
        self.nonlocal_text_features = []

        # if only one nonlocal context is provided, convert it to a list
        if not isinstance(self.nonlocal_ctx, list):
            self.nonlocal_ctx = [self.nonlocal_ctx]

        # iterate through different nonlocal contexts (global or other clients)
        for ctx in self.nonlocal_ctx:
            # load nonlocal ctx
            self.load_ctx(ctx)

            # compute nonlocal text features
            with torch.no_grad():
                text_features = self.text_encoder(self.prompt_learner(), self.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = self.pool(text_features)
                self.nonlocal_text_features.append(text_features.detach())
        
        # restore local state dict
        self.prompt_learner.load_state_dict(temp_local_state_dict)

    def load_ctx(self, ctx):
        self.prompt_learner.ctx.data.copy_(ctx.data)
        # temp_dict = self.prompt_learner.state_dict()
        # temp_dict['ctx']= ctx
        # self.prompt_learner.load_state_dict(temp_dict)

    def forward(self, image):
        
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        local_logits = logit_scale * image_features @ text_features.t()

        if self.nonlocal_text_features:
            q = self.pool(image_features).repeat(self.n_class, 1, 1) # (n_class, Batch, feature_dim)
            k = v = torch.stack([self.pool(text_features)] + self.nonlocal_text_features).permute(1, 0, 2) # (n_class, n_experts, feature_dim)
            new_features = self.gating(q, k, v)[0].permute(1, 2, 0) # (Batch, feature_dim, n_class)
            return self.lmbda * local_logits + logit_scale * torch.bmm(self.pool(image_features).unsqueeze(1), new_features).squeeze(1)
        else:
            return local_logits


    def forward_global(self, image):
        
        image_features = self.image_encoder(image.type(self.dtype))

        tokenized_prompts = self.prompt_learner.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        local_logits = logit_scale * image_features @ text_features.t()

        return local_logits
    

