

import time
import os
import torch
import torchvision.utils as vutils
import math
from torch import nn
from einops import rearrange
from dataclasses import dataclass
import sys
import random
import json
from torch.nn import LayerNorm
from torch.nn import functional as F
from functools import partial
from checks import check_tensors, check_model
from rotary_pos import apply_rotary_emb
from torch.nn import GELU


class MLP(nn.Module):
    def __init__(
        self, embed_size, bias, dropout, non_linearity="GELU", norm_class=LayerNorm, eps_norm=1e-5, out_channels=None
    ):
        super().__init__()
        self.embed_size = embed_size
        if out_channels is None:
            out_channels = self.embed_size
        self.bias = bias
        non_linearity = getattr(sys.modules[__name__], non_linearity)()

        self.lin_block = nn.Sequential(
            norm_class(self.embed_size, bias=self.bias, eps=eps_norm),
            nn.Linear(self.embed_size, 4 * self.embed_size, bias=self.bias),
            non_linearity,
        )
        self.c_proj = nn.Linear(4 * self.embed_size, out_channels, bias=self.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.lin_block(x)
        out = self.c_proj(out)
        return self.drop(out)


def att_mask(att, causal_mask, attention_mask):
    B, H, T, N = att.size()
    # Apply causal mask
    causal_mask = causal_mask[:, :, :T, :T]
    att = att.masked_fill(causal_mask == 0, float("-inf"))

    # Apply attention mask if provided
    if attention_mask is not None:
        attention_mask = attention_mask.view(B, 1, 1, T)  # Reshape to match dimensions
        att = att.masked_fill(attention_mask == 0, float("-inf"))
    return att


class SelfAttention(nn.Module):
    def __init__(
        self,
        heads,
        embed_size,
        bias=False,
        max_history=256,
        dropout=0.0,
        flash=True,
        norm_class="LayerNorm",
        non_linearity="GELU",
        eps_norm=1e-5,
        layer_id=-1,
    ):
        super().__init__()
        self.att = nn.Linear(embed_size, 3 * embed_size, bias=bias)
        self.c_proj = nn.Linear(embed_size, embed_size, bias=bias)
        self.dropout_prob = dropout
        self.att_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self.heads = heads
        self.bias = bias
        norm_class = getattr(sys.modules[__name__], norm_class)
        self.norm = norm_class(self.embed_size, bias=self.bias, eps=eps_norm)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(max_history, max_history)).view(1, 1, max_history, max_history)
        )
        self.mlp = MLP(
            self.embed_size,
            self.bias,
            self.dropout_prob,
            non_linearity=non_linearity,
            norm_class=norm_class,
            eps_norm=eps_norm,
        )
        self.flash = flash

    def forward(self, x, attention_mask=None, freqs=None):
        B, T, C = x.size()
        x_norm = self.norm(x)
        qkv = self.att(x_norm).split(self.embed_size, dim=2)
        q, k, v = map(lambda t: rearrange(t, "B T (H E) -> B H T E", H=self.heads), qkv)
        if freqs is not None:
            q_nope, q_pe = torch.split(q, [q.shape[-1] - freqs.shape[1] * 2, freqs.shape[1] * 2], dim=-1)
            assert q_nope.shape[-1] > 0 and q_pe.shape[-1] > 0
            q_pe = apply_rotary_emb(q_pe, freqs)
            q = torch.cat([q_nope, q_pe], dim=-1)
            k_nope, k_pe = torch.split(k, [k.shape[-1] - freqs.shape[1] * 2, freqs.shape[1] * 2], dim=-1)
            k_pe = apply_rotary_emb(k_pe, freqs)
            k = torch.cat([k_nope, k_pe], dim=-1)
        if self.flash:
            attention_mask = attention_mask.view(B, 1, 1, T).bool() if attention_mask is not None else None
            att = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=self.dropout_prob if self.training else 0, is_causal=True
            )
        else:
            scale = k.shape[-1] ** -0.5
            att = torch.einsum("B H T E, B H N E -> B H T N", q, k) * scale
            att = att_mask(att, self.causal_mask, attention_mask)
            att = torch.softmax(att, dim=-1)
            att = self.att_dropout(att)
            att = torch.einsum("B H T N, B H N E -> B H T E", att, v)

        att = rearrange(att, "B H T E -> B T (H E)")
        att = self.res_dropout(self.c_proj(att))

        x = x + att
        x = x + self.mlp(x)
        return x
