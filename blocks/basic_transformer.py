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
from utils.checks import check_tensors, check_model
from rotary_pos import apply_rotary_emb
from torch.nn import GELU, RMSNorm
import warnings
from blocks.blocks import ReshapeConv, ReshapeConvV2


class SwiGLU(nn.Module):
    """SwiGLU activation: split last dim into (x, gate) halves -> silu(gate) * x.
    Input expects shape [..., 2 * hidden]. Output shape [..., hidden]."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * x


def get_norm(norm_class, dim, bias, eps):
    if issubclass(norm_class, RMSNorm):
        return norm_class(dim, eps=eps)
    return norm_class(dim, bias=bias, eps=eps)


class LowRankLinear(nn.Module):
    """Low-rank factorization of a Linear layer: x -> W2(W1(x)).

    This is equivalent to a single Linear with rank <= r, but parameterized as two matrices.
    It supports fusing into a single nn.Linear for inference.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool, *, init_std: float = 0.02):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.bias = bool(bias)

        self.w1 = nn.Linear(self.in_features, self.rank, bias=False)
        self.w2 = nn.Linear(self.rank, self.out_features, bias=self.bias)

        # Tag children so the global model init can skip re-initializing them.
        self.w1._skip_init = True
        self.w2._skip_init = True

        self.reset_parameters(init_std=init_std)

    def reset_parameters(self, *, init_std: float = 0.02):
        # Match the *effective* fused weight std to init_std.
        # If W = W2 @ W1, with iid Normal(0, s^2) for both factors, then
        # Var(W_ij) = rank * s^4. We want std(W_ij) ~= init_std, i.e. Var ~= init_std^2:
        #   rank * s^4 = init_std^2  =>  s = sqrt(init_std) / rank^(1/4)
        s = math.sqrt(float(init_std)) / (self.rank**0.25)
        self.w1.weight.data.normal_(mean=0.0, std=s)
        self.w2.weight.data.normal_(mean=0.0, std=s)
        if self.w2.bias is not None:
            self.w2.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x))

    @torch.no_grad()
    def fused_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        w = self.w2.weight @ self.w1.weight
        b = self.w2.bias
        return w, b

    @torch.no_grad()
    def fuse(self) -> nn.Linear:
        w, b = self.fused_weight_bias()
        fused = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias,
            device=w.device,
            dtype=w.dtype,
        )
        fused.weight.copy_(w)
        if b is not None:
            fused.bias.copy_(b)
        return fused


def get_linear_layer(linear_layer_type, in_features, out_features, bias, **kwargs):
    rank_ratio = kwargs.get("rank_ratio", 1.0)
    if linear_layer_type == "standard":
        if rank_ratio == 1.0:
            return nn.Linear(in_features, out_features, bias=bias)
        else:
            min_size = min(in_features, out_features)
            rank = max(1, round(min_size * rank_ratio))
            print("rank:", rank, rank_ratio)
            return LowRankLinear(in_features=in_features, out_features=out_features, rank=rank, bias=bias)
    elif linear_layer_type == "conv2d":
        return ReshapeConv(
            in_features=in_features,
            out_features=out_features,
            channels=kwargs.get("conv2d_channels", None),
            height=kwargs.get("conv2d_height", None),
            kernel_size=kwargs.get("conv2d_kernel_size", 5),
            bias=bias,
        )
    elif linear_layer_type == "conv2dv2":
        if in_features < out_features:
            print("Using standard linear for linear layer", in_features, out_features)
            return nn.Linear(in_features, out_features, bias=bias)
        else:
            print(
                "Using ReshapeConvV2 for linear layer",
                in_features,
                out_features,
                kwargs.get("conv2d_channels", None),
                kwargs.get("conv2d_height", None),
                kwargs.get("conv2d_kernel_size", 5),
            )
            return ReshapeConvV2(
                in_features=in_features,
                out_features=out_features,
                channels=kwargs.get("conv2d_channels", None),
                height=kwargs.get("conv2d_height", None),
                kernel_size=kwargs.get("conv2d_kernel_size", 5),
                bias=bias,
            )
    else:
        raise ValueError(f"Unsupported linear layer type: {linear_layer_type}")


class MLP(nn.Module):
    def __init__(
        self,
        embed_size,
        bias,
        dropout,
        non_linearity="GELU",
        norm_class=LayerNorm,
        eps_norm=1e-5,
        out_channels=None,
        linear_layer="standard",
        **kwargs,
    ):
        super().__init__()
        self.embed_size = embed_size
        if out_channels is None:
            out_channels = self.embed_size
        self.bias = bias
        non_linearity_fn = getattr(sys.modules[__name__], non_linearity)()
        multiplier = kwargs.get("mlp_multiplier", 4)
        self.lin_block = nn.Sequential(
            get_norm(norm_class, self.embed_size, self.bias, eps_norm),
            get_linear_layer(
                linear_layer, self.embed_size, multiplier * self.embed_size, bias=self.bias, **kwargs
            ),
            non_linearity_fn,
        )
        self.c_proj = get_linear_layer(
            linear_layer, multiplier * self.embed_size, out_channels, bias=self.bias, **kwargs
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.lin_block(x)
        out = self.c_proj(out)
        return self.drop(out)


class MLPv2(nn.Module):
    """Enhanced MLP with internal RMSNorm (or provided norm), SwiGLU option, residual-friendly design."""

    def __init__(
        self,
        embed_size,
        bias,
        dropout,
        non_linearity="GELU",
        norm_class=RMSNorm,
        eps_norm=1e-5,
        out_channels=None,
        linear_layer="standard",
        **kwargs,
    ):
        super().__init__()
        self.embed_size = embed_size
        if out_channels is None:
            out_channels = self.embed_size
        self.bias = bias
        self.non_linearity_name = non_linearity
        self.norm = get_norm(norm_class, self.embed_size, self.bias, eps_norm)
        self.drop = nn.Dropout(dropout)
        multiplier = kwargs.get("mlp_multiplier", 4)
        if non_linearity == "SwiGLU":
            # Keep parameter budget parity with 4x GELU MLP: hidden_dim = 8/3 * d
            self.hidden_dim = int(self.embed_size * multiplier * 2 / 3)
            self.fc = get_linear_layer(
                linear_layer, self.embed_size, 2 * self.hidden_dim, bias=self.bias, **kwargs
            )
            self.act = SwiGLU()
            self.c_proj = get_linear_layer(
                linear_layer, self.hidden_dim, out_channels, bias=self.bias, **kwargs
            )
        else:
            act = getattr(sys.modules[__name__], non_linearity)()
            self.fc = get_linear_layer(
                linear_layer, self.embed_size, multiplier * self.embed_size, bias=self.bias, **kwargs
            )
            self.act = act
            self.c_proj = get_linear_layer(
                linear_layer, multiplier * self.embed_size, out_channels, bias=self.bias, **kwargs
            )

    def forward(self, x):
        x_in = self.norm(x)
        out = self.fc(x_in)
        out = self.act(out)
        out = self.c_proj(out)
        return self.drop(out)


def att_mask(att, causal_mask, attention_mask, skip=1):
    B, H, T, N = att.size()
    N *= skip
    # Apply causal mask
    if causal_mask is not None:
        assert T <= N
        assert N <= causal_mask.size(
            2
        ), f"{N}>{causal_mask.size(2)} Causal mask must be large enough for the attention dimensions"
        diff_steps = N - T
        causal_mask = causal_mask[:, :, :N, :N]
        causal_mask = causal_mask[:, :, diff_steps:, :]  # Ensure the mask matches the attention dimensions
        if skip > 1:
            causal_mask = causal_mask[:, :, :, ::skip]  # Skip connections in attention
        att = att.masked_fill(causal_mask == 0, float("-inf"))

    # Apply attention mask if provided
    if attention_mask is not None:
        warnings.warn(
            "attention_mask is not None — this code path is untested",
            UserWarning,
            stacklevel=2,
        )
        attention_mask = attention_mask.view(B, 1, 1, N)  # Reshape to match dimensions
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
        causal=True,
        custom_weight=False,
        linear_layer="standard",
        **kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.att = get_linear_layer("standard", embed_size, 3 * embed_size, bias, **kwargs)
        # self.att = nn.Linear(embed_size, 3 * embed_size, bias=bias)
        self.c_proj = nn.Linear(embed_size, embed_size, bias=bias)
        self.dropout_prob = dropout
        self.att_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self.heads = heads
        self.bias = bias
        norm_class_ref = getattr(sys.modules[__name__], norm_class)
        self.norm = get_norm(norm_class_ref, self.embed_size, self.bias, eps_norm)
        if self.causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_history, max_history)).view(1, 1, max_history, max_history),
            )
        else:
            self.causal_mask = None
        self.mlp = MLP(
            self.embed_size,
            self.bias,
            self.dropout_prob,
            non_linearity=non_linearity,
            norm_class=norm_class_ref,
            eps_norm=eps_norm,
            linear_layer=linear_layer,
            **kwargs,
        )
        self.flash = flash
        self.use_custom_weight = custom_weight
        if self.use_custom_weight:
            # Produce per-position weights for a local temporal aggregation over value vectors.
            self.window_size = 7
            self.custom_weights = nn.Sequential(
                nn.Linear(embed_size // self.heads, self.window_size),
            )

    def forward(
        self,
        x,
        attention_mask=None,
        freqs=None,
        k_cache=None,
        v_cache=None,
        return_cache=False,
    ):
        B, T, C = x.size()
        if k_cache is not None and T == 1:
            assert v_cache is not None
            causal_mask = None
            is_causal = False
        else:
            assert k_cache is None and v_cache is None
            causal_mask = self.causal_mask
            is_causal = self.causal
        x_norm = self.norm(x)
        qkv = self.att(x_norm).split(self.embed_size, dim=2)
        q, k, v = map(lambda t: rearrange(t, "B T (H E) -> B H T E", H=self.heads), qkv)
        if freqs is not None:
            if k_cache is not None:
                cache_len = k_cache.shape[-2]
                # If freqs is a global lookup table (T_total, half, 2) we slice by cache length.
                # If freqs is already aligned to the current input chunk (e.g. (B, T, half, 2)
                # from use_rot_emb_2d), we must NOT slice away the only timestep.
                if freqs.dim() == 3:
                    freqs = freqs[cache_len : cache_len + T]
                assert freqs.shape[-3] == T
            rope_half = freqs.shape[-2]
            q_nope, q_pe = torch.split(q, [q.shape[-1] - rope_half * 2, rope_half * 2], dim=-1)
            assert q_nope.shape[-1] > 0 and q_pe.shape[-1] > 0
            q_pe = apply_rotary_emb(q_pe, freqs)
            q = torch.cat([q_nope, q_pe], dim=-1)
            k_nope, k_pe = torch.split(k, [k.shape[-1] - rope_half * 2, rope_half * 2], dim=-1)
            k_pe = apply_rotary_emb(k_pe, freqs)
            k = torch.cat([k_nope, k_pe], dim=-1)
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=-2)
            assert v_cache is not None
            v = torch.cat([v_cache, v], dim=-2)
        else:
            assert v_cache is None
        v_cache_new = v
        if self.use_custom_weight:
            # custom_weights: (B, H, T, W) raw weights for windowed aggregation
            custom_weights = self.custom_weights(k)  # (B, H, T, W)

            # v_c currently has shape (B, H, T, E). We want to replace each v_c[:,:,t,:] with
            # sum_{i=0}^{W-1} w_{t,i} * v_c[:,:, t - (W-1) + i, :]
            # Implement via left replicate padding then unfold.
            B_, H_, T_, E_ = v.shape
            pad_len = self.window_size - 1
            # Zero padding for positions before t=0
            left_pad = v.new_zeros(B_, H_, pad_len, E_)
            v_pad = torch.cat([left_pad, v], dim=2)  # (B,H,T+pad_len,E)
            # Create sliding windows of length W ending at each original timestep t
            v_windows = v_pad.unfold(dimension=2, size=self.window_size, step=1)  # (B,H,T,E,W)
            # Weighted sum over window dimension
            v = (v_windows * custom_weights.unsqueeze(-2)).sum(dim=4)  # (B,H,T,E)

        if self.flash:
            attention_mask = (
                attention_mask.view(B, 1, 1, k.shape[-2]).bool() if attention_mask is not None else None
            )
            att = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout_prob if self.training else 0,
                is_causal=is_causal,
            )
        else:
            scale = k.shape[-1] ** -0.5
            att = torch.einsum("B H T E, B H N E -> B H T N", q, k) * scale
            att = att_mask(att, causal_mask, attention_mask)
            att = torch.softmax(att, dim=-1)
            att = self.att_dropout(att)
            att = torch.einsum("B H T N, B H N E -> B H T E", att, v)
        att = rearrange(att, "B H T E -> B T (H E)")
        att = self.res_dropout(self.c_proj(att))

        x = x + att
        x = x + self.mlp(x)
        if return_cache:
            return x, k, v_cache_new
        else:
            return x


class SelfAttentionV2(nn.Module):
    def __init__(
        self,
        heads,
        embed_size,
        bias=False,
        max_history=256,
        dropout=0.0,
        flash=True,
        norm_class="RMSNorm",
        non_linearity="GELU",
        eps_norm=1e-5,
        layer_id=-1,
        causal=True,
        qk_norm=False,
        output_norm=False,
        residual_scale_init=0.1,
        linear_layer="standard",
        **kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.att = get_linear_layer("standard", embed_size, 3 * embed_size, bias, **kwargs)
        # self.att = nn.Linear(embed_size, 3 * embed_size, bias=bias)
        self.c_proj = nn.Linear(embed_size, embed_size, bias=bias)
        self.dropout_prob = dropout
        self.att_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self.heads = heads
        self.bias = bias
        norm_class_ref = getattr(sys.modules[__name__], norm_class)
        self.norm = get_norm(norm_class_ref, self.embed_size, self.bias, eps_norm)
        if self.causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_history, max_history)).view(1, 1, max_history, max_history),
            )
        else:
            self.causal_mask = None
        self.mlp = MLPv2(
            self.embed_size,
            self.bias,
            self.dropout_prob,
            non_linearity=non_linearity,
            norm_class=norm_class_ref,
            eps_norm=eps_norm,
            linear_layer=linear_layer,
            **kwargs,
        )
        self.flash = flash
        self.qk_norm_enabled = qk_norm
        if qk_norm:
            head_dim = embed_size // heads
            self.q_norm = RMSNorm(head_dim, eps=eps_norm)
            self.k_norm = RMSNorm(head_dim, eps=eps_norm)
        else:
            self.q_norm = None
            self.k_norm = None
        self.output_norm_enabled = output_norm
        if output_norm:
            self.att_out_norm = get_norm(norm_class_ref, self.embed_size, self.bias, eps_norm)
        else:
            self.att_out_norm = None
        self.gamma_attn = nn.Parameter(torch.ones(1) * residual_scale_init)
        self.gamma_mlp = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(
        self,
        x,
        attention_mask=None,
        freqs=None,
        k_cache=None,
        v_cache=None,
        return_cache=False,
    ):
        B, T, C = x.size()
        if k_cache is not None and T == 1:
            assert v_cache is not None
            causal_mask = None
            is_causal = False
        else:
            assert k_cache is None and v_cache is None
            causal_mask = self.causal_mask
            is_causal = self.causal
        x_norm = self.norm(x)
        qkv = self.att(x_norm).split(self.embed_size, dim=2)
        q, k, v = map(lambda t: rearrange(t, "B T (H E) -> B H T E", H=self.heads), qkv)
        if self.qk_norm_enabled:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if freqs is not None:
            if k_cache is not None:
                cache_len = k_cache.shape[-2]
                if freqs.dim() == 3:
                    freqs = freqs[cache_len : cache_len + T]
                assert freqs.shape[-3] == T
            q_nope, q_pe = torch.split(q, [q.shape[-1] - freqs.shape[-2] * 2, freqs.shape[-2] * 2], dim=-1)
            assert q_nope.shape[-1] > 0 and q_pe.shape[-1] > 0
            q_pe = apply_rotary_emb(q_pe, freqs)
            q = torch.cat([q_nope, q_pe], dim=-1)
            k_nope, k_pe = torch.split(k, [k.shape[-1] - freqs.shape[-2] * 2, freqs.shape[-2] * 2], dim=-1)
            k_pe = apply_rotary_emb(k_pe, freqs)
            k = torch.cat([k_nope, k_pe], dim=-1)
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=-2)
            assert v_cache is not None
            v = torch.cat([v_cache, v], dim=-2)
        else:
            assert v_cache is None
        if self.flash:
            attention_mask = (
                attention_mask.view(B, 1, 1, k.shape[-2]).bool() if attention_mask is not None else None
            )
            att = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout_prob if self.training else 0,
                is_causal=is_causal,
            )
        else:
            scale = k.shape[-1] ** -0.5
            att = torch.einsum("B H T E, B H N E -> B H T N", q, k) * scale
            att = att_mask(att, causal_mask, attention_mask)
            att = torch.softmax(att, dim=-1)
            att = self.att_dropout(att)
            att = torch.einsum("B H T N, B H N E -> B H T E", att, v)
        att = rearrange(att, "B H T E -> B T (H E)")
        if self.output_norm_enabled:
            att = self.att_out_norm(att)
        att = self.res_dropout(self.c_proj(att))
        x = x + self.gamma_attn * att
        mlp_out = self.mlp(x)
        x = x + self.gamma_mlp * mlp_out
        if return_cache:
            return x, k, v
        else:
            return x
