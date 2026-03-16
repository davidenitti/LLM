import pytest
import torch
from einops import rearrange

from blocks.basic_transformer import SelfAttention, SelfAttentionV2
from rotary_pos import precompute_freqs


def _raw_v_projection(module, x):
    x_norm = module.norm(x)
    qkv = module.att(x_norm).split(module.embed_size, dim=2)
    v = qkv[2]
    return rearrange(v, "B T (H E) -> B H T E", H=module.heads)


def _cached_decode(module, x, freqs=None):
    module.eval()
    step_outputs = []
    k_cache = None
    v_cache = None
    for t in range(x.shape[1]):
        out_t, k_cache, v_cache = module(
            x[:, t : t + 1],
            freqs=freqs,
            k_cache=k_cache,
            v_cache=v_cache,
            return_cache=True,
        )
        step_outputs.append(out_t)
    return torch.cat(step_outputs, dim=1), k_cache, v_cache


@pytest.mark.parametrize(
    ("attn_cls", "extra_kwargs"),
    [
        (SelfAttention, {"custom_weight": True}),
        (SelfAttentionV2, {}),
    ],
)
@pytest.mark.parametrize("flash", [False, True])
@pytest.mark.parametrize("use_rope", [False, True])
def test_attention_cache_matches_full_forward(attn_cls, extra_kwargs, flash, use_rope):
    torch.manual_seed(0)
    module = attn_cls(
        heads=2,
        embed_size=8,
        max_history=16,
        dropout=0.0,
        flash=flash,
        **extra_kwargs,
    )
    module.eval()

    x = torch.randn(2, 5, 8)
    freqs = precompute_freqs(qk_rope_dim=2, max_seq_len=16) if use_rope else None

    with torch.no_grad():
        full = module(x, freqs=freqs)
        cached, _, _ = _cached_decode(module, x, freqs=freqs)

    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)


def test_selfattention_custom_weight_cache_keeps_raw_values():
    torch.manual_seed(0)
    module = SelfAttention(
        heads=2,
        embed_size=8,
        max_history=16,
        dropout=0.0,
        flash=False,
        custom_weight=True,
    )
    module.eval()

    x = torch.randn(1, 4, 8)

    expected_v_cache = None
    k_cache = None
    v_cache = None
    with torch.no_grad():
        for t in range(x.shape[1]):
            x_t = x[:, t : t + 1]
            v_t = _raw_v_projection(module, x_t)
            expected_v_cache = v_t if expected_v_cache is None else torch.cat([expected_v_cache, v_t], dim=-2)

            _, k_cache, v_cache = module(
                x_t,
                k_cache=k_cache,
                v_cache=v_cache,
                return_cache=True,
            )

            torch.testing.assert_close(v_cache, expected_v_cache, rtol=1e-6, atol=1e-6)
