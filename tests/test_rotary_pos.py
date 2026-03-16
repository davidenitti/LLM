import torch

from rotary_pos import precompute_freqs, apply_rotary_emb


def test_shapes_and_device_cpu():
    dim = 64
    T = 128
    B, H = 2, 4
    freqs = precompute_freqs(qk_rope_dim=dim, max_seq_len=T)
    assert freqs.shape == (T, dim // 2, 2)

    x = torch.randn(B, H, T, dim)
    y = apply_rotary_emb(x, freqs)
    assert y.shape == x.shape
    # dtype preserved
    assert y.dtype == x.dtype


def test_norm_preservation_per_pair():
    dim = 32
    T = 64
    B, H = 1, 1
    freqs = precompute_freqs(qk_rope_dim=dim, max_seq_len=T)
    x = torch.randn(B, H, T, dim)
    y = apply_rotary_emb(x, freqs)

    # Reshape into pairs (real, imag)
    xr, xi = x.reshape(B, H, T, -1, 2).unbind(-1)
    yr, yi = y.reshape(B, H, T, -1, 2).unbind(-1)

    x_norm = (xr.pow(2) + xi.pow(2)).sum()
    y_norm = (yr.pow(2) + yi.pow(2)).sum()

    assert torch.allclose(x_norm, y_norm, rtol=1e-5, atol=1e-6)


def test_inverse_property():
    # Applying with -freqs should invert the rotation
    dim = 32
    T = 50
    B, H = 2, 3
    freqs = precompute_freqs(qk_rope_dim=dim, max_seq_len=T)
    x = torch.randn(B, H, T, dim)
    y = apply_rotary_emb(x, freqs)

    # Construct negative frequencies by flipping sin sign
    neg_freqs = freqs.clone()
    neg_freqs[..., 1] = -neg_freqs[..., 1]
    z = apply_rotary_emb(y, neg_freqs)

    assert torch.allclose(x, z, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_accepts_batched_freqs_shape_equivalent_to_unbatched():
    # freqs_cis can be provided as (B, T, half, 2) and should match
    # the unbatched (T, half, 2) behavior when each batch uses the same positions.
    dim = 32
    T = 64
    B, H = 3, 2
    freqs = precompute_freqs(qk_rope_dim=dim, max_seq_len=T)
    x = torch.randn(B, H, T, dim)

    y_unbatched = apply_rotary_emb(x, freqs)
    freqs_batched = freqs.unsqueeze(0).expand(B, T, dim // 2, 2).contiguous()
    y_batched = apply_rotary_emb(x, freqs_batched)

    assert y_batched.shape == x.shape
    assert torch.allclose(y_unbatched, y_batched, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_accepts_per_head_freqs_shape_equivalent_to_batched():
    # freqs_cis can be provided as (B, H, T, half, 2) and should match
    # the (B, T, half, 2) behavior when replicated across heads.
    dim = 64
    T = 32
    B, H = 2, 4
    freqs = precompute_freqs(qk_rope_dim=dim, max_seq_len=T)
    x = torch.randn(B, H, T, dim)

    freqs_batched = freqs.unsqueeze(0).expand(B, T, dim // 2, 2).contiguous()
    freqs_per_head = freqs_batched.unsqueeze(1).expand(B, H, T, dim // 2, 2).contiguous()

    y_batched = apply_rotary_emb(x, freqs_batched)
    y_per_head = apply_rotary_emb(x, freqs_per_head)

    assert y_per_head.shape == x.shape
    assert torch.allclose(y_batched, y_per_head, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_dtype_preserved_bfloat16_if_available():
    # apply_rotary_emb internally uses float() but should return original dtype.
    if not hasattr(torch, "bfloat16"):
        return
    dim = 32
    T = 16
    B, H = 2, 2
    freqs = precompute_freqs(qk_rope_dim=dim, max_seq_len=T)
    x = torch.randn(B, H, T, dim).to(torch.bfloat16)
    y = apply_rotary_emb(x, freqs)
    assert y.dtype == x.dtype
