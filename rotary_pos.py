# based on https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

import torch
import math
import matplotlib.pyplot as plt


def precompute_freqs(
    qk_rope_dim=64,
    max_seq_len=1024,
    original_seq_len=1024,
    beta_fast=32,
    beta_slow=1,
    rope_theta=10000,
    rope_factor=40,
):
    dim = qk_rope_dim
    assert dim % 2 == 0
    seqlen = max_seq_len
    beta_fast = beta_fast
    beta_slow = beta_slow
    base = rope_theta
    factor = rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        # NOTE: Frequencies tensor has size dim/2; operate on half-dimension indices for alignment.
        half_dim = dim // 2  # in deepseek code this is dim, but we want to correct over half-dim
        return half_dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        half_dim = dim // 2  # Correcting over half-dimension not as in deepseek code
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, half_dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Standard RoPE base frequencies; index over half dimension
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)  # (seqlen, dim/2)
    # Instead of returning cos and sin separately, stack them along the last dimension
    cos = torch.cos(freqs)  # (seqlen, dim/2)
    sin = torch.sin(freqs)  # (seqlen, dim/2)
    freqs_cis = torch.stack([cos, sin], dim=-1)  # (seqlen, dim/2, 2)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings using stacked cos/sin.

    Shapes:
        x: (B, H, T, C)
        freqs_cis:
            - (T, C/2, 2) for standard 1D RoPE (broadcasts over batch/heads)
            - (B, T, C/2, 2) for per-sample positions (broadcasts over heads)
            - (B, H, T, C/2, 2) for fully per-head freqs

    Returns:
        Tensor of shape (B, H, T, C)
    """

    dtype = x.dtype
    assert x.dim() == 4, "x must be a 4D tensor"
    assert freqs_cis.shape[-1] == 2, "freqs_cis last dim must be 2 (cos,sin)"

    B, H, T, C = x.shape
    half = C // 2
    assert C % 2 == 0, "Rotary dim C must be even"

    # Slice freqs along the sequence dimension, keeping the last two dims (half, 2)
    if freqs_cis.dim() == 3:
        # (T, half, 2)
        freqs_cis = freqs_cis[:T, :, :]
        assert freqs_cis.shape[0] == T
        assert freqs_cis.shape[1] == half
        cos = freqs_cis[..., 0].view(1, 1, T, half)
        sin = freqs_cis[..., 1].view(1, 1, T, half)
    elif freqs_cis.dim() == 4:
        # (B, T, half, 2)
        freqs_cis = freqs_cis[:, :T, :, :]
        assert freqs_cis.shape[0] == B
        assert freqs_cis.shape[1] == T
        assert freqs_cis.shape[2] == half
        cos = freqs_cis[..., 0].unsqueeze(1)  # (B, 1, T, half)
        sin = freqs_cis[..., 1].unsqueeze(1)  # (B, 1, T, half)
    elif freqs_cis.dim() == 5:
        # (B, H, T, half, 2)
        freqs_cis = freqs_cis[:, :, :T, :, :]
        assert freqs_cis.shape[0] == B
        assert freqs_cis.shape[1] == H
        assert freqs_cis.shape[2] == T
        assert freqs_cis.shape[3] == half
        cos = freqs_cis[..., 0]  # (B, H, T, half)
        sin = freqs_cis[..., 1]  # (B, H, T, half)
    else:
        raise ValueError(f"Unsupported freqs_cis rank {freqs_cis.dim()} for apply_rotary_emb")

    x_float = x.float().contiguous().view(B, H, T, half, 2)
    x_real, x_imag = x_float.unbind(dim=-1)  # (B, H, T, half)
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    y = torch.stack([out_real, out_imag], dim=-1).flatten(3)
    return y.to(dtype)


def plot_freqs_cis(freqs_cis: torch.Tensor, num_points: int = 1024):
    """
    Plots the cosine and sine parts of the first few frequency components.

    Args:
        freqs_cis (torch.Tensor): Precomputed cosine and sine values for positional embeddings, stacked.
        num_points (int): Number of points to plot from the sequence length.
    """
    cos = freqs_cis[..., 0]
    # sin = freqs_cis[..., 1]  # Uncomment to plot sine as well
    plt.figure(figsize=(10, 5))
    for i in range(10, cos.shape[1]):
        plt.plot(cos[:num_points, i], label=f"real part {i}")
    plt.title("Real part of Rotary Embedding")
    plt.xlabel("Sequence Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("freqs_cis_plot.png")


if __name__ == "__main__":
    # Example usage
    freqs_cis = precompute_freqs()
    plot_freqs_cis(freqs_cis)
    # Example usage of apply_rotary_emb
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    # Create a random tensor to represent input embeddings
    x = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # Apply rotary embeddings
    rotary_embedded_x = apply_rotary_emb(x, freqs_cis)
    print("Rotary embedded tensor shape:", rotary_embedded_x.shape)
