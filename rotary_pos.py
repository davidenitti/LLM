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
) -> torch.Tensor:
    dim = qk_rope_dim
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
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

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
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

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

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs) # (seqlen, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied. (B, H, T, C)
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings. (T,C/2)

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    assert x.dim() == 4, "x must be a 4D tensor"
    freqs_cis = freqs_cis[:x.size(2), :]
    assert x.size(2) == freqs_cis.size(0), "x and freqs_cis must have the same sequence length"
    assert x.size(3) == freqs_cis.size(1)*2
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, 1, x.size(2), x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)



def plot_freqs_cis(freqs_cis: torch.Tensor, num_points: int = 100):
    """
    Plots the real and imaginary parts of the first few frequency components.

    Args:
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.
        num_points (int): Number of points to plot from the sequence length.
    """
    real_part = freqs_cis.real

    plt.figure(figsize=(10, 5))
    for i in range(10, real_part.shape[1]):
        plt.plot(real_part[:, i], label=f"real_part {i}")
    # plt.plot(imag_part, label="Imaginary Part")
    plt.title("Real and Imaginary Parts of freqs_cis")
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