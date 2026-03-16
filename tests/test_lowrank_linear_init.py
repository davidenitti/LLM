import math

import torch

from blocks.basic_transformer import LowRankLinear


def _assert_close_rel(actual: float, expected: float, rel_tol: float, *, msg: str):
    assert expected != 0.0
    rel_err = abs(actual - expected) / abs(expected)
    assert rel_err <= rel_tol, f"{msg}: actual={actual:.6g} expected={expected:.6g} rel_err={rel_err:.3g}"


def test_lowranklinear_fused_weight_std_matches_initializer_range():
    # Large dims make the empirical std stable, while still fast on CPU.
    torch.manual_seed(1)

    in_features = 900
    out_features = 2700
    rank = 720
    init_std = 0.02

    layer = LowRankLinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=True,
        init_std=init_std,
    )

    with torch.no_grad():
        w_fused, b_fused = layer.fused_weight_bias()

        # Bias is initialized to 0
        assert b_fused is not None
        assert float(b_fused.abs().max()) == 0.0

        fused_std = float(w_fused.std(unbiased=False))
        fused_mean = float(w_fused.mean())

    # Mean should be near 0 and std should match the requested init_std.
    # Tolerances are set to be robust across platforms while still catching regressions.
    assert abs(fused_mean) < init_std * 0.02
    _assert_close_rel(fused_std, init_std, rel_tol=0.08, msg="fused weight std")
    print(f"Fused weight mean: {fused_mean:.6g}, std: {fused_std:.6g}")


def test_lowranklinear_factor_std_scales_as_rank_quarter_power():
    torch.manual_seed(0)

    in_features = 256
    out_features = 512
    rank = 64
    init_std = 0.02

    layer = LowRankLinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=False,
        init_std=init_std,
    )

    expected_factor_std = math.sqrt(init_std) / (rank**0.25)
    with torch.no_grad():
        w1_std = float(layer.w1.weight.std(unbiased=False))
        w2_std = float(layer.w2.weight.std(unbiased=False))

    _assert_close_rel(w1_std, expected_factor_std, rel_tol=0.10, msg="w1 factor std")
    _assert_close_rel(w2_std, expected_factor_std, rel_tol=0.10, msg="w2 factor std")


def test_lowranklinear_forward_matches_fused_linear():
    torch.manual_seed(0)

    batch = 4
    seq = 7
    in_features = 128
    out_features = 256
    rank = 32
    init_std = 0.02

    layer = LowRankLinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=True,
        init_std=init_std,
    )
    fused = layer.fuse()

    x = torch.randn(batch, seq, in_features)

    layer.eval()
    fused.eval()

    with torch.no_grad():
        y = layer(x)
        y_fused = fused(x)

    # Allow tiny numerical differences due to different matmul association.
    torch.testing.assert_close(y, y_fused, rtol=1e-5, atol=1e-6)


def test_lowranklinear_fuse_preserves_dtype_and_device():
    layer = LowRankLinear(
        in_features=16,
        out_features=8,
        rank=4,
        bias=True,
        init_std=0.02,
    ).to(dtype=torch.float64)

    fused = layer.fuse()

    assert fused.weight.dtype == layer.w1.weight.dtype
    assert fused.weight.device == layer.w1.weight.device
    assert fused.bias is not None
    assert fused.bias.dtype == layer.w2.bias.dtype
    assert fused.bias.device == layer.w2.bias.device
