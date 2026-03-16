import torch
import torch.nn.functional as torch_F

import blocks.conv_block as conv_block
from blocks.conv_block import UnetNet, UnetNet3D


def test_unet_net_forward_shape_and_grad():
    model = UnetNet(
        in_channels=3,
        out_channels=5,
        base_channels=8,
        num_layers=3,
    )
    x = torch.randn(2, 3, 64, 48, requires_grad=True)
    y = model(x)

    assert y.shape == (2, 5, 64, 48)

    y.mean().backward()
    assert x.grad is not None
    assert model.final.weight.grad is not None


def test_unet_net_symmetric_padding_and_crop(monkeypatch):
    model = UnetNet(
        in_channels=1,
        out_channels=2,
        base_channels=4,
        num_layers=3,
    )
    x = torch.randn(1, 1, 31, 35)
    captured = {}

    orig_pad = torch_F.pad

    def pad_wrapper(input, pad, mode="constant", value=0):
        captured["pad"] = tuple(pad)
        return orig_pad(input, pad, mode=mode, value=value)

    monkeypatch.setattr(conv_block.F, "pad", pad_wrapper)
    y = model(x)

    assert y.shape == (1, 2, 31, 35)

    factor = 2 ** len(model.encoders)
    pad_h = (factor - (x.shape[-2] % factor)) % factor
    pad_w = (factor - (x.shape[-1] % factor)) % factor
    expected = (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
    )
    assert captured["pad"] == expected


def test_unet_net3d_forward_shape_and_grad():
    model = UnetNet3D(
        in_channels=2,
        out_channels=4,
        base_channels=6,
        num_layers=2,
    )
    x = torch.randn(2, 2, 10, 32, 24, requires_grad=True)
    y = model(x)

    assert y.shape == (2, 4, 10, 32, 24)

    y.mean().backward()
    assert x.grad is not None
    assert model.final.weight.grad is not None


def test_unet_net3d_symmetric_padding_and_crop(monkeypatch):
    model = UnetNet3D(
        in_channels=1,
        out_channels=3,
        base_channels=4,
        num_layers=2,
    )
    x = torch.randn(1, 1, 7, 19, 23)
    captured = {}

    orig_pad = torch_F.pad

    def pad_wrapper(input, pad, mode="constant", value=0):
        captured["pad"] = tuple(pad)
        return orig_pad(input, pad, mode=mode, value=value)

    monkeypatch.setattr(conv_block.F, "pad", pad_wrapper)
    y = model(x)

    assert y.shape == (1, 3, 7, 19, 23)

    factor = 2 ** len(model.encoders)
    pad_d = (factor - (x.shape[-3] % factor)) % factor
    pad_h = (factor - (x.shape[-2] % factor)) % factor
    pad_w = (factor - (x.shape[-1] % factor)) % factor
    expected = (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_d // 2,
        pad_d - pad_d // 2,
    )
    assert captured["pad"] == expected
