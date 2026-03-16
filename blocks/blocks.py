import sys
from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SoftGradHardTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.clamp(input, min=-1, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() <= 1] = grad_output[input.abs() <= 1]
        grad_input[input.abs() > 1] = grad_output[input.abs() > 1] * (
            1 - torch.tanh(input[input.abs() > 1]) ** 2
        )
        return grad_input


class SoftGradHardTanh(nn.Module):
    def forward(self, input):
        return SoftGradHardTanhFunction.apply(input)


class SoftGradHardSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.clamp((input + 1) / 2, min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        limit = 1
        grad_input[input.abs() <= limit] = grad_output[input.abs() <= limit]
        grad_input[input.abs() > limit] = grad_output[input.abs() > limit] * (
            1 - torch.tanh(input[input.abs() > limit]) ** 2
        )
        return grad_input


class SoftGradHardSigmoid(nn.Module):
    def forward(self, input):
        return SoftGradHardSigmoidFunction.apply(input)


class ReshapeConv(nn.Module):
    def __init__(
        self,
        in_features,  # total number of input channels
        out_features,  # total number of output channels
        channels,
        height,
        kernel_size,
        bias,
    ):
        super(ReshapeConv, self).__init__()

        if channels is not None:
            self.C = channels
            assert self.C > 0, f"channels must be > 0, got {self.C}"
            assert (
                in_features % self.C == 0
            ), f"in_features must be divisible by channels: {in_features} % {self.C} != 0"
            self.H = int((in_features // self.C) ** 0.5)
        elif height is not None:
            self.H = height
            assert self.H > 0, f"height must be > 0, got {self.H}"
            assert (
                in_features % (self.H * self.H) == 0
            ), f"in_features must be divisible by height^2: {in_features} % ({self.H}^2) != 0"
            self.C = in_features // self.H // self.H
        else:
            raise ValueError("Either channels or height must be specified")
        self.W = in_features // self.C // self.H
        assert (
            self.H > 0 and self.W > 0
        ), f"Derived spatial shape must be positive, got H={self.H}, W={self.W}"
        assert (
            out_features % (self.H * self.W) == 0
        ), f"out_features must be divisible by H*W: {out_features} % ({self.H}*{self.W}) != 0"
        self.out_C = out_features // (self.H * self.W)
        assert (
            in_features == self.C * self.H * self.W
        ), f"in_features is not compatible with C, H, W: {in_features} != {self.C}*{self.H}*{self.W}"
        self.block = nn.Conv3d(
            self.C,
            self.out_C,
            kernel_size=[1, kernel_size, kernel_size],
            padding=[0, kernel_size // 2, kernel_size // 2],
            bias=bias,
        )
        print(f"ReshapeConv: {in_features} -> {out_features} with C={self.C}, H={self.H}, W={self.W}")

    def forward(self, x):
        B, T, CWH = x.shape
        assert CWH == self.C * self.H * self.W, "Input shape is not compatible"
        x_reshaped = rearrange(x, "B T (C H W) -> B C T H W", C=self.C, H=self.H, W=self.W)
        out = self.block(x_reshaped)
        return rearrange(out, "B C T H W -> B T (C H W)", C=self.out_C, H=self.H, W=self.W)


class ReshapeConvV2(nn.Module):
    def __init__(
        self,
        in_features,  # total number of input channels
        out_features,  # total number of output channels
        channels,
        height,
        kernel_size,
        bias,
    ):
        super().__init__()
        if channels is not None:
            self.C = channels
            assert self.C > 0, f"channels must be > 0, got {self.C}"
            assert (
                in_features % self.C == 0
            ), f"in_features must be divisible by channels: {in_features} % {self.C} != 0"
            self.H = int((in_features // self.C) ** 0.5)
        elif height is not None:
            self.H = height
            assert self.H > 0, f"height must be > 0, got {self.H}"
            assert (
                in_features % (self.H * self.H) == 0
            ), f"in_features must be divisible by height^2: {in_features} % ({self.H}^2) != 0"
            self.C = in_features // self.H // self.H
        else:
            raise ValueError("Either channels or height must be specified")
        self.W = in_features // self.C // self.H
        assert (
            self.H > 0 and self.W > 0
        ), f"Derived spatial shape must be positive, got H={self.H}, W={self.W}"
        self.out_C = self.C
        assert (
            in_features == self.C * self.H * self.W
        ), f"in_features is not compatible with C, H, W: {in_features} != {self.C}*{self.H}*{self.W}"
        self.block = nn.Sequential(
            nn.Conv3d(
                self.C,
                self.out_C,
                kernel_size=[1, kernel_size, kernel_size],
                padding=[0, kernel_size // 2, kernel_size // 2],
                bias=False,
            ),
            nn.GELU(),
        )
        self.linear = nn.Linear(self.out_C * self.H * self.W, out_features, bias=bias)
        print(f"ReshapeConv: {in_features} -> {out_features} with C={self.C}, H={self.H}, W={self.W}")

    def forward(self, x):
        B, T, CWH = x.shape
        assert CWH == self.C * self.H * self.W, "Input shape is not compatible"
        x_reshaped = rearrange(x, "B T (C H W) -> B C T H W", C=self.C, H=self.H, W=self.W)
        out = self.block(x_reshaped)
        out = rearrange(out, "B C T H W -> B T (C H W)", C=self.out_C, H=self.H, W=self.W)
        out = self.linear(out + x)
        return out


def vis(model):
    input_tensor = torch.arange(-4, 4, 0.04, requires_grad=True)
    output = model(input_tensor)
    output_sum = output.sum()
    output_sum.backward()
    gradient = input_tensor.grad
    # Visualize the gradient
    plt.figure(figsize=(8, 6))
    plt.plot(
        input_tensor.detach().numpy(),
        output.detach().numpy(),
        color="blue",
        label=model.__class__.__name__,
    )
    plt.plot(
        input_tensor.detach().numpy(),
        gradient.detach().numpy(),
        color="red",
        label="Gradient " + model.__class__.__name__,
    )
    plt.legend()
    plt.xlabel("Input Index")
    plt.ylabel("Gradient")
    plt.show()


if __name__ == "__main__":
    vis(SoftGradHardTanh())
    vis(SoftGradHardSigmoid())
