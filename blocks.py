from torch import nn
import torch
import torch.nn.functional as F
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
        grad_input[input.abs() > 1] = grad_output[input.abs() > 1] * (1 - torch.tanh(input[input.abs() > 1]) ** 2)
        return grad_input


class SoftGradHardTanh(nn.Module):
    def forward(self, input):
        return SoftGradHardTanhFunction.apply(input)

class SoftGradHardSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.clamp((input+1)/2, min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        limit = 1
        grad_input[input.abs() <= limit] = grad_output[input.abs() <= limit]
        grad_input[input.abs() > limit] = grad_output[input.abs() > limit] * (1 - torch.tanh(input[input.abs() > limit]) ** 2)
        return grad_input


class SoftGradHardSigmoid(nn.Module):
    def forward(self, input):
        return SoftGradHardSigmoidFunction.apply(input)


def vis(model):
    import matplotlib.pyplot as plt
    input_tensor = torch.arange(-4, 4, 0.04, requires_grad=True)
    output = model(input_tensor)
    output_sum = output.sum()
    output_sum.backward()
    gradient = input_tensor.grad
    # Visualize the gradient
    plt.figure(figsize=(8, 6))
    plt.plot(input_tensor.detach().numpy(), output.detach().numpy(), color='blue', label=model.__class__.__name__)
    plt.plot(input_tensor.detach().numpy(), gradient.detach().numpy(), color='red', label="Gradient "+model.__class__.__name__)
    plt.legend()
    plt.xlabel('Input Index')
    plt.ylabel('Gradient')
    plt.show()

if __name__ == "__main__":
    vis(SoftGradHardTanh())
    vis(SoftGradHardSigmoid())