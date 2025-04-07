import torch
from torch import Tensor

__all__ = ["add"]


def add(a: Tensor, b: Tensor) -> Tensor:
    """Performs a + b"""
    return torch.ops.extension_cpp.add.default(a, b)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::add")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = grad  # Gradient for a is just the grad from downstream
    if ctx.needs_input_grad[1]:
        grad_b = grad  # Gradient for b is also just the grad from downstream
    return grad_a, grad_b


def _setup_context(ctx, inputs, output):
    a, b = inputs
    ctx.save_for_backward(a, b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::add", _backward, setup_context=_setup_context)
