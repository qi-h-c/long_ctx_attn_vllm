import torch
from. import _C

@torch.library.register_fake("extension_cuda_add::add_cuda")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float32)
    torch._check(b.dtype == torch.float32)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def _backward(ctx, grad):
    return grad, grad


def _setup_context(ctx, inputs, output):
    pass


torch.library.register_autograd(
    "extension_cuda_add::add_cuda", _backward, setup_context=_setup_context
)