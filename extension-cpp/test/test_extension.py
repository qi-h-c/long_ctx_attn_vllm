import torch
import extension_cpp  # 假设你已经正确安装了该扩展模块
import unittest


class TestAddOperator(unittest.TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        return [
            [make_tensor(3), make_tensor(3)],  # Simple tensors with the same size
            [make_tensor(20), make_tensor(20)],  # Larger tensors
        ]

    def test_correctness_cpu(self):
        """Test add operator on CPU"""
        samples = self.sample_inputs('cpu')
        for args in samples:
            result = extension_cpp.ops.add(*args)  # Call the custom add operator
            expected = torch.add(*args)  # Use PyTorch's built-in add for reference
            torch.testing.assert_close(result, expected)  # Check if results match

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        """Test add operator on CUDA (GPU)"""
        samples = self.sample_inputs('cuda')
        for args in samples:
            result = extension_cpp.ops.add(*args)  # Call the custom add operator
            expected = torch.add(*args)  # Use PyTorch's built-in add for reference
            torch.testing.assert_close(result, expected)  # Check if results match

    def test_gradients_cpu(self):
        """Test add operator's gradients on CPU"""
        samples = self.sample_inputs('cpu', requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = extension_cpp.ops.add(*args)  # Forward pass using custom add operator
            grad_out = torch.randn_like(out)  # Random gradient for backward pass
            result = torch.autograd.grad(out, diff_tensors, grad_out)  # Compute gradients
            expected = torch.autograd.grad(torch.add(*args), diff_tensors, grad_out)  # Expected gradients
            torch.testing.assert_close(result, expected)  # Check if gradients match

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        """Test add operator's gradients on CUDA (GPU)"""
        samples = self.sample_inputs('cuda', requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = extension_cpp.ops.add(*args)  # Forward pass using custom add operator
            grad_out = torch.randn_like(out)  # Random gradient for backward pass
            result = torch.autograd.grad(out, diff_tensors, grad_out)  # Compute gradients
            expected = torch.autograd.grad(torch.add(*args), diff_tensors, grad_out)  # Expected gradients
            torch.testing.assert_close(result, expected)  # Check if gradients match

if __name__ == '__main__':
    unittest.main()
