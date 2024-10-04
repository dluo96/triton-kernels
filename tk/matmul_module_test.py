import unittest

import torch
import torch.autograd

from tk.matmul_module import MatMul


class TestMatMul(unittest.TestCase):
    def test_forward(self):
        x = torch.ones(3, 4, device="cuda", requires_grad=False)
        y = torch.ones(4, 5, device="cuda", requires_grad=False)
        self.assertTrue(torch.allclose(MatMul.apply(x, y), x @ y))

    def test_backward(self):
        x = torch.randn(3, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 5, device="cuda", requires_grad=True)
        a = x.clone().detach().requires_grad_(True)
        b = y.clone().detach().requires_grad_(True)

        # Compute result using the Triton-powered add_vectors_with_autograd
        output = MatMul.apply(x, y)

        # Get gradients with Triton kernel in computation graph
        loss = output.sum()
        loss.backward()

        # Get gradient with standard PyTorch
        c = a @ b
        c.sum().backward()

        self.assertTrue(torch.allclose(x.grad, a.grad, atol=1e-2))
        self.assertTrue(torch.allclose(y.grad, b.grad, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
