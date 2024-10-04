import unittest

import torch
import torch.autograd

from tk.vector_addition_module import add_vectors_with_autograd


class TestAddVectors(unittest.TestCase):
    def test_forward(self):
        # Create test tensors for the forward pass
        x = torch.randn(64, 1000, device="cuda", requires_grad=False)
        y = torch.randn(64, 1000, device="cuda", requires_grad=False)

        # Compute result using the Triton-powered add_vectors_with_autograd
        output = add_vectors_with_autograd(x, y)

        # Expected result from simple PyTorch addition
        expected_output = x + y

        # Assert that the Triton kernel result matches the expected result
        self.assertTrue(
            torch.allclose(output, expected_output),
            msg="The forward pass result does not match expected output.",
        )

    def test_backward(self):
        # Create test tensors for the backward pass
        x = torch.randn(64, 1000, device="cuda", requires_grad=True)
        y = torch.randn(64, 1000, device="cuda", requires_grad=True)

        # Compute result using the Triton-powered add_vectors_with_autograd
        output = add_vectors_with_autograd(x, y)

        # Arbitrary loss function
        loss = output.sum()
        loss.backward()

        # The gradient of addition is just 1 for both inputs, so grad_x and grad_y should be ones
        expected_grad_x = torch.ones_like(x)
        expected_grad_y = torch.ones_like(y)

        # Assert that the computed gradients match the expected values (ones)
        self.assertTrue(
            torch.allclose(x.grad, expected_grad_x),
            msg="The gradient for x does not match expected output.",
        )
        self.assertTrue(
            torch.allclose(y.grad, expected_grad_y),
            msg="The gradient for y does not match expected output.",
        )


if __name__ == "__main__":
    unittest.main()
