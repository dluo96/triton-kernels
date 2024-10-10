import pytest
import torch
import torch.autograd

from tk.vector_addition_module import AddVectors


class TestAddVectors:
    @pytest.mark.skipif(torch.cuda.is_available() is False, reason="Requires CUDA GPU")
    @pytest.mark.parametrize("B, T", [(64, 1000), (331, 487)])
    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            (torch.bfloat16, 1e-8, 1e-5),
            (torch.float32, 1e-8, 1e-5),
        ],
    )
    def test_forward(
        self, B: int, T: int, dtype: torch.dtype, atol: float, rtol: float
    ):
        x = torch.randn(B, T, dtype=dtype, device="cuda", requires_grad=False)
        y = torch.randn(B, T, dtype=dtype, device="cuda", requires_grad=False)

        out = AddVectors.apply(x, y)
        expected_out = x + y

        assert torch.allclose(out, expected_out, atol=atol, rtol=rtol)

    @pytest.mark.skipif(torch.cuda.is_available() is False, reason="Requires CUDA GPU")
    @pytest.mark.parametrize("B, T", [(64, 1000), (331, 487)])
    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            (torch.bfloat16, 1e-8, 1e-5),
            (torch.float32, 1e-8, 1e-5),
        ],
    )
    def test_backward(
        self, B: int, T: int, dtype: torch.dtype, atol: float, rtol: float
    ):
        x = torch.randn(B, T, dtype=dtype, device="cuda", requires_grad=True)
        y = torch.randn(B, T, dtype=dtype, device="cuda", requires_grad=True)

        # Compute sum using Triton-powered add_vectors_with_autograd
        output = AddVectors.apply(x, y)

        # Arbitrary loss function so we can do backprop
        loss = output.sum()
        loss.backward()

        # For `f = x + y`, ∂f/∂x = 1 and ∂f/∂y = 1
        expected_grad_x = torch.ones_like(x)
        expected_grad_y = torch.ones_like(y)

        assert torch.allclose(x.grad, expected_grad_x, atol=atol, rtol=rtol)
        assert torch.allclose(y.grad, expected_grad_y, atol=atol, rtol=rtol)
