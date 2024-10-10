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
        a = x.clone().detach().requires_grad_(True)
        b = y.clone().detach().requires_grad_(True)

        # Forward and backward passes in Triton
        output = AddVectors.apply(x, y)
        loss = output.sum()  # Arbitrary loss function so we can backprop
        loss.backward()

        # Forward and backward passes in PyTorch
        loss = (a + b).sum()
        loss.backward()

        assert torch.allclose(x.grad, a.grad, atol=atol, rtol=rtol)
        assert torch.allclose(y.grad, b.grad, atol=atol, rtol=rtol)
