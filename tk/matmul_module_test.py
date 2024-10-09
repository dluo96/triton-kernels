import pytest
import torch
import torch.autograd

from tk.matmul_module import MatMul


class TestMatMul:
    @pytest.mark.skipif(torch.cuda.is_available() is False, reason="Requires CUDA GPU")
    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            (torch.float32, 1e-8, 1e-2),
            (torch.bfloat16, 1e-10, 0),
        ],
    )
    def test_forward(self, dtype: torch.dtype, atol: float, rtol: float):
        x = torch.randn(3, 4, dtype=dtype, device="cuda", requires_grad=False)
        y = torch.randn(4, 5, dtype=dtype, device="cuda", requires_grad=False)
        assert torch.allclose(MatMul.apply(x, y), x @ y, atol=atol, rtol=rtol)

    @pytest.mark.skipif(torch.cuda.is_available() is False, reason="Requires CUDA GPU")
    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            (torch.float32, 1e-8, 1e-2),
            (torch.bfloat16, 1e-10, 0),
        ],
    )
    def test_backward(self, dtype: torch.dtype, atol: float, rtol: float):
        x = torch.randn(3, 4, dtype=dtype, device="cuda", requires_grad=True)
        y = torch.randn(4, 5, dtype=dtype, device="cuda", requires_grad=True)
        a = x.clone().detach().requires_grad_(True)
        b = y.clone().detach().requires_grad_(True)

        # Triton
        output = MatMul.apply(x, y)
        output.sum().backward()

        # PyTorch
        c = a @ b
        c.sum().backward()

        # Compare
        assert torch.allclose(x.grad, a.grad, atol=atol, rtol=rtol)
        assert torch.allclose(y.grad, b.grad, atol=atol, rtol=rtol)
