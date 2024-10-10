import pytest
import torch
import torch.autograd

from tk.vector_addition_module import AddVectors


@pytest.mark.skipif(torch.cuda.is_available() is False, reason="Requires CUDA GPU")
@pytest.mark.parametrize("B, T", [(64, 1000), (331, 487)])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-8, 1e-5),
        (torch.float32, 1e-8, 1e-5),
    ],
)
def test_add_vectors_autograd_function(
    B: int, T: int, dtype: torch.dtype, atol: float, rtol: float
):
    x = torch.randn(B, T, dtype=dtype, device="cuda", requires_grad=True)
    y = torch.randn(B, T, dtype=dtype, device="cuda", requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = y.clone().detach().requires_grad_(True)

    # Forward pass (Triton)
    out = AddVectors.apply(x, y)
    loss = out.sum()  # Arbitrary loss function so we can backprop

    # Forward pass (PyTorch)
    out_torch = x_ref + y_ref
    loss_ref = out_torch.sum()

    # Check forward
    assert torch.allclose(out, out_torch, atol=atol, rtol=rtol)

    # Backward pass
    loss.backward()
    loss_ref.backward()

    assert torch.allclose(x.grad, x_ref.grad, atol=atol, rtol=rtol)
    assert torch.allclose(y.grad, x_ref.grad, atol=atol, rtol=rtol)
