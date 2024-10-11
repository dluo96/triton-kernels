import pytest
import torch
import torch.nn.functional as F

from tk.flash_attention import FlashAttention

torch.manual_seed(0)


@pytest.mark.skipif(torch.cuda.is_available() is False, reason="Requires CUDA GPU")
@pytest.mark.parametrize("B, H, T, D", [(1, 1, 2, 16)])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # Flash Attention only supports FP16 and BF16
        (torch.bfloat16, 1e-8, 1e-5),
        (torch.float16, 1e-3, 1e-5),
    ],
)
def test_flash_attention_forward(
    B: int,  # Batch size
    H: int,  # Number of heads
    T: int,  # Sequence length
    D: int,  # Dimension per head
    dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    q = torch.randn(B, H, T, D, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=dtype, device="cuda", requires_grad=True)

    q_torch = q.detach().clone().requires_grad_(True)
    k_torch = k.detach().clone().requires_grad_(True)
    v_torch = v.detach().clone().requires_grad_(True)

    # Define a softmax scaling factor
    sm_scale = 1.0 / (D**0.5)

    # Forward (Triton)
    out_triton = FlashAttention.apply(q, k, v, sm_scale)

    # Forward (PyTorch)
    out_torch = F.scaled_dot_product_attention(
        q_torch, k_torch, v_torch, is_causal=True
    )

    assert torch.allclose(out_triton, out_torch, atol=atol, rtol=rtol)

    # Backward (Triton)
    loss = out_triton.sum()
    loss.backward()

    # Backward (PyTorch)
    loss_torch = out_torch.sum()
    loss_torch.backward()

    # TODO: make the below work
    # assert torch.allclose(q.grad, q_.grad)
    # assert torch.allclose(k.grad, k_.grad)
    # assert torch.allclose(v.grad, v_.grad)
