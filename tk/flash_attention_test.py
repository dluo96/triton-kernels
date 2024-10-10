import pytest
import torch
import torch.nn.functional as F

from tk.flash_attention import FlashAttention

torch.manual_seed(0)


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
    q = torch.randn(B, H, T, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, T, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, T, D, dtype=dtype, device="cuda")

    # Define a softmax scaling factor
    sm_scale = 1.0 / (D**0.5)

    # Forward (Triton)
    out_triton = FlashAttention.apply(q, k, v, sm_scale)

    # Forward (PyTorch)
    out_torch = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.allclose(out_triton, out_torch, atol=atol, rtol=rtol)
