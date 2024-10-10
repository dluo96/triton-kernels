import pytest
import torch

from tk.fused_softmax import softmax

torch.manual_seed(23)


@pytest.mark.skipif(torch.cuda.is_available() is False, reason="Requires CUDA GPU")
@pytest.mark.parametrize("n_rows, n_cols", [(3, 4), (64, 512), (1823, 781)])
@pytest.mark.parametrize("dtype, atol, rtol", [(torch.float32, 1e-8, 1e-5)])
def test_softmax(
    n_rows: int, n_cols: int, dtype: torch.dtype, atol: float, rtol: float
):
    logits = torch.randn((n_rows, n_cols), dtype=dtype, device="cuda")
    probs = softmax(logits)
    assert probs.shape == (n_rows, n_cols)

    expected_probs = torch.softmax(logits, axis=1)
    assert torch.allclose(probs, expected_probs)
