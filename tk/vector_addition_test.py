import pytest
import torch

from tk.vector_addition import add_vectors

torch.manual_seed(0)


@pytest.mark.skipif(torch.cuda.is_available() is False, reason="Requires CUDA GPU")
@pytest.mark.parametrize("size", [1, 36, 100, 98432])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-8, 1e-5),
        (torch.float32, 1e-8, 1e-5),
    ],
)
def test_add(size: int, dtype: torch.dtype, atol: float, rtol: float):
    x = torch.rand(size, dtype=dtype, device="cuda")
    y = torch.rand(size, dtype=dtype, device="cuda")

    out = add_vectors(x, y)
    expected_out = x + y

    assert torch.allclose(out, expected_out, atol=atol, rtol=rtol)
