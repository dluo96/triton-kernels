import pytest
import torch

from tk.vector_addition import add_vectors


@pytest.mark.parametrize("size", [1, 36, 100, 98432])
def test_add(size: int):
    torch.manual_seed(0)

    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    out_torch = x + y
    out_triton = add_vectors(x, y)

    assert torch.allclose(out_torch, out_triton)
