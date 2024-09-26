import torch

from tk.vector_addition import add_vectors


def test_add():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    output_torch = x + y
    output_triton = add_vectors(x, y)

    assert torch.allclose(output_triton, output_torch)
