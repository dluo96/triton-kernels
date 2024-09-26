import functools

import torch

from tk.matmul import kernel_naive_matmul, matmul


def test_matmul_naive():
    a = torch.ones((3, 4), dtype=torch.float32, device="cuda")
    b = torch.ones((4, 5), dtype=torch.float32, device="cuda")

    matmul_naive = functools.partial(matmul, kernel_matmul=kernel_naive_matmul)

    assert torch.allclose(matmul_naive(a, b), a @ b)


def test_matmul_naive__large():
    a = torch.ones((300, 400), dtype=torch.float32, device="cuda")
    b = torch.ones((400, 500), dtype=torch.float32, device="cuda")

    matmul_naive = functools.partial(matmul, kernel_matmul=kernel_naive_matmul)

    assert torch.allclose(matmul_naive(a, b), a @ b)
