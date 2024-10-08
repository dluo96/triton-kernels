from typing import Callable

import pytest
import torch

from tk.matmul import kernel_matmul_grouped, kernel_matmul_naive, matmul
from tk.matmul_autotuned import matmul_grouped_autotuned


@pytest.mark.skipif(
    torch.cuda.is_available() is False, reason="Requires CUDA capable GPU"
)
@pytest.mark.parametrize(
    "m, k, n", [(8, 16, 8), (3, 4, 5), (30, 40, 50), (300, 400, 500)]
)
@pytest.mark.parametrize("group_size_m", [16, 32])
@pytest.mark.parametrize("kernel", [kernel_matmul_naive, kernel_matmul_grouped])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-7, 5e-2),
        (torch.float32, 1e-8, 1e-6),
    ],
)
def test_matmul(
    m: int,
    n: int,
    k: int,
    group_size_m: int,
    kernel: Callable,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    a = torch.ones((m, k), dtype=dtype, device="cuda")
    b = torch.ones((k, n), dtype=dtype, device="cuda")
    out = matmul(a, b, kernel_matmul=kernel, group_size=group_size_m)
    expected_out = a @ b
    assert torch.allclose(out, expected_out, atol=atol, rtol=rtol)
