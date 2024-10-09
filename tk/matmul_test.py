from typing import Callable

import pytest
import torch

from tk.matmul import kernel_matmul_grouped, kernel_matmul_naive, matmul
from tk.matmul_autotuned import matmul_grouped_autotuned


@pytest.mark.skipif(
    torch.cuda.is_available() is False, reason="Requires CUDA capable GPU"
)
@pytest.mark.parametrize(
    "m, k, n", [(8, 16, 32), (3, 4, 5), (2048, 128, 256), (13, 11, 17)]
)
@pytest.mark.parametrize("group_size_m", [16, 32])
@pytest.mark.parametrize("kernel", [kernel_matmul_naive, kernel_matmul_grouped])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-10, 0),
        (torch.float32, 5e-2, 0),
    ],
)
@pytest.mark.parametrize("tensor_factory", [torch.randn, torch.ones])
def test_matmul(
    m: int,
    n: int,
    k: int,
    group_size_m: int,
    kernel: Callable,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    tensor_factory: Callable,
):
    # PyTorch uses TF32 (not FP32) while Triton uses FP32, so
    # to get Triton's FP32 matmul to match Torch's more closely,
    # we need to disable TF32 in PyTorch.
    if dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False

    a = tensor_factory((m, k), dtype=dtype, device="cuda")
    b = tensor_factory((k, n), dtype=dtype, device="cuda")
    out = matmul(a, b, kernel_matmul=kernel, group_size=group_size_m)
    expected_out = torch.matmul(a, b)
    assert torch.allclose(out, expected_out, atol=atol, rtol=rtol)
