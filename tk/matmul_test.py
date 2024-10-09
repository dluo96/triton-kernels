from typing import Callable

import pytest
import torch

from tk.matmul import kernel_matmul_grouped, kernel_matmul_naive, matmul
from tk.matmul_autotuned import matmul_grouped_autotuned

torch.manual_seed(0)


@pytest.mark.skipif(torch.cuda.is_available() is False, reason="Requires CUDA GPU")
@pytest.mark.parametrize("m, k, n", [(3, 4, 5), (2048, 128, 256), (13, 11, 17)])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("group_size_m", [16, 32])
@pytest.mark.parametrize("kernel", [kernel_matmul_naive, kernel_matmul_grouped])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-10, 0),
        # On NVIDIA GPUs with Tensor Cores, when the inputs are float32,
        # Triton by default uses TF32 (TensorFloat-32) for tl.dot, whereas
        # PyTorch has disabled TF32 by default.
        (torch.float32, 5e-2, 0),
    ],
)
@pytest.mark.parametrize("tensor_factory", [torch.randn, torch.ones])
def test_matmul(
    m: int,
    n: int,
    k: int,
    block_size: int,
    group_size_m: int,
    kernel: Callable,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    tensor_factory: Callable,
):
    a = tensor_factory((m, k), dtype=dtype, device="cuda")
    b = tensor_factory((k, n), dtype=dtype, device="cuda")

    assert (
        torch.backends.cuda.matmul.allow_tf32 is False,
        "TensorFloat-32 should be disabled!",
    )
    expected_out = torch.matmul(a, b)

    out = matmul(a, b, kernel_matmul=kernel, bs=block_size, group_size=group_size_m)
    assert torch.allclose(out, expected_out, atol=atol, rtol=rtol)
