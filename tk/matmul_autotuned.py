import torch
import triton
import triton.language as tl

from tk.matmul import kernel_matmul_grouped


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({'bs_m': 128, 'bs_n': 256, 'bs_k': 64, 'group_size_m': 8}, num_stages=3, num_warps=8),
        triton.Config({'bs_m':  64, 'bs_n': 256, 'bs_k': 32, 'group_size_m': 8}, num_stages=4, num_warps=4),
        triton.Config({'bs_m': 128, 'bs_n': 128, 'bs_k': 32, 'group_size_m': 8}, num_stages=4, num_warps=4),
        triton.Config({'bs_m': 128, 'bs_n':  64, 'bs_k': 32, 'group_size_m': 8}, num_stages=4, num_warps=4),
        triton.Config({'bs_m':  64, 'bs_n': 128, 'bs_k': 32, 'group_size_m': 8}, num_stages=4, num_warps=4),
        triton.Config({'bs_m': 128, 'bs_n':  32, 'bs_k': 32, 'group_size_m': 8}, num_stages=4, num_warps=4),
        triton.Config({'bs_m':  64, 'bs_n':  32, 'bs_k': 32, 'group_size_m': 8}, num_stages=5, num_warps=2),
        triton.Config({'bs_m':  32, 'bs_n':  64, 'bs_k': 32, 'group_size_m': 8}, num_stages=5, num_warps=2),
    ],
    key=["m", "n", "k"],  # Problem size: everytime it changes, all the configs above will be evaluated
) # fmt: on
@triton.jit
def kernel_matmul_grouped_autotuned(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    bs_m: tl.constexpr,
    bs_n: tl.constexpr,
    bs_k: tl.constexpr,
    group_size_m: tl.constexpr,
):
    kernel_matmul_grouped(
        a_ptr,
        b_ptr,
        c_ptr,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        bs_m,
        bs_n,
        bs_k,
        group_size_m,
    )


def matmul_grouped_autotuned(a, b):
    assert a.shape[1] == b.shape[0], "Input shapes incompatible for matmul!"

    m, k = a.shape
    _, n = b.shape

    # Allocate space for output
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)

    # Define grid and call kernel
    grid = lambda meta: (triton.cdiv(m, meta["bs_m"]), triton.cdiv(n, meta["bs_n"]))
    kernel_matmul_grouped_autotuned[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        # bs_m=bs, bs_n=bs, bs_k=bs,    <-- This will be autotuned by @triton.autone
        # group_size_m                  <-- This will be autotuned by @triton.autone
    )
    return c
