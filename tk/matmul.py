import torch

import triton
import triton.language as tl


@triton.jit
def get_2d_offset(offsets_0, offsets_1, stride_0, stride_1):
    return tl.expand_dims(offsets_0, axis=1) * stride_0 + tl.expand_dims(offsets_1, axis=0) * stride_1

@triton.jit
def get_2d_mask(offsets_0, offsets_1, max_0, max_1):
    return (tl.expand_dims(offsets_0, axis=1) < max_0) & (tl.expand_dims(offsets_1, axis=0) < max_1)

@triton.jit
def kernel_naive_matmul(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr,
):
    """Compute matrix multiplication of `a` (m, k) and `b` (k, n) to get output matrix (m, n)."""
    # Get program IDs which determine which section of the output matrix `c` (m x n) to compute
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Get offsets in m/n/k dimensions
    offsets_m = pid_m * bm + tl.arange(0, bm)
    offsets_n = pid_n * bn + tl.arange(0, bn)
    offsets_k = tl.arange(0, bk)

    # Get offsets for input matrices `a` and `b`
    offsets_a = a_ptr + get_2d_offset(offsets_m, offsets_k, stride_am, stride_ak)
    offsets_b = b_ptr + get_2d_offset(offsets_k, offsets_n, stride_bk, stride_bn)

    # Horizontal and vertical masks
    mask_m = offsets_m < m
    mask_n = offsets_n < n
    mask_k = offsets_k < k

    # Combine masks
    mask_a = get_2d_mask(offsets_m, offsets_k, m, k)
    mask_b = get_2d_mask(offsets_k, offsets_n, k, n)

    # Initialise and iteratively update accumulator (loop over phases)
    accumulator = tl.zeros((bm, bn), dtype=tl.float32)
    for _ in range(0, k, bk):
        a = tl.load(offsets_a, mask=mask_a)
        b = tl.load(offsets_b, mask=mask_b)
        accumulator += tl.dot(a, b)

        # Increase offsets so that the next iteration loads the next chunks
        offsets_a += bk * stride_ak
        offsets_b += bk * stride_bk

    c = c_ptr + get_2d_offset(offsets_m, offsets_n, stride_cm, stride_cn)
    mask = get_2d_mask(offsets_m, offsets_n, m, n)
    tl.store(c, accumulator, mask=mask)


def matmul(a, b, kernel_matmul, bs=16, group_size=None):
    assert a.shape[1] == b.shape[0], "Input matrix shapes are incompatible for matrix multiplication"
    
    m, k = a.shape
    _, n = b.shape

    # Allocate space for output
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)

    # Define grid (this tells us how many program IDs horizontally and vertically)
    grid = lambda meta: (triton.cdiv(m, meta["bm"]), triton.cdiv(n, meta["bn"]))

    # Call kernel
    kernel_naive_matmul[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        bm=bs, bn=bs, bk=bs,
    )
    return c
