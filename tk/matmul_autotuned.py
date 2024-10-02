import torch
import triton
import triton.language as tl

from tk.matmul import get_2d_mask, get_2d_offset


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
)
@triton.jit
def kernel_matmul_grouped_autotuned(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bs_m: tl.constexpr, bs_n: tl.constexpr, bs_k: tl.constexpr,
    group_size_m: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.num_programs(axis=0)
    num_pid_n = tl.num_programs(axis=1)

    # Swizzle: this reordering of blocks can increase L2-cache hit rate and
    # thus make our kernel faster
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_size_m)

    # Get offsets in m/n/k dimensions
    offsets_m = pid_m * bs_m + tl.arange(0, bs_m)
    offsets_n = pid_n * bs_n + tl.arange(0, bs_n)
    offsets_k = tl.arange(0, bs_k)

    # Get offsets for input matrices `a` and `b`
    offsets_a = a_ptr + get_2d_offset(offsets_m, offsets_k, stride_am, stride_ak)
    offsets_b = b_ptr + get_2d_offset(offsets_k, offsets_n, stride_bk, stride_bn)

    # Initialise and iteratively update accumulator (loop over phases)
    accumulator = tl.zeros((bs_m, bs_n), dtype=tl.float32)
    for _ in range(0, k, bs_k):
        a = tl.load(offsets_a)
        b = tl.load(offsets_b)
        accumulator += tl.dot(a, b)

        # Increase offsets so that the next iteration loads the next chunks
        offsets_a += bs_k * stride_ak
        offsets_b += bs_k * stride_bk

    c = c_ptr + get_2d_offset(offsets_m, offsets_n, stride_cm, stride_cn)
    mask = get_2d_mask(offsets_m, offsets_n, m, n)
    tl.store(c, accumulator, mask=mask)


def matmul_grouped_autotuned(a, b):
    assert a.shape[1] == b.shape[0], "Input shapes incompatible for matmul!"
    
    m, k = a.shape
    _, n = b.shape
    
    # Allocate space for output
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)

    # Define grid and call kernel
    grid = lambda meta: (triton.cdiv(m, meta["bs_m"]), triton.cdiv(n, meta["bs_n"]))
    kernel_matmul_grouped_autotuned[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # bm=bs, bn=bs, bk=bs, <- will be autotuned
        # **group_sz <- will be autotuned
    )
    return c
