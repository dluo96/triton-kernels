import torch
import triton
import triton.language as tl

from tk.matmul import get_2d_mask, get_2d_offset


@triton.jit
def get_1d_offset(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)


@triton.jit
def kernel_swizzle(x_ptr, z_ptr, group_sz: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    offs_m = pid_m * 1 + tl.arange(0, 1)
    offs_n = pid_n * 1 + tl.arange(0, 1)

    offs = get_2d_offset(offs_m, offs_n, stride_0=num_pid_n, stride_1=1)
    mask = get_2d_mask(offs_m, offs_n, max_0=num_pid_m, max_1=num_pid_n)

    # NOTE: tl.swizzle2d doesn't work when simulating on CPU
    pid_m_, pid_n_ = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_sz)

    offs_sw_m = pid_m_ + tl.arange(0, 1)
    offs_sw_n = pid_n_ + tl.arange(0, 1)

    offs_sw = get_2d_offset(offs_sw_m, offs_sw_n, stride_0=num_pid_n, stride_1=1)
    mask_sw = get_2d_mask(offs_sw_m, offs_sw_n, max_0=num_pid_m, max_1=num_pid_n)

    # Load data from the original position of the input array and
    # store it in the swizzled position of the output array
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(z_ptr + offs_sw, x, mask=mask_sw)
