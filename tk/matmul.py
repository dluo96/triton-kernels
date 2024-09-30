import torch

import triton
import triton.language as tl


@triton.jit
def get_2d_offset(offsets_0, offsets_1, stride_0, stride_1):
    """
    For example, suppose we have the tensor:

                        pid_y
                 ------------------>
                |  0  1  2  3  4  5  6
          pid_x |  7  8  9 10 11 12 13
                | 14 15 16 17 18 19 20
                | 21 22 23 24 25 26 27
    
    This has
        - t.stride(0) = 7
        - t.stride(1) = 1
    
    Scenario 1: 
        - `pid_x = 0` which means `offsets_0` is [0, 1], assuming bs_x = 2.
        - `pid_y = 0` which means `offsets_1` is [0, 1], assuming bs_y = 2.
        - The combined 2D offset is:
                        [[0, 1],
                         [7, 8]]
          because:
                [0] * 7 + [0, 1] = [0] + [0, 1] = [0 0] + [0 1] = [0 1]
                [1]                [7]            [7 7]   [0 1]   [7 8]

    Scenario 2:
        - `pid_x = 0` which means `offsets_0` is [0, 1], assuming bs_x = 2.
        - `pid_y = 3` which means `offsets_1` is [6, 7], assuming bs_y = 2.
        - The combined 2D offset is:
                        [[ 6,  7],
                         [13, 14]]
            because:
                [0] * 7 + [6, 7] = [0] + [6, 7] = [0 0] + [6 7] = [ 6  7]
                [1]                [7]            [7 7]   [6 7]   [13 14]
    
    Scenario 3:
        - `pid_x = 1` which means `offsets_0` is [2, 3], assuming bs_x = 2.
        - `pid_y = 0` which means `offsets_1` is [0, 1, 2], assuming bs_y = 3.
        - The combined 2D offset is:
                        [[14, 15, 16],
                         [21, 22, 23]]
          because:
            [2] * 7 + [0, 1, 2] = [14] + [0, 1, 2] = [14 14 14] + [0 1 2] = [14 15 16]
            [3]                   [21]               [21 21 21]   [0 1 2]   [21 22 23]
    
    Scenario 4:
        - `pid_x = 1` which means `offsets_0` is [3, 4, 5], assuming bs_x = 3.
        - `pid_y = 3` which means `offsets_1` is [6, 7], assuming bs_y = 2.
        - The combined 2D offset is:
                        [[27, 28],
                         [34, 35]
                         [41, 42]]
          because:
            [3]               [21]           [21 21]   [6 7]   [27 28]
            [4] * 7 + [6 7] = [28] + [6 7] = [28 28] + [6 7] = [34 35]
            [5]               [35]           [35 35]   [6 7]   [41 42]
    """
    return (
        tl.expand_dims(offsets_0, axis=1) * stride_0 + 
        tl.expand_dims(offsets_1, axis=0) * stride_1
    )


@triton.jit
def get_2d_mask(offsets_0, offsets_1, max_0, max_1):
    """
    For example, suppose we have the tensor:

                    pid_y
             ------------------>
            |  0  1  2  3  4  5  6
      pid_x |  7  8  9 10 11 12 13
            | 14 15 16 17 18 19 20
            | 21 22 23 24 25 26 27
    
    Since the tensor is 4x7, we have `max_0 = 4` and `max_1` = 7.
    
    Scenario 1:
        - `pid_x = 0` which means `offsets_0` is [0, 1], assuming bs_x = 2.
        - `pid_y = 0` which means `offsets_1` is [0, 1], assuming bs_y = 2.
        - The 2D mask is:
                        [[True, True],
                         [True, True]]
          because:
                [0] < 4 & [0, 1] < 7 = [True] + [True True] = [True True]
                [1]                    [True]                 [True True]

    Scenario 2:
        - `pid_x = 0` which means `offsets_0` is [0, 1], assuming bs_x = 2.
        - `pid_y = 3` which means `offsets_1` is [6, 7], assuming bs_y = 2.
        - The 2D mask is:
                        [[True, False],
                         [True, False]]
            because:

                [0] < 4 & [6, 7] < 7 = [True] + [True, False] = [True True] & [True False] = [True False]
                [1]                    [True]                   [True True]   [True False]   [True False]
            
            This makes sense: we saw earlier that the combined 2D offset is
                [[ 6,  7],
                 [13, 14]]
            but 7 and 14 are out of bounds since there are only 7 columns (not 8).
    
    Scenario 3:
        - `pid_x = 1` which means `offsets_0` is [2, 3], assuming bs_x = 2.
        - `pid_y = 0` which means `offsets_1` is [0, 1, 2], assuming bs_y = 3.
        - The 2D mask is:
                        [[True, True, True],
                         [True, True, True]]
          because:
            [2] < 4 & [0, 1, 2] < 7 = [True] + [True, True, True] = [True True True]
            [3]                       [True]                        [True True True]
    
    Scenario 4:
        - `pid_x = 1` which means `offsets_0` is [3, 4, 5], assuming bs_x = 3.
        - `pid_y = 3` which means `offsets_1` is [6, 7], assuming bs_y = 2.
        - The 2D mask is:
                        [[True, False],
                         [False, False]
                         [False, False]]
          because:
            [3]                   [True]                    [True  True]    [True False]   [True False]
            [4] < 4 & [6 7] < 7 = [False] & [True, False] = [False False] & [True False] = [False False]
            [5]                   [False]                   [False False]   [True False]   [False False]
    """
    return (
        (tl.expand_dims(offsets_0, axis=1) < max_0) & 
        (tl.expand_dims(offsets_1, axis=0) < max_1)
    )

@triton.jit
def kernel_matmul_naive(
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
    kernel_matmul[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        bm=bs, bn=bs, bk=bs,
    )
    return c

