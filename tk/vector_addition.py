"""Adapted from the Triton documentation (https://triton-lang.org/main/index.html)."""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def kernel_add_vectors(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # There are multiple "programs" each processing different parts of the data.
    # Each program is uniquely identified by its `program_id`. Because we will
    # use a 1D launch grid, we set "axis" to 0.
    pid = tl.program_id(axis=0)

    # Each program handles `BLOCK_SIZE` elements. For example, if we have a vector
    # with 256 elements and set the block size to 64, the cdiv(256, 64) = 4 programs
    # would access the elements 0:64, 64:128, 128:192, and 192:256.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # List of pointers

    # Use a mask to avoid out-of-bounds memory accesses.
    mask = offsets < n_elements

    # Load x and y from DRAM, masking out any extra elements in case the size
    # of the vector (`n_elements`) is not a multiple of the block size (this
    # would only affect the last program as this handles the 'last' block).
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform the addition and write the result back to DRAM
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Need to pre-allocate device memory for the output
    output = torch.empty_like(x)

    # Verify that the input and output tensors are on the GPU
    assert x.is_cuda and y.is_cuda and output.is_cuda

    # Extract the size of the output
    n_elements = output.numel()

    # The kernel will be launched concurrently with different `program_id`s on a
    # grid of so-called "instances", following Single Program Multiple Data (SPDM).
    # This grid is analogous to a CUDA launch grid. In this case, we use a 1D grid
    # where each program (kernel instance) performs `BLOCK_SIZE` element-wise additions.
    # Hence the grid dimension (number of programs/instances) is given by the ceiling
    # division of `n_elements` by `BLOCK_SIZE`.
    grid = lambda metaparams: (triton.cdiv(n_elements, metaparams["BLOCK_SIZE"]),)

    # Launch the Triton kernel
    # Note that
    #  - Each `torch.tensor` object is implicitly converted into a pointer to its first element.
    #  - The `triton.jit`-decorated function is indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass metaparameters (needed in `grid`) as kwargs (`BLOCK_SIZE` in this case).
    kernel_add_vectors[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
