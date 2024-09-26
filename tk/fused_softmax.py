"""Adapted from the Triton documentation (https://triton-lang.org/main/index.html)."""
import torch
import triton
import triton.language as tl
from triton.runtime import driver

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGISTERS_PER_SM = properties["max_num_regs"]
SIZE_SHMEM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
dict_kernels = {}  # Cache precompiled kernels based on their block size

# import os
# os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Assuming the softmax calculation is done per row, the rows are independent.
    # Thus, we can parallelize across rows.

    # Each program starts from a different row. It handles one row at a time but
    # possibly multiple rows overall (depending of the number of programs).
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    # For each program, loop over the rows that the program will handle.
    # Each iteration of the loop corresponds to a program computing the softmax
    # for one row.
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The row stride denotes how much we need to increment the pointer to
        # advance 1 row. This is usually the number of columns of the input.
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # The block size was set to be the next power of two greater than `n_cols`,
        # so we can fit each row in a single block.
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets  # List of pointers to the row elements

        # Load the row into SRAM, using a mask since `BLOCK_SIZE` may be greater than
        # `n_cols`. Note out-of-bound elements won’t affect the sum since exp(-inf)=0.
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)

        # Note that exponentiation in Triton is fast but approximate.
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)  # Sum across the row
        softmax_output = numerator / denominator

        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than
    # the number of columns in `x`. Thus, each row fits in a single block.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Ask the compiler to use more threads per row by increasing the number of warps
    # (`num_warps`) over which the softmax calculation for each row is distributed.
    # In this case, each kernel instance (program) will be automatically parallelized
    # to cooperatively execute using 8 * 32 = 256 threads.
    num_warps = 8

    # Number of stages that the compiler should use when software-pipelining loops
    num_stages = 4 if SIZE_SHMEM_PER_SM > 200_000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # Compute the input and output row strides (the jump necessary to go from one
    # element to the next in the row dimension). For example, we set the input
    # row stride equal to the number of columns of x since this is the number of
    # elements per row. Similarly for the output row stride.
    input_row_stride = x.stride(dim=0)
    output_row_stride = y.stride(dim=0)

    # Check if a kernel for the given `BLOCK_SIZE` is already cached. If not, then
    # pre-compile the kernel to get register usage and compute thread occupancy.
    kernel, num_programs = dict_kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        # "warmup" is a process used to pre-compile a kernel and gather information
        # about its resource usage, including the number of registers and the amount
        # of shared memory it uses. This step is crucial for optimizing kernel
        # performance as it allows the Triton compiler to perform initial compilation
        # and prepare the kernel for efficient execution.
        # This step does not execute the kernel on actual data but ensures that when
        # the kernel is subsequently launched, it runs with the best possible settings.
        # The grid size of (1,) ensures minimal overhead and quick retrieval of kernel
        # resource usage information.
        kernel = softmax_kernel.warmup(
            x,
            y,
            input_row_stride,
            output_row_stride,
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )

        # Initialise internal handles and gather metadata about the kernel, including
        # register usage and shared memory requirements
        kernel._init_handles()
        n_registers_per_thread = kernel.n_regs
        size_shmem_per_instance = kernel.metadata.shared

        # Calculate occupancy (defined here as the maximum number of programs/instances
        # that can run simultaneously on a single SM) based on register usage. The
        # calculation divides the total number of registers in an SM by the number of
        # registers used per kernel instance (since WARP_SIZE * num_warps is the number
        # of threads per instance).
        occupancy = NUM_REGISTERS_PER_SM // (
            n_registers_per_thread * WARP_SIZE * num_warps
        )

        # Occupancy can be limited by register usage or shared memory.
        # The minimum of register-based occupancy and shared-memory based occupancy is
        # chosen.
        occupancy = min(occupancy, SIZE_SHMEM_PER_SM // size_shmem_per_instance)

        # Number of programs = (Number of SMs) * (Programs/SM).
        # This is number of concurrent kernel instances that can run on an SM.
        num_programs = NUM_SM * occupancy

        # Cache the compiled kernel and its number of programs/instances for future use.
        dict_kernels[BLOCK_SIZE] = (kernel, num_programs)

    # Ensures that the number of concurrent kernel instances in the grid does not exceed
    # the number of rows in the input tensor. While the GPU might be able to handle a
    # large number of kernel instances concurrently based on its hardware limits, the
    # actual number of kernel instances needed is limited by the number of rows in the
    # input tensor. If the number of programs that the hardware can support is larger
    # than the number of rows, simply make the number of programs actually used equal
    # to the number of rows such that each program handles exactly one row.
    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        x,
        y,
        input_row_stride,
        output_row_stride,
        n_rows,
        n_cols,
    )

    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    print("Success")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[128 * i for i in range(2, 100)],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="Global Memory Bandwidth (GB/s)",
            plot_name="softmax-performance",
            args={"M": 4096},  # Number of columns in input matrix
        )
    )
    def benchmark(M, N, provider):
        # Initialise input
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)

        # Create and set CUDA stream
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)

        # Benchmark the runtime, getting the mean (default)
        if provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        if provider == "triton":
            ms = triton.testing.do_bench(lambda: softmax(x))

        # Compute the global memory bandwidth in GB/s.
        # The factor of 2 is because we have two tensors (input `x` and output `y`).
        # `x.nelement()` gives the total number of elements in `x`.
        # `x.element_size()` gives the size in bytes of an individual element of `x`
        # Thus, the formula basically says:
        # Global memory bandwidth in GB/s = (#Bytes)/(Runtime in ms)*(1e-9 GB/B)*(1e3 ms/s)
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms)

    benchmark.run(
        print_data=True, save_path="/home/danielluo/cuda-c/benchmarks/softmax/"
    )
