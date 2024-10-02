import pathlib

import torch
import triton

from tk.vector_addition import add_vectors


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Possible values for `x_name` (here 2^12 to 2^27)
        x_log=True,
        line_arg="provider",  # Each value of `line_arg` identifies a line in the plot.
        line_vals=[
            "triton",
            "torch",
        ],  # Possible values of `line_arg` (here `triton` and `torch`)
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="Global Memory Bandwidth (GB/s)",
        plot_name="vector-add-benchmarks",
        args={},
    )
)
def benchmark(size, provider):
    # Random initialisation of inputs
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)

    # Benchmark the runtime, getting the median, 20th pecentile, and 80th percentile.
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add_vectors(x, y), quantiles=quantiles
        )

    # Compute the global memory bandwidth in GB/s. This indicates the data transfer rate.
    # The factor of 3 is because we have three tensors (`x`, `y`, `output`).
    # `x.numel()` gives the total number of elements in `x`.
    # `x.element_size()` gives the size in bytes of an individual element of `x`
    # Thus, the formula basically says:
    # Global memory bandwidth in GB/s = (#Bytes)/(Runtime in ms)*(1e-9 GB/B)*(1e3 ms/s)
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(
        print_data=True,
        save_path=pathlib.Path(__file__).parent / "profiling" / "vector_addition",
    )
