import functools
import pathlib

import torch
import triton

from tk.matmul import kernel_matmul_grouped, kernel_matmul_naive, matmul
from tk.matmul_autotuned import matmul_grouped_autotuned

matmul_naive = functools.partial(matmul, kernel_matmul=kernel_matmul_naive)
matmul_grouped = functools.partial(matmul, kernel_matmul=kernel_matmul_grouped)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["square_matrix_size"],
        x_vals=[2**i for i in range(5, 13, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["naive", "grouped", "grouped-autotuned", "torch"],
        line_names=["Naive", "Grouped", "Grouped+Autotuned", "Torch"],
        styles=[("green", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="GB/s",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(square_matrix_size: int, provider: str):
    sz = square_matrix_size
    a = torch.rand((sz, sz), device="cuda", dtype=torch.float32)
    b = torch.rand((sz, sz), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_naive(a, b), quantiles=quantiles
        )
    elif provider == "grouped":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_grouped(a, b, group_size=8), quantiles=quantiles
        )
    elif provider == "grouped-autotuned":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_grouped_autotuned(a, b), quantiles=quantiles
        )
    elif provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    else:
        raise ValueError("Unrecognized provider: {provider}")

    """
    Compute the global memory bandwidth in GB/s, which indicates data transfer rate.
    The factor of 12 is the number of bytes in the computation of each output element.

    The program reads from `a` and `b`, and writes the result into `c`. Assuming each
    matrix element is a 32-bit (4-byte) float, you need:
        - 4 bytes for each element in `a`,
        - 4 bytes for each element in `b`,
        - 4 bytes for each element in `c`.
    Thus, for every matrix element multiplied and added, you have a total of
    4 + 4 + 4 = 12 bytes involved in memory transfers.

    The formula says:
    Global memory bandwidth in GB/s = (#Bytes)/(Runtime in ms)*(1e-9 GB/B)*(1e3 ms/s)
    """
    gbps = lambda ms: 12 * (sz**2) / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(
        print_data=True,
        save_path=pathlib.Path(__file__).parent / "profiling" / "matmul",
    )
