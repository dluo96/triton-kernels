import torch
import triton

from tk.fused_softmax import softmax


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


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    print("Success")

    benchmark.run(
        print_data=True, save_path="/home/danielluo/cuda-c/benchmarks/softmax/"
    )
