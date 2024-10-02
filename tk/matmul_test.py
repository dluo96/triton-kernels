import functools
import unittest

import torch

from tk.matmul import kernel_matmul_grouped, kernel_matmul_naive, matmul
from tk.matmul_autotuned import matmul_grouped_autotuned


class TestMatmul(unittest.TestCase):
    def setUp(self):
        self.matmul_naive = functools.partial(matmul, kernel_matmul=kernel_matmul_naive)
        self.matmul_grouped = functools.partial(
            matmul, kernel_matmul=kernel_matmul_grouped
        )

    def tearDown(self):
        torch.cuda.empty_cache()

    def test_matmul_naive__small(self):
        a = torch.ones((3, 4), dtype=torch.float32, device="cuda")
        b = torch.ones((4, 5), dtype=torch.float32, device="cuda")
        assert torch.allclose(self.matmul_naive(a, b), a @ b)

    def test_matmul_naive__large(self):
        a = torch.randn((300, 400), dtype=torch.float32, device="cuda")
        b = torch.randn((400, 500), dtype=torch.float32, device="cuda")
        assert torch.allclose(self.matmul_naive(a, b), a @ b, atol=0.1, rtol=0)

    def test_matmul_grouped__small(self):
        a = torch.ones((3, 4), dtype=torch.float32, device="cuda")
        b = torch.ones((4, 5), dtype=torch.float32, device="cuda")
        assert torch.allclose(self.matmul_grouped(a, b, group_size=32), a @ b)

    def test_matmul_grouped__large(self):
        a = torch.randn((300, 400), dtype=torch.float32, device="cuda")
        b = torch.randn((400, 500), dtype=torch.float32, device="cuda")
        assert torch.allclose(
            self.matmul_grouped(a, b, group_size=32), a @ b, atol=0.1, rtol=0
        )

    def test_matmul_grouped_autotuned__small(self):
        a = torch.ones((3, 4), dtype=torch.float32, device="cuda")
        b = torch.ones((4, 5), dtype=torch.float32, device="cuda")
        assert torch.allclose(matmul_grouped_autotuned(a, b), a @ b)


if __name__ == "__main__":
    unittest.main()
