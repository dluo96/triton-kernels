import functools
import unittest
import torch


class TestMatmul(unittest.TestCase):
    def setUp(self):
        from tk.matmul import kernel_matmul_naive, matmul
        self.matmul_naive = functools.partial(matmul, kernel_matmul=kernel_matmul_naive)

    def test_matmul_naive__small(self):
        a = torch.ones((3, 4), dtype=torch.float32, device="cuda")
        b = torch.ones((4, 5), dtype=torch.float32, device="cuda")
        assert torch.allclose(self.matmul_naive(a, b), a @ b)


    def test_matmul_naive__large(self):
        a = torch.randn((300, 400), dtype=torch.float32, device="cuda")
        b = torch.randn((400, 500), dtype=torch.float32, device="cuda")
        assert torch.allclose(self.matmul_naive(a, b), a @ b, atol=0.1, rtol=0)


if __name__ == "__main__":
    unittest.main()