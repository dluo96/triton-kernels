import unittest

import torch

from tk.swizzle import kernel_swizzle


class TestSwizzle(unittest.TestCase):
    def test_swizzle(self):
        num_blocks_m = 5
        num_blocks_n = 4
        x = torch.arange(num_blocks_m * num_blocks_n, device="cuda").view(
            num_blocks_m, num_blocks_n
        )
        z = -torch.ones_like(x)
        kernel_swizzle[(num_blocks_m, num_blocks_n)](x, z, group_sz=3)

        # fmt: off
        expected_z = torch.tensor(
            [
                [ 0,  3,  6,  9],
                [ 1,  4,  7, 10],
                [ 2,  5,  8, 11],
                [12, 14, 16, 18],
                [13, 15, 17, 19],
            ],
            device=x.device,
        )
        # fmt: on
        assert torch.equal(z, expected_z)


if __name__ == "__main__":
    unittest.main()
