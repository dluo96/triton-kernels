import pytest
import torch

from tk.swizzle import kernel_swizzle


@pytest.mark.skipif(
    torch.cuda.is_available() is False, reason="Requires CUDA capable GPU"
)
@pytest.mark.parametrize(
    "num_blocks_m, num_blocks_n, group_size_m, expected_out",
    [
        # Check that nothing happens when the group size is set to 1
        pytest.param(
            5,
            4,
            1,
            torch.tensor(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                    [16, 17, 18, 19],
                ],
                device="cuda",
            ),
        ),
        pytest.param(
            5,
            4,
            2,
            torch.tensor(
                [
                    [0, 2, 4, 6],
                    [1, 3, 5, 7],
                    [8, 10, 12, 14],
                    [9, 11, 13, 15],
                    [16, 17, 18, 19],
                ],
                device="cuda",
            ),
        ),
        pytest.param(
            5,
            4,
            3,
            torch.tensor(
                [
                    [0, 3, 6, 9],
                    [1, 4, 7, 10],
                    [2, 5, 8, 11],
                    [12, 14, 16, 18],
                    [13, 15, 17, 19],
                ],
                device="cuda",
            ),
        ),
        pytest.param(
            5,
            4,
            5,
            torch.tensor(
                [
                    [0, 5, 10, 15],
                    [1, 6, 11, 16],
                    [2, 7, 12, 17],
                    [3, 8, 13, 18],
                    [4, 9, 14, 19],
                ],
                device="cuda",
            ),
        ),
    ],
)
def test_swizzle(
    num_blocks_m: int,
    num_blocks_n: int,
    group_size_m: int,
    expected_out: torch.Tensor,
):
    x = torch.arange(num_blocks_m * num_blocks_n, device="cuda").view(
        num_blocks_m, num_blocks_n
    )
    out = -torch.ones_like(x)
    kernel_swizzle[(num_blocks_m, num_blocks_n)](x, out, group_sz=group_size_m)
    assert torch.equal(out, expected_out)
