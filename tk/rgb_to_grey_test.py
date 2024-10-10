import pathlib

import pytest
import torch
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io

from tk.rgb_to_grey import rgb_to_grey


@pytest.mark.parametrize("block_size", [(16, 16), (32, 32), (64, 64)])
def test_rgb_to_grey(block_size: tuple[int, int]):
    # Load and resize image
    path_img = pathlib.Path(__file__).parent / "dog.jpg"
    if not path_img.exists():
        raise FileNotFoundError("Dog image not found!")
    img = io.read_image(path_img)
    img = tvf.resize(img, 150, antialias=True)
    assert img.shape == (3, 150, 225), "Resized image has incorrect shape!"
    img = img.to("cuda")

    # Convert to grey scale
    grey_img = rgb_to_grey(img, block_size=block_size)
    expected_grey_img = tv.transforms.Grayscale()(img)
    assert torch.allclose(grey_img, expected_grey_img)
