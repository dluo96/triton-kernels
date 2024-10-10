import pathlib

import torch
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io

from tk.rgb_to_grey import rgb_to_grey


def test_rgb_to_grey():
    # Load and resize image
    path_img = pathlib.Path(__file__).parent / "dog.jpg"
    if not path_img.exists():
        raise FileNotFoundError("Dog image not found!")
    img = io.read_image(path_img)
    img = tvf.resize(img, 150, antialias=True)
    assert img.shape == (3, 150, 225)
    img = img.to("cuda")

    # Convert to grey scale
    grey_img = rgb_to_grey(img, block_size=(32, 32))
    expected_grey_img = tv.transforms.Grayscale()(img)
    assert torch.allclose(grey_img, expected_grey_img)
