import pathlib
import unittest
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import torch
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io

from tk.rgb_to_grey import rgb_to_grey


class TestRGBToGrey(unittest.TestCase):
    def setUp(self):
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg?20140729055059"
        path_img = pathlib.Path("dog.jpg")
        if not path_img.exists():
            urlretrieve(url, path_img)
        assert path_img.exists(), "Image not downloaded correctly!"
        img = io.read_image(path_img)
        self.img = tvf.resize(img, 150, antialias=True)
        self.assertEqual(
            self.img.shape,
            (3, 150, 225),
            msg="After resizing, the image should have 3 channels, a height of 150, "
            "and a width of 225",
        )

    def test_rgb_to_grey(self):
        grey_img = rgb_to_grey(self.img.to("cuda"), block_size=(32, 32)).to("cpu")
        manual_rgb_to_grey = tv.transforms.Grayscale()
        expected_grey_img = manual_rgb_to_grey(self.img)
        self.assertTrue(torch.allclose(grey_img, expected_grey_img))


if __name__ == "__main__":
    unittest.main()
