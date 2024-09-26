import torch

from tk.fused_softmax import softmax


def test_softmax():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch)
