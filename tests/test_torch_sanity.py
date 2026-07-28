"""Quick test: verify torch CUDA is operational after install."""
import torch
import pytest


def test_torch_version():
    # Project requires torch>=2.1 (see pyproject.toml dependencies).
    # Reject anything older; accept any newer release (cu124, cu128, cu129, ...)
    # so the test is portable across H20 / A100 / T4 image variants.
    major_minor = tuple(int(x) for x in torch.__version__.split(".")[:2])
    assert major_minor >= (2, 1), (
        f"torch {torch.__version__} is too old; project requires >= 2.1"
    )


def test_cuda_available():
    assert torch.cuda.is_available(), "torch.cuda not available — driver mismatch?"


def test_cuda_compute():
    x = torch.randn(128, 128, device="cuda")
    y = x @ x.T
    assert y.shape == (128, 128)
    assert torch.isfinite(y).all()


def test_cuda_device_name():
    assert torch.cuda.device_count() >= 1
    print("\n[torch]", torch.__version__, "cuda=", torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}]", torch.cuda.get_device_name(i))
