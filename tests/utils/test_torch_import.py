# test importu torch po reinstalacji
import pytest


def test_torch_import():
    # sprawdza czy torch moze byc zaimportowany
    try:
        import torch

        assert torch is not None
    except ImportError:
        pytest.skip("PyTorch nie jest zainstalowany")


def test_torch_version():
    # sprawdza wersje torch
    try:
        import torch

        assert torch.__version__ is not None
        assert len(torch.__version__) > 0
    except ImportError:
        pytest.skip("PyTorch nie jest zainstalowany")


def test_torch_device():
    # sprawdza czy torch moze utworzyc device CPU
    try:
        import torch

        device = torch.device("cpu")
        assert device is not None
        assert str(device) == "cpu"
    except ImportError:
        pytest.skip("PyTorch nie jest zainstalowany")
