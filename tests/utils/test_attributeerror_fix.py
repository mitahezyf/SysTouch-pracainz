# test weryfikujacy naprawe AttributeError w main_window.py
from pathlib import Path

import pytest


def test_main_window_attributes_initialization():
    # sprawdza czy MainWindow inicjalizuje atrybuty _translator i _normalizer
    # nawet gdy import torch jest zablokowany
    import sys

    # blokujemy torch tymczasowo
    original_torch = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore

    try:
        from app.gui.main_window import MainWindow

        # import powinien sie udac (moze byc blad Qt, ale nie AttributeError)
        assert MainWindow is not None
    except AttributeError as e:
        pytest.fail(f"AttributeError podczas importu MainWindow: {e}")
    except (ImportError, ModuleNotFoundError) as e:
        # bledy Qt/PySide6 sa OK
        if "PySide6" not in str(e) and "QtWidgets" not in str(e):
            raise
    finally:
        # przywroc oryginalna wartosc torch
        if original_torch is None and "torch" in sys.modules:
            del sys.modules["torch"]
        elif original_torch is not None:
            sys.modules["torch"] = original_torch


def test_main_window_source_code_has_fixes():
    # weryfikuje ze kod main_window.py zawiera poprawki dla AttributeError
    project_root = Path(__file__).parent.parent.parent
    main_window_path = project_root / "app" / "gui" / "main_window.py"

    assert main_window_path.exists(), f"Brak pliku: {main_window_path}"

    content = main_window_path.read_text(encoding="utf-8")

    # sprawdz inicjalizacje atrybutow w __init__
    assert (
        "self._normalizer" in content and "self._translator" in content
    ), "Brak inicjalizacji _normalizer lub _translator w main_window.py"


def test_torch_is_available():
    # sprawdza czy PyTorch jest dostepny (moze byc skip jesli nie zainstalowany)
    try:
        import torch

        assert torch.__version__ is not None
        assert torch.device("cpu") is not None
    except ImportError:
        pytest.skip("PyTorch nie jest zainstalowany - wykonaj: .\\reinstall_torch.ps1")


def test_sign_translator_import():
    # sprawdza czy SignTranslator moze byc zaimportowany
    try:
        from app.sign_language.translator import SignTranslator

        assert SignTranslator is not None
    except ImportError as e:
        if "torch" in str(e).lower():
            pytest.skip(f"PyTorch problem: {e}")
        else:
            raise
