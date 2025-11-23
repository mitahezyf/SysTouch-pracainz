import os
import tempfile

import numpy as np
import pytest
import torch

from app.sign_language.model import SignLanguageMLP
from app.sign_language.translator import SignTranslator


def test_translator_missing_classes_file():
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.pth")
        # zapisujemy poprawny state_dict
        model = SignLanguageMLP(input_size=63, hidden_size=32, num_classes=2)
        torch.save(model.state_dict(), model_path)
        classes_path = os.path.join(td, "no_classes.npy")
        with pytest.raises(FileNotFoundError):
            SignTranslator(model_path=model_path, classes_path=classes_path)


def test_translator_missing_model_file():
    with tempfile.TemporaryDirectory() as td:
        classes_path = os.path.join(td, "classes.npy")
        np.save(classes_path, np.array(list("AB")))
        model_path = os.path.join(td, "no_model.pth")
        with pytest.raises(FileNotFoundError):
            SignTranslator(model_path=model_path, classes_path=classes_path)


def test_translator_bad_classes_file():
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.pth")
        classes_path = os.path.join(td, "classes.npy")
        # plik z losowymi bajtami zamiast numpy array
        with open(classes_path, "wb") as f:
            f.write(b"not-a-npy")
        model = SignLanguageMLP(input_size=63, hidden_size=16, num_classes=3)
        torch.save(model.state_dict(), model_path)
        with pytest.raises(RuntimeError):
            SignTranslator(model_path=model_path, classes_path=classes_path)


def test_translator_corrupted_model_file():
    with tempfile.TemporaryDirectory() as td:
        classes_path = os.path.join(td, "classes.npy")
        np.save(classes_path, np.array(list("AB")))
        model_path = os.path.join(td, "model.pth")
        with open(model_path, "wb") as f:
            f.write(b"garbage")
        with pytest.raises(RuntimeError):
            SignTranslator(model_path=model_path, classes_path=classes_path)
