import os
import tempfile

import numpy as np
import torch

from app.sign_language.model import SignLanguageMLP
from app.sign_language.translator import SignTranslator


def _save_model(state_dict_path: str, hidden_size: int, num_classes: int = 4):
    model = SignLanguageMLP(
        input_size=63, hidden_size=hidden_size, num_classes=num_classes
    )
    torch.save(model.state_dict(), state_dict_path)


def test_translator_infers_hidden_size():
    with tempfile.TemporaryDirectory() as td:
        classes = np.array(list("ABCD"))
        classes_path = os.path.join(td, "classes.npy")
        np.save(classes_path, classes)
        model_path = os.path.join(td, "model.pth")
        _save_model(model_path, hidden_size=48, num_classes=len(classes))
        tr = SignTranslator(model_path=model_path, classes_path=classes_path)
        first_linear = tr.model.network[0]
        assert first_linear.out_features == 48


def test_translator_fallback_hidden_size():
    with tempfile.TemporaryDirectory() as td:
        classes = np.array(list("AB"))
        classes_path = os.path.join(td, "classes.npy")
        np.save(classes_path, classes)
        # sztuczny state_dict bez network.0.weight -> fallback na 256
        # nowa architektura: network.12 to warstwa output (po 3 hidden + batchnorm)
        fake_state = {
            "network.12.weight": torch.randn(len(classes), 128),  # output layer
            "network.12.bias": torch.randn(len(classes)),
        }
        model_path = os.path.join(td, "model.pth")
        torch.save(fake_state, model_path)
        tr = SignTranslator(model_path=model_path, classes_path=classes_path)
        first_linear = tr.model.network[0]
        assert first_linear.out_features == 256  # fallback (bylo 128, teraz 256)
