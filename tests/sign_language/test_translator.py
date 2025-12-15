import os
import tempfile

import numpy as np
import torch

from app.sign_language.model import SignLanguageMLP
from app.sign_language.translator import SignTranslator


def test_translator_predict_single():
    # przygotowuje sztuczny model i klasy
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.pth")
        classes_path = os.path.join(td, "classes.npy")
        classes = np.array(list("AB"))
        np.save(classes_path, classes)

        # model 63->2
        model = SignLanguageMLP(input_size=63, hidden_size=32, num_classes=2)
        torch.save(model.state_dict(), model_path)

        translator = SignTranslator(model_path=model_path, classes_path=classes_path)
        # wektor wejsciowy 63 floatow
        sample = np.random.randn(63).astype(np.float32)
        pred = translator.predict(sample)
        assert pred in set(classes.tolist())


def test_translator_history_stabilization():
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.pth")
        classes_path = os.path.join(td, "classes.npy")
        classes = np.array(list("XYZ"))
        np.save(classes_path, classes)

        model = SignLanguageMLP(input_size=63, hidden_size=16, num_classes=3)
        torch.save(model.state_dict(), model_path)

        translator = SignTranslator(model_path=model_path, classes_path=classes_path)
        # symuluje wiele predykcji
        for _ in range(10):
            sample = np.random.randn(63).astype(np.float32)
            _ = translator.predict(sample)
        # po kilku predykcjach historia nie jest pusta
        assert len(translator.history) > 0
        # najczestszy element w historii to jeden z klas
        assert translator.history[0] in set(classes.tolist())
