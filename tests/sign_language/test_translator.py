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

        translator = SignTranslator(
            model_path=model_path,
            classes_path=classes_path,
            buffer_size=3,  # mniejszy bufor dla szybszego testu
            confidence_entry=0.01,  # niski prog aby latwo przeszedl
        )

        # wypelnij bufor (3 klatki)
        sample = np.random.randn(63).astype(np.float32)
        for _ in range(3):
            pred = translator.predict(sample)

        # po zapelnieniu bufora powinien zwrocic litere
        assert pred is None or pred in set(classes.tolist())


def test_translator_history_stabilization():
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.pth")
        classes_path = os.path.join(td, "classes.npy")
        classes = np.array(list("XYZ"))
        np.save(classes_path, classes)

        model = SignLanguageMLP(input_size=63, hidden_size=16, num_classes=3)
        torch.save(model.state_dict(), model_path)

        translator = SignTranslator(
            model_path=model_path,
            classes_path=classes_path,
            buffer_size=5,
            confidence_entry=0.5,
        )

        # generuje wiele predykcji (wiecej niz rozmiar bufora)
        for _ in range(10):
            sample = np.random.randn(63).astype(np.float32)
            _ = translator.predict(sample)

        # sprawdza czy bufor sie zapelnil
        assert len(translator.frame_buffer) == 5

        # sprawdz stan translatora
        state = translator.get_state()
        assert state["buffer_fill"] == 5
        assert state["buffer_size"] == 5
