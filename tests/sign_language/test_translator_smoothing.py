# testy dla translatora z buforem i histereza
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from app.sign_language.model import SignLanguageMLP
from app.sign_language.translator import SignTranslator


@pytest.fixture
def mock_model_and_classes():
    # tworzy tymczasowy model i klasy dla testow
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pth"
        classes_path = Path(tmpdir) / "test_classes.npy"

        # mini model
        model = SignLanguageMLP(input_size=63, hidden_size=32, num_classes=3)
        torch.save(model.state_dict(), model_path)

        # 3 klasy testowe
        classes = np.array(["A", "B", "C"])
        np.save(classes_path, classes)

        yield str(model_path), str(classes_path)


def test_translator_initialization(mock_model_and_classes):
    # test inicjalizacji translatora
    model_path, classes_path = mock_model_and_classes

    translator = SignTranslator(
        model_path=model_path,
        classes_path=classes_path,
        buffer_size=7,
        min_hold_ms=400,
    )

    assert translator.buffer_size == 7
    assert translator.min_hold_ms == 400
    assert len(translator.classes) == 3
    assert translator.current_letter is None


def test_translator_reset(mock_model_and_classes):
    # test resetowania stanu
    model_path, classes_path = mock_model_and_classes
    translator = SignTranslator(model_path=model_path, classes_path=classes_path)

    # symuluj jakis stan
    translator.current_letter = "A"
    translator.current_confidence = 0.8

    translator.reset()

    assert translator.current_letter is None
    assert translator.current_confidence == 0.0
    assert len(translator.frame_buffer) == 0


def test_translator_buffer_fill(mock_model_and_classes):
    # test zapelniania bufora
    model_path, classes_path = mock_model_and_classes
    translator = SignTranslator(
        model_path=model_path, classes_path=classes_path, buffer_size=3
    )

    # pierwsza klatka - bufor niezapelniony, brak predykcji
    frame1 = np.random.rand(63).astype(np.float32)
    result = translator.process_frame(frame1)
    assert result is None  # bufor niepelny

    # druga klatka
    frame2 = np.random.rand(63).astype(np.float32)
    result = translator.process_frame(frame2)
    assert result is None  # nadal niepelny

    # trzecia klatka - bufor zapelniony
    frame3 = np.random.rand(63).astype(np.float32)
    result = translator.process_frame(frame3)
    # teraz moze byc litera lub None (zalezne od confidence)
    assert result is None or isinstance(result, str)


def test_translator_invalid_input(mock_model_and_classes):
    # test walidacji rozmiaru wejscia
    model_path, classes_path = mock_model_and_classes
    translator = SignTranslator(model_path=model_path, classes_path=classes_path)

    # zly rozmiar wektora
    bad_frame = np.random.rand(50).astype(np.float32)
    result = translator.process_frame(bad_frame)

    # powinien zachowac aktualny stan (None) zamiast crashowac
    assert result is None or isinstance(result, str)


def test_translator_get_state(mock_model_and_classes):
    # test diagnostyki stanu
    model_path, classes_path = mock_model_and_classes
    translator = SignTranslator(model_path=model_path, classes_path=classes_path)

    state = translator.get_state()

    assert "current_letter" in state
    assert "confidence" in state
    assert "buffer_fill" in state
    assert "buffer_size" in state
    assert "time_held_ms" in state


def test_translator_missing_model():
    # test braku pliku modelu/klas
    with pytest.raises(FileNotFoundError, match="Brak pliku"):
        SignTranslator(model_path="nonexistent.pth", classes_path="nonexistent.npy")


def test_translator_predict_alias(mock_model_and_classes):
    # test kompatybilnosci wstecznej metody predict
    model_path, classes_path = mock_model_and_classes
    translator = SignTranslator(model_path=model_path, classes_path=classes_path)

    frame = np.random.rand(63).astype(np.float32)
    result = translator.predict(frame)

    # predict to alias dla process_frame
    assert result is None or isinstance(result, str)
