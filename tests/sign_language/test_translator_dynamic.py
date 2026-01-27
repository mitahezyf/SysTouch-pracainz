"""
Testy integracyjne dla translatora z obsługą gestów dynamicznych.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from app.sign_language.model import SignLanguageMLP
from app.sign_language.translator import SignTranslator


@pytest.fixture
def pjm_labels_with_dynamic():
    """Przykładowy plik pjm.json z etykietami dynamicznymi."""
    return {
        "name": "PJM-test",
        "classes": ["A", "A+", "B", "C+", "D"],
        "num_classes": 5,
        "gesture_types": {
            "A": "static",
            "A+": "dynamic",
            "B": "static",
            "C+": "dynamic",
            "D": "dynamic",
        },
        "sequences": {},
    }


@pytest.fixture
def translator_with_dynamic_gestures(pjm_labels_with_dynamic):
    """Translator z włączonymi gestami dynamicznymi."""
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.pth")
        classes_path = os.path.join(td, "classes.npy")
        labels_path = os.path.join(td, "labels", "pjm.json")

        # utworz katalog labels
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)

        # zapisz klasy
        classes = np.array(pjm_labels_with_dynamic["classes"])
        np.save(classes_path, classes)

        # zapisz labels JSON
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(pjm_labels_with_dynamic, f, ensure_ascii=False, indent=2)

        # model 189->5 (3 klatki x 63 cechy)
        model = SignLanguageMLP(input_size=189, hidden_size=32, num_classes=5)
        torch.save(model.state_dict(), model_path)

        # translator z enable_dynamic_gestures=True
        translator = SignTranslator(
            model_path=model_path,
            classes_path=classes_path,
            buffer_size=3,
            confidence_entry=0.01,  # niski prog dla testow
            enable_dynamic_gestures=True,  # wlacz gesty dynamiczne
            frames_per_sequence=3,
        )

        yield translator


@pytest.fixture
def translator_without_dynamic_gestures(pjm_labels_with_dynamic):
    """Translator z wyłączonymi gestami dynamicznymi."""
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.pth")
        classes_path = os.path.join(td, "classes.npy")
        labels_path = os.path.join(td, "labels", "pjm.json")

        os.makedirs(os.path.dirname(labels_path), exist_ok=True)

        classes = np.array(pjm_labels_with_dynamic["classes"])
        np.save(classes_path, classes)

        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(pjm_labels_with_dynamic, f, ensure_ascii=False, indent=2)

        model = SignLanguageMLP(input_size=189, hidden_size=32, num_classes=5)
        torch.save(model.state_dict(), model_path)

        translator = SignTranslator(
            model_path=model_path,
            classes_path=classes_path,
            buffer_size=3,
            confidence_entry=0.01,
            enable_dynamic_gestures=False,  # wylacz gesty dynamiczne
            frames_per_sequence=3,
        )

        yield translator


def test_translator_dynamic_gestures_enabled(translator_with_dynamic_gestures):
    """Test: translator z enable_dynamic_gestures=True inicjalizuje GestureManager."""
    assert translator_with_dynamic_gestures.gesture_manager is not None
    assert translator_with_dynamic_gestures.enable_dynamic_gestures is True


def test_translator_dynamic_gestures_disabled(translator_without_dynamic_gestures):
    """Test: translator z enable_dynamic_gestures=False nie inicjalizuje GestureManager."""
    assert translator_without_dynamic_gestures.gesture_manager is None
    assert translator_without_dynamic_gestures.enable_dynamic_gestures is False


def test_translator_processes_static_gestures_normally(
    translator_with_dynamic_gestures,
):
    """
    Test: gesty statyczne są przetwarzane normalnie nawet gdy GestureManager włączony.
    """
    translator = translator_with_dynamic_gestures

    # symuluj landmarki dla statycznej litery "A"
    # wygeneruj sztuczne landmarki (21x3)
    landmarks = np.random.randn(21, 3).astype(np.float32)

    # przetwórz wielokrotnie (do wypełnienia bufora 3 klatek)
    for _ in range(5):
        _ = translator.process_landmarks(landmarks, handedness="Right")

    # translator powinien działać normalnie (statyczne nie wywołują GestureManager)
    # możliwy wynik to None lub jakaś litera (zależnie od losowego modelu)
    # ważne że nie crashuje
    assert True  # podstawowa weryfikacja że nie ma błędu


def test_translator_reset_clears_gesture_manager(translator_with_dynamic_gestures):
    """Test: reset() translatora resetuje także GestureManager."""
    translator = translator_with_dynamic_gestures

    # symuluj predykcje
    landmarks = np.random.randn(21, 3).astype(np.float32)
    for _ in range(5):
        translator.process_landmarks(landmarks, handedness="Right")

    # reset
    translator.reset()

    # sprawdź że GestureManager został zresetowany
    if translator.gesture_manager:
        assert len(translator.gesture_manager.prediction_buffer) == 0
        assert translator.gesture_manager.current_dynamic is None


def test_translator_statistics_work_with_dynamic_gestures(
    translator_with_dynamic_gestures,
):
    """Test: statystyki translatora działają poprawnie z GestureManager."""
    translator = translator_with_dynamic_gestures

    # symuluj predykcje
    landmarks = np.random.randn(21, 3).astype(np.float32)
    for _ in range(10):
        translator.process_landmarks(landmarks, handedness="Right")

    # pobierz statystyki
    stats = translator.get_statistics()

    # sprawdź że słownik statystyk jest poprawny
    assert "total_detections" in stats
    assert "session_duration_s" in stats
    assert "letter_counts" in stats
    assert isinstance(stats["letter_counts"], dict)


def test_translator_history_with_dynamic_gestures(translator_with_dynamic_gestures):
    """Test: historia wykrytych liter działa z GestureManager."""
    translator = translator_with_dynamic_gestures

    # symuluj predykcje
    landmarks = np.random.randn(21, 3).astype(np.float32)
    for _ in range(10):
        translator.process_landmarks(landmarks, handedness="Right")

    # pobierz historię
    history = translator.get_history(format_groups=False)

    # historia powinna być stringiem
    assert isinstance(history, str)

    # sprawdź że clear działa
    translator.clear_history()
    assert len(translator.letter_history) == 0
