# testy licznika statystyk w translatorze
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from app.sign_language.model import SignLanguageMLP
from app.sign_language.translator import SignTranslator


@pytest.fixture
def translator_with_stats():
    # tworzy translator do testow statystyk
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pth"
        classes_path = Path(tmpdir) / "classes.npy"

        model = SignLanguageMLP(input_size=63, hidden_size=16, num_classes=3)
        torch.save(model.state_dict(), model_path)
        classes = np.array(["A", "B", "C"])
        np.save(classes_path, classes)

        translator = SignTranslator(
            model_path=str(model_path),
            classes_path=str(classes_path),
            buffer_size=3,
            min_hold_ms=10,
            confidence_entry=0.01,
        )
        yield translator


def test_statistics_initialization(translator_with_stats):
    # test inicjalizacji licznika
    translator = translator_with_stats

    assert hasattr(translator, "letter_stats")
    assert hasattr(translator, "total_detections")
    assert hasattr(translator, "session_start_time")
    assert translator.total_detections == 0
    assert len(translator.letter_stats) == 0


def test_statistics_counting(translator_with_stats):
    # sprawdza czy statystyki sa prawidlowo zliczane
    translator = translator_with_stats

    # generuje kilka klatek do przetworzenia
    for _ in range(5):
        frame = np.random.rand(63).astype(np.float32)
        translator.process_frame(frame)

    stats = translator.get_statistics()

    assert "total_detections" in stats
    assert "session_duration_s" in stats
    assert "letter_counts" in stats
    assert "most_common" in stats
    assert stats["session_duration_s"] > 0


def test_statistics_reset_keep(translator_with_stats):
    # test resetu z zachowaniem statystyk
    translator = translator_with_stats

    # wygeneruj jakies wykrycia
    for _ in range(5):
        frame = np.random.rand(63).astype(np.float32)
        translator.process_frame(frame)

    total_before = translator.total_detections

    # reset z zachowaniem statystyk
    translator.reset(keep_stats=True)

    assert translator.total_detections == total_before
    assert translator.current_letter is None  # stan zresetowany


def test_statistics_reset_clear(translator_with_stats):
    # test resetu z czyszczeniem statystyk
    translator = translator_with_stats

    # wygeneruj wykrycia
    for _ in range(5):
        frame = np.random.rand(63).astype(np.float32)
        translator.process_frame(frame)

    # reset bez zachowania statystyk
    translator.reset(keep_stats=False)

    assert translator.total_detections == 0
    assert len(translator.letter_stats) == 0
    assert translator.current_letter is None


def test_state_includes_stats(translator_with_stats):
    # test czy get_state zawiera statystyki
    translator = translator_with_stats

    state = translator.get_state()

    assert "total_detections" in state
    assert "session_duration_s" in state
    assert "detections_per_minute" in state
    assert "unique_letters" in state
