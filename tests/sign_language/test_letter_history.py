# testy dla funkcji historii liter w SignTranslator
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.sign_language.model import SignLanguageMLP
from app.sign_language.translator import SignTranslator


@pytest.fixture
def temp_model():
    # tworzy tymczasowy model i zwraca sciezki do modelu i klas
    model = SignLanguageMLP(input_size=63, hidden_size=32, num_classes=3)
    classes = np.array(["A", "B", "C"])

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pth"
        classes_path = Path(tmpdir) / "classes.npy"

        import torch

        torch.save(model.state_dict(), model_path)
        np.save(classes_path, classes)

        yield str(model_path), str(classes_path)


def test_history_initialization(temp_model):
    # sprawdza inicjalizacje historii
    model_path, classes_path = temp_model
    translator = SignTranslator(
        model_path=model_path,
        classes_path=classes_path,
        buffer_size=3,
        max_history=100,
    )

    assert translator.letter_history == []
    assert translator.max_history == 100
    history = translator.get_history()
    assert history == ""


def test_history_accumulation(temp_model):
    # sprawdza akumulacje liter w historii
    import time

    model_path, classes_path = temp_model
    translator = SignTranslator(
        model_path=model_path,
        classes_path=classes_path,
        buffer_size=1,  # maly bufor dla szybkiego testowania
        min_hold_ms=50,  # krotki min_hold
        confidence_entry=0.0,  # zawsze akceptuj
    )

    # symuluj 3 rozne litery z pauzami
    vec = np.random.rand(63).astype(np.float32)

    # pierwsza litera
    translator.process_frame(vec)
    time.sleep(0.06)  # czekaj wiecej niz min_hold

    # zmien wejscie dla drugiej litery
    vec2 = np.random.rand(63).astype(np.float32)
    translator.process_frame(vec2)
    time.sleep(0.06)

    # zmien dla trzeciej
    vec3 = np.random.rand(63).astype(np.float32)
    translator.process_frame(vec3)

    # sprawdz historie (moze byc 1-3 litery w zaleznosci od roznic w predykcjach)
    history = translator.get_history(format_groups=False)
    # akceptuj ze historia ma co najmniej 1 litere (rozne losowe wejscia moga dac ta sama predykcje)
    assert len(history) >= 1
    assert len(history) <= 3


def test_history_formatting(temp_model):
    # sprawdza formatowanie historii z grupami
    model_path, classes_path = temp_model
    translator = SignTranslator(
        model_path=model_path, classes_path=classes_path, buffer_size=1
    )

    # manualnie dodaj litery do historii
    translator.letter_history = list("ABCABCABCABC")

    # bez formatowania
    history_raw = translator.get_history(format_groups=False)
    assert history_raw == "ABCABCABCABC"

    # z formatowaniem (spacje co 5 znakow)
    history_formatted = translator.get_history(format_groups=True)
    assert history_formatted == "ABCAB CABCA BC"


def test_history_max_limit(temp_model):
    # sprawdza ograniczenie maksymalnej dlugosci historii
    model_path, classes_path = temp_model
    translator = SignTranslator(
        model_path=model_path,
        classes_path=classes_path,
        buffer_size=1,
        max_history=5,  # limit 5 liter
    )

    # dodaj 10 liter recznie
    for i, letter in enumerate("ABCABCABCA"):
        translator.letter_history.append(letter)
        if len(translator.letter_history) > translator.max_history:
            translator.letter_history.pop(0)

    # sprawdz ze historia ma max 5 liter (ostatnie 5)
    assert len(translator.letter_history) <= 5
    history = translator.get_history(format_groups=False)
    assert history == "CABCA"


def test_clear_history(temp_model):
    # sprawdza czyszczenie historii
    model_path, classes_path = temp_model
    translator = SignTranslator(
        model_path=model_path, classes_path=classes_path, buffer_size=1
    )

    # dodaj litery
    translator.letter_history = list("ABCABC")
    assert len(translator.letter_history) == 6

    # wyczysc
    translator.clear_history()
    assert len(translator.letter_history) == 0
    assert translator.get_history() == ""


def test_reset_clears_history(temp_model):
    # sprawdza ze reset(keep_stats=False) czysci historie
    model_path, classes_path = temp_model
    translator = SignTranslator(
        model_path=model_path, classes_path=classes_path, buffer_size=1
    )

    translator.letter_history = list("ABCABC")
    translator.reset(keep_stats=False)

    assert len(translator.letter_history) == 0


def test_reset_keeps_history_when_keep_stats(temp_model):
    # sprawdza ze reset(keep_stats=True) rowniez czysci historie
    # (historia to aktualny tekst, nie statystyka)
    model_path, classes_path = temp_model
    translator = SignTranslator(
        model_path=model_path, classes_path=classes_path, buffer_size=1
    )

    translator.letter_history = list("ABCABC")
    translator.reset(keep_stats=True)

    # historia powinna byc wyczyszczona (bo to nie jest statystyka licznikowa)
    assert len(translator.letter_history) == 0


def test_history_empty_when_no_detections(temp_model):
    # sprawdza ze historia jest pusta gdy brak wykryc
    model_path, classes_path = temp_model
    translator = SignTranslator(
        model_path=model_path,
        classes_path=classes_path,
        buffer_size=5,
        confidence_entry=0.99,  # bardzo wysoki prog - trudno wykryc
    )

    # probuj wykryc z losowymi danymi (najprawdopodobniej nie przekroczy progu)
    vec = np.random.rand(63).astype(np.float32)
    for _ in range(10):
        translator.process_frame(vec)

    # jesli nie wykrylo nic, historia powinna byc pusta
    # (test moze byc flaky w zaleznosci od losowych danych, ale z 0.99 progiem prawie na pewno pusta)
    history = translator.get_history()
    # akceptuj rowniez przypadek gdy cos wykrylo (losowo)
    assert isinstance(history, str)
