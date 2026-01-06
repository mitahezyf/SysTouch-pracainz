# testy diagnostyczne dla debugowania modelu PJM
from collections import Counter

import numpy as np
import torch

from app.sign_language.dataset import INPUT_SIZE
from app.sign_language.model import SignLanguageMLP


def test_threshold_logic() -> None:
    # testuje logike threshold na sztucznie wygenerowanych probach
    threshold = 0.7
    classes = ["A", "B", "C"]

    # przypadek 1: proba 0.6 < 0.7 -> None
    probas = np.array([0.6, 0.2, 0.2], dtype=np.float32)
    best_idx = int(np.argmax(probas))
    best_prob = float(probas[best_idx])
    result = classes[best_idx] if best_prob >= threshold else None
    assert (
        result is None
    ), f"oczekiwano None dla prob={best_prob} < threshold={threshold}"

    # przypadek 2: proba 0.8 >= 0.7 -> klasa A
    probas2 = np.array([0.8, 0.1, 0.1], dtype=np.float32)
    best_idx2 = int(np.argmax(probas2))
    best_prob2 = float(probas2[best_idx2])
    result2 = classes[best_idx2] if best_prob2 >= threshold else None
    assert (
        result2 == "A"
    ), f"oczekiwano A dla prob={best_prob2} >= threshold={threshold}"


def test_smoothing_stabilizer() -> None:
    # testuje logike smoothing/stabilizacji
    window = 10
    seq = ["A"] * 9 + ["B"]

    last_n = seq[-window:]
    counts = Counter(last_n)
    top = counts.most_common(1)[0]
    got = top[0] if top[1] >= 8 else None

    assert got == "A", f"Stabilizacja powinna zwrocic A. seq={seq} counts={counts}"


def test_model_input_output_shape() -> None:
    # sprawdza ksztalt wejscia/wyjscia modelu
    input_size = INPUT_SIZE
    num_classes = 36

    model = SignLanguageMLP(
        input_size=input_size, hidden_size=128, num_classes=num_classes
    )

    batch = torch.rand(4, input_size, dtype=torch.float32)
    output = model(batch)

    assert output.shape == (4, num_classes), f"zly shape wyjscia: {output.shape}"


def test_model_forward_no_nan() -> None:
    # sprawdza czy model nie zwraca NaN
    input_size = INPUT_SIZE
    num_classes = 36

    model = SignLanguageMLP(
        input_size=input_size, hidden_size=128, num_classes=num_classes
    )
    model.eval()

    batch = torch.rand(8, input_size, dtype=torch.float32)
    with torch.no_grad():
        output = model(batch)

    assert not torch.isnan(output).any(), "Model zwraca NaN"
    assert not torch.isinf(output).any(), "Model zwraca Inf"


def test_train_smoke_small() -> None:
    # szybki smoke test treningu na minimalnych danych
    n_classes = 3
    n_samples = 30
    input_size = INPUT_SIZE

    X = np.random.rand(n_samples, input_size).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)

    model = SignLanguageMLP(
        input_size=input_size, hidden_size=32, num_classes=n_classes
    )

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # mini trening
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        output = model(X_t)
        loss = criterion(output, y_t)
        loss.backward()
        optimizer.step()

    # sprawdz predykcje
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(X_t[:1]), dim=1).numpy()

    assert proba.shape == (1, n_classes), f"zly shape: {proba.shape}"
    assert np.isfinite(proba).all(), "proba zawiera NaN/Inf"


def test_translator_model_loads() -> None:
    # sprawdza czy translator poprawnie laduje model
    from app.sign_language.translator import SignTranslator

    translator = SignTranslator()

    assert translator.model is not None, "Translator nie zaladowal modelu"
    assert len(translator.classes) > 0, "Translator nie zaladowal klas"
    assert translator.model_input_size == INPUT_SIZE, (
        f"Rozmiar wejscia modelu powinien byc {INPUT_SIZE}, "
        f"jest {translator.model_input_size}"
    )


def test_translator_handles_valid_landmarks() -> None:
    # sprawdza czy translator przetwarza poprawne landmarki
    from app.sign_language.translator import SignTranslator

    translator = SignTranslator(confidence_entry=0.0, buffer_size=1, min_hold_ms=0)

    # poprawne landmarki (21, 3)
    landmarks = np.random.rand(21, 3).astype(np.float32)
    # nie powinien crashowac
    result = translator.process_landmarks(landmarks)
    # result moze byc None lub string (zalezy od modelu)
    assert result is None or isinstance(result, str)
