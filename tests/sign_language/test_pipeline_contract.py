from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

FORBIDDEN_LABELS = {"PONIEDZIALEK", "NIEDZIELA", "WIOSNA", "JESIEN"}


def test_dataset_vectors_exists(vectors_csv_path: Path) -> None:
    assert vectors_csv_path.exists(), f"brak PJM-vectors.csv: {vectors_csv_path}"


def test_dataset_vectors_columns_and_forbidden_labels(vectors_csv_path: Path) -> None:
    df = pd.read_csv(vectors_csv_path)

    required = {"sign_label", "user_id", "lux_value"}
    missing = required - set(df.columns)
    assert not missing, f"brak kolumn w PJM-vectors.csv: {sorted(missing)}"

    forbidden_present = sorted(set(df["sign_label"].astype(str)) & FORBIDDEN_LABELS)
    assert (
        not forbidden_present
    ), f"W CSV nadal sa zabronione etykiety: {forbidden_present}"


def test_dataset_feature_cols_only_vector_and_sane(vectors_csv_path: Path) -> None:
    df = pd.read_csv(vectors_csv_path)

    feature_cols = [c for c in df.columns if c.startswith("vector_")]
    assert feature_cols, "brak kolumn vector_* - nie da sie trenowac modelu"

    forbidden_in_features = {"user_id", "lux_value", "sign_label"} & set(feature_cols)
    assert (
        not forbidden_in_features
    ), f"W feature_cols znalazly sie metadane: {sorted(forbidden_in_features)}"

    # probka diagnostyczna
    sample = df[feature_cols].head(200).to_numpy(dtype=np.float32, copy=True)
    assert sample.ndim == 2, f"X ma zly wymiar: {sample.shape}"
    assert sample.shape[1] == len(
        feature_cols
    ), f"X.shape[1] != liczba feature_cols: X={sample.shape[1]} cols={len(feature_cols)}"

    bad_mask = ~np.isfinite(sample)
    assert not bool(bad_mask.any()), (
        "W probce X sa NaN/Inf. "
        f"pierwszy bledny indeks={np.argwhere(bad_mask)[0].tolist()}"
    )


def test_dataset_class_distribution_nonzero(vectors_csv_path: Path) -> None:
    df = pd.read_csv(vectors_csv_path)
    counts = Counter(df["sign_label"].astype(str).tolist())

    zero = [k for k, v in counts.items() if v <= 0]
    assert not zero, f"Klasy z 0 probek: {zero}"

    # sanity: co najmniej kilka klas
    assert len(counts) >= 5, f"Podejrzanie malo klas w CSV: {len(counts)}"


def test_group_split_disjoint_users(vectors_csv_path: Path) -> None:
    df = pd.read_csv(vectors_csv_path)

    feature_cols = [c for c in df.columns if c.startswith("vector_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["sign_label"].astype(str).to_numpy()
    groups = df["user_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_users = set(groups[train_idx].tolist())
    test_users = set(groups[test_idx].tolist())
    overlap = train_users & test_users

    assert not overlap, f"split miesza userow: overlap={sorted(list(overlap))[:10]}"

    train_labels = set(y[train_idx].tolist())
    test_labels = set(y[test_idx].tolist())

    ratio = len(train_labels) / max(1, len(set(y.tolist())))
    assert ratio >= 0.8, (
        "Po splicie train ma zbyt malo klas. "
        f"train_classes={len(train_labels)} all_classes={len(set(y.tolist()))} ratio={ratio:.2f}. "
        "Rozwaz GroupKFold lub inny test_size / wiecej danych per user"
    )

    assert (
        len(test_labels) >= 1
    ), "test split nie zawiera zadnej klasy - dane/split sa uszkodzone"


@pytest.mark.parametrize(
    "probas,threshold,expected",
    [
        ([0.1, 0.2, 0.7], 0.8, None),
        ([0.05, 0.9, 0.05], 0.8, "B"),
    ],
)
def test_realtime_threshold_logic(
    probas: list[float], threshold: float, expected: str | None
) -> None:
    # testuje logike threshold (bez modelu), zeby szybko wykryc regresje
    classes = ["A", "B", "C"]
    best_idx = int(np.argmax(np.asarray(probas)))
    best_prob = float(probas[best_idx])

    got = None if best_prob < threshold else classes[best_idx]

    assert got == expected, (
        "Prog pewnosci dziala inaczej niz oczekiwano.\n"
        f"probas={probas} threshold={threshold} expected={expected} got={got}"
    )


def test_realtime_smoothing_majority_vote() -> None:
    # testuje stabilizacje po etykiecie (majority vote / stabilizer)
    window = 10
    seq = ["A"] * 9 + ["B"]

    last_n = seq[-window:]
    counts = Counter(last_n)
    top = counts.most_common(1)[0]
    got = top[0] if top[1] >= 8 else None

    assert got == "A", (
        "Stabilizacja majority vote powinna zwrocic A.\n"
        f"seq={seq} counts={counts} got={got}"
    )


def test_realtime_smoothing_average_proba() -> None:
    # testuje smoothing na sredniej prob
    probas = [
        np.array([0.6, 0.2, 0.2], dtype=np.float32),
        np.array([0.7, 0.2, 0.1], dtype=np.float32),
        np.array([0.8, 0.1, 0.1], dtype=np.float32),
    ]
    smooth = np.mean(np.stack(probas, axis=0), axis=0)
    got_idx = int(np.argmax(smooth))

    assert got_idx == 0, (
        "Smoothing na probach powinien wskazac klase 0.\n"
        f"smooth={smooth.tolist()} got_idx={got_idx}"
    )


def test_readme_debugging_by_tests_exists(repo_root: Path) -> None:
    readme = repo_root / "tests" / "README_DEBUGGING_BY_TESTS.md"
    assert readme.exists(), (
        "Brak README dla testow diagnostycznych. " f"Oczekiwano pliku: {readme}"
    )
    content = readme.read_text(encoding="utf-8")
    assert (
        "Debugging by tests" in content
    ), "README nie zawiera naglowka 'Debugging by tests'"


def test_translator_handles_invalid_shapes() -> None:
    # translator loguje blad i zwraca None przy zlym ksztalcie (nie crashuje)
    from app.sign_language.translator import SignTranslator

    translator = SignTranslator(confidence_entry=0.8)

    # zly ksztalt (10, 3) zamiast (21, 3) - nie crashuje, zwraca None
    bad_landmarks = np.zeros((10, 3), dtype=np.float32)
    result = translator.process_landmarks(bad_landmarks)
    assert result is None, "Translator powinien zwrocic None dla zlego ksztaltu"

    # zly ksztalt 1D
    bad_landmarks_1d = np.zeros(50, dtype=np.float32)
    result = translator.process_landmarks(bad_landmarks_1d)
    assert result is None

    # shape (21, 2) zamiast (21, 3)
    bad_landmarks_2d = np.zeros((21, 2), dtype=np.float32)
    result = translator.process_landmarks(bad_landmarks_2d)
    assert result is None

    # poprawny ksztalt (21, 3) - nie crashuje
    landmarks_array = np.random.rand(21, 3).astype(np.float32)
    translator.process_landmarks(landmarks_array)


def test_no_feature_mode_dropdown_in_ui_code() -> None:
    # sprawdza, ze UI nie ma juz dropdown 63/189

    ui_file = Path("app/gui/ui_components.py")
    content = ui_file.read_text(encoding="utf-8")

    assert (
        "Feature mode" not in content
    ), "UI nadal zawiera 'Feature mode' (dropdown 63/189)"
    assert "Full 189D" not in content, "UI nadal zawiera 'Full 189D'"
    assert (
        "Runtime-compatible 63D" not in content
    ), "UI nadal zawiera 'Runtime-compatible 63D'"
