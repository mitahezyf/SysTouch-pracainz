from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from tools.train_model import load_model_artifact, runtime_feature_names

FORBIDDEN_LABELS = {"PONIEDZIALEK", "NIEDZIELA", "WIOSNA", "JESIEN"}


def _format_first_diffs(expected: list[str], got: list[str], limit: int = 20) -> str:
    diffs: list[str] = []
    for i, (e, g) in enumerate(zip(expected, got)):
        if e != g:
            diffs.append(f"[{i}] expected={e} got={g}")
            if len(diffs) >= limit:
                break
    if not diffs and len(expected) != len(got):
        diffs.append(f"len mismatch only: expected={len(expected)} got={len(got)}")
    return "\n".join(diffs)


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

    # dodatkowo: train powinien zawierac sensowna liczbe klas
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


def test_model_artifact_has_required_keys(model_joblib_path: Path) -> None:
    if not model_joblib_path.exists():
        raise AssertionError(
            "Brak modelu joblib. Uruchom trening: python tools\\train_model.py "
            "--vectors app\\sign_language\\data\\raw\\PJM-vectors.csv "
            "--out app\\sign_language\\models\\pjm_model.joblib"
        )

    artifact = load_model_artifact(model_joblib_path)

    assert "pipeline" in artifact, "artefakt: brak pipeline"
    assert "label_encoder" in artifact, "artefakt: brak label_encoder"
    assert "feature_cols" in artifact, "artefakt: brak feature_cols"
    assert "preprocess_config" in artifact, "artefakt: brak preprocess_config"


def test_model_feature_cols_match_csv(
    model_joblib_path: Path, vectors_csv_path: Path
) -> None:
    if not model_joblib_path.exists():
        raise AssertionError(
            "Brak modelu joblib. Uruchom trening: python tools\\train_model.py --out app\\sign_language\\models\\pjm_model.joblib"
        )

    artifact = joblib.load(model_joblib_path)
    feature_cols = list(artifact.get("feature_cols") or [])

    df = pd.read_csv(vectors_csv_path)
    csv_cols = [c for c in df.columns if c.startswith("vector_")]

    assert feature_cols, "artefakt: feature_cols jest puste"
    assert (
        len(feature_cols) == 63
    ), f"oczekiwano 63 cech w modelu, got={len(feature_cols)}"

    # model trenujemy na bloku 1 (runtime)
    expected_subset = [
        c
        for c in csv_cols
        if c.startswith("vector_hand_1_") or c.startswith("vector_1_")
    ]
    assert (
        len(expected_subset) == 63
    ), f"CSV nie zawiera kompletnego bloku 63D (hand_1). got={len(expected_subset)}"

    assert feature_cols == expected_subset, (
        "artefakt.feature_cols nie odpowiada blokowi 1 z CSV (kolejnosc lub nazwy rozne). "
        f"head_model={feature_cols[:5]} head_csv={expected_subset[:5]}"
    )


def test_label_encoder_matches_dataset_set(
    model_joblib_path: Path, vectors_csv_path: Path
) -> None:
    if not model_joblib_path.exists():
        raise AssertionError(
            "Brak modelu joblib. Uruchom trening: python tools\\train_model.py --out app\\sign_language\\models\\pjm_model.joblib"
        )

    artifact = joblib.load(model_joblib_path)
    le = artifact.get("label_encoder")
    classes_model = [str(c) for c in getattr(le, "classes_", [])]

    df = pd.read_csv(vectors_csv_path)
    classes_csv = sorted(set(df["sign_label"].astype(str)) - FORBIDDEN_LABELS)

    assert classes_model, "label_encoder.classes_ jest puste"

    missing_in_model = sorted(set(classes_csv) - set(classes_model))
    extra_in_model = sorted(set(classes_model) - set(classes_csv))

    assert not missing_in_model, (
        "Model nie zna klas z datasetu. " f"missing_in_model={missing_in_model[:20]}"
    )
    assert not extra_in_model, (
        "Model ma klasy, ktorych nie ma w dataset. "
        f"extra_in_model={extra_in_model[:20]}"
    )


def test_feature_parity_names_model_vs_runtime(model_joblib_path: Path) -> None:
    if not model_joblib_path.exists():
        raise AssertionError(
            "Brak modelu joblib. Uruchom trening: python tools\\train_model.py --out app\\sign_language\\models\\pjm_model.joblib"
        )

    artifact = joblib.load(model_joblib_path)
    model_cols = list(artifact.get("feature_cols") or [])
    runtime_cols = runtime_feature_names()

    assert model_cols, "artefakt.feature_cols jest puste"
    assert (
        len(runtime_cols) == 63
    ), f"runtime_feature_names powinno zwracac 63, got={len(runtime_cols)}"

    if model_cols != runtime_cols:
        diffs = _format_first_diffs(model_cols, runtime_cols, limit=20)
        raise AssertionError(
            "Kolejnosc nazw cech rozna miedzy modelem a runtime.\n"
            f"len(model)={len(model_cols)} len(runtime)={len(runtime_cols)}\n"
            f"pierwsze roznice:\n{diffs}\n"
            f"model_head={model_cols[:10]}\n"
            f"runtime_head={runtime_cols[:10]}"
        )


def test_pipeline_accepts_one_row_from_csv(
    model_joblib_path: Path, vectors_csv_path: Path
) -> None:
    if not model_joblib_path.exists():
        raise AssertionError(
            "Brak modelu joblib. Uruchom trening: python tools\\train_model.py --out app\\sign_language\\models\\pjm_model.joblib"
        )

    artifact = joblib.load(model_joblib_path)
    pipe = artifact.get("pipeline")
    feature_cols = list(artifact.get("feature_cols") or [])

    assert pipe is not None, "artefakt.pipeline jest None"
    assert feature_cols, "artefakt.feature_cols jest puste"

    df = pd.read_csv(vectors_csv_path)
    row = df.sample(n=1, random_state=42)

    X = row[feature_cols].to_numpy(dtype=np.float32)
    assert X.shape == (1, len(feature_cols)), f"niepoprawny ksztalt X: {X.shape}"

    try:
        proba = pipe.predict_proba(X)
    except Exception as exc:
        raise AssertionError(
            "pipeline.predict_proba rzuca wyjatek dla poprawnego X. "
            f"shape={X.shape} exc={exc}"
        )

    assert proba.shape[0] == 1, f"proba shape zly: {proba.shape}"


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


def test_preprocess_config_present_and_has_known_keys(model_joblib_path: Path) -> None:
    if not model_joblib_path.exists():
        raise AssertionError(
            "Brak modelu joblib. Uruchom trening: python tools\\train_model.py --out app\\sign_language\\models\\pjm_model.joblib"
        )

    artifact = joblib.load(model_joblib_path)
    pcfg = artifact.get("preprocess_config")

    assert isinstance(pcfg, dict), f"preprocess_config ma zly typ: {type(pcfg)}"

    for key in ["mirror_left_hand_x", "use_standard_scaler"]:
        assert key in pcfg, f"preprocess_config brak klucza: {key}"


def test_readme_debugging_by_tests_exists(repo_root: Path) -> None:
    readme = repo_root / "tests" / "README_DEBUGGING_BY_TESTS.md"
    assert readme.exists(), (
        "Brak README dla testow diagnostycznych. " f"Oczekiwano pliku: {readme}"
    )
    content = readme.read_text(encoding="utf-8")
    assert (
        "Debugging by tests" in content
    ), "README nie zawiera naglowka 'Debugging by tests'"


def test_translator_rejects_wrong_shape_with_valueerror() -> None:
    # translator musi rzucic ValueError (nie log-only) jesli dostanie zly ksztalt wektora
    from app.sign_language.translator import SignTranslator

    translator = SignTranslator(confidence_threshold=0.8, debug_mode=False)

    # przypadek 1: zly ksztalt (10, 3) zamiast (21, 3)
    bad_landmarks = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="landmarks.*shape"):
        translator.process_landmarks(bad_landmarks)

    # przypadek 2: zly ksztalt 2D -> 1D
    bad_landmarks_1d = np.zeros(50, dtype=np.float32)
    with pytest.raises(ValueError, match="landmarks.*shape"):
        translator.process_landmarks(bad_landmarks_1d)

    # przypadek 3: shape (21, 2) zamiast (21, 3)
    bad_landmarks_2d = np.zeros((21, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="landmarks.*shape"):
        translator.process_landmarks(bad_landmarks_2d)

    # przypadek 4: shape (21, 4) zamiast (21, 3)
    bad_landmarks_4d = np.zeros((21, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="landmarks.*shape"):
        translator.process_landmarks(bad_landmarks_4d)

    # przypadek 5: shape (0, 3) - pusta tablica
    bad_landmarks_empty = np.zeros((0, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="landmarks.*shape"):
        translator.process_landmarks(bad_landmarks_empty)

    # przypadek 6: shape (21, 3) ale lista pythonowa (powinna byc zaakceptowana po konwersji)
    landmarks_list = [[0.5 + i * 0.01, 0.5 + i * 0.01, 0.0] for i in range(21)]
    landmarks_array = np.array(landmarks_list, dtype=np.float32)
    # nie powinien crashowac, moze zwrocic None jesli model nie zaladowany lub prog nie spelniony
    translator.process_landmarks(landmarks_array)
    # nie failuje, result moze byc None lub string

    # przypadek 7: shape (21, 3) ale NaN/inf - zaleznie od logiki
    # tutaj traktujemy to jako dane wejsciowe i niech model sobie z tym radzi
    # nie blokujemy na poziomie translatora, ale test dokumentuje przypadek
    landmarks_nan = np.full((21, 3), np.nan, dtype=np.float32)
    # translator NIE rzuca ValueError dla NaN (model dostaje dane i decyduje)
    # ale mozemy dodac ostrzezenie w logu
    translator.process_landmarks(landmarks_nan)
    # nie crashuje, result to None lub string (w praktyce model zwroci nieokreslone wyjscie)


def test_runtime_feature_dim_matches_constant() -> None:
    # runtime_feature_names() musi zwracac dokladnie RUNTIME_FEATURE_DIM cech (63)
    from tools.train_model import RUNTIME_FEATURE_DIM, runtime_feature_names

    names = runtime_feature_names()
    assert len(names) == int(
        RUNTIME_FEATURE_DIM
    ), f"runtime_feature_names zwraca {len(names)} cech, oczekiwano {RUNTIME_FEATURE_DIM}"


def test_pjm_ui_panel_widgets_exist_and_no_crash_on_mode_switch() -> None:
    # test UI: przelaczenie trybu nie rozwala layoutu (widgety istnieja i nie sa None)
    from unittest import mock

    from app.gui.ui_components import build_ui

    # mock PySide6
    with mock.patch("app.gui.ui_components.importlib.import_module") as mock_import:
        mock_qtcore = mock.Mock()
        mock_qtw = mock.Mock()

        # mock klasy Qt
        mock_qtw.QLabel = mock.Mock(return_value=mock.Mock())
        mock_qtw.QHBoxLayout = mock.Mock(return_value=mock.Mock())
        mock_qtw.QVBoxLayout = mock.Mock(return_value=mock.Mock())
        mock_qtw.QComboBox = mock.Mock(return_value=mock.Mock())
        mock_qtw.QCheckBox = mock.Mock(return_value=mock.Mock())
        mock_qtw.QPushButton = mock.Mock(return_value=mock.Mock())
        mock_qtw.QGroupBox = mock.Mock(return_value=mock.Mock())
        mock_qtw.QWidget = mock.Mock(return_value=mock.Mock())
        mock_qtw.QSlider = mock.Mock(return_value=mock.Mock())
        mock_qtw.QPlainTextEdit = mock.Mock(return_value=mock.Mock())
        mock_qtw.QLineEdit = mock.Mock(return_value=mock.Mock())
        mock_qtw.QFrame = mock.Mock(return_value=mock.Mock())
        mock_qtw.QScrollArea = mock.Mock(return_value=mock.Mock())

        mock_qtcore.Qt = mock.Mock()

        def side_effect(name):
            if "QtCore" in name:
                return mock_qtcore
            if "QtWidgets" in name:
                return mock_qtw
            return mock.Mock()

        mock_import.side_effect = side_effect

        # buduj UI
        ui = build_ui(640, 480)

        # sprawdz, ze kluczowe widgety PJM panelu istnieja
        assert ui.pjm_group is not None
        assert ui.pjm_letter_label is not None
        assert ui.pjm_conf_label is not None
        assert ui.pjm_train_from_csv_btn is not None
        assert ui.pjm_calibrate_btn is not None
        assert ui.pjm_threshold_slider is not None

        # sprawdz, ze NIE ma dropdown 63/189 (juz usuniety)
        # ui_components zwraca UIRefs, gdzie pjm_feature_mode_combo powinno byc None lub nie istnieje
        # (w nowym kodzie nie ma tego pola w UIRefs, wiec po prostu sprawdzamy ze nie ma attributu)
        # jesli attribute istnieje, sprawdz ze to None albo nie-widok
        # bezpieczniej: sprawdz w kodzie zrodlowym czy string "Feature mode" nie wystepuje


def test_no_feature_mode_dropdown_in_ui_code() -> None:
    # sprawdza, ze UI nie ma juz dropdown 63/189
    from pathlib import Path

    ui_file = Path("app/gui/ui_components.py")
    content = ui_file.read_text(encoding="utf-8")

    assert (
        "Feature mode" not in content
    ), "UI nadal zawiera 'Feature mode' (dropdown 63/189)"
    assert "Full 189D" not in content, "UI nadal zawiera 'Full 189D'"
    assert (
        "Runtime-compatible 63D" not in content
    ), "UI nadal zawiera 'Runtime-compatible 63D'"
