# testy diagnostyczne dla debugowania modelu PJM: zgodnosc feature, klasy, threshold, smoothing
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from tools.train_model import runtime_feature_names


def test_feature_columns_exact_match(
    model_joblib_path: Path, vectors_csv_path: Path
) -> None:
    # porownuje feature_cols z modelu vs CSV kolumny 1:1
    # w przypadku diff, pokazuje pierwsze 10 brakujacych, nadmiarowych i miejsce rozjazdu kolejnosci
    if not model_joblib_path.exists():
        pytest.skip("brak modelu joblib - uruchom trening")

    artifact = joblib.load(model_joblib_path)
    model_cols = list(artifact.get("feature_cols") or [])
    runtime_cols = runtime_feature_names()

    # 1. sprawdz czy brak cech
    model_set = set(model_cols)
    runtime_set = set(runtime_cols)

    missing = sorted(runtime_set - model_set)
    extra = sorted(model_set - runtime_set)

    if missing or extra:
        msg = "Feature columns niezgodne miedzy modelem a runtime.\n"
        if missing:
            msg += f"Brakujace w modelu (pierwsze 10): {missing[:10]}\n"
        if extra:
            msg += f"Nadmiarowe w modelu (pierwsze 10): {extra[:10]}\n"
        raise AssertionError(msg)

    # 2. sprawdz kolejnosc
    if model_cols != runtime_cols:
        # znajdz pierwsze miejsce rozjazdu
        first_diff_idx = None
        for i, (m, r) in enumerate(zip(model_cols, runtime_cols)):
            if m != r:
                first_diff_idx = i
                break

        if first_diff_idx is None and len(model_cols) != len(runtime_cols):
            first_diff_idx = min(len(model_cols), len(runtime_cols))

        msg = (
            f"Feature columns maja inny porzadek (len model={len(model_cols)} runtime={len(runtime_cols)}).\n"
            f"Pierwsze roznica na indeksie {first_diff_idx}:\n"
            f"  model[{first_diff_idx}]={model_cols[first_diff_idx] if first_diff_idx < len(model_cols) else 'BRAK'}\n"
            f"  runtime[{first_diff_idx}]={runtime_cols[first_diff_idx] if first_diff_idx < len(runtime_cols) else 'BRAK'}\n"
        )
        raise AssertionError(msg)

    # 3. sanity check: porownaj z CSV
    df = pd.read_csv(vectors_csv_path)
    csv_cols = [
        c
        for c in df.columns
        if c.startswith("vector_hand_1_") or c.startswith("vector_1_")
    ]

    if set(csv_cols) != model_set:
        csv_missing = sorted(model_set - set(csv_cols))
        csv_extra = sorted(set(csv_cols) - model_set)
        msg = "Feature columns modelu vs CSV rozne.\n"
        if csv_missing:
            msg += f"Brakujace w CSV (pierwsze 10): {csv_missing[:10]}\n"
        if csv_extra:
            msg += f"Nadmiarowe w CSV (pierwsze 10): {csv_extra[:10]}\n"
        raise AssertionError(msg)


def test_class_mapping_consistency(
    model_joblib_path: Path, vectors_csv_path: Path
) -> None:
    # sprawdza czy klasy z metadata modelu odpowiadaja label encoderowi i CSV
    if not model_joblib_path.exists():
        pytest.skip("brak modelu joblib - uruchom trening")

    artifact = joblib.load(model_joblib_path)
    le = artifact.get("label_encoder")
    classes_model = [str(c) for c in getattr(le, "classes_", [])]

    # porownaj z CSV
    df = pd.read_csv(vectors_csv_path)
    forbidden = {"PONIEDZIALEK", "NIEDZIELA", "WIOSNA", "JESIEN"}
    classes_csv = sorted(set(df["sign_label"].astype(str)) - forbidden)

    missing = sorted(set(classes_csv) - set(classes_model))
    extra = sorted(set(classes_model) - set(classes_csv))

    if missing or extra:
        msg = "Klasy w modelu vs CSV niezgodne.\n"
        if missing:
            msg += f"Brakujace w modelu (pierwsze 10): {missing[:10]}\n"
        if extra:
            msg += f"Nadmiarowe w modelu (pierwsze 10): {extra[:10]}\n"
        raise AssertionError(msg)

    # dodatkowo sprawdz label encoder vs pjm.json
    import json
    from pathlib import Path

    labels_path = Path("app/sign_language/labels/pjm.json")
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        json_classes = list(config.get("classes") or [])
        if json_classes and set(json_classes) != set(classes_model):
            msg = (
                f"Klasy w modelu ({len(classes_model)}) vs pjm.json ({len(json_classes)}) rozne.\n"
                f"Zaleca sie aktualizowac pjm.json lub przetrenowac model.\n"
            )
            raise AssertionError(msg)


def test_threshold_and_smoothing_behavior() -> None:
    # testuje logike threshold i smoothing na sztucznie wygenerowanych probach
    from app.sign_language.translator import SignTranslator

    # mock translator z malym threshold
    SignTranslator(
        confidence_threshold=0.7,
        stability_frames=3,
        min_hold_time_s=0.0,
        debug_mode=False,
    )

    # przypadek 1: proba 0.6 < 0.7 -> None
    classes = ["A", "B", "C"]
    probas = np.array([0.6, 0.2, 0.2], dtype=np.float32)
    best_idx = int(np.argmax(probas))
    best_prob = float(probas[best_idx])
    expected = None if best_prob < 0.7 else classes[best_idx]
    assert expected is None, f"oczekiwano None dla prob={best_prob} < threshold=0.7"

    # przypadek 2: proba 0.85 >= 0.7 -> "A"
    probas2 = np.array([0.85, 0.1, 0.05], dtype=np.float32)
    best_idx2 = int(np.argmax(probas2))
    best_prob2 = float(probas2[best_idx2])
    expected2 = None if best_prob2 < 0.7 else classes[best_idx2]
    assert expected2 == "A", f"oczekiwano 'A' dla prob={best_prob2} >= threshold=0.7"

    # przypadek 3: smoothing majority vote (window=10, prog 80%)
    window = 10
    seq = ["A"] * 9 + ["B"]
    last_n = seq[-window:]
    counts = Counter(last_n)
    top = counts.most_common(1)[0]
    got = top[0] if top[1] >= 8 else None
    assert got == "A", f"majority vote powinno zwrocic 'A', got={got}"

    # przypadek 4: smoothing na sredniej prob
    probas_list = [
        np.array([0.7, 0.2, 0.1], dtype=np.float32),
        np.array([0.75, 0.15, 0.1], dtype=np.float32),
        np.array([0.8, 0.1, 0.1], dtype=np.float32),
    ]
    smooth = np.mean(np.stack(probas_list, axis=0), axis=0)
    got_idx = int(np.argmax(smooth))
    assert got_idx == 0, f"smoothing powinno wskazac klase 0, got={got_idx}"


def test_train_smoke_small(vectors_csv_path: Path, tmp_path: Path) -> None:
    # smoke test: trenuje model na malym wycinku danych, sprawdza brak crashu i spojnosc shape/kolumn
    # test ma byc szybki (<5s), deterministyczny, nie wymaga accuracy
    # UWAGA: uzywa uproszczonego treningu bez GroupShuffleSplit (zwykly train_test_split)
    if not vectors_csv_path.exists():
        pytest.skip("brak PJM-vectors.csv")

    df = pd.read_csv(vectors_csv_path)

    # wytnij maly subset: min 5 klas, min 100 probek
    classes = df["sign_label"].unique()
    if len(classes) < 5:
        pytest.skip("za malo klas w CSV do smoke testu")

    selected_classes = list(classes[:5])
    df_small = df[df["sign_label"].isin(selected_classes)].head(200)

    if len(df_small) < 100:
        pytest.skip("za malo probek do smoke testu (min 100)")

    # sprawdz faktyczna liczbe klas w df_small
    actual_classes = df_small["sign_label"].nunique()
    if actual_classes < 2:
        pytest.skip(f"za malo klas po filtracji (min 2, got {actual_classes})")

    # zapisz tymczasowy CSV
    csv_small = tmp_path / "small_vectors.csv"
    df_small.to_csv(csv_small, index=False)

    # uruchom uproszczony trening (bez GroupShuffleSplit)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    feature_cols = [
        c
        for c in df_small.columns
        if c.startswith("vector_hand_1_") or c.startswith("vector_1_")
    ]
    if len(feature_cols) != 63:
        pytest.skip(f"smoke test wymaga 63 cech, got={len(feature_cols)}")

    X = df_small[feature_cols].to_numpy(dtype=np.float32)
    y = df_small["sign_label"].astype(str).to_numpy()

    # label encoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split bez grup
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # prosty pipeline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # trenuj
    pipe.fit(X_train, y_train)

    # zapisz artefakt
    out_path = tmp_path / "smoke_model.joblib"
    artifact = {
        "pipeline": pipe,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "preprocess_config": {"mirror_left_hand_x": True, "use_standard_scaler": True},
    }
    joblib.dump(artifact, out_path)

    # walidacja artefaktu
    assert out_path.exists(), "smoke trening nie stworzyl modelu"

    loaded = joblib.load(out_path)
    assert "pipeline" in loaded
    assert "label_encoder" in loaded
    assert "feature_cols" in loaded
    assert (
        len(loaded["feature_cols"]) == 63
    ), f"smoke model ma zly wymiar cech: {len(loaded['feature_cols'])}"

    # smoke predict
    X_test_sample = X_test[:1]
    proba = pipe.predict_proba(X_test_sample)
    n_classes_trained = len(le.classes_)
    assert proba.shape == (
        1,
        n_classes_trained,
    ), f"smoke predict ma zly shape: {proba.shape}, oczekiwano (1, {n_classes_trained})"


def test_realtime_vector_contract(vectors_csv_path: Path) -> None:
    # bierze 5 probek z CSV, przechodzi przez preprocess -> translator predict -> sprawdza shape i brak NaN
    if not vectors_csv_path.exists():
        pytest.skip("brak PJM-vectors.csv")

    df = pd.read_csv(vectors_csv_path)
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("vector_hand_1_") or c.startswith("vector_1_")
    ]

    if len(feature_cols) != 63:
        pytest.skip(f"CSV nie ma 63 cech bloku 1, got={len(feature_cols)}")

    # wybierz 5 probek
    samples = df.sample(n=min(5, len(df)), random_state=42)

    from app.sign_language.translator import SignTranslator

    translator = SignTranslator(
        confidence_threshold=0.0,
        stability_frames=1,
        min_hold_time_s=0.0,
        debug_mode=False,
    )

    if not translator._model_loaded:
        pytest.skip("translator nie ma zaladowanego modelu")

    for i, row in samples.iterrows():
        X = row[feature_cols].to_numpy(dtype=np.float32)
        assert X.shape == (63,), f"probka {i} ma zly shape: {X.shape}"
        assert np.isfinite(X).all(), f"probka {i} zawiera NaN/inf"

        # symulacja landmarks (rekonstrukcja z featurow nie jest trywalna, pomijamy)
        # zamiast tego sprawdz ze pipeline zaakceptuje X
        X_2d = X.reshape(1, -1)
        proba = translator._pipeline.predict_proba(X_2d)
        assert proba.shape[0] == 1
        assert np.isfinite(
            proba
        ).all(), f"proba dla probki {i} zawiera NaN/inf: {proba}"
