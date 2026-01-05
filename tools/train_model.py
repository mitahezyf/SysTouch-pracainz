import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

FORBIDDEN_LABELS = {"PONIEDZIALEK", "NIEDZIELA", "WIOSNA", "JESIEN"}

# kontrakt CSV (historyczny 189D): 3 bloki po 63 cechy
# - blok 1: vector_hand_1_{x|y|z} + vector_1_{1..20}_{x|y|z}
# - blok 2: vector_hand_2_{x|y|z} + vector_2_{1..20}_{x|y|z}
# - blok 3: vector_hand_3_{x|y|z} + vector_3_{1..20}_{x|y|z}
# razem 189 cech; runtime w aplikacji wylicza tylko jeden blok (63D) bez replikacji

RUNTIME_FEATURE_DIM = 63


@dataclass(frozen=True)
class PreprocessConfig:
    # opisuje zasady preprocessingu, zeby trening i runtime byly spojne
    # mirror lewej dloni jest realizowany na poziomie feature buildera (from_mediapipe_landmarks)
    mirror_left_hand_x: bool = True
    # dane vectors sa juz znormalizowane do unit vectors, scaler dalej pomaga dla klasyfikatora
    use_standard_scaler: bool = True


def _load_vectors_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"brak pliku: {path}")

    df = pd.read_csv(path)

    # filtruje zabronione klasy (dni tygodnia i pory roku)
    if "sign_label" in df.columns:
        df = df[~df["sign_label"].astype(str).isin(FORBIDDEN_LABELS)]

    required = {"user_id", "sign_label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"brak wymaganych kolumn: {sorted(missing)}")

    # krytyczne: X = tylko vector_*; user_id i lux_value nie sa cechami
    feature_cols = [c for c in df.columns if c.startswith("vector_")]
    if not feature_cols:
        raise RuntimeError("brak kolumn vector_* w pliku")

    # stabilizuje kolejnosc cech: hand_1 potem 1..20
    # jesli CSV ma inna kolejnosc, tutaj zawsze sortujemy deterministycznie
    def _feature_key(col: str) -> tuple[int, int, int, int]:
        # sortuje deterministycznie:
        # 0) vector_hand_{h}_{axis} -> (0, h, 0, axis)
        # 1) vector_{h}_{i}_{axis} -> (1, h, i, axis)
        axis_map = {"x": 0, "y": 1, "z": 2}

        if col.startswith("vector_hand_"):
            parts = col.split("_")
            hand_idx = 0
            axis = 99
            if len(parts) >= 4:
                try:
                    hand_idx = int(parts[2])
                except Exception:
                    hand_idx = 0
                axis = axis_map.get(parts[3], 99)
            return (0, hand_idx, 0, axis)

        parts = col.split("_")
        # vector_{hand}_{i}_{axis}
        hand_idx = 0
        bone_idx = 0
        axis = 99
        if len(parts) >= 4:
            try:
                hand_idx = int(parts[1])
            except Exception:
                hand_idx = 0
            try:
                bone_idx = int(parts[2])
            except Exception:
                bone_idx = 0
            axis = axis_map.get(parts[3], 99)

        return (1, hand_idx, bone_idx, axis)

    feature_cols_sorted = sorted(feature_cols, key=_feature_key)

    # sanity check na liczbe cech (oczekujemy 63, ale nie blokujemy gdy repo sie zmieni)
    logger.info(
        "features: count=%d first=%s last=%s",
        len(feature_cols_sorted),
        feature_cols_sorted[0],
        feature_cols_sorted[-1],
    )

    # usuwa ew. wiersze z NaN w cechach
    df = df.dropna(subset=feature_cols_sorted + ["sign_label", "user_id"])

    df.attrs["feature_cols"] = feature_cols_sorted
    return df


def _build_pipeline(random_state: int) -> Pipeline:
    # stabilny baseline: scaler + logistic regression
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    n_jobs=None,
                    multi_class="auto",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(
    vectors_csv: Path,
    output_path: Path,
    test_size: float,
    random_state: int,
    preprocess_config: PreprocessConfig,
    feature_mode: int = 63,
) -> None:
    df = _load_vectors_csv(vectors_csv)
    all_feature_cols = list(df.attrs["feature_cols"])

    if feature_mode != 63:
        raise ValueError(
            "obslugiwany jest tylko format 63D (bez wyboru w GUI). "
            "Jesli masz stary model 189D, uruchom retrain na nowym pipeline"
        )

    # bierze tylko pierwszy blok zgodny z runtime (hand_1 + vector_1_1..20)
    feature_cols = [
        c
        for c in all_feature_cols
        if c.startswith("vector_hand_1_") or c.startswith("vector_1_")
    ]

    if not feature_cols:
        raise RuntimeError("feature_cols puste po wyborze 63D")

    # krytyczne: X = tylko vector_* w tej kolejnosci
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["sign_label"].astype(str).to_numpy()
    groups = df["user_id"].to_numpy()

    if "lux_value" in df.columns:
        logger.info(
            "info: lux_value wykryte w CSV, ale zgodnie z wymaganiami nie jest uzywane jako cecha"
        )

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_users = set(groups[train_idx].tolist())
    test_users = set(groups[test_idx].tolist())
    overlap = train_users & test_users
    if overlap:
        raise RuntimeError(
            f"split nieprawidlowy - train/test dziela user_id: {sorted(list(overlap))[:10]}"
        )

    logger.info(
        "split groups: train_users=%d test_users=%d", len(train_users), len(test_users)
    )
    logger.info("split samples: train=%d test=%d", len(train_idx), len(test_idx))

    pipe = _build_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    labels_sorted = list(range(len(le.classes_)))
    report = classification_report(
        y_test,
        y_pred,
        labels=labels_sorted,
        target_names=le.classes_.tolist(),
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    logger.info("\n" + "=" * 60)
    logger.info("classification_report (group split):\n%s", report)
    logger.info("confusion_matrix:\n%s", cm)

    artifact: dict[str, Any] = {
        "pipeline": pipe,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "preprocess_config": {
            "mirror_left_hand_x": preprocess_config.mirror_left_hand_x,
            "use_standard_scaler": preprocess_config.use_standard_scaler,
        },
        "meta": {
            "vectors_csv": str(vectors_csv),
            "test_size": test_size,
            "random_state": random_state,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(le.classes_)),
            "feature_mode": int(feature_mode),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    logger.info("zapisano model do: %s", output_path)


def runtime_feature_names() -> list[str]:
    """Zwraca nazwy cech runtime w stalej kolejnosci.

    Kontrakt runtime/trening: 63D (jeden blok) bez replikacji.

    Kolejnosc:
    - vector_hand_1 (x,y,z)
    - vector_1_{1..20} (x,y,z)
    """

    cols: list[str] = []

    cols.extend(
        [
            "vector_hand_1_x",
            "vector_hand_1_y",
            "vector_hand_1_z",
        ]
    )

    for i in range(1, 21):
        cols.extend(
            [
                f"vector_1_{i}_x",
                f"vector_1_{i}_y",
                f"vector_1_{i}_z",
            ]
        )

    return cols


def load_model_artifact(path: Path) -> dict[str, Any]:
    """Laduje artefakt modelu i waliduje podstawowe wymagania kontraktu."""

    if not path.exists():
        raise FileNotFoundError(
            f"brak artefaktu modelu: {path}. uruchom: python tools\\train_model.py --out {path}"
        )

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"niepoprawny format artefaktu: {type(artifact)}")

    required_keys = {"pipeline", "label_encoder", "feature_cols", "preprocess_config"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise KeyError(f"artefakt nie zawiera kluczy: {sorted(missing)}")

    feature_cols = artifact.get("feature_cols")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("artefakt: feature_cols musi byc niepusta lista")

    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trenuje model PJM z pliku PJM-vectors.csv"
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        required=True,
        help="Sciezka do PJM-vectors.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("app/sign_language/models/pjm_model.joblib"),
        help="Sciezka wyjsciowa modelu joblib",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--feature-mode",
        type=int,
        default=63,
        help="DEPRECATED: zawsze 63 (jeden format bez wyboru)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    train_model(
        vectors_csv=args.vectors,
        output_path=args.out,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        preprocess_config=PreprocessConfig(),
        feature_mode=int(args.feature_mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
