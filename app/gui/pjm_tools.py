import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


CALIBRATION_PATH = Path("app/sign_language/models/calibration.json")


@dataclass(frozen=True)
class ModelInfo:
    model_path: Path
    mtime_s: float
    n_classes: int
    n_features: int
    threshold: float
    window: int
    classes: list[str]


def load_model_info(
    model_path: Path, default_threshold: float, default_window: int
) -> ModelInfo:
    if not model_path.exists():
        raise FileNotFoundError(f"brak modelu: {model_path}")

    artifact = joblib.load(model_path)
    classes = [str(c) for c in getattr(artifact.get("label_encoder"), "classes_", [])]
    feature_cols = list(artifact.get("feature_cols") or [])

    threshold = float(default_threshold)
    window = int(default_window)

    # jesli jest kalibracja, nadpisuje
    calib = load_calibration(CALIBRATION_PATH)
    if calib is not None:
        threshold = float(calib.get("threshold", threshold))
        window = int(calib.get("window", window))

    return ModelInfo(
        model_path=model_path,
        mtime_s=model_path.stat().st_mtime,
        n_classes=int(len(classes)),
        n_features=int(len(feature_cols)),
        threshold=float(threshold),
        window=int(window),
        classes=classes,
    )


def load_calibration(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("nie mozna odczytac kalibracji: %s", exc)
        return None


def save_calibration(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def suggest_threshold_from_max_proba(max_probas: np.ndarray) -> float:
    # bierze 10 percentyl jako prog (niski prog na start), ogranicza do [0.05, 0.95]
    if max_probas.size == 0:
        return 0.8
    q = float(np.quantile(max_probas, 0.10))
    return float(min(0.95, max(0.05, q)))
