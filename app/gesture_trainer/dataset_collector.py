from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.gesture_engine.logger import logger
from app.sign_language.features import FeatureConfig, FeatureExtractor


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    output_dir: Path
    session_id: str
    mirror_left: bool = True
    include_landmarks: bool = True
    include_features: bool = True


def _now_ts_ms() -> int:
    return int(time.time() * 1000)


def build_session_id() -> str:
    """Generuje ID sesji z timestampem dla czytelnosci.

    Format: YYYYMMDD_HHMMSS_nanoseconds
    Przyklad: 20260111_012410_123456789
    """
    import time
    from datetime import datetime

    now = datetime.now()
    # YYYYMMDD_HHMMSS
    date_str = now.strftime("%Y%m%d_%H%M%S")
    # nanoseconds dla unikalnosci
    nanos = time.time_ns() % 1_000_000_000
    return f"{date_str}_{nanos:09d}"


def ensure_dirs(cfg: CollectionConfig) -> tuple[Path, Path, Path]:
    session_dir = cfg.output_dir / cfg.session_id
    clips_dir = session_dir / "clips"
    features_dir = session_dir / "features"
    session_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, clips_dir, features_dir


def write_session_meta(
    cfg: CollectionConfig, *, labels: list[str], extra: dict[str, object] | None = None
) -> Path:
    session_dir, _, _ = ensure_dirs(cfg)
    meta_path = session_dir / "session.json"
    payload: dict[str, object] = {
        "session_id": cfg.session_id,
        "created_ts_ms": _now_ts_ms(),
        "labels": labels,
        "mirror_left": cfg.mirror_left,
        "include_landmarks": cfg.include_landmarks,
        "include_features": cfg.include_features,
    }
    if extra:
        payload.update(extra)

    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta_path


def csv_header(*, include_landmarks: bool, include_features: bool) -> list[str]:
    base = [
        "session_id",
        "clip_id",
        "label",
        "frame_idx",
        "timestamp_ms",
        "handedness",
        "has_hand",
        "mirror_applied",
    ]

    cols: list[str] = list(base)

    if include_landmarks:
        for i in range(21):
            cols.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])

    if include_features:
        for i in range(63):
            cols.append(f"feat_{i}")

    return cols


def build_row(
    *,
    session_id: str,
    clip_id: str,
    label: str,
    frame_idx: int,
    timestamp_ms: int,
    handedness: str | None,
    has_hand: bool,
    mirror_applied: bool,
    landmarks21: np.ndarray | None,
    features63: np.ndarray | None,
    include_landmarks: bool,
    include_features: bool,
) -> list[object]:
    row: list[object] = [
        session_id,
        clip_id,
        label,
        int(frame_idx),
        int(timestamp_ms),
        handedness or "",
        int(bool(has_hand)),
        int(bool(mirror_applied)),
    ]

    if include_landmarks:
        if landmarks21 is None:
            row.extend([""] * (21 * 3))
        else:
            flat = np.asarray(landmarks21, dtype=np.float32).reshape(-1)
            row.extend([float(x) for x in flat])

    if include_features:
        if features63 is None:
            row.extend([""] * 63)
        else:
            feat = np.asarray(features63, dtype=np.float32).reshape(-1)
            if feat.shape != (63,):
                raise ValueError(
                    f"niepoprawny ksztalt features: {feat.shape}, oczekiwano (63,)"
                )
            row.extend([float(x) for x in feat])

    return row


class ClipCSVWriter:
    def __init__(
        self,
        path: Path,
        *,
        include_landmarks: bool,
        include_features: bool,
    ) -> None:
        self.path = path
        self.include_landmarks = include_landmarks
        self.include_features = include_features
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(
            csv_header(
                include_landmarks=include_landmarks, include_features=include_features
            )
        )

    def write(self, row: list[object]) -> None:
        self._writer.writerow(row)

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()


class FrameFeaturePipeline:
    def __init__(self, *, mirror_left: bool) -> None:
        self._cfg = FeatureConfig(mirror_left=mirror_left)
        self._extractor = FeatureExtractor(self._cfg)

    def compute(
        self, landmarks21: np.ndarray, handedness: str | None
    ) -> tuple[np.ndarray, bool]:
        # mirror jest aplikowany tylko dla handedness=Left (patrz features.py)
        mirror_applied = (
            self._cfg.mirror_left
            and handedness is not None
            and handedness.lower().startswith("left")
        )
        feat = self._extractor.extract(landmarks21, handedness=handedness)
        return feat, mirror_applied


def safe_close(writer: ClipCSVWriter | None) -> None:
    if writer is None:
        return
    try:
        writer.close()
    except Exception:
        logger.exception("nie mozna zamknac writer")
