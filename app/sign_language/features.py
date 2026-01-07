from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

EPS = 1e-9


def unit(vector: np.ndarray) -> np.ndarray:
    """Zwraca wektor znormalizowany z ochrona przed zerem."""
    norm = np.linalg.norm(vector)
    if norm < EPS:
        return np.zeros_like(vector, dtype=np.float32)
    result: np.ndarray = (vector / norm).astype(np.float32)
    return result


@dataclass
class FeatureConfig:
    mirror_left: bool = True
    scale_by_mcp: bool = False


# wagi do rekonstrukcji brakujacych baz MCP (dopasowane do datasetu)
_BASE_INDEX_WEIGHTS = np.array(
    [1.1675016, -2.9088350, 2.0806787, 4.1512337, -3.4905784], dtype=np.float32
)
_BASE_MIDDLE_WEIGHTS = np.array(
    [1.1216259, -3.2278988, 2.1774454, 4.5146465, -3.5858188], dtype=np.float32
)
_BASE_RING_WEIGHTS = np.array(
    [1.0546410, -3.1840618, 1.9982234, 4.3267107, -3.1955132], dtype=np.float32
)
_BASE_PINKY_WEIGHTS = np.array(
    [0.92379045, -2.2040052, 1.1102204, 2.7982580, -1.6282636], dtype=np.float32
)


def _build_landmarks21_from_points25(points25: np.ndarray) -> np.ndarray:
    """Mapowanie 25->21 zgodne z tools/verify_mediapipe_reconstruction_parity."""
    mp = np.zeros((21, 3), dtype=np.float32)
    mp[0] = points25[0]
    mp[1] = points25[1]
    mp[2] = points25[2]
    mp[3] = points25[3]
    mp[4] = points25[4]

    mp[5] = points25[6]
    mp[6] = points25[7]
    mp[7] = points25[8]
    mp[8] = points25[9]

    mp[9] = points25[11]
    mp[10] = points25[12]
    mp[11] = points25[13]
    mp[12] = points25[14]

    mp[13] = points25[16]
    mp[14] = points25[17]
    mp[15] = points25[18]
    mp[16] = points25[19]

    mp[17] = points25[21]
    mp[18] = points25[22]
    mp[19] = points25[23]
    mp[20] = points25[24]
    return mp


def _compute_hand_normal(rel: np.ndarray) -> np.ndarray:
    # normalna dloni: cross(index_mcp - wrist, pinky_mcp - wrist)
    wrist = rel[0]
    idx = rel[5]
    pinky = rel[17]
    normal = np.cross(idx - wrist, pinky - wrist)
    return unit(normal)


def _compute_feature_vectors(rel: np.ndarray) -> np.ndarray:
    """63D: hand normal + 20 unit bone vectors w zadanej kolejnosci."""
    # kolejnosc kosci
    pairs: Iterable[tuple[int, int]] = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    )

    feats = [_compute_hand_normal(rel)]
    for pa, pb in pairs:
        vec = rel[pb] - rel[pa]
        feats.append(unit(vec))

    result: np.ndarray = np.concatenate(feats, axis=0).astype(np.float32)
    return result


def _features_from_landmarks21(
    landmarks21: np.ndarray, handedness: str | None, cfg: FeatureConfig
) -> np.ndarray:
    if landmarks21.shape != (21, 3):
        raise ValueError(
            f"niepoprawny ksztalt landmarks: {landmarks21.shape}, oczekiwano (21, 3)"
        )

    rel = np.asarray(landmarks21, dtype=np.float32) - np.asarray(
        landmarks21[0], dtype=np.float32
    )

    # mirror dla lewej reki (z perspektywy uzytkownika)
    # MediaPipe "Left" = twoja prawa reka, "Right" = twoja lewa reka
    # Dataset byl nagrany lewa reka, wiec mirrorujemy "Left" (prawa reka uzytkownika)
    if (
        cfg.mirror_left
        and handedness is not None
        and handedness.lower().startswith("left")
    ):
        rel[:, 0] *= -1.0

    if cfg.scale_by_mcp:
        ref = rel[9]
        norm = float(np.linalg.norm(ref))
        if norm > EPS:
            rel = rel / norm

    return _compute_feature_vectors(rel)


def _build_points25_from_mediapipe21(mp21: np.ndarray) -> np.ndarray:
    if mp21.shape != (21, 3):
        raise ValueError(
            f"niepoprawny ksztalt mediapipe: {mp21.shape}, oczekiwano (21, 3)"
        )

    mp = np.asarray(mp21, dtype=np.float32)
    pts = np.zeros((25, 3), dtype=np.float32)

    pts[0:5] = mp[0:5]
    pts[6:10] = mp[5:9]
    pts[11:15] = mp[9:13]
    pts[16:20] = mp[13:17]
    pts[21:25] = mp[17:21]

    base_inputs = mp[[0, 5, 9, 13, 17]]
    pts[5] = np.tensordot(_BASE_INDEX_WEIGHTS, base_inputs, axes=(0, 0))
    pts[10] = np.tensordot(_BASE_MIDDLE_WEIGHTS, base_inputs, axes=(0, 0))
    pts[15] = np.tensordot(_BASE_RING_WEIGHTS, base_inputs, axes=(0, 0))
    pts[20] = np.tensordot(_BASE_PINKY_WEIGHTS, base_inputs, axes=(0, 0))

    return pts


def _features_from_points25(
    points25: np.ndarray, handedness: str | None, cfg: FeatureConfig
) -> np.ndarray:
    pts = np.asarray(points25, dtype=np.float32)
    if pts.shape != (25, 3):
        raise ValueError(
            f"niepoprawny ksztalt points25: {pts.shape}, oczekiwano (25, 3)"
        )

    rel = pts - pts[0]

    # mirror dla lewej reki (z perspektywy uzytkownika)
    # MediaPipe "Left" = twoja prawa reka, "Right" = twoja lewa reka
    # Dataset byl nagrany lewa reka, wiec mirrorujemy "Left" (prawa reka uzytkownika)
    if (
        cfg.mirror_left
        and handedness is not None
        and handedness.lower().startswith("left")
    ):
        rel[:, 0] *= -1.0

    if cfg.scale_by_mcp:
        ref = rel[11]
        norm = float(np.linalg.norm(ref))
        if norm > EPS:
            rel = rel / norm

    idx = rel[6]
    mid = rel[11]
    ring = rel[16]
    pinky = rel[21]
    center = (rel[0] + idx + mid + ring + pinky) / 5.0
    hand = unit(np.cross(idx - center, pinky - center))

    pairs: Iterable[tuple[int, int]] = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
    )

    feats = [hand]
    for parent_idx, child_idx in pairs:
        vec = rel[parent_idx] - rel[child_idx]
        feats.append(unit(vec))

    result: np.ndarray = np.concatenate(feats, axis=0).astype(np.float32)
    return result


def from_points25(points25: np.ndarray, handedness: str | None = None) -> np.ndarray:
    return _features_from_points25(points25, handedness, FeatureConfig())


def from_mediapipe_landmarks(
    landmarks21: np.ndarray,
    handedness: str | None = None,
    cfg: FeatureConfig | None = None,
) -> np.ndarray:
    """
    Konwertuje landmarki MediaPipe (21 punktow) na 63 cechy.
    """
    lm = np.asarray(landmarks21, dtype=np.float32).copy()

    # bez transformacji
    pts25 = _build_points25_from_mediapipe21(lm)
    return _features_from_points25(pts25, handedness, cfg or FeatureConfig())


def normalize_hand_points(points: np.ndarray) -> np.ndarray:
    """Pomocniczo zwraca 63 cechy z 21 punktow (bez mirrora), uzywane w skryptach diagnostycznych."""
    return from_mediapipe_landmarks(points, handedness=None)


class FeatureExtractor:
    """Wrapper dla ekstrakcji cech, kompatybilny z dataset.py i translator.py"""

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()

    def extract_batch(
        self, landmarks_batch: np.ndarray, handedness_batch: Optional[list[str]] = None
    ) -> np.ndarray:
        if landmarks_batch.ndim != 3 or landmarks_batch.shape[1:] != (21, 3):
            raise ValueError(
                f"niepoprawny ksztalt batcha: {landmarks_batch.shape}, oczekiwano (N, 21, 3)"
            )

        features_list: list[np.ndarray] = []
        for i, lm in enumerate(landmarks_batch):
            handed: str | None = (
                handedness_batch[i] if handedness_batch is not None else None
            )
            feat_63 = from_mediapipe_landmarks(lm, handedness=handed, cfg=self.cfg)
            if np.isnan(feat_63).any() or np.isinf(feat_63).any():
                raise ValueError("wykryto NaN/Inf w cechach batch")
            features_list.append(feat_63)

        return np.array(features_list, dtype=np.float32)

    def extract(
        self, landmarks: np.ndarray, handedness: str | None = None
    ) -> np.ndarray:
        if landmarks.shape != (21, 3):
            raise ValueError(
                f"niepoprawny ksztalt landmarks: {landmarks.shape}, oczekiwano (21, 3)"
            )

        feat_63 = from_mediapipe_landmarks(
            landmarks, handedness=handedness, cfg=self.cfg
        )
        if np.isnan(feat_63).any() or np.isinf(feat_63).any():
            raise ValueError("wykryto NaN/Inf w cechach")

        return feat_63


__all__ = [
    "from_mediapipe_landmarks",
    "from_points25",
    "normalize_hand_points",
    "unit",
    "FeatureConfig",
    "FeatureExtractor",
]
