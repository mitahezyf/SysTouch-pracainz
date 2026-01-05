from typing import Iterable

import numpy as np

EPS = 1e-9


def unit(vector: np.ndarray) -> np.ndarray:
    """Zwraca wektor znormalizowany z ochrona przed zerem."""
    norm = np.linalg.norm(vector)
    if norm < EPS:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _build_points25(
    landmarks21: np.ndarray, handedness: str | None = None
) -> np.ndarray:
    """Buduje 25 punktow z 21 landmarkow mediapipe, z opcjonalnym mirrorem lewej dloni."""
    if landmarks21.shape != (21, 3):
        raise ValueError(
            f"niepoprawny ksztalt landmarks: {landmarks21.shape}, oczekiwano (21, 3)"
        )

    lm = np.asarray(landmarks21, dtype=np.float32)
    wrist = lm[0]
    relative = lm - wrist

    if handedness is not None and handedness.lower().startswith("left"):
        # mirror osi X wzgledem nadgarstka, aby lewa/prawa dawaly te same cechy
        relative[:, 0] *= -1.0

    points = np.zeros((25, 3), dtype=np.float32)

    # podstawowe punkty
    points[0] = relative[0]  # P0 wrist (0,0,0)
    points[1] = relative[0]  # P1 thumb_base = wrist (duplikat)

    points[2] = relative[2]
    points[3] = relative[3]
    points[4] = relative[4]

    points[6] = relative[5]
    points[7] = relative[6]
    points[8] = relative[7]
    points[9] = relative[8]

    points[11] = relative[9]
    points[12] = relative[10]
    points[13] = relative[11]
    points[14] = relative[12]

    points[16] = relative[13]
    points[17] = relative[14]
    points[18] = relative[15]
    points[19] = relative[16]

    points[21] = relative[17]
    points[22] = relative[18]
    points[23] = relative[19]

    # brakujace punkty bazowe na podstawie liniowej kombinacji MCP (punkty nie lez na prostej wrist->mcp)
    idx = relative[5]
    mid = relative[9]
    ring = relative[13]
    pinky = relative[17]

    points[5] = -2.863790 * idx + 1.982240 * mid + 4.222725 * ring - 3.505714 * pinky
    points[10] = -3.199167 * idx + 2.114646 * mid + 4.560266 * ring - 3.595484 * pinky
    points[15] = -3.171155 * idx + 1.969997 * mid + 4.347232 * ring - 3.199870 * pinky
    points[20] = -2.202918 * idx + 1.107818 * mid + 2.800028 * ring - 1.628653 * pinky

    return points


def _compute_feature_vectors(points25: np.ndarray) -> np.ndarray:
    """Liczy 63-cechowy wektor (vector_hand_1 + vector_1_1..20) z 25 punktow."""
    if points25.shape != (25, 3):
        raise ValueError(
            f"niepoprawny ksztalt points25: {points25.shape}, oczekiwano (25, 3)"
        )

    # centrowanie na nadgarstek
    centered = points25 - points25[0]

    # vector_hand_1 = normalna dloni z punktow MCP (ring, middle) i bazy malego (pinky_base)
    middle_mcp = centered[11]
    ring_mcp = centered[16]
    pinky_base = centered[20]
    vector_hand_1 = unit(np.cross(ring_mcp - middle_mcp, pinky_base - middle_mcp))

    vectors: list[np.ndarray] = []
    vectors.append(np.zeros(3, dtype=np.float32))  # vector_1_1 = (0,0,0)

    pairs: Iterable[tuple[int, int]] = (
        (0, 2),
        (2, 3),
        (3, 4),  # thumb
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),  # index
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),  # middle
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),  # ring
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),  # pinky
    )

    for pa, pb in pairs:
        direction = unit(centered[pa] - centered[pb])
        vectors.append(direction)

    feature_vector = np.concatenate([vector_hand_1] + vectors, axis=0)
    return feature_vector.astype(np.float32)


def from_points25(points25: np.ndarray) -> np.ndarray:
    """Ekstrahuje 63 cechy z 25 punktow (np. po rekonstrukcji z csv)."""
    return _compute_feature_vectors(np.asarray(points25, dtype=np.float32))


def from_mediapipe_landmarks(
    landmarks21: np.ndarray, handedness: str | None = None
) -> np.ndarray:
    """Ekstrahuje 63 cechy z 21 punktow mediapipe (x,y,z), z mirrorem lewej dloni."""
    points25 = _build_points25(np.asarray(landmarks21, dtype=np.float32), handedness)
    return _compute_feature_vectors(points25)


def normalize_hand_points(points: np.ndarray) -> np.ndarray:
    """Pomocniczo zwraca 63 cechy z 21 punktow (bez mirrora), uzywane w skryptach diagnostycznych."""
    return from_mediapipe_landmarks(points)


class FeatureExtractor:
    """Wrapper dla ekstrakcji cech, kompatybilny z dataset.py i translator.py"""

    def extract_batch(self, landmarks_batch: np.ndarray) -> np.ndarray:
        """
        Ekstrahuje cechy z batcha landmarkow [N, 21, 3].

        Args:
            landmarks_batch: tablica [N, 21, 3] z raw landmarks

        Returns:
            features: tablica [N, 88] z cechami (rozszerzone do 88)
        """
        if landmarks_batch.ndim != 3 or landmarks_batch.shape[1:] != (21, 3):
            raise ValueError(
                f"niepoprawny ksztalt batcha: {landmarks_batch.shape}, oczekiwano (N, 21, 3)"
            )

        features_list = []
        for lm in landmarks_batch:
            # ekstrahuje 63 cechy
            feat_63 = from_mediapipe_landmarks(lm, handedness=None)
            # rozszerz do 88 (padding zerami)
            feat_88 = np.zeros(88, dtype=np.float32)
            feat_88[:63] = feat_63
            features_list.append(feat_88)

        return np.array(features_list, dtype=np.float32)

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Ekstrahuje cechy z pojedynczego zestawu landmarkow (21, 3).

        Args:
            landmarks: tablica [21, 3] z raw landmarks

        Returns:
            features: tablica [88] z cechami (rozszerzone do 88)
        """
        if landmarks.shape != (21, 3):
            raise ValueError(
                f"niepoprawny ksztalt landmarks: {landmarks.shape}, oczekiwano (21, 3)"
            )

        # ekstrahuje 63 cechy
        feat_63 = from_mediapipe_landmarks(landmarks, handedness=None)
        # rozszerz do 88 (padding zerami)
        feat_88 = np.zeros(88, dtype=np.float32)
        feat_88[:63] = feat_63

        return feat_88


__all__ = [
    "from_mediapipe_landmarks",
    "from_points25",
    "normalize_hand_points",
    "unit",
    "FeatureExtractor",
]
