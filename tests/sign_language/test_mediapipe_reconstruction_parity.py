import csv
from pathlib import Path

import numpy as np

from app.sign_language.features import from_mediapipe_landmarks, from_points25
from tools.verify_mediapipe_reconstruction_parity import build_landmarks21_from_points25


def _load_points(points_path: Path, n: int) -> list[np.ndarray]:
    items: list[np.ndarray] = []
    with points_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            if idx >= n:
                break
            values = [float(row[f"point_1_{i}"]) for i in range(1, 76)]
            items.append(np.array(values, dtype=np.float32).reshape(25, 3))
    if not items:
        raise RuntimeError("Brak rekordow w pliku points")
    return items


def test_mediapipe_reconstruction_parity() -> None:
    points_path = Path("app/sign_language/data/raw/PJM-points.csv")
    samples = _load_points(points_path, 20)

    tol_bones = 0.005
    tol_hand = 0.08

    for points25 in samples:
        landmarks21 = build_landmarks21_from_points25(points25)
        feat_gold = from_points25(points25)
        feat_mp = from_mediapipe_landmarks(landmarks21, handedness="Right")

        diff = np.abs(feat_gold - feat_mp)
        hand_diff = diff[:3]
        bones_diff = diff[3:]

        assert hand_diff.max() < tol_hand
        assert bones_diff.max() < tol_bones
