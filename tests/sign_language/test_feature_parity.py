import csv
from pathlib import Path

import numpy as np
import pytest

from app.sign_language.features import from_points25

POINTS_DEFAULT = Path("app/sign_language/data/raw/PJM-points.csv")
VECTORS_DEFAULT = Path("app/sign_language/data/raw/PJM-vectors.csv")


def _iter_rows(path: Path, limit: int):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            yield row


def test_feature_parity_small_sample():
    if not POINTS_DEFAULT.exists() or not VECTORS_DEFAULT.exists():
        pytest.skip("PJM CSV niedostepne w srodowisku testowym")

    n = 50
    tol_bones = 0.005
    tol_hand = 0.08

    max_hand = 0.0
    max_bones = 0.0

    for p_row, v_row in zip(
        _iter_rows(POINTS_DEFAULT, n), _iter_rows(VECTORS_DEFAULT, n)
    ):
        values = [float(p_row[f"point_1_{i}"]) for i in range(1, 76)]
        points25 = np.array(values, dtype=np.float32).reshape(25, 3)
        feat = from_points25(points25)

        cols = ["vector_hand_1_x", "vector_hand_1_y", "vector_hand_1_z"]
        for i in range(1, 21):
            cols.extend([f"vector_1_{i}_x", f"vector_1_{i}_y", f"vector_1_{i}_z"])
        vec_expected = np.array([float(v_row[c]) for c in cols], dtype=np.float32)

        diff = np.abs(feat - vec_expected)
        max_hand = max(max_hand, float(diff[:3].max()))
        max_bones = max(max_bones, float(diff[3:].max()))

    assert max_bones < tol_bones
    assert max_hand < tol_hand
