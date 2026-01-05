"""Uruchomienie (PowerShell):
1) cd <repo_root> ; python tools\verify_feature_parity.py --n 200
2) cd <repo_root> ; python -m tools.verify_feature_parity --n 200
(optional) $env:PYTHONPATH="." ; python tools\verify_feature_parity.py --n 200
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from app.sign_language.features import from_points25


def parse_points(row: dict) -> np.ndarray:
    values = []
    for i in range(1, 76):
        key = f"point_1_{i}"
        if key not in row:
            raise KeyError(f"brak kolumny {key} w PJM-points.csv")
        values.append(float(row[key]))
    points = np.array(values, dtype=np.float32).reshape(25, 3)
    return points


def parse_vectors(row: dict) -> np.ndarray:
    cols = ["vector_hand_1_x", "vector_hand_1_y", "vector_hand_1_z"]
    for i in range(1, 21):
        cols.extend([f"vector_1_{i}_x", f"vector_1_{i}_y", f"vector_1_{i}_z"])
    values = []
    for key in cols:
        if key not in row:
            raise KeyError(f"brak kolumny {key} w PJM-vectors.csv")
        values.append(float(row[key]))
    return np.array(values, dtype=np.float32)


def verify(
    points_path: Path, vectors_path: Path, n: int, tol_bones: float, tol_hand: float
) -> tuple[float, float, float, float, int]:
    with points_path.open(newline="", encoding="utf-8") as fp, vectors_path.open(
        newline="", encoding="utf-8"
    ) as fv:
        points_reader = csv.DictReader(fp)
        vectors_reader = csv.DictReader(fv)

        max_hand = 0.0
        max_bones = 0.0
        sum_hand = 0.0
        sum_bones = 0.0
        count = 0

        for p_row, v_row in zip(points_reader, vectors_reader):
            if count >= n:
                break
            points25 = parse_points(p_row)
            feat = from_points25(points25)
            vec_expected = parse_vectors(v_row)

            diff = np.abs(feat - vec_expected)
            hand_diff = diff[:3]
            bones_diff = diff[3:]

            max_hand = max(max_hand, float(hand_diff.max()))
            max_bones = max(max_bones, float(bones_diff.max()))
            sum_hand += float(hand_diff.mean())
            sum_bones += float(bones_diff.mean())
            count += 1

        if count == 0:
            raise RuntimeError("Brak rekordow do weryfikacji")

        mean_hand = sum_hand / count
        mean_bones = sum_bones / count

    return max_hand, max_bones, mean_hand, mean_bones, count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Weryfikacja parzystosci featurów PJM 63D"
    )
    parser.add_argument(
        "--points", type=Path, default=Path("app/sign_language/data/raw/PJM-points.csv")
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=Path("app/sign_language/data/raw/PJM-vectors.csv"),
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--tol_bones", type=float, default=0.005)
    parser.add_argument("--tol_hand", type=float, default=0.08)

    args = parser.parse_args()

    if not args.points.exists():
        return 1
    if not args.vectors.exists():
        return 1

    max_hand, max_bones, mean_hand, mean_bones, count = verify(
        args.points, args.vectors, args.n, args.tol_bones, args.tol_hand
    )

    if max_bones > args.tol_bones or max_hand > args.tol_hand:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
