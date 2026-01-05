import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_points(row: dict) -> np.ndarray:
    values = []
    for i in range(1, 76):
        key = f"point_1_{i}"
        if key not in row:
            raise KeyError(f"brak kolumny {key} w PJM-points.csv")
        values.append(float(row[key]))
    return np.array(values, dtype=np.float32).reshape(25, 3)


def load_points(points_path: Path, n: int) -> np.ndarray:
    records: list[np.ndarray] = []
    with points_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            if idx >= n:
                break
            records.append(parse_points(row))
    if not records:
        raise RuntimeError("brak rekordow do dopasowania")
    return np.stack(records, axis=0)


def fit_coeffs(
    sources: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], target: np.ndarray
) -> tuple[np.ndarray, float]:
    """Dopasowuje wspolczynniki a,b,c,d w modelu target ~= a*idx + b*mid + c*ring + d*pinky."""
    idx, mid, ring, pinky = sources
    n = target.shape[0]
    a_rows: list[list[float]] = []
    b_vals: list[float] = []
    for i in range(n):
        for dim in range(3):
            a_rows.append(
                [
                    float(idx[i, dim]),
                    float(mid[i, dim]),
                    float(ring[i, dim]),
                    float(pinky[i, dim]),
                ]
            )
            b_vals.append(float(target[i, dim]))
    a_mat = np.asarray(a_rows, dtype=np.float64)
    b_vec = np.asarray(b_vals, dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    pred = a_mat @ coeffs
    rmse = float(np.sqrt(np.mean((pred - b_vec) ** 2)))
    return coeffs, rmse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dopasowanie wspolczynnikow punktow bazowych P5/P10/P15/P20"
    )
    parser.add_argument(
        "--points", type=Path, default=Path("app/sign_language/data/raw/PJM-points.csv")
    )
    parser.add_argument(
        "--n", type=int, default=5000, help="liczba rekordow do dopasowania"
    )

    args = parser.parse_args()

    if not args.points.exists():
        return 1

    points_all = load_points(args.points, args.n)
    rel = points_all - points_all[:, 0:1, :]

    idx = rel[:, 6]
    mid = rel[:, 11]
    ring = rel[:, 16]
    pinky = rel[:, 21]

    targets = {
        "P5": rel[:, 5],
        "P10": rel[:, 10],
        "P15": rel[:, 15],
        "P20": rel[:, 20],
    }

    for name, target in targets.items():
        coeffs, rmse = fit_coeffs((idx, mid, ring, pinky), target)
        a, b, c, d = coeffs.tolist()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
