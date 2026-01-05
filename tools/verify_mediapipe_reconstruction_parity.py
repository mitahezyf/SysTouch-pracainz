import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.sign_language.features import from_mediapipe_landmarks, from_points25


def parse_points(row: dict) -> np.ndarray:
    values = []
    for i in range(1, 76):
        key = f"point_1_{i}"
        if key not in row:
            raise KeyError(f"brak kolumny {key} w PJM-points.csv")
        values.append(float(row[key]))
    return np.array(values, dtype=np.float32).reshape(25, 3)


def build_landmarks21_from_points25(points25: np.ndarray) -> np.ndarray:
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


def verify(
    points_path: Path, n: int, tol_bones: float, tol_hand: float
) -> tuple[float, float, float, float, int]:
    with points_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)

        max_hand = 0.0
        max_bones = 0.0
        sum_hand = 0.0
        sum_bones = 0.0
        count = 0

        for row in reader:
            if count >= n:
                break
            points25 = parse_points(row)
            landmarks21 = build_landmarks21_from_points25(points25)

            feat_gold = from_points25(points25)
            feat_mp = from_mediapipe_landmarks(landmarks21, handedness="Right")

            diff = np.abs(feat_gold - feat_mp)
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
        description="Weryfikacja parzystosci rekonstrukcji MediaPipe -> cechy 63D"
    )
    parser.add_argument(
        "--points", type=Path, default=Path("app/sign_language/data/raw/PJM-points.csv")
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--tol_bones", type=float, default=0.005)
    parser.add_argument("--tol_hand", type=float, default=0.08)

    args = parser.parse_args()

    if not args.points.exists():
        return 1

    max_hand, max_bones, mean_hand, mean_bones, count = verify(
        args.points, args.n, args.tol_bones, args.tol_hand
    )

    if max_bones > args.tol_bones or max_hand > args.tol_hand:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
