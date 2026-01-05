import argparse
import csv
import json
import random
from pathlib import Path


def extract_x63(row, col_index):
    feats = []
    # 3: vector_hand_1_(x,y,z)
    for a in ("x", "y", "z"):
        feats.append(float(row[col_index[f"vector_hand_1_{a}"]]))

    # 60: vector_1_1..vector_1_20 (x,y,z)
    for i in range(1, 21):
        for a in ("x", "y", "z"):
            feats.append(float(row[col_index[f"vector_1_{i}_{a}"]]))

    if len(feats) != 63:
        raise ValueError(f"Expected 63 features, got {len(feats)}")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to PJM-vectors.csv")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--labels", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--k", type=int, default=50, help="Samples per label")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buckets = {lab: [] for lab in args.labels}

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_index = {name: i for i, name in enumerate(header)}
        label_col = "sign_label"
        if label_col not in col_index:
            raise RuntimeError(f"Missing column: {label_col}")

        for row in reader:
            lab = row[col_index[label_col]]
            if lab in buckets:
                try:
                    x = extract_x63(row, col_index)
                except Exception:
                    continue
                buckets[lab].append(x)

    data = []
    for lab in args.labels:
        xs = buckets[lab]
        if len(xs) < args.k:
            raise RuntimeError(f"Label {lab}: found {len(xs)} samples, need {args.k}")
        rng.shuffle(xs)
        for x in xs[: args.k]:
            data.append({"label": lab, "x": x})

    out_path.write_text(
        json.dumps({"labels": args.labels, "samples": data}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
