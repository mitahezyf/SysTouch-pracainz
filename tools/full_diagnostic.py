# pelna diagnostyka - sprawdza wszystkie transformacje
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np

# wczytaj dane treningowe
train = np.load("app/sign_language/data/processed/train.npz", allow_pickle=True)
X_train = train["X"]
y_train = train["y"]
meta = json.loads(str(train["meta"]))
classes = meta["classes"]

# srednia A
a_idx = classes.index("A")
a_mean = X_train[y_train == a_idx][:, :63].mean(axis=0)

print("Srednie cechy A z datasetu (pierwsze 12):")
print(f"  {a_mean[:12]}")

# wczytaj surowe punkty z PJM-points dla porownania
import pandas as pd

df = pd.read_csv("app/sign_language/data/raw/PJM-points.csv")
row_a = df[df["sign_label"] == "A"].iloc[0]
pts_cols = [f"point_1_{i}" for i in range(1, 76)]
pts25 = row_a[pts_cols].to_numpy(dtype=np.float32).reshape(25, 3)

print("\nPunkty A z datasetu (nadgarstek i pierwszy palec):")
print(f"  wrist: {pts25[0]}")
print(f"  thumb_tip: {pts25[4]}")
print(f"  index_tip: {pts25[9]}")

# oblicz cechy z tych punktow
from app.sign_language.features import from_points25

feat_from_pts = from_points25(pts25)

print("\nCechy obliczone z from_points25 (pierwsze 12):")
print(f"  {feat_from_pts[:12]}")

# roznica
diff = np.abs(a_mean - feat_from_pts)
print(f"\nRoznica vs srednia A: max={diff.max():.4f}")

# teraz symuluj MediaPipe - konwertuj 25->21 i zobacz co sie dzieje
from app.sign_language.features import (
    FeatureConfig,
    _build_landmarks21_from_points25,
    from_mediapipe_landmarks,
)

lm21 = _build_landmarks21_from_points25(pts25)
print("\nPo konwersji 25->21 (nadgarstek i palce):")
print(f"  wrist: {lm21[0]}")
print(f"  thumb_tip: {lm21[4]}")
print(f"  index_tip: {lm21[8]}")

# oblicz cechy przez from_mediapipe_landmarks
feat_from_mp = from_mediapipe_landmarks(
    lm21, handedness=None, cfg=FeatureConfig(mirror_left=False)
)
print("\nCechy z from_mediapipe_landmarks (pierwsze 12):")
print(f"  {feat_from_mp[:12]}")

diff_mp = np.abs(feat_from_pts - feat_from_mp)
print(f"Roznica vs from_points25: max={diff_mp.max():.4f}")

if diff_mp.max() > 0.1:
    print("\n[PROBLEM] Duza roznica miedzy from_points25 a from_mediapipe_landmarks!")
    bad_idx = np.where(diff_mp > 0.1)[0]
    print(f"  Indeksy: {bad_idx[:10]}")
