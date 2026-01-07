# sprawdzenie zgodnosci osi Y miedzy datasetem a runtime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from app.sign_language.features import (
    FeatureConfig,
    _build_landmarks21_from_points25,
    _build_points25_from_mediapipe21,
    _features_from_points25,
    from_mediapipe_landmarks,
    from_points25,
)

print("=" * 60)
print("SPRAWDZENIE ZGODNOSCI OSI Y")
print("=" * 60)

# wczytaj dane
df_pts = pd.read_csv("app/sign_language/data/raw/PJM-points.csv")
df_vec = pd.read_csv("app/sign_language/data/raw/PJM-vectors.csv")

# pierwsza probka
row_pts = df_pts.iloc[0]
row_vec = df_vec.iloc[0]
label = row_pts["sign_label"]
print(f"\nProbka: litera '{label}'")

# wektory z datasetu (blok 1 = 63 cechy)
cols = ["vector_hand_1_x", "vector_hand_1_y", "vector_hand_1_z"]
for i in range(1, 21):
    cols.extend([f"vector_1_{i}_x", f"vector_1_{i}_y", f"vector_1_{i}_z"])
vec_dataset = np.array([row_vec[c] for c in cols], dtype=np.float32)
print("\n[DATASET] Wektory z PJM-vectors.csv:")
print(f"  Pierwsze 6: {vec_dataset[:6]}")
print(f"  Range: [{vec_dataset.min():.4f}, {vec_dataset.max():.4f}]")

# wyciagnij punkty i oblicz cechy przez from_points25
pts_cols = [f"point_1_{i}" for i in range(1, 76)]
pts_flat = row_pts[pts_cols].to_numpy(dtype=np.float32)
pts25 = pts_flat.reshape(25, 3)
vec_from_pts = from_points25(pts25)

print("\n[FROM_POINTS25] Cechy obliczone z PJM-points.csv:")
print(f"  Pierwsze 6: {vec_from_pts[:6]}")
diff_pts = np.abs(vec_dataset - vec_from_pts)
print(f"  Max roznica vs dataset: {diff_pts.max():.6f}")

# teraz symuluj MediaPipe: 25->21 punktow
lm21 = _build_landmarks21_from_points25(pts25)
print("\n[21 PUNKTOW] Po konwersji 25->21:")
print(f"  Shape: {lm21.shape}")

# TEST 1: from_mediapipe_landmarks (Z odwroceniem Y - obecna implementacja)
vec_mp_flip = from_mediapipe_landmarks(
    lm21.copy(), handedness=None, cfg=FeatureConfig()
)
print("\n[MEDIAPIPE Z FLIP Y] Obecna implementacja:")
print(f"  Pierwsze 6: {vec_mp_flip[:6]}")
diff_flip = np.abs(vec_dataset - vec_mp_flip)
print(f"  Max roznica vs dataset: {diff_flip.max():.4f}")

# TEST 2: Bez odwrocenia Y (recznie)
lm21_no_flip = lm21.copy()
pts25_recon = _build_points25_from_mediapipe21(lm21_no_flip)
vec_no_flip = _features_from_points25(pts25_recon, handedness=None, cfg=FeatureConfig())
print("\n[BEZ FLIP Y] Recznie bez odwrocenia:")
print(f"  Pierwsze 6: {vec_no_flip[:6]}")
diff_no_flip = np.abs(vec_dataset - vec_no_flip)
print(f"  Max roznica vs dataset: {diff_no_flip.max():.4f}")

print("\n" + "=" * 60)
print("WNIOSKI:")
print("=" * 60)
if diff_flip.max() > diff_no_flip.max():
    print(
        f">>> BEZ FLIP Y jest lepsze! ({diff_no_flip.max():.4f} < {diff_flip.max():.4f})"
    )
    print(">>> Odwracanie Y PSUJE cechy!")
else:
    print(
        f">>> Z FLIP Y jest lepsze lub rowne ({diff_flip.max():.4f} <= {diff_no_flip.max():.4f})"
    )
    print(">>> Odwracanie Y jest poprawne")

if diff_pts.max() < 0.01:
    print(f"\n>>> from_points25 jest ZGODNE z datasetem (diff={diff_pts.max():.6f})")
else:
    print(
        f"\n>>> from_points25 NIE jest zgodne z datasetem (diff={diff_pts.max():.6f})"
    )
