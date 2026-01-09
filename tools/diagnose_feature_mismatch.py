# diagnostyka - porownanie cech z datasetu vs ekstrakcji na zywo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from app.sign_language.features import (
    FeatureConfig,
    _features_from_points25,
    from_points25,
)

print("=== DIAGNOSTYKA ZGODNOSCI CECH ===\n")

# wczytaj przykladowe dane z PJM-points.csv
points_csv = Path("app/sign_language/data/raw/PJM-points.csv")
df = pd.read_csv(points_csv)

print(f"Wczytano {len(df)} probek z PJM-points.csv")

# wybierz jedna probke (np. litera A)
sample_idx = 0
row = df.iloc[sample_idx]
label = row["sign_label"]
print(f"\nProbka {sample_idx}: litera '{label}'")

# ekstrakcja cech z points25 (jak w datasecie)
block1_cols = [f"point_1_{i}" for i in range(1, 76)]
raw_flat = row[block1_cols].to_numpy(dtype=np.float32)
points25 = raw_flat.reshape(25, 3)

print(f"\nPoints25 shape: {points25.shape}")
print(f"Points25 range: [{points25.min():.2f}, {points25.max():.2f}]")

# cechy standardowe (63D) - z from_points25
feat_std = from_points25(points25, handedness=None)
print(f"\nStandardowe cechy (63D): {feat_std.shape}")
print(f"  Range: [{feat_std.min():.4f}, {feat_std.max():.4f}]")
print(f"  First 10: {feat_std[:10]}")

# cechy rozszerzone (82D) - bezposrednio z _features_from_points25
cfg_ext = FeatureConfig(extended_features=True)  # type: ignore[call-arg]
feat_ext = _features_from_points25(points25, handedness=None, cfg=cfg_ext)
print(f"\nRozszerzone cechy (82D): {feat_ext.shape}")
print(f"  Range: [{feat_ext.min():.4f}, {feat_ext.max():.4f}]")
print(f"  First 10: {feat_ext[:10]}")

# porownaj pierwsze 63 cechy
diff = np.abs(feat_std - feat_ext[:63])
print(f"\nRoznica pierwszych 63 cech: max={diff.max():.6f}, mean={diff.mean():.6f}")

if diff.max() < 0.001:
    print("[OK] Cechy bazowe sa zgodne")
else:
    print("[BLAD] Cechy bazowe sie roznia!")
    print(f"  Indeksy roznic: {np.where(diff > 0.001)[0]}")

# pokaz dodatkowe cechy
print("\nDodatkowe cechy (19D):")
print(f"  Finger angles [63:67]: {feat_ext[63:67]}")
print(f"  Finger curls [67:72]: {feat_ext[67:72]}")
print(f"  Fingertip distances [72:82]: {feat_ext[72:82]}")

# teraz sprawdz co generuje FeatureExtractor z MediaPipe landmarks
# symuluj MediaPipe 21 punktow (mapowanie 25->21)
print("\n" + "=" * 50)
print("SYMULACJA MEDIAPIPE (21 punktow)")
print("=" * 50)

from app.sign_language.features import (
    _build_landmarks21_from_points25,
    from_mediapipe_landmarks,
)

# konwersja 25->21 (odwrotna do tego co robi translator)
landmarks21 = _build_landmarks21_from_points25(points25)
print(f"\nLandmarks21 shape: {landmarks21.shape}")
print(f"Landmarks21 range: [{landmarks21.min():.2f}, {landmarks21.max():.2f}]")

# ekstrakcja cech przez from_mediapipe_landmarks (jak robi translator)
# UWAGA: from_mediapipe_landmarks ODWRACA Y!
feat_mp_std = from_mediapipe_landmarks(
    landmarks21, handedness=None, cfg=FeatureConfig()
)
print(f"\nCechy z MediaPipe (63D, std): {feat_mp_std.shape}")

feat_mp_ext = from_mediapipe_landmarks(landmarks21, handedness=None, cfg=FeatureConfig(extended_features=True))  # type: ignore[call-arg]
print(f"Cechy z MediaPipe (82D, ext): {feat_mp_ext.shape}")

# porownaj z oryginalem
diff_mp = np.abs(feat_std - feat_mp_std)
print(f"\nRoznica std vs mp_std: max={diff_mp.max():.4f}")

if diff_mp.max() > 0.1:
    print("[PROBLEM] Duza roznica miedzy cechami z points25 a z MediaPipe!")
    print("To moze byc przyczyna zlego rozpoznawania!")
    print(f"  Indeksy duzych roznic: {np.where(diff_mp > 0.1)[0][:10]}")

# sprawdz czy problem to odwrocenie Y
print("\n" + "=" * 50)
print("TEST ODWROCENIA OSI Y")
print("=" * 50)

# bez odwrocenia Y
lm_no_flip = landmarks21.copy()
from app.sign_language.features import (
    _build_points25_from_mediapipe21,
    _features_from_points25,
)

pts25_recon = _build_points25_from_mediapipe21(lm_no_flip)
feat_no_flip = _features_from_points25(
    pts25_recon, handedness=None, cfg=FeatureConfig()
)

# z odwroceniem Y (jak robi from_mediapipe_landmarks)
lm_flip = landmarks21.copy()
lm_flip[:, 1] *= -1
pts25_flip = _build_points25_from_mediapipe21(lm_flip)
feat_flip = _features_from_points25(pts25_flip, handedness=None, cfg=FeatureConfig())

diff_no_flip = np.abs(feat_std - feat_no_flip)
diff_flip = np.abs(feat_std - feat_flip)

print(f"Roznica BEZ odwrocenia Y: max={diff_no_flip.max():.4f}")
print(f"Roznica Z odwroceniem Y: max={diff_flip.max():.4f}")

if diff_no_flip.max() < diff_flip.max():
    print("\n[WNIOSEK] Cechy sa lepsze BEZ odwrocenia Y!")
    print("To sugeruje ze dane PJM juz maja prawidlowy uklad Y.")
else:
    print("\n[WNIOSEK] Odwrocenie Y jest poprawne.")

print("\n=== KONIEC DIAGNOSTYKI ===")
