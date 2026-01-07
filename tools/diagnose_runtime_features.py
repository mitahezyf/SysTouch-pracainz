# diagnostyka roznic cech runtime vs trening
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np

from app.sign_language.features import FeatureConfig, FeatureExtractor

print("=" * 60)
print("DIAGNOSTYKA ROZNIC CECH RUNTIME VS TRENING")
print("=" * 60)

# wczytaj dane treningowe
train_data = np.load("app/sign_language/data/processed/train.npz", allow_pickle=True)
X_train = train_data["X"]
y_train = train_data["y"]
meta = json.loads(str(train_data["meta"]))
classes = meta["classes"]

print(f"\nDane treningowe: {X_train.shape}")
print(f"Klasy: {len(classes)}")

# statystyki per-feature danych treningowych
train_means = X_train.mean(axis=0)
train_stds = X_train.std(axis=0)

# symuluj cechy runtime - losowe landmarki jak z MediaPipe
extractor = FeatureExtractor(FeatureConfig())

# MediaPipe zwraca landmarki w zakresie [0, 1] (znormalizowane do rozmiaru obrazu)
# z jest glebokoscia w zakresie ok. [-0.3, 0.3]
np.random.seed(42)
fake_landmarks = np.random.rand(21, 3).astype(np.float32)
fake_landmarks[:, 2] = (np.random.rand(21) - 0.5) * 0.3  # z blizej 0

# wyciagnij cechy dla 3 klatek (symulacja sekwencji)
feats = []
for _ in range(3):
    f = extractor.extract(fake_landmarks, handedness="Right")
    feats.append(f)
runtime_vec = np.concatenate(feats)

print(f"\nCechy runtime (symulowane): {runtime_vec.shape}")
print(f"  Mean: {runtime_vec.mean():.4f} (trening: {train_means.mean():.4f})")
print(f"  Std: {runtime_vec.std():.4f} (trening: {train_stds.mean():.4f})")
print(f"  Min: {runtime_vec.min():.4f} (trening: {X_train.min():.4f})")
print(f"  Max: {runtime_vec.max():.4f} (trening: {X_train.max():.4f})")

# sprawdz ktore cechy sa najbardziej rozne
diff_means = np.abs(runtime_vec - train_means)
worst_idx = np.argsort(diff_means)[-10:]
print(f"\nNajwieksza roznica mean (indeksy): {worst_idx}")
for idx in worst_idx[-5:]:
    print(
        f"  cecha {idx}: runtime={runtime_vec[idx]:.3f}, train_mean={train_means[idx]:.3f}"
    )

# teraz wczytaj prawdziwa probke L z danych treningowych i porownaj
l_idx = classes.index("L")
l_samples = X_train[y_train == l_idx]
print(f"\nProbki L w treningu: {len(l_samples)}")
l_mean = l_samples.mean(axis=0)

# znajdz najbardziej charakterystyczne cechy dla L
l_std = l_samples.std(axis=0)
char_feats = np.where(l_std < 0.1)[0]  # cechy o malej wariancji = stabilne
print(f"Stabilne cechy dla L (std < 0.1): {len(char_feats)}")
if len(char_feats) > 0:
    print(f"  Przyklad: cechy {char_feats[:5]}")
    for idx in char_feats[:3]:
        print(f"    cecha {idx}: L_mean={l_mean[idx]:.3f}, L_std={l_std[idx]:.3f}")

# podobnie dla U
u_idx = classes.index("U")
u_samples = X_train[y_train == u_idx]
u_mean = u_samples.mean(axis=0)

# roznica miedzy L i U
diff_l_u = np.abs(l_mean - u_mean)
most_diff_idx = np.argsort(diff_l_u)[-10:]
print("\nCechy najbardziej roznicujace L vs U:")
for idx in most_diff_idx[-5:]:
    print(
        f"  cecha {idx}: L={l_mean[idx]:.3f}, U={u_mean[idx]:.3f}, diff={diff_l_u[idx]:.3f}"
    )

print("\n" + "=" * 60)
print("WNIOSKI:")
print("=" * 60)
print(
    """
Jesli cechy runtime sa bardzo rozne od treningowych:
1. Sprawdz czy kamera jest podobnie ustawiona jak przy nagrywaniu datasetu
2. Sprawdz oswietlenie
3. Sprawdz czy uzywasz tej samej reki (prawa vs lewa)
4. Sprawdz czy dlon jest dobrze widoczna (nie obcieta)
"""
)
