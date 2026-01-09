# skrypt diagnostyczny dla struktury danych PJM
# force flush
import functools

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

print("=== DIAGNOSTYKA STRUKTURY DANYCH PJM ===")
print()

# wczytaj dane
vec_path = "app/sign_language/data/raw/PJM-vectors.csv"
pts_path = "app/sign_language/data/raw/PJM-points.csv"

df_vec = pd.read_csv(vec_path)
df_pts = pd.read_csv(pts_path)

print("--- PJM-vectors.csv ---")
print(f"Wierszy: {len(df_vec)}")
print(f"Kolumn: {len(df_vec.columns)}")
print(f"\nPierwsze 20 kolumn: {df_vec.columns.tolist()[:20]}")
print(f"\nOstatnie 10 kolumn: {df_vec.columns.tolist()[-10:]}")

# znajdz kolumny cech
feature_cols = [c for c in df_vec.columns if c.startswith("vector_")]
print(f"\nKolumn vector_*: {len(feature_cols)}")

# metadane
meta_cols = [c for c in df_vec.columns if not c.startswith("vector_")]
print(f"Kolumny metadanych: {meta_cols}")

print("\n--- PJM-points.csv ---")
print(f"Wierszy: {len(df_pts)}")
print(f"Kolumn: {len(df_pts.columns)}")
print(f"\nPierwsze 20 kolumn: {df_pts.columns.tolist()[:20]}")

# znajdz kolumny punktow
point_cols = [c for c in df_pts.columns if c.startswith("point_")]
print(f"\nKolumn point_*: {len(point_cols)}")

# sprawdz rozklad klas
print("\n--- ROZKLAD KLAS ---")
vc = df_vec["sign_label"].value_counts()
print(f"Unikalne klasy: {len(vc)}")
print(f"Min probek: {vc.min()}")
print(f"Max probek: {vc.max()}")
print("\nBrakujace klasy (vs pelny alfabet):")
expected = set(
    [
        "A",
        "A+",
        "B",
        "C",
        "C+",
        "CH",
        "CZ",
        "D",
        "E",
        "E+",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "L+",
        "M",
        "N",
        "N+",
        "O",
        "O+",
        "P",
        "R",
        "RZ",
        "S",
        "S+",
        "SZ",
        "T",
        "U",
        "W",
        "Y",
        "Z",
        "Z+",
        "Z++",
    ]
)
found = set(df_vec["sign_label"].unique())
missing = expected - found
print(f"Brakuje: {missing if missing else 'zadnych'}")

# statystyki cech
print("\n--- STATYSTYKI CECH (vectors) ---")
X = df_vec[feature_cols].values.astype(np.float32)
print(f"Ksztalt X: {X.shape}")
print(f"Min: {X.min():.4f}")
print(f"Max: {X.max():.4f}")
print(f"Mean: {X.mean():.4f}")
print(f"Std: {X.std():.4f}")

# sprawdz czy sa sekwencje 3-klatkowe
if len(feature_cols) == 189:
    print("\n--- ANALIZA SEKWENCJI 3-KLATKOWYCH ---")
    print("Wektor 189D = 3 bloki x 63 cech")
    block1 = X[:, :63]
    block2 = X[:, 63:126]
    block3 = X[:, 126:189]
    print(f"Block 1 (poczatek) - mean: {block1.mean():.4f}, std: {block1.std():.4f}")
    print(f"Block 2 (srodek) - mean: {block2.mean():.4f}, std: {block2.std():.4f}")
    print(f"Block 3 (koniec) - mean: {block3.mean():.4f}, std: {block3.std():.4f}")

# analiza points
print("\n--- STATYSTYKI PUNKTOW (points) ---")
point_cols_pts = [c for c in df_pts.columns if c.startswith("point_")]
if point_cols_pts:
    Xp = df_pts[point_cols_pts].values.astype(np.float32)
    print(f"Ksztalt punktow: {Xp.shape}")
    print(f"Punktow na probke: {len(point_cols_pts) // 3}")  # x, y, z
    print(f"Min: {Xp.min():.4f}")
    print(f"Max: {Xp.max():.4f}")

print("\n=== KONIEC DIAGNOSTYKI ===")
