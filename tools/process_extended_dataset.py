# skrypt do przetworzenia datasetu PJM z rozszerzonymi cechami
# uruchamiany z katalogu glownego projektu
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from app.sign_language.features import (
    FeatureConfig,
    _features_from_points25,
)

# sciezki
BASE_DIR = Path(__file__).parent.parent / "app" / "sign_language"
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed_extended"
VECTORS_CSV = RAW_DIR / "PJM-vectors.csv"
POINTS_CSV = RAW_DIR / "PJM-points.csv"

# rozmiary - rozszerzone cechy
BLOCK_SIZE_EXTENDED = 82  # 63 + 4 + 5 + 10
NUM_BLOCKS = 3
INPUT_SIZE_EXTENDED = BLOCK_SIZE_EXTENDED * NUM_BLOCKS  # 246


def extract_extended_features_from_points(
    points25: np.ndarray, handedness: str | None = None
) -> np.ndarray:
    """Ekstrahuje 82D cechy z 25 punktow."""
    cfg = FeatureConfig(extended_features=True)  # type: ignore[call-arg]
    return _features_from_points25(points25, handedness, cfg)


def process_points_csv() -> tuple[np.ndarray, Any]:
    """
    Przetwarza PJM-points.csv na rozszerzone cechy (246D).

    Returns:
        X: macierz cech (N, 246)
        y: wektor etykiet string
    """
    print(f"Wczytywanie {POINTS_CSV}...")
    df = pd.read_csv(POINTS_CSV)

    print(f"Wierszy: {len(df)}")
    print(f"Kolumn: {len(df.columns)}")

    # znajdz kolumne etykiet
    label_col = "sign_label" if "sign_label" in df.columns else "label"
    y_raw = df[label_col].astype(str).values
    y = np.array(y_raw, dtype=str)

    print(f"Unikalne klasy: {len(np.unique(y))}")

    all_features = []

    # przetwarzaj kazdą probke
    print("Ekstrakcja cech...")
    for idx in tqdm(range(len(df)), desc="Przetwarzanie"):
        row = df.iloc[idx]

        block_features = []
        for block in range(1, 4):  # 3 bloki (poczatek, srodek, koniec)
            # wyciagnij 75 wartosci dla bloku (25 punktow x 3)
            point_cols = [f"point_{block}_{i}" for i in range(1, 76)]
            raw_flat = row[point_cols].values.astype(np.float32)
            points25 = raw_flat.reshape(25, 3)

            # ekstrahuj rozszerzone cechy (82D)
            feat = extract_extended_features_from_points(points25)
            block_features.append(feat)

        # polacz 3 bloki -> 246D
        full_features = np.concatenate(block_features)
        all_features.append(full_features)

    X = np.array(all_features, dtype=np.float32)
    print(f"Ksztalt X: {X.shape}")

    return X, y


def split_and_save(X: np.ndarray, y: Any) -> None:
    """Dzieli dane na train/val/test i zapisuje."""
    print("\nPodział danych...")

    # enkoduj etykiety
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_

    print(f"Klasy ({len(classes)}): {list(classes)}")

    # split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # zapisz
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "classes": list(classes),
        "num_classes": len(classes),
        "input_size": INPUT_SIZE_EXTENDED,
        "block_size": BLOCK_SIZE_EXTENDED,
        "num_blocks": NUM_BLOCKS,
        "version": "extended_82x3",
    }

    np.savez_compressed(
        PROCESSED_DIR / "train.npz", X=X_train, y=y_train, meta=json.dumps(meta)
    )
    np.savez_compressed(
        PROCESSED_DIR / "val.npz", X=X_val, y=y_val, meta=json.dumps(meta)
    )
    np.savez_compressed(
        PROCESSED_DIR / "test.npz", X=X_test, y=y_test, meta=json.dumps(meta)
    )

    print(f"\nZapisano do {PROCESSED_DIR}/")
    print(f"  train.npz: {X_train.shape}")
    print(f"  val.npz: {X_val.shape}")
    print(f"  test.npz: {X_test.shape}")


def main():
    print("=== PRZETWARZANIE DATASETU PJM Z ROZSZERZONYMI CECHAMI ===")
    print(f"Rozmiar bloku: {BLOCK_SIZE_EXTENDED}D (bylo 63D)")
    print(f"Rozmiar sekwencji: {INPUT_SIZE_EXTENDED}D (bylo 189D)")
    print()

    X, y = process_points_csv()

    # walidacja
    if np.isnan(X).any() or np.isinf(X).any():
        print("[BLAD] Wykryto NaN/Inf w cechach!")
        nan_rows = np.where(np.isnan(X).any(axis=1))[0]
        print(f"Wiersze z NaN: {nan_rows[:10]}")
        return

    print("\n[OK] Brak NaN/Inf")
    print(f"Min: {X.min():.4f}, Max: {X.max():.4f}, Mean: {X.mean():.4f}")

    split_and_save(X, y)

    print("\n=== ZAKONCZONO ===")


if __name__ == "__main__":
    main()
