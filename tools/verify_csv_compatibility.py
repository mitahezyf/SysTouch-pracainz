# -*- coding: utf-8 -*-
"""Weryfikacja kompatybilnosci formatow CSV w pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    print("=" * 60)
    print("WERYFIKACJA KOMPATYBILNOSCI CSV")
    print("=" * 60)

    errors = []

    # 1. Sprawdz format plikow w data/collected/*/features/
    collected_dir = Path("data/collected")
    if collected_dir.exists():
        sessions = [s for s in collected_dir.iterdir() if s.is_dir()]
        print(f"\n1. Sesje w data/collected/: {len(sessions)}")

        for session in sessions[:1]:
            features_dir = session / "features"
            if features_dir.exists():
                csvs = list(features_dir.glob("*.csv"))[:1]
                if csvs:
                    df = pd.read_csv(csvs[0])
                    print(f"\n   Przykladowy CSV: {csvs[0].name}")

                    feat_cols = [c for c in df.columns if c.startswith("feat_")]
                    print(f"   Kolumny feat_*: {len(feat_cols)}")

                    if len(feat_cols) != 63:
                        errors.append(
                            f"Oczekiwano 63 kolumn feat_*, jest {len(feat_cols)}"
                        )

                    if "label" in df.columns:
                        print("   Kolumna label: TAK")
                    else:
                        errors.append("Brak kolumny 'label'")

                    if "has_hand" in df.columns:
                        print("   Kolumna has_hand: TAK")
    else:
        print("UWAGA: Brak katalogu data/collected/")

    # 2. Sprawdz custom_dataset.csv
    custom_csv = Path("app/sign_language/data/raw/custom_dataset.csv")
    print("\n" + "-" * 60)
    if custom_csv.exists():
        df = pd.read_csv(custom_csv)
        print("2. custom_dataset.csv: ISTNIEJE")
        print(f"   Liczba wierszy: {len(df)}")

        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        print(f"   Kolumny feat_*: {len(feat_cols)}")

        if len(feat_cols) != 63:
            errors.append(
                f"custom_dataset.csv: oczekiwano 63 kolumn feat_*, jest {len(feat_cols)}"
            )

        if "label" in df.columns:
            labels = sorted(df["label"].unique())
            print(f"   Etykiety: {labels}")
    else:
        print("2. custom_dataset.csv: NIE ISTNIEJE")
        print("   (zostanie utworzony przy konsolidacji)")

    # 3. Sprawdz train.npz
    train_npz = Path("app/sign_language/data/processed/train.npz")
    print("\n" + "-" * 60)
    if train_npz.exists():
        data = np.load(train_npz, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        print("3. train.npz: ISTNIEJE")
        print(f"   X.shape: {X.shape}")
        print(f"   y.shape: {y.shape}")

        if X.shape[1] != 189:
            errors.append(f"train.npz: oczekiwano 189 cech, jest {X.shape[1]}")
    else:
        print("3. train.npz: NIE ISTNIEJE")
        print("   (zostanie utworzony przy preprocessing)")

    # Werdykt
    print("\n" + "=" * 60)
    if errors:
        print("WERDYKT: BLEDY KOMPATYBILNOSCI")
        for e in errors:
            print(f"  - {e}")
    else:
        print("WERDYKT: Formaty sa KOMPATYBILNE")
    print("=" * 60)

    print()
    print("Pipeline danych:")
    print("  data/collected/*.csv (feat_0..feat_62)")
    print("       -> consolidate_collected.py")
    print("       -> custom_dataset.csv (feat_0..feat_62)")
    print("       -> dataset.py._extract_custom() (63 -> 189)")
    print("       -> train.npz (X: [N, 189])")
    print("       -> trainer.py (PyTorch MLP)")
    print()

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
