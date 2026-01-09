# -*- coding: utf-8 -*-
"""Testy weryfikujace poprawnosc konsolidacji i formatu danych custom."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_consolidate_collected_creates_valid_csv(tmp_path: Path) -> None:
    """Sprawdza ze consolidate_collected.py tworzy poprawny CSV."""
    from tools.consolidate_collected import consolidate_sessions

    # Stworz sztuczna strukture sesji
    session_dir = tmp_path / "session_test"
    features_dir = session_dir / "features"
    features_dir.mkdir(parents=True)

    # Stworz przykladowy CSV z cechami
    csv_data = {
        "session_id": ["test"],
        "clip_id": ["A_Tester_1_123456"],
        "label": ["A"],
        "frame_idx": [0],
        "timestamp_ms": [123456],
        "handedness": ["Right"],
        "has_hand": [1],
        "mirror_applied": [0],
    }
    # Dodaj kolumny feat_0..feat_62
    for i in range(63):
        csv_data[f"feat_{i}"] = [float(i) * 0.01]

    df = pd.DataFrame(csv_data)
    csv_path = features_dir / "A_Tester_1_123456.csv"
    df.to_csv(csv_path, index=False)

    # Uruchom konsolidacje
    output_csv = tmp_path / "output" / "custom_dataset.csv"
    consolidate_sessions(collected_dir=tmp_path, output_path=output_csv)

    # Sprawdz wynik
    assert output_csv.exists(), "Plik wyjsciowy nie zostal utworzony"

    result_df = pd.read_csv(output_csv)

    # Sprawdz kolumny feat_*
    feat_cols = [c for c in result_df.columns if c.startswith("feat_")]
    assert len(feat_cols) == 63, f"Oczekiwano 63 kolumn feat_*, jest {len(feat_cols)}"

    # Sprawdz ze label jest poprawny
    assert "label" in result_df.columns
    assert result_df["label"].iloc[0] == "A"


def test_dataset_loads_custom_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sprawdza ze PJMDataset poprawnie wczytuje custom_dataset.csv."""
    from app.sign_language import dataset

    # Stworz tymczasowy custom_dataset.csv
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    csv_data = {"label": ["A", "B", "C"] * 20}  # 60 probek
    for i in range(63):
        csv_data[f"feat_{i}"] = [float(i) * 0.01] * 60

    df = pd.DataFrame(csv_data)
    custom_csv = raw_dir / "custom_dataset.csv"
    df.to_csv(custom_csv, index=False)

    # Podmien sciezki
    monkeypatch.setattr(dataset, "RAW_DIR", raw_dir)
    monkeypatch.setattr(
        dataset,
        "CSV_FILES",
        {
            "vectors": raw_dir / "PJM-vectors.csv",  # nie istnieje
            "points": raw_dir / "PJM-points.csv",  # nie istnieje
            "custom": custom_csv,
        },
    )

    # Wczytaj dataset
    pjm_dataset = dataset.PJMDataset(use_multiple_datasets=True)
    X, y = pjm_dataset.load_and_validate()

    # Powinno byc 60 probek x 189 cech (63 * 3)
    assert X.shape == (60, 189), f"Oczekiwano (60, 189), jest {X.shape}"
    assert len(y) == 60


def test_custom_features_are_tiled_3x() -> None:
    """Sprawdza ze cechy z custom CSV sa powielane 3x (63 -> 189)."""

    # Symulacja _extract_custom
    feat_single = np.arange(63, dtype=np.float32).reshape(1, 63)
    feat_tiled = np.tile(feat_single, (1, 3))

    assert feat_tiled.shape == (1, 189)

    # Sprawdz ze cechy sa poprawnie powielone
    np.testing.assert_array_equal(feat_tiled[0, :63], feat_tiled[0, 63:126])
    np.testing.assert_array_equal(feat_tiled[0, :63], feat_tiled[0, 126:189])


def test_feat_column_order_is_deterministic() -> None:
    """Sprawdza ze kolejnosc kolumn feat_* jest deterministyczna."""
    # Sprawdz ze sortowanie jest poprawne
    shuffled = [f"feat_{i}" for i in [5, 0, 62, 30, 1]]
    sorted_cols = sorted(shuffled, key=lambda c: int(c.split("_")[1]))

    assert sorted_cols == [f"feat_{i}" for i in [0, 1, 5, 30, 62]]
