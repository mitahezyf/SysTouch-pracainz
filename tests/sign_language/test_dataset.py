# testy dla dataset loadera
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.sign_language.dataset import PJMDataset, load_processed_split


def test_dataset_load_missing_file():
    # test braku pliku CSV
    dataset = PJMDataset(csv_path="nonexistent.csv")

    with pytest.raises(FileNotFoundError, match="Brak pliku datasetu"):
        dataset.load_and_validate()


def test_dataset_load_valid():
    # test wczytywania prawidlowego CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # tworzy mini CSV: 1 kolumna label + 63 cechy
        data = {"label": ["A", "B", "C", "A", "B"]}
        for i in range(63):
            data[f"p{i // 3}_{['x', 'y', 'z'][i % 3]}"] = np.random.rand(5)

        df = pd.DataFrame(data)
        df.to_csv(f.name, index=False)
        temp_path = f.name

    # wczytaj po zamknieciu pliku
    dataset = PJMDataset(csv_path=temp_path)
    X, y = dataset.load_and_validate()

    assert X.shape == (5, 63)
    assert y.shape == (5,)
    assert X.dtype == np.float32
    assert not np.isnan(X).any()

    # sprzatanie
    Path(temp_path).unlink()


def test_dataset_split_and_save():
    # test podzialu i zapisu - uzywamy mniejszej liczby klas ale z wystarczajaca liczba probek
    n_classes = 5
    n_samples_per_class = 20
    X = np.random.rand(n_classes * n_samples_per_class, 63).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_samples_per_class)  # balans klas

    with tempfile.TemporaryDirectory() as tmpdir:
        # mockujemy PROCESSED_DIR
        import app.sign_language.dataset as ds_module

        original_dir = ds_module.PROCESSED_DIR
        ds_module.PROCESSED_DIR = Path(tmpdir)

        dataset = PJMDataset()
        dataset.label_encoder.fit(["A", "B", "C", "D", "E"])  # 5 klas testowych
        dataset.split_and_save(X, y, test_size=0.2, val_size=0.2)

        # sprawdza czy pliki zostaly utworzone
        assert (Path(tmpdir) / "train.npz").exists()
        assert (Path(tmpdir) / "val.npz").exists()
        assert (Path(tmpdir) / "test.npz").exists()

        # przywraca oryginalny katalog
        ds_module.PROCESSED_DIR = original_dir


def test_load_processed_split_missing():
    # test braku przetworzonych danych
    with pytest.raises(FileNotFoundError, match="Brak pliku"):
        load_processed_split("nonexistent")


def test_dataset_invalid_columns():
    # test CSV z bledna liczba kolumn
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data = {"label": ["A", "B"], "feat1": [0.1, 0.2]}  # tylko 2 kolumny zamiast 64
        df = pd.DataFrame(data)
        df.to_csv(f.name, index=False)
        temp_path = f.name

    dataset = PJMDataset(csv_path=temp_path)

    with pytest.raises(ValueError, match="Za malo cech w datasecie"):
        dataset.load_and_validate()

    Path(temp_path).unlink()
