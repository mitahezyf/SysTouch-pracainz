import json
import os
from pathlib import Path

import numpy as np

from app.sign_language.trainer import train


def _make_dummy_processed_data(
    tmpdir: Path, num_samples: int = 60, num_classes: int = 3
):
    # tworzy syntetyczne processed splits (train/val/test.npz)
    # balans klas: po 20 probek na klase
    samples_per_class = num_samples // num_classes
    X = np.random.randn(num_samples, 63).astype(np.float32)
    y = np.repeat(np.arange(num_classes), samples_per_class)

    classes = [chr(65 + i) for i in range(num_classes)]  # A, B, C, ...
    meta = {
        "classes": classes,
        "num_classes": num_classes,
        "version": "1.0",
    }

    # split: 60% train, 20% val, 20% test
    split_train = int(num_samples * 0.6)
    split_val = int(num_samples * 0.8)

    np.savez_compressed(
        tmpdir / "train.npz",
        X=X[:split_train],
        y=y[:split_train],
        meta=json.dumps(meta),
    )
    np.savez_compressed(
        tmpdir / "val.npz",
        X=X[split_train:split_val],
        y=y[split_train:split_val],
        meta=json.dumps(meta),
    )
    np.savez_compressed(
        tmpdir / "test.npz",
        X=X[split_val:],
        y=y[split_val:],
        meta=json.dumps(meta),
    )


def test_trainer_runs_and_saves(tmp_path):
    # przygotuj katalog processed
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _make_dummy_processed_data(processed_dir, num_samples=60, num_classes=3)

    # mockuj PROCESSED_DIR
    import app.sign_language.dataset as ds_module

    original_dir = ds_module.PROCESSED_DIR
    ds_module.PROCESSED_DIR = processed_dir

    model_file = tmp_path / "model.pth"
    classes_file = tmp_path / "classes.npy"

    try:
        metrics = train(
            model_path=str(model_file),
            classes_path=str(classes_file),
            epochs=5,
            lr=0.01,
            hidden_size=32,
        )

        assert os.path.exists(model_file)
        assert os.path.exists(classes_file)
        assert (
            "accuracy" in metrics and "loss" in metrics and metrics["num_classes"] == 3
        )
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["loss"] >= 0.0
    finally:
        ds_module.PROCESSED_DIR = original_dir


def test_trainer_repro_dimensions(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    # wiecej probek aby model mial szanse sie nauczyc
    _make_dummy_processed_data(processed_dir, num_samples=150, num_classes=3)

    import app.sign_language.dataset as ds_module

    original_dir = ds_module.PROCESSED_DIR
    ds_module.PROCESSED_DIR = processed_dir

    model_file = tmp_path / "model.pth"
    classes_file = tmp_path / "classes.npy"

    try:
        metrics = train(
            model_path=str(model_file),
            classes_path=str(classes_file),
            epochs=10,  # wiecej epok
            hidden_size=32,  # wiekszy model
            lr=0.01,
        )

        assert metrics["num_classes"] == 3
        # accuracy moze byc niska na sztucznych danych, ale powinien sie kompilowac
        assert 0.0 <= metrics["accuracy"] <= 1.0
    finally:
        ds_module.PROCESSED_DIR = original_dir
