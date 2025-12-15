import os

import numpy as np
import pandas as pd

from app.sign_language.trainer import train


def _make_dummy_dataset(path: str, num_samples: int = 50, num_features: int = 63):
    # tworzy syntetyczny dataset: kolumna label + 63 cech
    labels = np.random.choice(list("ABC"), size=num_samples)  # 3 klasy
    data = np.random.randn(num_samples, num_features).astype(np.float32)
    # sklada dataframe
    cols = ["label"] + [f"p{i}_{ax}" for i in range(21) for ax in ["x", "y", "z"]]
    df = pd.DataFrame(
        [[labels[i]] + data[i].tolist() for i in range(num_samples)], columns=cols
    )
    df.to_csv(path, index=False)


def test_trainer_runs_and_saves(tmp_path):
    data_file = tmp_path / "dataset.csv"
    model_file = tmp_path / "model.pth"
    classes_file = tmp_path / "classes.npy"
    _make_dummy_dataset(str(data_file))

    metrics = train(
        data_file=str(data_file),
        model_path=str(model_file),
        classes_path=str(classes_file),
        epochs=5,
        lr=0.01,
    )

    assert os.path.exists(model_file)
    assert os.path.exists(classes_file)
    assert "accuracy" in metrics and "loss" in metrics and metrics["num_classes"] == 3
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] >= 0.0


def test_trainer_repro_dimensions(tmp_path):
    data_file = tmp_path / "dataset.csv"
    model_file = tmp_path / "model.pth"
    classes_file = tmp_path / "classes.npy"
    _make_dummy_dataset(str(data_file), num_samples=30)

    metrics = train(
        data_file=str(data_file),
        model_path=str(model_file),
        classes_path=str(classes_file),
        epochs=3,
    )

    assert metrics["num_classes"] == 3
