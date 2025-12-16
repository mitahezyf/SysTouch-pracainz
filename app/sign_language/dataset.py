# loader i preprocessing datasetu PJM-vectors.csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.gesture_engine.logger import logger

# sciezki domyslne
DATA_DIR = Path(__file__).parent / "data"
RAW_CSV = DATA_DIR / "raw" / "PJM-vectors.csv"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_PATH = Path(__file__).parent / "labels" / "pjm.json"


class PJMDataset:
    """Loader datasetu PJM z walidacja i preprocessingiem."""

    def __init__(
        self,
        csv_path: str | Path = RAW_CSV,
        labels_path: str | Path = LABELS_PATH,
    ):
        self.csv_path = Path(csv_path)
        self.labels_path = Path(labels_path)
        self.label_encoder = LabelEncoder()
        self.expected_classes = self._load_expected_classes()

    def _load_expected_classes(self) -> list[str]:
        # wczytuje oczekiwane klasy z pjm.json
        if not self.labels_path.exists():
            logger.warning(
                "Brak pliku etykiet %s, uzywam domyslnych A-Z", self.labels_path
            )
            return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        with open(self.labels_path, encoding="utf-8") as f:
            labels_config = json.load(f)
            classes = labels_config.get("classes", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            # zapewnienie ze zwracamy list[str]
            return (
                list(classes)
                if isinstance(classes, list)
                else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            )

    def load_and_validate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wczytuje CSV i waliduje format.

        Returns:
            X: macierz cech [N, 63], dtype=float32
            y: wektor etykiet [N], dtype=int64 (zakodowane)
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Brak pliku datasetu: {self.csv_path}. Pobierz PJM-vectors.csv z Kaggle i umiesc w {self.csv_path.parent}"
            )

        logger.info("Wczytuje dataset z %s", self.csv_path)
        df = pd.read_csv(self.csv_path)

        # walidacja struktury - dataset moze miec sign_label lub label
        label_col = None
        if "sign_label" in df.columns:
            label_col = "sign_label"
        elif "label" in df.columns:
            label_col = "label"
        else:
            raise ValueError("Brak kolumny 'label' lub 'sign_label' w CSV")

        logger.info("Kolumna etykiet: %s", label_col)

        # dataset PJM ma 3 dlonie (hand_1, hand_2, hand_3), bierzemy tylko hand_1 (63 cechy)
        # kolumny: vector_hand_1_x/y/z, vector_1_1_x/y/z, ..., vector_1_20_x/y/z
        hand1_cols = [
            col
            for col in df.columns
            if col.startswith("vector_hand_1_") or col.startswith("vector_1_")
        ]

        if len(hand1_cols) == 63:
            # mamy dokladnie 63 cechy dla hand_1
            feature_cols = hand1_cols
            logger.info("Uzywam 63 cech z hand_1")
        elif len(hand1_cols) > 63:
            # przytnij do 63
            feature_cols = hand1_cols[:63]
            logger.warning(
                "Znaleziono %d cech hand_1, uzywam pierwszych 63", len(hand1_cols)
            )
        else:
            # fallback: wszystkie kolumny oprocz metadanych
            exclude_cols = [label_col, "user_id", "lux_value"]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            if len(feature_cols) < 63:
                raise ValueError(
                    f"Za malo cech w datasecie: {len(feature_cols)} (oczekiwano 63)"
                )
            # bierz pierwsze 63
            feature_cols = feature_cols[:63]
            logger.warning("Fallback: uzywam pierwszych 63 cech z dostepnych")

        X = df[feature_cols].values.astype(np.float32)
        y_raw = df[label_col].values

        # walidacja NaN i Inf
        if np.isnan(X).any():
            logger.warning("Dataset zawiera NaN, usuwam takie wiersze")
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            y_raw = y_raw[valid_mask]

        if np.isinf(X).any():
            logger.warning("Dataset zawiera Inf, usuwam takie wiersze")
            valid_mask = ~np.isinf(X).any(axis=1)
            X = X[valid_mask]
            y_raw = y_raw[valid_mask]

        # enkoduje etykiety
        self.label_encoder.fit(self.expected_classes)
        y = self.label_encoder.transform(y_raw)

        logger.info(
            "Zaladowano dataset: %d probek, %d klas", len(X), len(self.expected_classes)
        )
        logger.info("Klasy: %s", list(self.label_encoder.classes_))

        return X, y

    def split_and_save(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> None:
        """
        Dzieli dane na train/val/test i zapisuje do .npz.

        Args:
            X: macierz cech [N, 63]
            y: wektor etykiet [N]
            test_size: frakcja danych testowych
            val_size: frakcja danych walidacyjnych (z pozostalych po test)
            random_state: seed dla powtarzalnosci
        """
        # dzieli dane na train+val i test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # dzieli pozostale dane na train i val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp,
        )

        # tworzy katalog wyjsciowy
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # zapisuje do .npz
        meta = {
            "classes": list(self.label_encoder.classes_),
            "num_classes": len(self.label_encoder.classes_),
            "version": "1.0",
        }

        np.savez_compressed(
            PROCESSED_DIR / "train.npz",
            X=X_train,
            y=y_train,
            meta=json.dumps(meta),
        )
        np.savez_compressed(
            PROCESSED_DIR / "val.npz", X=X_val, y=y_val, meta=json.dumps(meta)
        )
        np.savez_compressed(
            PROCESSED_DIR / "test.npz", X=X_test, y=y_test, meta=json.dumps(meta)
        )

        logger.info("Train: %d probek", len(X_train))
        logger.info("Val: %d probek", len(X_val))
        logger.info("Test: %d probek", len(X_test))
        logger.info("Zapisano do %s", PROCESSED_DIR)


def load_processed_split(split: str = "train") -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Wczytuje przetworzony split (train/val/test).

    Args:
        split: nazwa splitu ('train', 'val', 'test')

    Returns:
        X: macierz cech [N, 63]
        y: wektor etykiet [N]
        meta: slownik metadanych
    """
    npz_path = PROCESSED_DIR / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Brak pliku {npz_path}. Uruchom preprocessing: python -m app.sign_language.dataset"
        )

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    meta = json.loads(str(data["meta"]))

    return X, y, meta


def main():
    # punkt wejsciowy do preprocessingu
    logger.info("=== PJM Dataset Preprocessing ===")

    dataset = PJMDataset()
    X, y = dataset.load_and_validate()
    dataset.split_and_save(X, y)

    logger.info("Preprocessing zakonczony. Pliki zapisane w %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
