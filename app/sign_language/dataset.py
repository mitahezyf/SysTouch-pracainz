# loader i preprocessing datasetu PJM - obsluga wielu CSV
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.gesture_engine.logger import logger
from app.sign_language.normalizer import MediaPipeNormalizer

# sciezki domyslne
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_PATH = Path(__file__).parent / "labels" / "pjm.json"

# nazwy plikow CSV
CSV_FILES = {
    "vectors": RAW_DIR / "PJM-vectors.csv",
    "points": RAW_DIR / "PJM-points.csv",
    # images pomijamy - tylko metadane
}


class PJMDataset:
    """Loader datasetu PJM z walidacja i preprocessingiem - obsluga wielu CSV."""

    def __init__(
        self,
        labels_path: str | Path = LABELS_PATH,
        use_multiple_datasets: bool = True,
    ):
        self.labels_path = Path(labels_path)
        self.use_multiple = use_multiple_datasets
        self.label_encoder = LabelEncoder()
        self.expected_classes = self._load_expected_classes()
        self.normalizer = MediaPipeNormalizer()

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

    def _load_single_csv(
        self, csv_path: Path, dataset_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wczytuje pojedynczy plik CSV i ekstrahuje cechy 63D.

        Args:
            csv_path: sciezka do pliku CSV
            dataset_type: typ datasetu ('vectors', 'points')

        Returns:
            X: macierz cech [N, 63]
            y_raw: wektor etykiet string [N]
        """
        if not csv_path.exists():
            logger.warning("Plik %s nie istnieje, pomijam", csv_path)
            return np.array([]).reshape(0, 63), np.array([])

        logger.info("Wczytuje %s z %s", dataset_type, csv_path)
        df = pd.read_csv(csv_path)

        # znajdz kolumne etykiet
        label_col = None
        if "sign_label" in df.columns:
            label_col = "sign_label"
        elif "label" in df.columns:
            label_col = "label"
        else:
            logger.error("Brak kolumny etykiet w %s, pomijam", csv_path)
            return np.array([]).reshape(0, 63), np.array([])

        y_raw = df[label_col].values

        # ekstrahuj cechy w zaleznosci od typu
        if dataset_type == "vectors":
            X = self._extract_vectors(df, label_col)
        elif dataset_type == "points":
            X = self._extract_and_normalize_points(df, label_col)
        else:
            raise ValueError(f"Nieznany typ datasetu: {dataset_type}")

        # walidacja NaN/Inf
        if len(X) > 0:
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                logger.warning(
                    "Usunieto %d wierszy z NaN/Inf z %s", invalid_count, dataset_type
                )
                X = X[valid_mask]
                y_raw = y_raw[valid_mask]

        logger.info("Zaladowano %d probek z %s", len(X), dataset_type)
        return X, y_raw

    def _extract_vectors(self, df: pd.DataFrame, label_col: str) -> np.ndarray:
        """Ekstrahuje gotowe wektory 63D z PJM-vectors.csv"""
        # kolumny vector_hand_1_* lub vector_1_*
        hand1_cols = [
            col
            for col in df.columns
            if col.startswith("vector_hand_1_") or col.startswith("vector_1_")
        ]

        if len(hand1_cols) >= 63:
            feature_cols = hand1_cols[:63]
        else:
            # fallback: wszystkie kolumny oprocz metadanych
            exclude_cols = [label_col, "user_id", "lux_value"]
            feature_cols = [col for col in df.columns if col not in exclude_cols][:63]

        if len(feature_cols) < 63:
            logger.error(
                "Za malo cech w PJM-vectors: %d (oczekiwano 63)", len(feature_cols)
            )
            return np.array([]).reshape(0, 63)

        X = df[feature_cols].values.astype(np.float32)
        return X

    def _extract_and_normalize_points(
        self, df: pd.DataFrame, label_col: str
    ) -> np.ndarray:
        """Ekstrahuje i normalizuje raw landmarki z PJM-points.csv"""
        # PJM-points ma 75 punktow, ale MediaPipe uzywa tylko 21
        # kolumny: point_1_1 do point_1_63 (21 punktow x 3 wspolrzedne)
        point_cols = [f"point_1_{i}" for i in range(1, 64)]  # 1..63

        # sprawdz czy wszystkie kolumny istnieja
        missing_cols = [col for col in point_cols if col not in df.columns]
        if missing_cols:
            logger.error(
                "Brak wymaganych kolumn point w PJM-points: %s", missing_cols[:5]
            )
            return np.array([]).reshape(0, 63)

        # wczytaj raw landmarki
        raw_landmarks = df[point_cols].values.astype(np.float32)

        # reshape do [N, 21, 3]
        try:
            raw_landmarks = raw_landmarks.reshape(-1, 21, 3)
        except ValueError as e:
            logger.error("Nie mozna przeksztalcic do (N, 21, 3): %s", e)
            return np.array([]).reshape(0, 63)

        # normalizuj batch
        logger.info("Normalizuje %d probek z PJM-points", len(raw_landmarks))
        X_normalized = self.normalizer.normalize_batch(raw_landmarks)

        return X_normalized

    def load_and_validate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wczytuje wszystkie dostepne CSV i laczy dane.

        Returns:
            X: macierz cech [N, 63], dtype=float32
            y: wektor etykiet [N], dtype=int64 (zakodowane)
        """
        X_list = []
        y_raw_list = []

        if self.use_multiple:
            # laduj wszystkie dostepne datasety
            for dataset_type, csv_path in CSV_FILES.items():
                X_part, y_part = self._load_single_csv(csv_path, dataset_type)
                if len(X_part) > 0:
                    X_list.append(X_part)
                    y_raw_list.append(y_part)
        else:
            # tylko PJM-vectors (legacy)
            vectors_path = CSV_FILES["vectors"]
            X_part, y_part = self._load_single_csv(vectors_path, "vectors")
            if len(X_part) > 0:
                X_list.append(X_part)
                y_raw_list.append(y_part)

        if not X_list:
            raise FileNotFoundError(
                f"Brak datasetow w {RAW_DIR}. Pobierz PJM-vectors.csv i/lub PJM-points.csv z Kaggle"
            )

        # lacz wszystkie datasety
        X = np.vstack(X_list)
        y_raw = np.concatenate(y_raw_list)

        logger.info("Polaczono datasety: %d probek lacznie", len(X))

        # enkoduj etykiety
        self.label_encoder.fit(self.expected_classes)
        y = self.label_encoder.transform(y_raw)

        logger.info("Liczba klas: %d", len(self.expected_classes))
        logger.info("Klasy: %s", list(self.label_encoder.classes_))

        # sprawdz balans klas
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Rozklad klas:")
        for cls_idx, count in zip(unique, counts):
            cls_name = self.label_encoder.classes_[cls_idx]
            logger.info("  %s: %d probek", cls_name, count)

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
            "version": "2.0",  # wersja 2.0 - multi-dataset
            "use_multiple": self.use_multiple,
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
    logger.info("=== PJM Dataset Preprocessing (Multi-CSV) ===")

    dataset = PJMDataset(use_multiple_datasets=True)
    X, y = dataset.load_and_validate()
    dataset.split_and_save(X, y)

    logger.info("Preprocessing zakonczony. Pliki zapisane w %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
