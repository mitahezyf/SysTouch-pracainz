# loader i preprocessing datasetu PJM - obsluga wielu CSV
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.gesture_engine.logger import logger
from app.sign_language.features import FeatureExtractor, from_points25

# 3 bloki po 63 cechy = 189 cech (poczatek, srodek, koniec gestu)
INPUT_SIZE = 189
BLOCK_SIZE = 63

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
        include_points: bool = False,
        input_size: int = INPUT_SIZE,
    ):
        self.labels_path = Path(labels_path)
        self.use_multiple = use_multiple_datasets
        self.include_points = include_points
        self.input_size = input_size
        self.label_encoder = LabelEncoder()
        self.expected_classes = self._load_expected_classes()
        self.gesture_types: dict[str, str] = {}
        self.sequences: dict[str, list[str]] = {}
        self._load_gesture_metadata()
        self.feature_extractor = FeatureExtractor()
        self._vector_feature_cols: list[str] | None = None

    def _load_expected_classes(self) -> list[str]:
        # wczytuje oczekiwane klasy z pjm.json
        if not self.labels_path.exists():
            logger.warning(
                "Brak pliku etykiet %s, uzywam domyslnych A-Z", self.labels_path
            )
            return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        has_bom = False
        try:
            has_bom = self.labels_path.read_bytes().startswith(b"\xef\xbb\xbf")
        except Exception:
            has_bom = False

        with open(self.labels_path, "r", encoding="utf-8-sig") as f:
            labels_config = json.load(f)

        if has_bom:
            logger.warning("Wykryto BOM w %s (utf-8-sig)", self.labels_path)
        classes = labels_config.get("classes", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        # zapewnienie ze zwracamy list[str]
        return (
            list(classes)
            if isinstance(classes, list)
            else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        )

    def _load_gesture_metadata(self) -> None:
        # wczytuje metadane typow gestow (static/dynamic) oraz sekwencji
        if not self.labels_path.exists():
            logger.warning("Brak pliku etykiet, wszystkie gesty domyslnie statyczne")
            self.gesture_types = {c: "static" for c in self.expected_classes}
            self.sequences = {}
            return

        has_bom = False
        try:
            has_bom = self.labels_path.read_bytes().startswith(b"\xef\xbb\xbf")
        except Exception:
            has_bom = False

        with open(self.labels_path, "r", encoding="utf-8-sig") as f:
            labels_config = json.load(f)

            # gesture_types: mapa litera -> "static" lub "dynamic"
            types_raw = labels_config.get("gesture_types", {})
            self.gesture_types = {
                c: types_raw.get(c, "static") for c in self.expected_classes
            }

            # sequences: dwuznaki i ich komponenty (np. CH -> [C, H])
            self.sequences = labels_config.get("sequences", {})

            logger.info(
                "Zaladowano typy gestow: %d statycznych, %d dynamicznych",
                sum(1 for t in self.gesture_types.values() if t == "static"),
                sum(1 for t in self.gesture_types.values() if t == "dynamic"),
            )
            if self.sequences:
                logger.info("Zaladowano sekwencje: %s", list(self.sequences.keys()))
            if has_bom:
                logger.warning("Wykryto BOM w %s (utf-8-sig)", self.labels_path)

    def _load_single_csv(
        self, csv_path: Path, dataset_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wczytuje pojedynczy plik CSV i ekstrahuje cechy (domyslnie 63D).

        Args:
            csv_path: sciezka do pliku CSV
            dataset_type: typ datasetu ('vectors', 'points')

        Returns:
            X: macierz cech [N, input_size]
            y_raw: wektor etykiet string [N]
        """
        if not csv_path.exists():
            logger.warning("Plik %s nie istnieje, pomijam", csv_path)
            return np.array([]).reshape(0, self.input_size), np.array([])

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
            empty_X = np.array([]).reshape(0, self.input_size)
            empty_y: np.ndarray = np.array([], dtype=object)
            return empty_X, empty_y

        # filtruj etykiety spoza puli expected_classes, aby uniknac bledow enkodera
        label_series = df[label_col].astype(str)
        valid_mask = label_series.isin(self.expected_classes)
        if not valid_mask.all():
            dropped = int((~valid_mask).sum())
            logger.warning(
                "Pominieto %d wierszy z nieznanymi etykietami w %s", dropped, csv_path
            )
            df = df.loc[valid_mask].reset_index(drop=True)
            label_series = df[label_col].astype(str)

        if df.empty:
            return np.array([]).reshape(0, self.input_size), np.array([])

        y_raw = label_series.values

        # konwersja do numpy array z explicit typem
        y_raw_array: np.ndarray = np.asarray(y_raw, dtype=object)

        # ekstrahuj cechy w zaleznosci od typu
        if dataset_type == "vectors":
            X = self._extract_vectors(df, label_col)
        elif dataset_type == "points":
            if not self.include_points:
                logger.info("Dataset points wylaczony - pomijam")
                X = np.array([]).reshape(0, self.input_size)
            else:
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
                y_raw_array = y_raw_array[valid_mask]

        logger.info("Zaladowano %d probek z %s", len(X), dataset_type)
        return X, y_raw_array

    def _extract_vectors(self, df: pd.DataFrame, label_col: str) -> np.ndarray:
        """Ekstrahuje 189 cech z PJM-vectors.csv (3 bloki x 63 cechy)."""
        feature_cols = self._get_vector_feature_cols(df, label_col)
        if not feature_cols:
            return np.array([]).reshape(0, self.input_size)

        X_raw = df[feature_cols].to_numpy(dtype=np.float32)
        X: np.ndarray = np.asarray(X_raw, dtype=np.float32)

        if X.shape[1] != self.input_size:
            logger.error(
                "Nieoczekiwany ksztalt cech vectors: %s (oczekiwano %d kolumn)",
                X.shape,
                self.input_size,
            )
            return np.array([]).reshape(0, self.input_size)

        return X

    def _extract_and_normalize_points(
        self, df: pd.DataFrame, label_col: str
    ) -> np.ndarray:
        """Ekstrahuje cechy z PJM-points.csv (3 bloki x 25 punktow -> 189 cech)."""
        # 3 bloki po 75 kolumn (25 punktow x 3 wspolrzedne)
        all_features: list[np.ndarray] = []

        for block in range(1, 4):
            point_cols = [f"point_{block}_{i}" for i in range(1, 76)]

            missing_cols = [col for col in point_cols if col not in df.columns]
            if missing_cols:
                logger.error(
                    "Brak wymaganych kolumn point_%d_* w PJM-points: %s",
                    block,
                    missing_cols[:5],
                )
                return np.array([]).reshape(0, self.input_size)

            raw_flat = df[point_cols].to_numpy(dtype=np.float32)
            try:
                raw_points = raw_flat.reshape(-1, 25, 3)
            except ValueError as e:
                logger.error(
                    "Nie mozna przeksztalcic bloku %d do (N, 25, 3): %s", block, e
                )
                return np.array([]).reshape(0, self.input_size)

            # generuj 63 cechy dla kazdej probki w bloku
            block_features = []
            for pts in raw_points:
                feat_63 = from_points25(pts)
                block_features.append(feat_63)
            all_features.append(np.array(block_features, dtype=np.float32))

        # polacz 3 bloki (N, 63) -> (N, 189)
        X: np.ndarray = np.concatenate(all_features, axis=1)
        logger.info("Ekstrahowano cechy z %d probek z PJM-points (3 bloki)", len(X))

        return X

    def load_and_validate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wczytuje wszystkie dostepne CSV i laczy dane.

        Returns:
            X: macierz cech [N, input_size], dtype=float32
            y: wektor etykiet [N], dtype=int64 (zakodowane)
        """
        X_list = []
        y_raw_list = []

        if self.use_multiple:
            # laduj wszystkie dostepne datasety
            for dataset_type, csv_path in CSV_FILES.items():
                if dataset_type == "points" and not self.include_points:
                    logger.info("Pomijam PJM-points (include_points=False)")
                    continue
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
            X: macierz cech [N, input_size]
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
            "input_size": self.input_size,
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

    def _get_vector_feature_cols(self, df: pd.DataFrame, label_col: str) -> list[str]:
        """
        Zwraca deterministyczna liste kolumn cech z PJM-vectors.csv.
        Zawiera wszystkie 3 bloki (poczatek, srodek, koniec gestu) = 189 cech.

        Args:
            df: DataFrame z wczytanym PJM-vectors.csv
            label_col: nazwa kolumny z etykietami

        Returns:
            lista nazw kolumn cech (lub pusta lista, jesli nie znaleziono)
        """
        if self._vector_feature_cols is not None:
            return self._vector_feature_cols

        # 3 bloki po 63 cechy = 189 cech lacznie
        required_cols = []
        for block in range(1, 4):  # bloki 1, 2, 3
            # wektor normalny dloni dla bloku
            required_cols.extend(
                [
                    f"vector_hand_{block}_x",
                    f"vector_hand_{block}_y",
                    f"vector_hand_{block}_z",
                ]
            )
            # 20 wektorow kosci dla bloku
            for i in range(1, 21):
                for axis in ("x", "y", "z"):
                    required_cols.append(f"vector_{block}_{i}_{axis}")

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error("Brak wymaganych kolumn w PJM-vectors: %s", missing[:10])
            return []

        self._vector_feature_cols = required_cols
        return required_cols


def load_processed_split(split: str = "train") -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Wczytuje przetworzony split (train/val/test).

    Args:
        split: nazwa splitu ('train', 'val', 'test')

    Returns:
        X: macierz cech [N, input_size]
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
