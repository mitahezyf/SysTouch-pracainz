# skrypt walidacyjny - sprawdza cechy relatywne i ich jakosc
from pathlib import Path

import numpy as np

from app.gesture_engine.logger import logger
from app.sign_language.dataset import load_processed_split


def validate_features() -> None:
    """Waliduje cechy relatywne w przetworzonych danych."""
    logger.info("=== Walidacja cech relatywnych ===")

    try:
        X_train, y_train, meta_train = load_processed_split("train")
        X_val, y_val, meta_val = load_processed_split("val")
        X_test, y_test, meta_test = load_processed_split("test")
    except FileNotFoundError as e:
        logger.error(
            "Brak przetworzonych danych. Uruchom: python -m app.sign_language.dataset"
        )
        raise e

    logger.info("Train: %d probek, shape=%s", len(X_train), X_train.shape)
    logger.info("Val: %d probek, shape=%s", len(X_val), X_val.shape)
    logger.info("Test: %d probek, shape=%s", len(X_test), X_test.shape)

    # sprawdz wymiary cech
    expected_dim = 88
    if X_train.shape[1] != expected_dim:
        logger.warning("Oczekiwano %dD, otrzymano %dD", expected_dim, X_train.shape[1])
    else:
        logger.info("Wymiar cech: %dD - OK", X_train.shape[1])

    # sprawdz statystyki cech
    logger.info("Statystyki cech (train):")
    logger.info(
        "  Mean: min=%.4f, max=%.4f",
        X_train.mean(axis=0).min(),
        X_train.mean(axis=0).max(),
    )
    logger.info(
        "  Std: min=%.4f, max=%.4f",
        X_train.std(axis=0).min(),
        X_train.std(axis=0).max(),
    )

    # wykryj cechy z zerowa wariancja
    zero_var_mask = X_train.std(axis=0) < 1e-6
    zero_var_indices = np.where(zero_var_mask)[0]

    if len(zero_var_indices) > 0:
        logger.warning(
            "Znaleziono %d cech z zerowa wariancja: %s",
            len(zero_var_indices),
            zero_var_indices[:10],
        )

        # zapisz indeksy do pliku
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        np.save(models_dir / "zero_var_indices.npy", zero_var_indices)
        logger.info(
            "Zapisano indeksy cech z zerowa wariancja: models/zero_var_indices.npy"
        )
    else:
        logger.info("Wszystkie cechy maja niezerowa wariancje - OK")

    # sprawdz rozklad klas
    classes = np.array(meta_train["classes"])
    unique, counts = np.unique(y_train, return_counts=True)

    logger.info("Rozklad klas (train):")
    class_balance = {}
    for cls_idx, count in zip(unique, counts):
        cls_name = classes[cls_idx]
        class_balance[cls_name] = count
        logger.info("  %s: %d probek", cls_name, count)

    # znajdz najbardziej i najmniej reprezentowane klasy
    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    logger.info("Balans klas:")
    logger.info("  Min probek: %d", min_count)
    logger.info("  Max probek: %d", max_count)
    logger.info("  Imbalance ratio: %.2f:1", imbalance_ratio)

    if imbalance_ratio > 3.0:
        logger.warning("Duza nierownowaga klas (>3:1), rozważ augmentacje")
        # znajdz klasy z najmniejsza reprezentacja
        underrepresented = [
            (classes[i], counts[list(unique).index(i)])
            for i in range(len(classes))
            if i in unique and counts[list(unique).index(i)] < np.median(counts)
        ]
        logger.info("Klasy podreprezentowane (<mediana): %s", underrepresented)
    else:
        logger.info("Balans klas w normie (<3:1)")

    # sprawdz NaN/Inf
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        logger.error("BLAD: Train zawiera NaN lub Inf!")
    else:
        logger.info("Train: brak NaN/Inf - OK")

    if np.isnan(X_val).any() or np.isinf(X_val).any():
        logger.error("BLAD: Val zawiera NaN lub Inf!")
    else:
        logger.info("Val: brak NaN/Inf - OK")

    if np.isnan(X_test).any() or np.isinf(X_test).any():
        logger.error("BLAD: Test zawiera NaN lub Inf!")
    else:
        logger.info("Test: brak NaN/Inf - OK")

    logger.info("=== Walidacja zakonczona ===")


if __name__ == "__main__":
    validate_features()
