# skrypt uruchamiajacy pelny pipeline treningu modelu PJM z cechami relatywnymi
import sys

from app.gesture_engine.logger import logger


def run_preprocessing() -> bool:
    """Uruchamia preprocessing danych."""
    logger.info("=== KROK 1/4: Preprocessing danych ===")
    try:
        from app.sign_language.dataset import main as dataset_main

        dataset_main()
        return True
    except Exception as e:
        logger.error("Blad preprocessingu: %s", e)
        return False


def run_validation() -> bool:
    """Uruchamia walidacje cech."""
    logger.info("=== KROK 2/4: Walidacja cech ===")
    try:
        from app.sign_language.validate_features import validate_features

        validate_features()
        return True
    except Exception as e:
        logger.error("Blad walidacji: %s", e)
        return False


def run_training(epochs: int = 150, augment: bool = True) -> bool:
    """Uruchamia trening modelu."""
    logger.info("=== KROK 3/4: Trening modelu ===")
    try:
        from app.sign_language.trainer import train

        metrics = train(
            epochs=epochs,
            lr=0.001,
            hidden_size=256,
            augment_low_accuracy=augment,
            augment_multiplier=15,  # wiecej augmentacji dla problematycznych liter
        )

        logger.info("Metryki treningu:")
        logger.info("  Test Accuracy: %.2f%%", metrics["accuracy"] * 100)
        logger.info("  Val Accuracy: %.2f%%", metrics["val_accuracy"] * 100)
        logger.info("  Final Loss: %.4f", metrics["loss"])

        return True
    except Exception as e:
        logger.error("Blad treningu: %s", e)
        import traceback

        traceback.print_exc()
        return False


def run_diagnosis() -> bool:
    """Uruchamia diagnoze modelu."""
    logger.info("=== KROK 4/4: Diagnoza modelu ===")
    try:
        from app.sign_language.diagnose import main as diagnose_main

        diagnose_main()
        return True
    except Exception as e:
        logger.error("Blad diagnozy: %s", e)
        return False


def main():
    """Glowny pipeline treningu."""
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline treningu modelu PJM")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Pomija preprocessing (uzyj jesli dane juz przetworzone)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Liczba epok treningu (default: 150)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Wylacz augmentacje danych",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PIPELINE TRENINGU MODELU PJM Z CECHAMI RELATYWNYMI (88D)")
    logger.info("=" * 60)

    # krok 1: preprocessing
    if not args.skip_preprocessing:
        if not run_preprocessing():
            logger.error("Pipeline przerwany: blad preprocessingu")
            sys.exit(1)
    else:
        logger.info("Pominieto preprocessing (--skip-preprocessing)")

    # krok 2: walidacja
    if not run_validation():
        logger.warning("Walidacja nie powiodla sie, ale kontynuuje...")

    # krok 3: trening
    if not run_training(epochs=args.epochs, augment=not args.no_augment):
        logger.error("Pipeline przerwany: blad treningu")
        sys.exit(1)

    # krok 4: diagnoza
    if not run_diagnosis():
        logger.warning("Diagnoza nie powiodla sie, ale model zapisany")

    logger.info("=" * 60)
    logger.info("PIPELINE ZAKONCZONY POMYSLNIE")
    logger.info("=" * 60)
    logger.info("Model zapisany w: app/sign_language/models/pjm_model.pth")
    logger.info("Uruchom aplikacje: python -m app.main")


if __name__ == "__main__":
    main()
