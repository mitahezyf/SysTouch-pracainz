"""
Zunifikowany skrypt treningu modelu PyTorch MLP na wlasnych danych.

Pipeline:
1. Konsolidacja sesji z data/collected/ -> custom_dataset.csv
2. Preprocessing (walidacja, split train/val/test)
3. Trening PyTorch MLP

Uzycie:
    python tools/run_full_training.py [--epochs=100] [--hidden-size=256]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Upewniamy sie ze importy dzialaja
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.gesture_engine.logger import logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pelny pipeline treningu modelu na wlasnych danych"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Liczba epok treningu (domyslnie: 100)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Rozmiar warstwy ukrytej (domyslnie: 256)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (domyslnie: 0.001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (domyslnie: 32)"
    )
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Pomin konsolidacje (jesli custom_dataset.csv juz istnieje)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Pomin preprocessing (jesli train.npz/val.npz/test.npz juz istnieja)",
    )
    parser.add_argument(
        "--collected-dir",
        type=Path,
        default=Path("data/collected"),
        help="Katalog z sesjami (domyslnie: data/collected)",
    )

    args = parser.parse_args(argv)

    logger.info("=" * 60)
    logger.info("PELNY PIPELINE TRENINGU MODELU")
    logger.info("=" * 60)

    # Sciezki
    project_root = Path(__file__).parent.parent
    custom_csv = (
        project_root / "app" / "sign_language" / "data" / "raw" / "custom_dataset.csv"
    )
    processed_dir = project_root / "app" / "sign_language" / "data" / "processed"

    # KROK 1: Konsolidacja sesji
    if not args.skip_consolidate:
        logger.info("")
        logger.info("KROK 1: Konsolidacja sesji z %s", args.collected_dir)
        logger.info("-" * 40)

        from tools.consolidate_collected import consolidate_sessions

        consolidate_sessions(
            collected_dir=args.collected_dir.resolve(),
            output_path=custom_csv,
        )

        if not custom_csv.exists():
            logger.error("Konsolidacja nie utworzyla pliku %s", custom_csv)
            return 1

        logger.info("OK: Utworzono %s", custom_csv)
    else:
        logger.info("")
        logger.info("KROK 1: Konsolidacja pomieta (--skip-consolidate)")
        if not custom_csv.exists():
            logger.error("Brak pliku %s - uruchom bez --skip-consolidate", custom_csv)
            return 1

    # KROK 2: Preprocessing (tworzenie train/val/test.npz)
    if not args.skip_preprocess:
        logger.info("")
        logger.info("KROK 2: Preprocessing i podzial danych")
        logger.info("-" * 40)

        from app.sign_language.dataset import PJMDataset

        # Uzywamy TYLKO custom datasetu (use_multiple_datasets=False wylacza PJM-vectors)
        # ale _extract_custom jest wolany tylko dla custom CSV
        dataset = PJMDataset(use_multiple_datasets=True)

        # Sprawdz czy custom_dataset.csv istnieje
        from app.sign_language.dataset import CSV_FILES

        if not CSV_FILES["custom"].exists():
            logger.error("Brak custom_dataset.csv - najpierw uruchom konsolidacje")
            return 1

        X, y = dataset.load_and_validate()

        if len(X) == 0:
            logger.error("Brak danych po wczytaniu - sprawdz format CSV")
            return 1

        dataset.split_and_save(X, y)

        logger.info("OK: Utworzono train.npz, val.npz, test.npz w %s", processed_dir)
    else:
        logger.info("")
        logger.info("KROK 2: Preprocessing pominiety (--skip-preprocess)")
        train_npz = processed_dir / "train.npz"
        if not train_npz.exists():
            logger.error("Brak %s - uruchom bez --skip-preprocess", train_npz)
            return 1

    # KROK 3: Trening modelu PyTorch MLP
    logger.info("")
    logger.info("KROK 3: Trening modelu PyTorch MLP")
    logger.info("-" * 40)
    logger.info(
        "  epochs=%d, hidden_size=%d, lr=%f, batch_size=%d",
        args.epochs,
        args.hidden_size,
        args.lr,
        args.batch_size,
    )

    from app.sign_language.trainer import train

    train(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("TRENING ZAKONCZONY POMYSLNIE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Model zapisany w: app/sign_language/models/pjm_model.pth")
    logger.info("Klasy zapisane w: app/sign_language/models/classes.npy")
    logger.info("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
