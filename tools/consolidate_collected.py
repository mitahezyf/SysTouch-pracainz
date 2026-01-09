# konsoliduje zebrane CSV z data/collected/* do jednego pliku
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.gesture_engine.logger import logger

# sciezki domyslne
COLLECTED_DIR = Path(__file__).parent.parent / "data" / "collected"
OUTPUT_DIR = Path(__file__).parent.parent / "app" / "sign_language" / "data" / "raw"


def consolidate_sessions(
    collected_dir: Path = COLLECTED_DIR, output_path: Path | None = None
) -> None:
    """
    Konsoliduje wszystkie sesje zbierania do jednego CSV.

    Args:
        collected_dir: katalog z sesjami (data/collected)
        output_path: sciezka wyjsciowa (domyslnie: app/sign_language/data/raw/custom_dataset.csv)
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "custom_dataset.csv"

    if not collected_dir.exists():
        logger.error("Brak katalogu %s", collected_dir)
        return

    # zbierz wszystkie pliki CSV z features/
    all_csv_files: list[Path] = []
    for session_dir in collected_dir.iterdir():
        if not session_dir.is_dir():
            continue
        features_dir = session_dir / "features"
        if not features_dir.exists():
            continue
        csv_files = list(features_dir.glob("*.csv"))
        all_csv_files.extend(csv_files)
        logger.info("Sesja %s: %d plikow CSV", session_dir.name, len(csv_files))

    if not all_csv_files:
        logger.error("Brak plikow CSV w %s", collected_dir)
        return

    logger.info("Znaleziono %d plikow CSV", len(all_csv_files))

    # wczytaj wszystkie CSV
    dfs: list[pd.DataFrame] = []
    stats: dict[str, int] = {}

    for csv_file in all_csv_files:
        try:
            df = pd.read_csv(csv_file)

            # filtruj wiersze gdzie has_hand=1 (tylko klatki z wykryta reka)
            if "has_hand" in df.columns:
                df = df[df["has_hand"] == 1].copy()

            if df.empty:
                logger.warning(
                    "Plik %s nie ma klatek z has_hand=1, pomijam", csv_file.name
                )
                continue

            # statystyki
            if "label" in df.columns:
                label = df["label"].iloc[0]
                stats[label] = stats.get(label, 0) + len(df)

            dfs.append(df)
            logger.debug("Wczytano %s: %d klatek", csv_file.name, len(df))
        except Exception as e:
            logger.warning("Nie mozna wczytac %s: %s", csv_file.name, e)

    if not dfs:
        logger.error("Nie wczytano zadnych danych")
        return

    # polacz wszystkie DataFrame
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info("Polaczono %d klatek", len(df_all))

    # statystyki
    logger.info("Rozklad etykiet:")
    for label, count in sorted(stats.items()):
        logger.info("  %s: %d klatek", label, count)

    # zapisz do CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False)
    logger.info("Zapisano do %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Konsoliduje zebrane CSV z data/collected/* do jednego pliku"
    )
    parser.add_argument(
        "--collected-dir",
        type=Path,
        default=COLLECTED_DIR,
        help="katalog z sesjami (domyslnie: data/collected)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "custom_dataset.csv",
        help="sciezka wyjsciowa (domyslnie: app/sign_language/data/raw/custom_dataset.csv)",
    )

    args = parser.parse_args()

    logger.info("=== Konsolidacja zebranych danych ===")
    consolidate_sessions(args.collected_dir, args.output)
    logger.info("Gotowe!")

    return 0


if __name__ == "__main__":
    exit(main())
