from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.gesture_engine.logger import logger
from app.gesture_trainer.collection_summary import (
    format_session_summary,
    summarize_session_dir,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Podsumowanie zebranej sesji datasetu")
    parser.add_argument(
        "session_dir", type=Path, help="np. data/collected/<session_id>"
    )

    args = parser.parse_args(argv)

    try:
        summary = summarize_session_dir(args.session_dir)
        logger.info("\n%s", format_session_summary(summary))
    except Exception as e:
        logger.error("nie mozna podsumowac sesji: %s", e)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
