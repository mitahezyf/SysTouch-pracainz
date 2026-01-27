import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.gesture_engine.config import DEBUG_MODE, LOG_LEVEL

# ustala poziom logowania na podstawie LOG_LEVEL (np. "DEBUG", "INFO"); w razie bledu wraca do DEBUG_MODE
_level: int
_tmp = getattr(logging, str(LOG_LEVEL).upper(), None)
if isinstance(_tmp, int):
    _level = _tmp
else:
    _level = logging.DEBUG if DEBUG_MODE else logging.INFO

logging.basicConfig(
    level=_level,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger("SysTouch")
logger.setLevel(_level)


def _ensure_file_handler() -> None:
    # tworzy handler plikowy z rotacja w katalogu reports/logs
    try:
        # __file__ -> .../app/gesture_engine/logger.py, parent[3] to katalog projektu
        project_root = Path(__file__).resolve().parents[3]
        logs_dir = project_root / "reports" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logfile = logs_dir / "app.log"

        root_logger = logging.getLogger()
        # sprawdza, czy juz dodano RotatingFileHandler na root
        for h in root_logger.handlers:
            if isinstance(h, RotatingFileHandler):
                return

        fh = RotatingFileHandler(
            filename=str(logfile),
            maxBytes=1000000,  # ~1 MB
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(_level)
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        root_logger.addHandler(fh)
    except Exception as e:
        # nie blokuje aplikacji w razie bledu IO; pozostawia logi tylko na konsoli
        logging.getLogger(__name__).debug("logger file handler init error: %s", e)


_ensure_file_handler()
