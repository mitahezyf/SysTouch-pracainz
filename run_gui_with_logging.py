# uruchamia GUI z pelnym logowaniem DEBUG do pliku i konsoli
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# WYMUSZAJ DEBUG w config PRZED jakimkolwiek importem app
os.environ["LOG_LEVEL"] = "DEBUG"

# najpierw ustaw logging PRZED importem app
# aby wszystkie moduły dziedziczyły DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# stworz katalog na logi
log_dir = Path("logs_debug")
log_dir.mkdir(exist_ok=True)

# nazwa pliku z timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"gui_landmarks_{timestamp}.log"

# dodaj handler do pliku
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
)

# dodaj do root loggera
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("GUI uruchomione z pelnym logowaniem DEBUG")
logger.info("Plik logow: %s", log_file)
logger.info("Poziom logowania: DEBUG (wszystkie szczegoly landmarks)")
logger.info("=" * 80)

# teraz zaimportuj i uruchom GUI
try:
    # wymus DEBUG poziom w app.gesture_engine.config
    import app.gesture_engine.config as cfg

    cfg.LOG_LEVEL = "DEBUG"
    cfg.DEBUG_MODE = True

    # wymus DEBUG dla wszystkich loggerów app
    logging.getLogger("app").setLevel(logging.DEBUG)
    logging.getLogger("inzynierka").setLevel(logging.DEBUG)

    # rekonfiguruj logger po zmianie config
    from app.gesture_engine import logger as app_logger

    app_logger.logger.setLevel(logging.DEBUG)

    logger.info("Wymuszono DEBUG mode w app.gesture_engine.config")
    logger.info("config.LOG_LEVEL = %s", cfg.LOG_LEVEL)
    logger.info("config.DEBUG_MODE = %s", cfg.DEBUG_MODE)
    logger.info("=" * 80)

    from app.gui.ui_app import main

    main()
except KeyboardInterrupt:
    logger.info("GUI zamkniete przez uzytkownika (Ctrl+C)")
except Exception as e:
    logger.error("Blad uruchomienia GUI: %s", e, exc_info=True)
finally:
    logger.info("=" * 80)
    logger.info("Sesja zakonczona. Logi zapisane w: %s", log_file)
    logger.info("=" * 80)
    print("\nAby zobaczyc logi:")
    print(f"  notepad {log_file}")
    print("  lub")
    print(f"  type {log_file}")
