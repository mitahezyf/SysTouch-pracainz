import importlib
import inspect
import os
import sys
from pathlib import Path

from app.gesture_engine.logger import logger

_log_counter = 0  # licznik do throttlingu

# sciezka do folderu gestures
GESTURE_DIR = Path(__file__).parent.parent / "gestures"


# wczytuje funkcje zaczynajace sie od detect_
def load_gesture_detectors():
    detectors = []

    # uniewaznia cache importu (dla nowych/zmienionych plikow)
    importlib.invalidate_caches()

    for file in os.listdir(GESTURE_DIR):
        if file.endswith(".py") and not file.startswith("__"):
            base = file[:-3]
            # proba importu zgodna z testami (krotsza nazwa), potem pelna nazwa pakietu
            module = None
            module_names = [f"gestures.{base}", f"app.gesture_engine.gestures.{base}"]
            for module_name in module_names:
                try:
                    if module_name in sys.modules:
                        module = importlib.reload(sys.modules[module_name])
                    else:
                        module = importlib.import_module(module_name)
                    break
                except Exception as e:
                    # log na debug, przechodzi do kolejnej nazwy
                    logger.debug(
                        "Nie udalo sie zaladowac %s: %s (sprobuje fallback)",
                        module_name,
                        e,
                    )
                    module = None

            if module is None:
                logger.warning(
                    "Nie udalo sie zaladowac zadnej wersji modulu dla pliku %s", file
                )
                continue

            try:
                members = inspect.getmembers(module, inspect.isfunction)
            except Exception as e:
                # w testach mock side_effect moze wyczerpac iterator -> StopIteration
                logger.debug("inspect.getmembers error for %s: %s", module.__name__, e)
                continue

            for name, obj in members:
                if name.startswith("detect_"):
                    detectors.append(obj)
                    logger.debug(f"Zaladowano gest: {name} z {module.__name__}")

    logger.info(f"Zaladowano {len(detectors)} detektorow gestow.")
    return detectors


gesture_detectors = load_gesture_detectors()


def reload_gesture_detectors() -> None:
    """Przeladowuje liste detektorow gestow z katalogu gestures.

    uzywane przy starcie lub okresowo, aby odswiezyc logike bez restartu aplikacji.
    """
    global gesture_detectors
    gesture_detectors = load_gesture_detectors()


def detect_gesture(landmarks):
    global _log_counter
    _log_counter += 1

    for detector in gesture_detectors:
        gesture = detector(landmarks)
        if gesture:
            name, confidence = gesture
            if _log_counter % 10 == 0:
                logger.debug(f"[gesture] Right: {name} ({confidence:.2f})")
            return gesture
    return None
