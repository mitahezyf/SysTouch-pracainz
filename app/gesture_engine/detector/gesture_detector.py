import importlib
import inspect
import os
from pathlib import Path

from app.gesture_engine.logger import logger

_log_counter = 0  # licznik do throttlingu

# ścieżka do folderu gestures
GESTURE_DIR = Path(__file__).parent.parent / "gestures"


# wczytanie funkcji zaczynających się od detect_
def load_gesture_detectors():
    detectors = []

    for file in os.listdir(GESTURE_DIR):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = f"gestures.{file[:-3]}"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                logger.warning(f"Nie udało się załadować modułu {module_name}: {e}")
                continue

            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("detect_"):
                    detectors.append(obj)
                    logger.debug(f"Załadowano gest: {name} z {module_name}")

    logger.info(f"Załadowano {len(detectors)} detektorów gestów.")
    return detectors


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
