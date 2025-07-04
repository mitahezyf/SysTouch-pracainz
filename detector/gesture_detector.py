import importlib
import os
import inspect
from pathlib import Path

#sciezka do folderu gestures
GESTURE_DIR = Path(__file__).parent.parent / "gestures"

#wczytanie funkcji zaczynajacych sie od detect_
def load_gesture_detectors():
    detectors = []

    for file in os.listdir(GESTURE_DIR):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = f"gestures.{file[:-3]}"
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("detect_"):
                    detectors.append(obj)

    return detectors

#tylko przy starcie, laduje gesty
gesture_detectors = load_gesture_detectors()

#glowna funkcja wykrywania gestu
def detect_gesture(landmarks):
    for detector in gesture_detectors:
        gesture = detector(landmarks)
        if gesture:
            return gesture
    return None



















# from gestures.click_gesture import detect_click_gesture
#
# def detect_gesture(landmarks):
#     for detector in [detect_click_gesture]:
#         gesture = detector(landmarks)
#         if gesture:
#             return gesture
#         return None