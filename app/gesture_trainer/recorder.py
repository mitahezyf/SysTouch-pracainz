# zbiera landmarki od uzytkownika dla danego gestu i zapisuje jako probki
import json
from pathlib import Path

from app.gesture_trainer.normalizer import normalize_landmarks

DATA_PATH = Path(__file__).parent / "data" / "raw_landmarks.json"


def record_sample(gesture_name, landmarks):
    """
    zapisuje pojedynczy przyklad gestu (po normalizacji)
    """
    normalized = normalize_landmarks(landmarks)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        with open(DATA_PATH) as f:
            data = json.load(f)
    else:
        data = {}

    if gesture_name not in data:
        data[gesture_name] = []

    data[gesture_name].append(normalized)

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_all_samples():
    if not DATA_PATH.exists():
        return {}
    with open(DATA_PATH) as f:
        return json.load(f)  # dict: gesture_name -> list of landmark vectors
