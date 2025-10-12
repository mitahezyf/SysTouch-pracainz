import json
from pathlib import Path

from app.gesture_trainer.classifier import GestureClassifier
from app.gesture_trainer.recorder import load_all_samples

MAP_PATH = Path(__file__).parent / "data" / "gesture_action_map.json"


def assign_action(gesture_name, action_name):
    """Przypisuje akcję do gestu"""
    if MAP_PATH.exists():
        with open(MAP_PATH) as f:
            mapping = json.load(f)
    else:
        mapping = {}

    mapping[gesture_name] = action_name

    with open(MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=2)


def get_action_for_gesture(gesture_name):
    if not MAP_PATH.exists():
        return None
    with open(MAP_PATH) as f:
        mapping = json.load(f)
    return mapping.get(gesture_name)


def train_and_save_model():
    samples = load_all_samples()
    clf = GestureClassifier()
    clf.train(samples)
    clf.save()
    return clf
