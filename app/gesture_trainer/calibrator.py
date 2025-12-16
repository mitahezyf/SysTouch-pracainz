# mierzy wymiary dloni z landmarkow MediaPipe
import json
from pathlib import Path

from app.gesture_engine.utils.geometry import distance
from app.gesture_engine.utils.landmarks import INDEX_MCP, MIDDLE_TIP, RING_MCP, WRIST

CALIBRATION_PATH = Path(__file__).parent / "data" / "calibration.json"


def calibrate(hand_landmarks):
    """Mierzy rozmiar i szerokosc dloni z landmarkow i zapisuje do pliku."""
    wrist = hand_landmarks[WRIST]
    middle_tip = hand_landmarks[MIDDLE_TIP]
    index_mcp = hand_landmarks[INDEX_MCP]
    ring_mcp = hand_landmarks[RING_MCP]

    hand_size = distance(wrist, middle_tip)
    hand_width = distance(index_mcp, ring_mcp)

    calibration_data = {
        "hand_size": hand_size,
        "hand_width": hand_width,
    }

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(calibration_data, f, indent=2)

    return calibration_data


def load_calibration():
    """Wczytuje dane kalibracji z pliku."""
    if not CALIBRATION_PATH.exists():
        return None
    with open(CALIBRATION_PATH) as f:
        return json.load(f)
