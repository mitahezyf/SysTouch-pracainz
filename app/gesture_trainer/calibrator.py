# oblicza wielkosc dloni (skale) na podstawie dystansu miedzy punktami landmarkow
import json
from pathlib import Path

from app.gesture_engine.utils.geometry import distance
from app.gesture_engine.utils.landmarks import INDEX_MCP
from app.gesture_engine.utils.landmarks import MIDDLE_TIP
from app.gesture_engine.utils.landmarks import RING_MCP
from app.gesture_engine.utils.landmarks import WRIST


CALIBRATION_PATH = Path(__file__).parent / "data" / "calibration.json"


def calibrate(hand_landmarks):
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
    if not CALIBRATION_PATH.exists():
        return None
    with open(CALIBRATION_PATH) as f:
        return json.load(f)
