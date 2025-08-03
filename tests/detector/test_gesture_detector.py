import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import types
from unittest.mock import patch

import app.detector.gesture_detector as gesture_detector


# ladowanie gestow z mockowanych plikow gestures/ i funkcji detect_
@patch("app.detector.gesture_detector.os.listdir")
@patch("app.detector.gesture_detector.importlib.import_module")
@patch("app.detector.gesture_detector.inspect.getmembers")
@patch("app.detector.gesture_detector.logger")
def test_load_gesture_detectors(_, mock_getmembers, mock_import, mock_listdir):
    # czyszczenie globalnego stanu
    gesture_detector.gesture_detectors.clear()

    mock_listdir.return_value = ["click.py", "scroll.py"]

    def dummy1(x): return x
    def dummy2(x): return x
    mock_getmembers.side_effect = [
        [("detect_click", dummy1)],
        [("detect_scroll", dummy2)],
    ]
    mock_import.side_effect = [
        types.ModuleType("click"),
        types.ModuleType("scroll"),
    ]

    detectors = gesture_detector.load_gesture_detectors()

    assert len(detectors) == 2
    assert all(callable(fn) for fn in detectors)


# blad importu modulu gestures powinien zostac zlapany
@patch("app.detector.gesture_detector.os.listdir")
@patch("app.detector.gesture_detector.importlib.import_module")
@patch("app.detector.gesture_detector.logger")
def test_load_with_import_error(_, mock_import, mock_listdir):
    gesture_detector.gesture_detectors.clear()
    mock_listdir.return_value = ["broken.py"]

    def import_side_effect(name):
        if name == "gestures.broken":
            raise Exception("Boom")
        return types.ModuleType("dummy")

    mock_import.side_effect = import_side_effect

    detectors = gesture_detector.load_gesture_detectors()
    assert detectors == []


# pierwszy detector zwraca gest
@patch("app.detector.gesture_detector.gesture_detectors", [
    lambda lm: ("click", 0.95),
    lambda lm: None
])
@patch("app.detector.gesture_detector.logger")
def test_detect_gesture_first_match(_):
    result = gesture_detector.detect_gesture("fake_landmarks")
    assert result == ("click", 0.95)


# zaden detector nie zwraca nic
@patch("app.detector.gesture_detector.gesture_detectors", [
    lambda lm: None,
    lambda lm: None
])
@patch("app.detector.gesture_detector.logger")
def test_detect_gesture_no_match(_):
    result = gesture_detector.detect_gesture("fake_landmarks")
    assert result is None


# logowanie co 10 wywolanie
@patch("app.detector.gesture_detector.gesture_detectors", [
    lambda lm: ("test", 0.88)
])
@patch("app.detector.gesture_detector.logger")
def test_detect_gesture_logs_every_10th(mock_logger):
    gesture_detector._log_counter = 9

    result = gesture_detector.detect_gesture("whatever")
    assert result == ("test", 0.88)
    mock_logger.debug.assert_called_once()
