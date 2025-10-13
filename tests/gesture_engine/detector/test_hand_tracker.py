import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import MagicMock, patch

from app.gesture_engine.detector.hand_tracker import HandTracker


# inicjalizacja tworzy obiekt mediapipe Hands z odpowiednimi argumentami
@patch("app.detector.hand_tracker.mp.solutions.hands.Hands")
@patch("app.detector.hand_tracker.logger")
def test_hand_tracker_init(mock_logger, mock_hands):
    tracker = HandTracker(
        max_num_hands=3, detection_confidence=0.8, tracking_confidence=0.6
    )

    mock_hands.assert_called_once_with(
        max_num_hands=3,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.6,
        model_complexity=1,
    )

    mock_logger.info.assert_called_once()
    assert tracker.hands == mock_hands.return_value


# metoda process wywoluje hands.process z danym obrazem
@patch("app.detector.hand_tracker.mp.solutions.hands.Hands")
def test_hand_tracker_process(mock_hands_class):
    mock_hands_instance = MagicMock()
    mock_hands_class.return_value = mock_hands_instance

    tracker = HandTracker()
    frame = "fake_rgb_frame"

    result = tracker.process(frame)

    mock_hands_instance.process.assert_called_once_with("fake_rgb_frame")
    assert result == mock_hands_instance.process.return_value
