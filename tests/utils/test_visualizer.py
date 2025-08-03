import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.utils.visualizer import Visualizer

# sztuczna ramka do testow - czarne tło
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

# inicjalizacja i skalowanie
def test_visualizer_scaling():
    vis = Visualizer((1920, 1080), (640, 480))
    assert vis.scale_x == 640 / 1920
    assert vis.scale_y == 480 / 1080

# draw_label: rysuje nazwe gestu i confidence
@patch("app.utils.visualizer.cv2.putText")
def test_draw_label(mock_putText):
    frame = dummy_frame()
    vis = Visualizer((1920, 1080), (640, 480))
    vis.draw_label(frame, "click", 0.87)

    mock_putText.assert_called_with(
        frame,
        "click: 87%",
        (10, 60),
        0,
        pytest.approx(0.5, rel=1),
        (255, 255, 255),
        pytest.approx(1, rel=1),
    )

# draw_fps: rysuje FPS
@patch("app.utils.visualizer.cv2.putText")
def test_draw_fps(mock_putText):
    frame = dummy_frame()
    vis = Visualizer((1920, 1080), (640, 480))
    vis.draw_fps(frame, 32.9)

    mock_putText.assert_called_with(
        frame,
        "FPS: 32",
        (10, 20),
        0,
        0.5,
        (0, 255, 0),
        1,
    )

# draw_frametime: rysuje czas klatki
@patch("app.utils.visualizer.cv2.putText")
def test_draw_frametime(mock_putText):
    frame = dummy_frame()
    vis = Visualizer((1920, 1080), (640, 480))
    vis.draw_frametime(frame, 17.2)

    mock_putText.assert_called_with(
        frame,
        "FrameTime: 17 ms",
        (10, 40),
        0,
        0.5,
        (0, 255, 255),
        1,
    )

# draw_current_gesture z nazwa
@patch("app.utils.visualizer.cv2.putText")
def test_draw_current_gesture_with_name(mock_putText):
    frame = dummy_frame()
    vis = Visualizer((1920, 1080), (640, 480))
    vis.draw_current_gesture(frame, "scroll", 0.6)

    mock_putText.assert_called_with(
        frame,
        "Gesture: scroll (60%)",
        (10, 60),
        0,
        0.5,
        (255, 255, 0),
        1,
    )

# raw_current_gesture bez nazwy - None
@patch("app.utils.visualizer.cv2.putText")
def test_draw_current_gesture_none(mock_putText):
    frame = dummy_frame()
    vis = Visualizer((1920, 1080), (640, 480))
    vis.draw_current_gesture(frame, None, 0.0)

    mock_putText.assert_called_with(
        frame,
        "Gesture: None",
        (10, 60),
        0,
        0.5,
        (255, 255, 0),
        1,
    )

# draw_landmarks: czy mediapipe.draw_landmarks zostalo wywolane
@patch("app.utils.visualizer.mp_drawing.draw_landmarks")
def test_draw_landmarks(mock_draw):
    frame = dummy_frame()
    landmarks = MagicMock()

    vis = Visualizer((1920, 1080), (640, 480))
    vis.draw_landmarks(frame, landmarks)

    mock_draw.assert_called_once()

# test draw_hand_box z etykieta
@patch("app.utils.visualizer.cv2.putText")
@patch("app.utils.visualizer.cv2.rectangle")
def test_draw_hand_box_with_label(mock_rect, mock_text):
    frame = dummy_frame()
    mock_hand = MagicMock()
    mock_hand.landmark = [MagicMock(x=0.5, y=0.5) for _ in range(21)]

    vis = Visualizer((1920, 1080), (640, 480))
    vis.draw_hand_box(frame, mock_hand, label="Right")

    mock_rect.assert_called_once()
    mock_text.assert_called_once()

# test draw_hand_box bez etykiety
@patch("app.utils.visualizer.cv2.rectangle")
def test_draw_hand_box_no_label(mock_rect):
    frame = dummy_frame()
    mock_hand = MagicMock()
    mock_hand.landmark = [MagicMock(x=0.4, y=0.6) for _ in range(21)]

    vis = Visualizer((1920, 1080), (640, 480))
    vis.draw_hand_box(frame, mock_hand)

    mock_rect.assert_called_once()
