import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import MagicMock, patch
from app.gesture_engine.utils import ThreadedCapture

# czy poprawnie ustawia parametry kamery i startuje watek
@patch("app.utils.video_capture.cv2.VideoCapture")
def test_threaded_capture_init(mock_cv2):
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, "dummy_frame")
    mock_cv2.return_value = mock_cap

    from app.gesture_engine.config import CAPTURE_WIDTH, CAPTURE_HEIGHT, TARGET_CAMERA_FPS

    capture = ThreadedCapture()

    # czy ustawienia kamery zostaly przypisane poprawnie
    mock_cap.set.assert_any_call(3, CAPTURE_WIDTH)      # CAP_PROP_FRAME_WIDTH
    mock_cap.set.assert_any_call(4, CAPTURE_HEIGHT)     # CAP_PROP_FRAME_HEIGHT
    mock_cap.set.assert_any_call(5, TARGET_CAMERA_FPS)  # CAP_PROP_FPS

    # czy ramka zostala poprawnie ustawiona
    assert capture.ret is True
    assert capture.frame == "dummy_frame"

    capture.stop()

# read() - powinna zwrocic aktualny frame
@patch("app.utils.video_capture.cv2.VideoCapture")
def test_threaded_capture_read(mock_cv2):
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, "some_frame")
    mock_cv2.return_value = mock_cap

    capture = ThreadedCapture()
    ret, frame = capture.read()

    assert ret is True
    assert frame == "some_frame"

    capture.stop()

# stop() - zatrzymuje watek i zwalnia zasoby kamery
@patch("app.utils.video_capture.cv2.VideoCapture")
def test_threaded_capture_stop(mock_cv2):
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, "frame")
    mock_cv2.return_value = mock_cap

    capture = ThreadedCapture()

    # zatrzymanie powinno ustawić running = False i zamknac watek
    capture.stop()
    assert capture.running is False
    mock_cap.release.assert_called_once()
