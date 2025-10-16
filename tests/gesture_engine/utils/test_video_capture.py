import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
from unittest.mock import MagicMock, patch

from app.gesture_engine.utils import ThreadedCapture


# czy poprawnie ustawia parametry kamery i startuje watek
@patch("app.utils.video_capture.cv2.VideoCapture")
def test_threaded_capture_init(mock_cv2):
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, "dummy_frame")
    mock_cv2.return_value = mock_cap

    from app.gesture_engine.config import (
        CAPTURE_HEIGHT,
        CAPTURE_WIDTH,
        TARGET_CAMERA_FPS,
    )

    capture = ThreadedCapture()

    # czy ustawienia kamery zostaly przypisane poprawnie
    mock_cap.set.assert_any_call(3, CAPTURE_WIDTH)  # CAP_PROP_FRAME_WIDTH
    mock_cap.set.assert_any_call(4, CAPTURE_HEIGHT)  # CAP_PROP_FRAME_HEIGHT
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

    # zatrzymanie ustawia running = false i zamyka watek
    capture.stop()
    assert capture.running is False
    mock_cap.release.assert_called_once()


@patch("app.gesture_engine.utils.video_capture.cv2")
def test_threaded_capture_raises_when_camera_not_opened(mock_cv2_mod):
    # przygotowuje stub VideoCapture zwracajacy obiekt bez otwartej kamery
    class DummyCap:
        def isOpened(self):
            return False

        def release(self):
            pass

    def make_cap(*args, **kwargs):
        return DummyCap()

    mock_cv2_mod.VideoCapture = make_cap
    # zapewnia obecnosci backendow, ale i tak nie otworzy
    setattr(mock_cv2_mod, "CAP_DSHOW", 1)
    setattr(mock_cv2_mod, "CAP_MSMF", 2)

    try:
        ThreadedCapture()
        assert (
            False
        ), "ThreadedCapture powinien rzucic RuntimeError, gdy kamera nie jest otwarta"
    except RuntimeError:
        pass


@patch("app.utils.video_capture.cv2.VideoCapture")
def test_threaded_capture_handles_read_exception(mock_cv2):
    # pierwszy read ok przy init, kolejne rzucaja wyjatek w watku
    mock_cap = MagicMock()
    mock_cap.read.side_effect = [(True, "first")]
    mock_cv2.return_value = mock_cap

    capture = ThreadedCapture()

    # teraz spraw, zeby kolejne wywolania read() rzucaly wyjatek
    def boom():
        raise RuntimeError("cap.read fail")

    mock_cap.read.side_effect = boom

    # daj watkowi chwile na jedna iteracje; TARGET_CAMERA_FPS ~ 50 -> ~20ms
    time.sleep(0.06)

    # po wyjatku w update ret powinno byc False, frame None
    ret, frame = capture.read()
    assert ret is False
    assert frame is None

    capture.stop()
