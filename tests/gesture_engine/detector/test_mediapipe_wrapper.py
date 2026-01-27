from types import SimpleNamespace

import app.gesture_engine.detector.mediapipe_wrapper as mw


class DummyCap:
    def __init__(self, opened=True, ret=True, frame=None):
        self._opened = opened
        self._ret = ret
        self._frame = frame if frame is not None else object()
        self.props = {}

    def set(self, prop, val):
        self.props[prop] = val

    def isOpened(self):
        return self._opened

    def read(self):
        return self._ret, self._frame

    def release(self):
        pass


def test_camera_not_opened(monkeypatch):
    monkeypatch.setattr(
        mw,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda *_: DummyCap(opened=False),
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
        ),
    )
    assert mw.get_hand_landmarks() is None


def test_read_failed(monkeypatch):
    cap = DummyCap(opened=True, ret=False, frame=None)
    monkeypatch.setattr(
        mw,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda *_: cap, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4
        ),
    )
    assert mw.get_hand_landmarks() is None


def test_no_hand_detected(monkeypatch):
    # cv2.cvtColor zwraca obiekt ramki bez zmian
    class CV2Mock(SimpleNamespace):
        def __init__(self):
            super().__init__(
                VideoCapture=lambda *_: DummyCap(opened=True, ret=True, frame=object()),
                CAP_PROP_FRAME_WIDTH=3,
                CAP_PROP_FRAME_HEIGHT=4,
                COLOR_BGR2RGB=4,
            )

        @staticmethod
        def cvtColor(frame, _code):
            return frame

    monkeypatch.setattr(mw, "cv2", CV2Mock())

    class Result:
        multi_hand_landmarks = None

    # patch _get_hand_tracker aby zwracal mocka z naszym process
    class MockTracker:
        @staticmethod
        def process(_fr):
            return Result()

    monkeypatch.setattr(mw, "_get_hand_tracker", lambda: MockTracker())

    assert mw.get_hand_landmarks() is None


def test_hand_detected(monkeypatch):
    class CV2Mock(SimpleNamespace):
        def __init__(self):
            super().__init__(
                VideoCapture=lambda *_: DummyCap(opened=True, ret=True, frame=object()),
                CAP_PROP_FRAME_WIDTH=3,
                CAP_PROP_FRAME_HEIGHT=4,
                COLOR_BGR2RGB=4,
            )

        @staticmethod
        def cvtColor(frame, _code):
            return frame

    monkeypatch.setattr(mw, "cv2", CV2Mock())

    class LM:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class Hand:
        def __init__(self):
            self.landmark = [LM(0.1, 0.2, 0.0) for _ in range(21)]

    class Result:
        def __init__(self):
            self.multi_hand_landmarks = [Hand()]

    # patch _get_hand_tracker aby zwracal mocka z naszym process
    class MockTracker:
        @staticmethod
        def process(_fr):
            return Result()

    monkeypatch.setattr(mw, "_get_hand_tracker", lambda: MockTracker())

    lms = mw.get_hand_landmarks()
    assert isinstance(lms, list) and len(lms) == 21
    assert isinstance(lms[0], tuple) and len(lms[0]) == 3
