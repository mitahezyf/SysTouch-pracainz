# testy akcji scroll
import sys
from unittest import mock

from app.gesture_engine.utils.landmarks import WRIST


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(wrist_y: float = 0.5) -> list[P]:
    pts = [P(0.5, 0.5, 0.0) for _ in range(21)]
    pts[WRIST] = P(0.5, wrist_y, 0.0)
    return pts


def test_set_scroll_anchor():
    # lazy import zeby uniknac circular import
    import app.gesture_engine.actions.scroll_action as scroll

    scroll.reset_scroll()
    assert scroll.scroll_anchor_y is None

    scroll.set_scroll_anchor(100)
    assert scroll.scroll_anchor_y == 100

    scroll.reset_scroll()
    assert scroll.scroll_anchor_y is None


def test_reset_scroll():
    import app.gesture_engine.actions.scroll_action as scroll

    scroll.scroll_anchor_y = 123
    scroll.position_buffer.append(50)
    scroll.last_scroll_time = 99.0

    scroll.reset_scroll()

    assert scroll.scroll_anchor_y is None
    assert len(scroll.position_buffer) == 0
    assert scroll.last_scroll_time == 0


def test_handle_scroll_sets_anchor_if_none():
    import app.gesture_engine.actions.scroll_action as scroll

    scroll.reset_scroll()
    pts = make_landmarks(wrist_y=0.5)
    frame_shape = (480, 640, 3)

    # pierwsze wywolania buduja bufor
    for _ in range(3):
        scroll.handle_scroll(pts, frame_shape)

    # anchor powinien byc ustawiony
    assert scroll.scroll_anchor_y is not None


def test_handle_scroll_does_not_scroll_below_sensitivity():
    import app.gesture_engine.actions.scroll_action as scroll

    scroll.reset_scroll()
    frame_shape = (480, 640, 3)

    # ustaw anchor
    scroll.set_scroll_anchor(240)

    # wrist blisko anchor (maly delta)
    pts = make_landmarks(wrist_y=0.5)  # 0.5 * 480 = 240
    for _ in range(3):
        scroll.handle_scroll(pts, frame_shape)

    # nie powinno wywolac scroll_windows bo delta < sensitivity


def test_scroll_windows_skips_on_non_windows():
    import app.gesture_engine.actions.scroll_action as scroll

    # na nie-windows powinien tylko zalogowac warning
    if sys.platform == "win32":
        # na windows mockujemy ctypes
        with mock.patch("ctypes.windll.user32.mouse_event") as mock_event:
            scroll.scroll_windows(1)
            mock_event.assert_called_once()
    else:
        # na linux/mac po prostu nie crashuje
        scroll.scroll_windows(1)
