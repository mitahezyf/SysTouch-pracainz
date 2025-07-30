import ctypes
import time
from collections import deque

from app.actions.hooks import register_gesture_start_hook
from app.config import MAX_SCROLL_SPEED
from app.config import SCROLL_BASE_INTERVAL
from app.config import SCROLL_SENSITIVITY
from app.utils.landmarks import WRIST

position_buffer = deque(maxlen=3)
scroll_anchor_y = None
last_scroll_time = 0


def scroll_start_hook(landmarks, frame_shape):
    wrist = landmarks[WRIST]
    screen_h = frame_shape[0]
    anchor_y = int(wrist.y * screen_h)
    set_scroll_anchor(anchor_y)
    reset_scroll()


register_gesture_start_hook("scroll", scroll_start_hook)


def scroll_windows(amount):
    # Windows native scroll using user32.mouse_event
    ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(amount * 120), 0)


def set_scroll_anchor(y):
    global scroll_anchor_y
    scroll_anchor_y = y


def handle_scroll(landmarks, frame_shape):
    global scroll_anchor_y, last_scroll_time

    wrist = landmarks[WRIST]
    screen_h = frame_shape[0]
    current_y = int(wrist.y * screen_h)

    position_buffer.append(current_y)

    if len(position_buffer) < 3:
        return

    avg_y = sum(position_buffer) / len(position_buffer)

    if scroll_anchor_y is None:
        scroll_anchor_y = avg_y
        return

    delta = avg_y - scroll_anchor_y

    if abs(delta) < SCROLL_SENSITIVITY:
        return

    # dynamiczne tempo
    scale = min(abs(delta) / 30, MAX_SCROLL_SPEED)
    direction = -1 if delta > 0 else 1

    now = time.time()
    interval = SCROLL_BASE_INTERVAL / scale

    if now - last_scroll_time >= interval:
        scroll_windows(direction)
        last_scroll_time = now


def reset_scroll():
    global scroll_anchor_y, position_buffer, last_scroll_time
    scroll_anchor_y = None
    position_buffer.clear()
    last_scroll_time = 0
