# todo do poprawy hooki i akceleracja scrolla
import ctypes
import time
from collections import deque

from app.gesture_engine.config import MAX_SCROLL_SPEED
from app.gesture_engine.config import SCROLL_BASE_INTERVAL
from app.gesture_engine.config import SCROLL_SENSITIVITY
from app.gesture_engine.core.hooks import register_gesture_start_hook
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.landmarks import WRIST

position_buffer = deque(maxlen=3)
scroll_anchor_y = None
last_scroll_time = 0


def scroll_start_hook(landmarks, frame_shape):
    wrist = landmarks[WRIST]
    screen_h = frame_shape[0]
    anchor_y = int(wrist.y * screen_h)
    set_scroll_anchor(anchor_y)
    reset_scroll()
    logger.debug(f"[scroll] start hook: anchor_y={anchor_y}")


register_gesture_start_hook("scroll", scroll_start_hook)


def scroll_windows(amount):
    # Windows native scroll using user32.mouse_event
    logger.debug(f"[scroll] wykonanie scrolla: amount={amount}")
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
        logger.debug(f"[scroll] brak anchor, ustawiam na: {scroll_anchor_y}")
        return

    delta = avg_y - scroll_anchor_y

    if abs(delta) < SCROLL_SENSITIVITY:
        return

    scale = min(abs(delta) / 30, MAX_SCROLL_SPEED)
    direction = -1 if delta > 0 else 1

    now = time.time()
    interval = SCROLL_BASE_INTERVAL / scale

    if now - last_scroll_time >= interval:
        scroll_windows(direction)
        last_scroll_time = now
        logger.debug(
            f"[scroll] delta={delta:.1f}, scale={scale:.2f}, direction={direction}, interval={interval:.3f}s"
        )


def reset_scroll():
    global scroll_anchor_y, position_buffer, last_scroll_time
    scroll_anchor_y = None
    position_buffer.clear()
    last_scroll_time = 0
    logger.debug("[scroll] reset_scroll()")
