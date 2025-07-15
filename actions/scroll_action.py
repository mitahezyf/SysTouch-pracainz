import threading
import pyautogui
import time
from collections import deque

from actions.move_mouse_action import position_buffer
from utils.landmarks import FINGER_TIPS, WRIST
from config import SCROLL_SENSITIVITY

#bufor wygladzania
position_buffer = deque(maxlen=5)
lock = threading.Lock()

#punkt wyjscia scroll
scroll_anchor_y = None
scrolling = False

def handle_scroll(landmarks, frame_shape):
    global scroll_anchor_y, scrolling

    wrist = landmarks[WRIST]
    screen_h = pyautogui.size()[1]
    current_y = int(wrist.y * screen_h)

    with lock:
        position_buffer.append(current_y)

        if len(position_buffer) < 3:
            return

        avg_y = int(sum(position_buffer) / len(position_buffer))

        if not scrolling:
            scroll_anchor_y = avg_y
            scrolling = True
        else:
            delta = avg_y - scroll_anchor_y


            if abs(delta) > SCROLL_SENSITIVITY:
                # skaluje intensywnosc scrolla
                scroll_amount = int(delta / 10)

                # ograniczenie zakresu
                scroll_amount = max(min(scroll_amount, 10), -10)

                pyautogui.scroll(-scroll_amount)

def reset_scroll():
    global scroll_anchor_y, scrolling
    with lock:
        scrolling = False
        scroll_anchor_y = None
        position_buffer.clear()

