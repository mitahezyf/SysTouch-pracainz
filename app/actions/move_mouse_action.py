import threading
import time
from collections import deque

import pyautogui

from app.logger import logger
from app.utils.landmarks import FINGER_TIPS

# bufor wygładzania
position_buffer = deque(maxlen=5)
latest_position = None
lock = threading.Lock()
running = True


def move_worker():
    global latest_position
    logger.info("[mouse] Wątek poruszania myszą wystartował")
    first_move_logged = False

    while running:
        with lock:
            if latest_position:
                position_buffer.append(latest_position)

        if position_buffer:
            avg_x = int(sum(p[0] for p in position_buffer) / len(position_buffer))
            avg_y = int(sum(p[1] for p in position_buffer) / len(position_buffer))
            pyautogui.moveTo(avg_x, avg_y, duration=0)

            if not first_move_logged:
                logger.debug(f"[mouse] pierwszy ruch kursora: ({avg_x}, {avg_y})")
                first_move_logged = True

        time.sleep(0.005)

    logger.info("[mouse] Wątek myszki zakończony")


worker_thread = threading.Thread(target=move_worker, daemon=True)
worker_thread.start()


def handle_move_mouse(landmarks, frame_shape):
    global latest_position
    index_tip = landmarks[FINGER_TIPS["index"]]
    screen_w, screen_h = pyautogui.size()

    screen_x = int(index_tip.x * screen_w)
    screen_y = int(index_tip.y * screen_h)

    with lock:
        latest_position = (screen_x, screen_y)


def stop_mouse_thread():
    global running
    running = False
    worker_thread.join()
