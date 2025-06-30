import pyautogui
import time
import threading
import queue
from collections import deque

# kolejka i kontrola watku
mouse_queue = queue.Queue()
mouse_thread_running = True

# throttling i wygladzanie
last_mouse_time = 0
mouse_history = deque(maxlen=8)

# funkcja w tle do moveto()
def mouse_worker():
    while mouse_thread_running:
        try:
            x, y = mouse_queue.get(timeout=0.05)
            pyautogui.moveTo(x, y)
        except queue.Empty:
            continue

# poczatek watku
mouse_thread = threading.Thread(target=mouse_worker, daemon=True)
mouse_thread.start()

# obsluga klikniecia
def handle_click(gesture, _):
    pyautogui.click()

# odsluga ruchu myszy
def handle_move_mouse(gesture, frame_shape):
    global last_mouse_time, mouse_history

    now = time.time()
    if now - last_mouse_time < 1 / 30:
        return
    last_mouse_time = now

    try:
        # czubek srodkowego palca jako kursor
        landmark = gesture.landmarks.landmark[12]
    except (AttributeError, IndexError):
        return

    frame_height, frame_width = frame_shape

    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)

    screen_w, screen_h = pyautogui.size()
    screen_x = int(x / frame_width * screen_w)
    screen_y = int(y / frame_height * screen_h)

    mouse_history.append((screen_x, screen_y))
    if len(mouse_history) < 2:
        return

    avg_x = int(sum(p[0] for p in mouse_history) / len(mouse_history))
    avg_y = int(sum(p[1] for p in mouse_history) / len(mouse_history))

    while not mouse_queue.empty():
        try:
            mouse_queue.get_nowait()
        except queue.Empty:
            break

    mouse_queue.put((avg_x, avg_y))


# mapa gestow
gesture_handlers = {
    "click": handle_click,
    "move_mouse": handle_move_mouse
}

# funkcja wykonujaca akcje
def execute_action(gesture, frame_shape=None):
    handler = gesture_handlers.get(gesture.name)
    if handler:
        handler(gesture, frame_shape)
