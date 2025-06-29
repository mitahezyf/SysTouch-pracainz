import pyautogui


def handle_click(gesture, _):
    pyautogui.click()


def handle_move_mouse(gesture, frame_shape):
    if gesture.landmarks and frame_shape:
        index_tip = gesture.landmarks.landmark[8]
        screen_w, screen_h = pyautogui.size()
        x = int(index_tip.x * screen_w)
        y = int(index_tip.y * screen_h)
        pyautogui.moveTo(x, y)


gesture_handlers = {
    "click": handle_click,
    "move_mouse": handle_move_mouse
}

def execute_action(gesture, frame_shape=None):
    handler = gesture_handlers.get(gesture.name)
    if handler:
        handler(gesture, frame_shape)
