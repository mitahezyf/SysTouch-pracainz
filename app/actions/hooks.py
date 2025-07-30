from app.actions.click_action import handle_click
from app.actions.click_action import update_click_state

# ostatni wykryty gest
last_gesture_name = None

# funkcje startowe dla danego gestu
gesture_start_hooks = {}


def register_gesture_start_hook(gesture_name, func):
    gesture_start_hooks[gesture_name] = func


def handle_gesture_start_hook(gesture_name, landmarks, frame_shape):
    global last_gesture_name

    # jeśli kończy się gest click lub click-hold – puść klik
    if handle_click.active and gesture_name not in ("click", "click-hold"):
        update_click_state(False)
        handle_click.active = False

    # jeśli gest się zmienił – wywołaj hook dla nowego
    if gesture_name != last_gesture_name:
        hook = gesture_start_hooks.get(gesture_name)
        if hook:
            hook(landmarks, frame_shape)

    if gesture_name is None and last_gesture_name in ("click", "click-hold"):
        update_click_state(False)
        handle_click.active = False

    last_gesture_name = gesture_name
