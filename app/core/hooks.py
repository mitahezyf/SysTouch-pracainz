# todo usunac powielone wywolanie update_click_state(False) przy zmianie gestu z click
from app.actions.click_action import handle_click
from app.actions.click_action import update_click_state
from app.logger import logger

last_gesture_name = None
gesture_start_hooks = {}


def register_gesture_start_hook(gesture_name, func):
    gesture_start_hooks[gesture_name] = func


def handle_gesture_start_hook(gesture_name, landmarks, frame_shape):
    global last_gesture_name

    if handle_click.active and gesture_name not in ("click", "click-hold"):
        update_click_state(False)
        handle_click.active = False
        logger.debug("[hook] click released (gest zmienił się)")

    if gesture_name != last_gesture_name:
        logger.debug(f"[hook] zmiana gestu: {last_gesture_name} -> {gesture_name}")
        hook = gesture_start_hooks.get(gesture_name)
        if hook:
            logger.debug(f"[hook] wywołanie hooka dla: {gesture_name}")
            hook(landmarks, frame_shape)

    if gesture_name is None and last_gesture_name in ("click", "click-hold"):
        update_click_state(False)
        handle_click.active = False
        logger.debug("[hook] click released (gest się zakończył)")

    last_gesture_name = gesture_name

    def test_scroll_hook(landmarks, frame_shape):
        logger.debug("[hook] scroll hook wywołany")

    register_gesture_start_hook("scroll", test_scroll_hook)
