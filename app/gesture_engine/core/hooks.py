# todo usunac powielone wywolanie update_click_state(False) przy zmianie gestu z click
from app.gesture_engine.actions.click_action import handle_click
from app.gesture_engine.actions.click_action import update_click_state
from app.gesture_engine.logger import logger

last_gesture_name = None
gesture_start_hooks = {}


def register_gesture_start_hook(gesture_name, func):
    gesture_start_hooks[gesture_name] = func


def handle_gesture_start_hook(gesture_name, landmarks, frame_shape):
    global last_gesture_name

    # zwalnia click przy zmianie na inny gest (ale nie None - to obsluzymy nizej)
    if (
        handle_click.active
        and gesture_name is not None
        and gesture_name not in ("click", "click-hold")
    ):
        update_click_state(False)
        handle_click.active = False
        logger.debug("[hook] click released (gest zmienił się)")

    if gesture_name != last_gesture_name:
        logger.debug(f"[hook] zmiana gestu: {last_gesture_name} -> {gesture_name}")
        hook = gesture_start_hooks.get(gesture_name)
        if hook:
            logger.debug(f"[hook] wywołanie hooka dla: {gesture_name}")
            if landmarks is not None and frame_shape is not None:
                hook(landmarks, frame_shape)
            else:
                logger.debug("[hook] pomijam wywołanie hooka (brak danych)")

    # specjalny przypadek: koniec gestu (None) po clicku
    if gesture_name is None and last_gesture_name in ("click", "click-hold"):
        update_click_state(False)
        handle_click.active = False
        logger.debug("[hook] click released (gest się zakończył)")

    last_gesture_name = gesture_name

    def test_scroll_hook(landmarks, frame_shape):
        logger.debug("[hook] scroll hook wywołany")

    if "scroll" not in gesture_start_hooks:
        register_gesture_start_hook("scroll", test_scroll_hook)
