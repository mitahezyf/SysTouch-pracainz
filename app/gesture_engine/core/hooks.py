# todo usunac powielone wywolanie update_click_state(False) przy zmianie gestu z click

from typing import Dict, Optional, cast

from app.gesture_engine.actions.click_action import handle_click, update_click_state
from app.gesture_engine.logger import logger

# deklaracja typu na poziomie modulu
volume_state: Optional[Dict[str, object]] = None

# volume: import stanu do resetu/przejsc
try:
    from app.gesture_engine.gestures.volume_gesture import volume_state as _volume_state

    volume_state = cast(Dict[str, object], _volume_state)
except Exception:
    # gdy modul gestu nie jest dostepny (np. w minimalnym trybie testowym)
    volume_state = None

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

    # reset volume przy wyjsciu z gestu
    if last_gesture_name == "volume" and gesture_name != "volume":
        if volume_state is not None:
            volume_state["phase"] = "idle"
            volume_state["_extend_start"] = None
            logger.debug("[hook] volume reset (wyjscie z gestu)")

    if gesture_name != last_gesture_name:
        logger.debug(
            "[hook] zmiana gestu: {} -> {}".format(last_gesture_name, gesture_name)
        )
        hook = gesture_start_hooks.get(gesture_name)
        if hook:
            logger.debug("[hook] wywołanie hooka dla: {}".format(gesture_name))
            if landmarks is not None and frame_shape is not None:
                hook(landmarks, frame_shape)
            else:
                logger.debug("[hook] pomijam wywolanie hooka (brak danych)")

    # specjalny przypadek: koniec gestu (None) po clicku
    if gesture_name is None and last_gesture_name in ("click", "click-hold"):
        update_click_state(False)
        handle_click.active = False
        logger.debug("[hook] click released (gest się zakończył)")

    last_gesture_name = gesture_name

    def test_scroll_hook(landmarks, frame_shape):
        logger.debug("[hook] scroll hook wywolany")

    if "scroll" not in gesture_start_hooks:
        register_gesture_start_hook("scroll", test_scroll_hook)

    # start hook dla volume: ustawia faze na adjusting gdy gest aktywowany
    def volume_start_hook(_landmarks, _frame_shape):
        if volume_state is not None:
            volume_state["phase"] = "adjusting"
            volume_state["_extend_start"] = None
            logger.debug("[hook] volume start -> adjusting")

    if "volume" not in gesture_start_hooks:
        register_gesture_start_hook("volume", volume_start_hook)
