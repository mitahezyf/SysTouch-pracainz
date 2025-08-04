# todo naprawic blad logiczny w release_click() – short tap nie zawsze wywołuje pyautogui.click()
import time

import pyautogui

from app.gesture_engine.config import HOLD_THRESHOLD
from app.gesture_engine.logger import logger


click_state = {
    "start_time": None,
    "holding": False,
    "mouse_down": False,
    "click_sent": False,
    "was_active": False,
}


def start_click():
    click_state["start_time"] = time.time()
    click_state["holding"] = False
    click_state["mouse_down"] = False
    click_state["click_sent"] = False
    logger.debug("[click] start_click()")


def handle_active():
    if click_state["start_time"] is None:
        start_click()

    current_time = time.time()
    duration = current_time - click_state["start_time"]

    if duration >= HOLD_THRESHOLD:
        if not click_state["holding"]:
            click_state["holding"] = True
            logger.debug("[click] HOLD aktywowany")

        if not click_state["mouse_down"]:
            pyautogui.mouseDown()
            click_state["mouse_down"] = True
            logger.debug("[click] mouseDown()")


def release_click():
    if click_state["start_time"] is None:
        logger.debug("[click] Ignoruję release – brak start_time")
        return

    duration = time.time() - click_state["start_time"]
    logger.debug(
        f"[click] release_click(): duration={duration:.3f}, holding={click_state['holding']}, sent={click_state['click_sent']}"
    )

    if click_state["holding"]:
        if click_state["mouse_down"]:
            pyautogui.mouseUp()
            logger.debug("[click] mouseUp() (hold)")
    elif not click_state["click_sent"] and duration < HOLD_THRESHOLD:
        pyautogui.click()
        click_state["click_sent"] = True
        logger.debug("[click] click() (short tap)")
    else:
        logger.debug("[click] release_click() – brak akcji")

    click_state["start_time"] = None
    click_state["holding"] = False
    click_state["mouse_down"] = False
    click_state["click_sent"] = False
    handle_click.active = False


def update_click_state(active: bool):
    if active:
        if not click_state["was_active"]:
            start_click()
        handle_active()
        click_state["was_active"] = True
    else:
        if click_state["was_active"]:
            release_click()
        click_state["was_active"] = False


def get_click_state_name():
    if click_state["holding"]:
        return "click-hold"
    elif click_state["start_time"] is not None and not click_state["click_sent"]:
        return "click"
    else:
        return None


def handle_click(_landmarks, _frame_shape):
    handle_click.active = True
    update_click_state(True)


handle_click.active = False
