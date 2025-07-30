import time

import pyautogui

from app.config import HOLD_THRESHOLD

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


def handle_active():
    if click_state["start_time"] is None:
        start_click()

    current_time = time.time()
    duration = current_time - click_state["start_time"]

    if duration >= HOLD_THRESHOLD:
        if not click_state["holding"]:
            click_state["holding"] = True
        if not click_state["mouse_down"]:
            pyautogui.mouseDown()
            click_state["mouse_down"] = True


def release_click():
    if click_state["start_time"] is None:
        print("[release_click] Ignored – no start_time")
        return

    duration = time.time() - click_state["start_time"]
    print(
        f"[release_click] Duration: {duration:.3f}, holding={click_state['holding']}, sent={click_state['click_sent']}"
    )

    if click_state["holding"]:
        if click_state["mouse_down"]:
            print("[release_click] mouseUp()")
            pyautogui.mouseUp()
    elif not click_state["click_sent"] and duration < HOLD_THRESHOLD:
        print("[release_click] click()")
        pyautogui.click()
        click_state["click_sent"] = True
    else:
        print("[release_click] Nothing to do")

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
