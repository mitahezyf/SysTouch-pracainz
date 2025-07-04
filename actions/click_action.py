import pyautogui
import time
from config import HOLD_THRESHOLD

#stan klikniecia
click_state = {
    "active": False,
    "start_time": None,
    "holding": False
}

#obsuga zmiany click/hold jezeli click trwa dluzej niz 1s wtedy zmiania sie na hold brak spamu kliknieciami
def handle_click(_landmarks, _frame_shape):
    global click_state

    current_time = time.time()

    if not click_state["active"]:
        click_state["active"] = True
        click_state["start_time"] = current_time
        click_state["holding"] = False
        pyautogui.mouseDown()
    else:
        duration = current_time - click_state["start_time"]
        if duration >= HOLD_THRESHOLD and click_state["holding"]:
            click_state["holding"] = True

#zmiana na puszcenie klikniecia
def release_click():
    global click_state

    if click_state["active"]:
        pyautogui.mouseUp()
        click_state = {
            "active": False,
            "start_time": None,
            "holding": False
        }
def get_click_state_name():
    if click_state["holding"]:
        return "click-hold"
    elif click_state["active"]:
        return "click"
    else:
        return None