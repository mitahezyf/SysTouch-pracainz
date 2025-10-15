# todo naprawic blad logiczny w release_click() - short tap nie zawsze wywoluje pyautogui.click()
import time

from app.gesture_engine.config import HOLD_THRESHOLD
from app.gesture_engine.logger import logger

# leniwy import pyautogui z no-op stubem, aby nie wysypywac sie w srodowiskach bez GUI
try:  # pragma: no cover - gałąź zależna od srodowiska CI
    import pyautogui as _pyautogui
except Exception:  # pragma: no cover

    class _PyAutoGuiStub:
        def click(self, *_, **__):
            pass

        def mouseDown(self, *_, **__):
            pass

        def mouseUp(self, *_, **__):
            pass

        def size(self):
            return (1920, 1080)

    logger.warning("pyautogui niedostepne – uzywam no-op stubu")
    pyautogui = _PyAutoGuiStub()
else:
    pyautogui = _pyautogui


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
    is_hold = duration >= HOLD_THRESHOLD
    logger.debug(
        f"[click] release_click(): duration={duration:.3f}, holding={click_state['holding']}, mouse_down={click_state['mouse_down']}, is_hold={is_hold}"
    )

    if is_hold:
        if click_state["mouse_down"]:
            pyautogui.mouseUp()
            logger.debug("[click] mouseUp() (hold)")
        else:
            # przekroczony prog, ale nie zdazyl nacisnac - bezpieczny fallback
            pyautogui.click()
            click_state["click_sent"] = True
            logger.debug("[click] click() (fallback po przekroczeniu progu)")
    else:
        pyautogui.click()
        click_state["click_sent"] = True
        logger.debug("[click] click() (short tap)")

    # resetuje stan
    click_state["start_time"] = None
    click_state["holding"] = False
    click_state["mouse_down"] = False
    # nie zeruje click_sent, bo test ma to zweryfikowac
    setattr(handle_click, "active", False)


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
    setattr(handle_click, "active", True)
    update_click_state(True)


setattr(handle_click, "active", False)
