import ctypes
import threading
import time
from typing import Optional, Tuple

from app.gesture_engine.config import MOUSE_DEADZONE_PX, MOUSE_MOVING_SMOOTHING
from app.gesture_engine.logger import logger

# leniwy import pyautogui z no-op stubem (zapobiega awarii gdy brak zaleznosci)
try:  # pragma: no cover
    import pyautogui as _pyautogui
except Exception:  # pragma: no cover

    class _PyAutoGuiStub:
        FAILSAFE: bool = False  # udostepnia atrybut dla zgodnosci z kodem

        def moveTo(self, *_args, **_kwargs) -> None:
            pass

        def size(self) -> tuple[int, int]:
            return (1920, 1080)

    logger.warning("pyautogui niedostepne - uzywam no-op stuba (move_mouse)")
    pyautogui = _PyAutoGuiStub()
else:
    pyautogui = _pyautogui
    # wylacza failsafe (ruch do (0,0) nie zatrzymuje akcji)
    try:  # pragma: no cover
        pyautogui.FAILSAFE = False
    except Exception as e:
        logger.debug("[mouse] nie udalo sie wylaczyc pyautogui.FAILSAFE: %s", e)

from app.gesture_engine.utils.landmarks import FINGER_TIPS

# --- Windows SendInput dla prawidlowego drag ---
# pyautogui.moveTo moze uzywac SetCursorPos ktore nie generuje WM_MOUSEMOVE
# uzywamy SendInput z MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE dla prawidlowego drag w Paint

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT),
    ]


INPUT_MOUSE = 0


def _send_mouse_move(x: int, y: int) -> None:
    """Wysyla zdarzenie ruchu myszy przez SendInput (generuje WM_MOUSEMOVE)."""
    try:
        # Przelicz na wspolrzedne absolutne (0-65535)
        screen_w, screen_h = pyautogui.size()
        abs_x = int(x * 65535 / screen_w)
        abs_y = int(y * 65535 / screen_h)

        extra = ctypes.c_ulong(0)
        mi = MOUSEINPUT(
            dx=abs_x,
            dy=abs_y,
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
            time=0,
            dwExtraInfo=ctypes.pointer(extra),
        )
        inp = INPUT(type=INPUT_MOUSE, mi=mi)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    except Exception as e:
        logger.debug("[mouse] SendInput wyjatek: %s", e)
        # fallback do pyautogui
        try:
            pyautogui.moveTo(x, y, duration=0)
        except Exception:
            pass


# ostatnia zadana pozycja kursora oraz pozycja wygladzona
latest_position: Optional[Tuple[int, int]] = None
_smoothed_position: Optional[Tuple[float, float]] = None

lock = threading.Lock()
running = True


def get_move_debug_state() -> dict[str, object]:
    """Snapshot stanu ruchu (do logow)."""
    with lock:
        return {
            "latest_position": latest_position,
            "smoothed_position": _smoothed_position,
        }


def _clamp(val: int, low: int, high: int) -> int:
    # ogranicza wartosc do przedzialu
    return high if val > high else (low if val < low else val)


def move_worker():
    global _smoothed_position
    logger.info("[mouse] Watek poruszania myszka wystartowal")
    first_move_logged = False

    # alpha okresla, jak silnie utrzymywac poprzednia pozycje
    alpha = max(0.0, min(1.0, float(MOUSE_MOVING_SMOOTHING)))

    while running:
        with lock:
            target = latest_position

        if target is None:
            # brak celu – odczekuje krotko i kontynuuje
            time.sleep(0.005)
            continue

        tx, ty = target

        if _smoothed_position is None:
            # inicjalizuje pozycje wygladzona jako aktualny cel
            _smoothed_position = (float(tx), float(ty))
        else:
            sx, sy = _smoothed_position
            dx = tx - sx
            dy = ty - sy

            # stosuje strefe martwa (deadzone) w pikselach
            if abs(dx) < MOUSE_DEADZONE_PX and abs(dy) < MOUSE_DEADZONE_PX:
                # ruch zbyt maly – pomija
                time.sleep(0.005)
                continue

            # wygladza eksponencjalnie: nowa = alpha*stara + (1-alpha)*cel
            _smoothed_position = (
                alpha * sx + (1.0 - alpha) * tx,
                alpha * sy + (1.0 - alpha) * ty,
            )

        # wykonuje ruch do pozycji wygladzonej - uzywamy SendInput dla prawidlowego drag
        mx = int(_smoothed_position[0])
        my = int(_smoothed_position[1])
        try:
            _send_mouse_move(mx, my)
        except Exception as e:  # pragma: no cover
            logger.debug("[mouse] move wyjatek: %s", e)
        else:
            if not first_move_logged:
                logger.debug(f"[mouse] pierwszy ruch kursora: ({mx}, {my})")
                first_move_logged = True

        time.sleep(0.005)

    logger.info("[mouse] Watek myszki zakonczony")


worker_thread = threading.Thread(target=move_worker, daemon=True)
worker_thread.start()


def handle_move_mouse(landmarks, frame_shape):
    global latest_position
    index_tip = landmarks[FINGER_TIPS["index"]]
    screen_w, screen_h = pyautogui.size()

    # przelicza wspolrzedne czubka palca na koordynaty ekranu w zakresie (mirror X)
    screen_x = _clamp(int((1.0 - index_tip.x) * screen_w), 0, screen_w - 1)
    screen_y = _clamp(int(index_tip.y * screen_h), 0, screen_h - 1)

    with lock:
        latest_position = (screen_x, screen_y)


def stop_mouse_thread():
    global running
    running = False
    worker_thread.join()
