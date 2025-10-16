import threading
import time
from typing import Optional, Tuple

from app.gesture_engine.config import MOUSE_DEADZONE_PX, MOUSE_MOVING_SMOOTHING
from app.gesture_engine.logger import logger

# leniwy import pyautogui z no-op stubem
try:  # pragma: no cover
    import pyautogui as _pyautogui
except Exception:  # pragma: no cover

    class _PyAutoGuiStub:
        FAILSAFE: bool = False  # dodany atrybut dla zgodnosci z mypy

        def moveTo(self, *_args, **_kwargs) -> None:
            pass

        def size(self) -> tuple[int, int]:
            return (1920, 1080)

    logger.warning("pyautogui niedostepne - uzywam no-op stuba (move_mouse)")
    pyautogui = _PyAutoGuiStub()
else:
    pyautogui = _pyautogui
    # wylacza failsafe (ruch do (0,0) nie powinien zatrzymywac akcji)
    try:  # pragma: no cover
        pyautogui.FAILSAFE = False
    except Exception as e:
        logger.debug("[mouse] nie udalo sie wylaczyc pyautogui.FAILSAFE: %s", e)

from app.gesture_engine.utils.landmarks import FINGER_TIPS

# ostatnia zadana pozycja (cel) oraz pozycja wygladzona
latest_position: Optional[Tuple[int, int]] = None
_smoothed_position: Optional[Tuple[float, float]] = None

lock = threading.Lock()
running = True


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
            # brak nowego celu, odczekuje krotko
            time.sleep(0.005)
            continue

        tx, ty = target

        if _smoothed_position is None:
            # inicjalizuje pozycje wygladzona od razu celem
            _smoothed_position = (float(tx), float(ty))
        else:
            sx, sy = _smoothed_position
            dx = tx - sx
            dy = ty - sy

            # stosuje deadzone w pikselach
            if abs(dx) < MOUSE_DEADZONE_PX and abs(dy) < MOUSE_DEADZONE_PX:
                # ruch zbyt maly - pomija, ale nadal usypia aby nie obciazac CPU
                time.sleep(0.005)
                continue

            # wygladzanie eksponencjalne: nowa = alpha*stara + (1-alpha)*cel
            _smoothed_position = (
                alpha * sx + (1.0 - alpha) * tx,
                alpha * sy + (1.0 - alpha) * ty,
            )

        # wykonuje ruch do pozycji wygladzonej
        mx = int(_smoothed_position[0])
        my = int(_smoothed_position[1])
        try:
            pyautogui.moveTo(mx, my, duration=0)
        except Exception as e:  # pragma: no cover
            # nie przerywa watku w razie bledow pyautogui (np. failsafe)
            logger.debug("[mouse] moveTo wyjatek: %s", e)
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

    # przelicza wspolrzedne na ekran i ogranicza do zakresu
    screen_x = _clamp(int(index_tip.x * screen_w), 0, screen_w - 1)
    screen_y = _clamp(int(index_tip.y * screen_h), 0, screen_h - 1)

    with lock:
        latest_position = (screen_x, screen_y)


def stop_mouse_thread():
    global running
    running = False
    worker_thread.join()
