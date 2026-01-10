from __future__ import annotations

from typing import Callable, Dict, Optional

from app.gesture_engine.logger import logger

"""Hooki na zmiane gestu.

WAŻNE:
Poprzednia implementacja miala globalny `last_gesture_name` wspolny dla obu rak.
W multi-hand powodowalo to losowe "zwalnianie" clicka / inne skutki uboczne.

W tej wersji hooki sa *celowo minimalistyczne*:
- NIE dotykamy clicka (click jest obslugiwany wprost w workerze + click_action).
- Hooki zostawiamy glownie pod volume/scroll itp. (opcjonalnie).
"""

# volume: import stanu do resetu/przejsc
volume_state: Optional[Dict[str, object]]
try:
    from app.gesture_engine.gestures.volume_gesture import volume_state as _volume_state

    volume_state = _volume_state
except Exception:
    volume_state = None


_last_gesture_per_hand: Dict[int, Optional[str]] = {}
_gesture_start_hooks: Dict[str, Callable[[object, object], None]] = {}


def register_gesture_start_hook(
    gesture_name: str, func: Callable[[object, object], None]
) -> None:
    _gesture_start_hooks[gesture_name] = func


def handle_gesture_start_hook(
    gesture_name: Optional[str],
    landmarks: object,
    frame_shape: object,
    *,
    hand_id: int = 0,
) -> None:
    """Wywolywane NA ZMIANIE gestu dla danej reki."""
    prev = _last_gesture_per_hand.get(int(hand_id))

    if gesture_name != prev:
        logger.debug("[hook] hand=%s: %s -> %s", hand_id, prev, gesture_name)
        hook = _gesture_start_hooks.get(str(gesture_name)) if gesture_name else None
        if hook and landmarks is not None and frame_shape is not None:
            try:
                hook(landmarks, frame_shape)
            except Exception as e:
                logger.debug("[hook] hand=%s hook EXC: %r", hand_id, e)

        # volume overlay: jak gest volume zniknie na tej rece, czyscimy faze
        if prev == "volume" and gesture_name != "volume" and volume_state is not None:
            try:
                volume_state["phase"] = None
            except Exception:
                pass

        _last_gesture_per_hand[int(hand_id)] = gesture_name


def reset_hooks_state() -> None:
    """Reset hookow przed startem przetwarzania."""
    _last_gesture_per_hand.clear()

    # volume: reset do idle
    if volume_state is not None:
        try:
            volume_state["phase"] = "idle"
            volume_state["_extend_start"] = None
        except Exception:
            pass

    logger.debug("reset_hooks_state: cleared")
