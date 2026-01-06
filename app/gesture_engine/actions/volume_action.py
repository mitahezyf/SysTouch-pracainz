import sys
from collections import deque
from statistics import median
from time import monotonic
from typing import cast

from app.gesture_engine.gestures.volume_gesture import volume_state
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.geometry import calculate_hand_roll

try:  # importuje kontroler glosnosci dla Windows
    from app.gesture_engine.utils.pycaw_controller import (
        poke_volume_osd,
        set_system_volume,
    )
except Exception:  # pragma: no cover
    poke_volume_osd = None  # type: ignore[assignment]
    set_system_volume = None  # type: ignore[assignment]


def _get_smoothed_roll(landmarks) -> float:
    """oblicza roll dloni z smoothingiem (mediana z ostatnich 5 wartosci)"""
    roll = calculate_hand_roll(landmarks)

    # inicjalizuj bufor jesli nie istnieje
    if "roll_buffer" not in volume_state:
        volume_state["roll_buffer"] = deque(maxlen=5)

    buffer = cast(deque[float], volume_state["roll_buffer"])
    buffer.append(roll)

    # jesli bufor niepelny, zwroc surowa wartosc
    if len(buffer) < 3:
        return roll

    # zwroc mediane (odporna na outliers)
    return median(buffer)


def _normalize_delta_deg(delta_deg: float, apply_dead_zone: bool = True) -> float:
    # normalizuje kat do przedzialu [-180, 180] i opcjonalnie dodaje dead zone
    while delta_deg > 180.0:
        delta_deg -= 360.0
    while delta_deg < -180.0:
        delta_deg += 360.0

    # dead zone: ±5 stopni od baseline nie zmienia wartosci
    # tylko dla malych odchylen (nie redukujemy duzych katow)
    if apply_dead_zone and abs(delta_deg) < 5.0:
        return 0.0

    return delta_deg


def _map_angle_to_percent(delta_deg: float, range_deg: float, invert: bool) -> int:
    # mapuje odchylenie katowe do 0..100, gdzie -range/2 -> 0, +range/2 -> 100
    if invert:
        delta_deg = -delta_deg
    half = max(1.0, range_deg / 2.0)
    raw = 50.0 + (delta_deg / half) * 50.0
    pct = int(max(0.0, min(100.0, raw)))
    # kwantyzacja 1% (bylo 5%, teraz lepsza precyzja)
    return pct


def _maybe_apply_system_volume(pct: int) -> None:
    # opcjonalnie ustawia glosnosc systemu (Windows) z rate limit
    try:
        if sys.platform != "win32":  # pragma: no cover
            return
        # domyslnie True - jesli pycaw dostepny, stosuj glosnosc
        # _report_capabilities() moze to wylaczyc jesli brak pycaw
        if not bool(volume_state.get("apply_system", True)):
            return
        # rate limit w ms (domyslnie 50 ms)
        rate_ms = 50.0
        rate_raw = volume_state.get("apply_rate_ms")
        if isinstance(rate_raw, (int, float, str)):
            try:
                rate_ms = float(rate_raw)
            except Exception:
                rate_ms = 50.0
        now = monotonic()
        last_ts = 0.0
        last_raw = volume_state.get("_last_apply_ts")
        if isinstance(last_raw, (int, float, str)):
            try:
                last_ts = float(last_raw)
            except Exception:
                last_ts = 0.0
        if (now - last_ts) * 1000.0 < rate_ms:
            return
        if set_system_volume is not None:
            set_system_volume(int(pct))
        if poke_volume_osd is not None:
            try:
                poke_volume_osd()
            except Exception:
                pass
        volume_state["_last_apply_ts"] = now
    except Exception as e:  # pragma: no cover
        logger.debug("[volume_action] apply_system failed: %s", e)


def handle_volume(landmarks, frame_shape):
    """ustawia glosnosc na bazie kata obrotu dloni (roll).

    Prosty model bez faz - dziala jak move_mouse:
    wykryto gest -> oblicz kat -> ustaw glosnosc NATYCHMIAST.
    """
    try:
        # oblicz kat obrotu dloni
        roll = _get_smoothed_roll(landmarks)

        # inicjalizacja baseline przy pierwszym wywolaniu
        if volume_state.get("hand_roll_baseline_deg") is None:
            volume_state["hand_roll_baseline_deg"] = roll
            volume_state["hand_roll_deg"] = roll
            volume_state["hand_roll_delta_deg"] = 0.0
            volume_state["pct"] = 50
            logger.info("[volume] INIT baseline=%.2f -> pct=50", roll)
            _maybe_apply_system_volume(50)  # ustaw na srodku od razu
            return

        # oblicz delta od baseline
        base_deg = float(volume_state.get("hand_roll_baseline_deg") or 0.0)
        delta = _normalize_delta_deg(roll - base_deg)

        # jednolite miejsce inwersji znaku (roll_invert domyslnie True gdy mirror podgladu)
        if bool(volume_state.get("roll_invert", True)):
            delta = -delta

        # mapuj na procent
        pct = _map_angle_to_percent(
            delta,
            float(volume_state.get("roll_range_deg") or 90.0),
            False,
        )

        # zapisz stan
        volume_state["hand_roll_deg"] = roll
        volume_state["hand_roll_delta_deg"] = delta
        volume_state["pct"] = pct

        logger.info(
            "[volume] roll=%.2f base=%.2f delta=%.2f -> pct=%d",
            roll,
            base_deg,
            delta,
            pct,
        )

        # USTAW GLOSNOSC NATYCHMIAST (nie czekaj na zadne fazy)
        _maybe_apply_system_volume(pct)

    except Exception as e:
        logger.error("[volume] error: %s", e)


def finalize_volume_if_stable() -> bool:
    """No-op: nie dokonuje zadnej finalizacji.

    Zwraca zawsze False.
    """
    return False
