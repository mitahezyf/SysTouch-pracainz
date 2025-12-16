import sys
from math import atan2, degrees
from time import monotonic

from app.gesture_engine.gestures.volume_gesture import volume_state
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.landmarks import FINGER_MCPS

try:  # importuje kontroler glosnosci dla Windows
    from app.gesture_engine.utils.pycaw_controller import (
        poke_volume_osd,
        set_system_volume,
    )
except Exception:  # pragma: no cover
    poke_volume_osd = None  # type: ignore[assignment]
    set_system_volume = None  # type: ignore[assignment]


def _angle_index_pinky(landmarks) -> float:
    # oblicza kat (radiany) wektora MCP index -> MCP pinky w plaszczyznie obrazu
    a = landmarks[FINGER_MCPS["index"]]
    b = landmarks[FINGER_MCPS["pinky"]]
    # wektor od index do pinky
    vx = b.x - a.x
    vy = b.y - a.y
    return atan2(vy, vx)


def _normalize_delta_deg(delta_deg: float) -> float:
    # normalizuje kat do przedzialu [-180, 180]
    while delta_deg > 180.0:
        delta_deg -= 360.0
    while delta_deg < -180.0:
        delta_deg += 360.0
    return delta_deg


def _map_angle_to_percent(delta_deg: float, range_deg: float, invert: bool) -> int:
    # mapuje odchylenie katowe do 0..100, gdzie -range/2 -> 0, +range/2 -> 100
    if invert:
        delta_deg = -delta_deg
    half = max(1.0, range_deg / 2.0)
    raw = 50.0 + (delta_deg / half) * 50.0
    pct = int(max(0.0, min(100.0, raw)))
    # kwantyzacja 5%
    pct = int(round(pct / 5.0) * 5)
    return pct


def _maybe_apply_system_volume(pct: int) -> None:
    # opcjonalnie ustawia glosnosc systemu (Windows) z rate limit
    try:
        if sys.platform != "win32":  # pragma: no cover
            return
        if not bool(volume_state.get("apply_system", False)):
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
    """Odczytuje procent glosnosci na bazie odchylenia kata MCP index->pinky.

    Wymagane: volume_state['phase'] == 'adjusting' (ustawiane przez hook).
    Testy zakladaja brak modyfikacji gdy phase != 'adjusting'.
    """
    try:
        if volume_state.get("phase") != "adjusting":
            return
        # inicjalizacja baseline
        if volume_state.get("knob_baseline_angle_deg") is None:
            ang0 = _angle_index_pinky(landmarks)
            ang0_deg = degrees(ang0)
            volume_state["knob_baseline_angle_deg"] = ang0_deg
            volume_state["angle_deg"] = ang0_deg
            volume_state["angle_delta_deg"] = 0.0
            volume_state["pct"] = 50
            logger.debug("[volume_action] baseline set %.2f deg -> pct=50" % ang0_deg)
            return
        # kolejne wywolania: liczy delta
        ang = _angle_index_pinky(landmarks)
        ang_deg = degrees(ang)
        base_deg = float(volume_state.get("knob_baseline_angle_deg") or 0.0)
        delta = _normalize_delta_deg(ang_deg - base_deg)
        volume_state["angle_deg"] = ang_deg
        volume_state["angle_delta_deg"] = delta
        pct = _map_angle_to_percent(
            delta,
            float(volume_state.get("knob_range_deg") or 180.0),
            bool(volume_state.get("knob_invert") or False),
        )
        volume_state["pct"] = pct
        logger.debug(
            "[volume_action] ang=%.2f base=%.2f delta=%.2f pct=%d"
            % (ang_deg, base_deg, delta, pct)
        )
        _maybe_apply_system_volume(pct)
    except Exception as e:
        logger.debug("[volume_action] error: %s" % e)


def finalize_volume_if_stable() -> bool:
    """No-op: nie dokonuje zadnej finalizacji.

    Zwraca zawsze False.
    """
    return False
