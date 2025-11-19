from __future__ import annotations

from app.gesture_engine.logger import logger


def quantize_pct(pct: float, step: int = 5) -> int:
    """Zaokragla procent do najblizszego kroku (domyslnie 5%)."""
    try:
        v = int(max(0.0, min(100.0, pct)))
        return int(round(v / float(step)) * step)
    except Exception as e:  # bardzo defensywnie
        logger.debug("quantize_pct error: %s", e)
        return 0
