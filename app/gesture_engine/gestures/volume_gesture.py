from app.gesture_engine.utils.geometry import distance
from app.gesture_engine.utils.landmarks import (
    FINGER_MCPS,
    FINGER_PIPS,
    FINGER_TIPS,
    WRIST,
)

# minimalna definicja gestu volume: tylko stale i stan dla GUI/akcji

# prog zacisku (ok. 50% dloni); uzywany przy liczeniu pct
PINCH_RATIO: float = 0.5

# globalny, minimalny stan gestu wykorzystywany przez GUI i testy
volume_state: dict[str, object] = {
    # podstawowe pola kontroli
    "phase": "idle",
    "_extend_start": None,
    # wartosci do obliczen i wizualizacji
    "pct": None,  # ostatnio policzony procent (0..100)
    "pinch_th": None,  # nadpisany prog pincha (opcjonalnie)
    "ref_max": None,  # nadpisany rozstaw dla 100% (opcjonalnie)
}


def detect_volume_gesture(landmarks):
    """Wykrywa gest triggera 'volume' (bez JSON).

    Kryteria (heurystyka):
    - pinch: odleglosc kciuk-wskazujacy < PINCH_RATIO * hand_size
    - wskazujacy i srodkowy proste (tip.y < pip.y)
    - serdeczny i maly zgięte (tip.y - mcp.y > 0)
    Zwraca ("volume", 1.0) albo None.
    """
    try:
        # wielkosc dloni (od wrist do pinky_mcp)
        hand_size = distance(landmarks[WRIST], landmarks[FINGER_MCPS["pinky"]])
        if hand_size <= 0:
            return None

        # pinch: kciuk + serdeczny (ring)
        thumb_tip = landmarks[FINGER_TIPS["thumb"]]
        ring_tip = landmarks[FINGER_TIPS["ring"]]
        pinch_dist_ring = distance(thumb_tip, ring_tip)
        pinch_th = float(volume_state.get("pinch_th") or (hand_size * PINCH_RATIO))
        pinch_ok = pinch_dist_ring < pinch_th

        # anty-konflikt z kliknieciem: kciuk+wskaz nie moga byc blisko
        index_tip = landmarks[FINGER_TIPS["index"]]
        pinch_dist_index = distance(thumb_tip, index_tip)
        not_click = pinch_dist_index > (pinch_th * 1.1)

        index_straight = (
            landmarks[FINGER_TIPS["index"]].y < landmarks[FINGER_PIPS["index"]].y
        )
        middle_straight = (
            landmarks[FINGER_TIPS["middle"]].y < landmarks[FINGER_PIPS["middle"]].y
        )

        if pinch_ok and not_click and index_straight and middle_straight:
            return "volume", 1.0
    except Exception:
        # defensywnie pomija bledy danych
        return None

    return None
