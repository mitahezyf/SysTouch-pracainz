import math


# funkcja do liczenia odleglosci miedzy 2 punktami
def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def angle_between(p1, p2, p3):
    # zwraca kat w stopniach miedzy wektorami p2 -> p1 i p2 -> p3 - srodek w p2
    v1 = (p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)
    v2 = (p3.x - p2.x, p3.y - p2.y, p3.z - p2.z)

    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    return math.degrees(math.acos(cos_theta))


def calculate_hand_roll(landmarks) -> float:
    """oblicza kat obrotu (roll) dloni w stopniach.

    metoda: mierzy kat wektora wrist->middle_mcp wzgledem osi poziomej
    obrot w prawo (clockwise) -> dodatni, w lewo (counter-clockwise) -> ujemny

    Args:
        landmarks: lista landmarkow MediaPipe (21 punktow)

    Returns:
        kat obrotu w stopniach [-180, 180]
    """
    from app.gesture_engine.utils.landmarks import FINGER_MCPS, WRIST

    wrist = landmarks[WRIST]
    middle_mcp = landmarks[FINGER_MCPS["middle"]]

    # wektor od wrist do middle_mcp
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y

    # kat w radianach, potem konwersja na stopnie
    # atan2 zwraca kat wzgledem osi X (poziomej)
    # dodatni kat = reka obrocona w prawo (w dol), ujemny = w lewo (w gore)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return angle_deg
