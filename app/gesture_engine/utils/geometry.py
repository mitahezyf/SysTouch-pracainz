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
