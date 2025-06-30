import math

#funkcja do liczenia odleglosci miedzy 2 punktami
def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)