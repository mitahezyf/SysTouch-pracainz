# narzedzia cech dloni: normalizacja, curl, pinch, projekcje w ukladzie dloni
# lekki szkic, bezpieczny do importu
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Landmark = Tuple[float, float, float]
Frame = Sequence[Landmark]


def normalize_landmarks(frame: Frame) -> Frame:
    # centrowanie do wrist i skala rozmiarem dloni (wrist->middle_mcp)
    # gdy dane niepelne zwraca bez zmian
    # indeksy jak w mediapipe hands
    if not frame or len(frame) < 10:
        return frame
    WRIST = 0
    MIDDLE_MCP = 9
    wx, wy, wz = frame[WRIST]
    mx, my, mz = frame[MIDDLE_MCP]
    scale = math.dist((wx, wy), (mx, my)) or 1.0
    norm: List[Landmark] = []
    for x, y, z in frame:
        norm.append(((x - wx) / scale, (y - wy) / scale, (z - wz) / scale))
    return norm


def finger_direction(
    frame: Frame, mcp_idx: int, tip_idx: int
) -> Tuple[float, float, float]:
    # wektor jednostkowy od mcp do tip; gdy dlugosc 0 zwraca 0
    mx, my, mz = frame[mcp_idx]
    tx, ty, tz = frame[tip_idx]
    vx, vy, vz = (tx - mx, ty - my, tz - mz)
    norm = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
    return (vx / norm, vy / norm, vz / norm)


def project_along(
    vec: Tuple[float, float, float], dir_vec: Tuple[float, float, float]
) -> float:
    # rzut skalarny vec na dir_vec (dir_vec znormalizowany)
    return vec[0] * dir_vec[0] + vec[1] * dir_vec[1] + vec[2] * dir_vec[2]


def curl_score(
    frame: Frame, mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int
) -> float:
    # przyblizony curl 0..1; 0 prosty, 1 zgiecie; heurystyka
    # wektory odcinkow
    mx, my, mz = frame[mcp_idx]
    px, py, pz = frame[pip_idx]
    dx, dy, dz = frame[dip_idx]
    tx, ty, tz = frame[tip_idx]
    v1 = (px - mx, py - my, pz - mz)
    v2 = (dx - px, dy - py, dz - pz)
    v3 = (tx - dx, ty - dy, tz - dz)

    def _angle(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        ax, ay, az = a
        bx, by, bz = b
        na = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        nb = math.sqrt(bx * bx + by * by + bz * bz) or 1.0
        cosv = max(-1.0, min(1.0, (ax * bx + ay * by + az * bz) / (na * nb)))
        return math.acos(cosv)

    a_pip = _angle(v1, v2)
    a_dip = _angle(v2, v3)

    # mapowanie katow do 0..1
    # prosty ~ male katy; zgiecie ~ duze katy
    def _map(a: float, lo: float, hi: float) -> float:
        return max(0.0, min(1.0, (a - lo) / max(1e-6, (hi - lo))))

    s_pip = _map(a_pip, math.radians(15), math.radians(90))
    s_dip = _map(a_dip, math.radians(10), math.radians(80))
    return max(0.0, min(1.0, 0.6 * s_pip + 0.4 * s_dip))


def pinch_distance_norm(frame: Frame, thumb_tip_idx: int, index_tip_idx: int) -> float:
    # odleglosc kciuk-wskaz znormalizowana; zaklada ramke po normalizacji
    tx, ty, tz = frame[thumb_tip_idx]
    ix, iy, iz = frame[index_tip_idx]
    return math.dist((tx, ty), (ix, iy))
