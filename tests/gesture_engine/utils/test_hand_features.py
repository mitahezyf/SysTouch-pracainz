from typing import List, Tuple

import pytest

from app.gesture_engine.utils.hand_features import (
    Frame,
    curl_score,
    finger_direction,
    normalize_landmarks,
    pinch_distance_norm,
)


def make_frame(points: List[Tuple[float, float, float]]) -> Frame:
    return points


def test_normalize_landmarks_short_input_returns_same():
    # zwraca oryginal dla zbyt krotkich danych
    frame: Frame = []
    assert normalize_landmarks(frame) == frame


def test_normalize_landmarks_centers_and_scales():
    # wrist (0) = (0,0), middle_mcp (9) = (0,2) -> skala 2
    pts = [(0.0, 0.0, 0.0)] * 21
    pts[9] = (0.0, 2.0, 0.0)
    pts[8] = (2.0, 2.0, 0.0)
    norm = normalize_landmarks(make_frame(pts))
    # po normalizacji wrist jest (0,0), a punkt (2,2)-> (1,1)
    assert pytest.approx(norm[8][0], rel=1e-6) == 1.0
    assert pytest.approx(norm[8][1], rel=1e-6) == 1.0


def test_finger_direction_unit_vector():
    # wektor od (0,0,0) do (3,4,0) -> znormalizowany (0.6,0.8,0)
    frame = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)]
    vx, vy, vz = finger_direction(frame, 0, 1)
    assert pytest.approx(vx, rel=1e-6) == 0.6
    assert pytest.approx(vy, rel=1e-6) == 0.8
    assert pytest.approx(vz, rel=1e-6) == 0.0


def test_curl_score_extremes():
    # prosty palec: segmenty w jednej linii, male katy -> niskie curl
    frame: Frame = []
    # zbuduje punkty mcp->pip->dip->tip w linii na osi X
    frame.extend([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)])
    score_straight = curl_score(frame, 0, 1, 2, 3)
    assert score_straight <= 0.05

    # zgiecie: ustaw katy blisko 90 deg -> wysoki curl
    frame2: Frame = []
    frame2.extend([(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 2.0, 0.0)])
    score_curled = curl_score(frame2, 0, 1, 2, 3)
    assert score_curled >= 0.7


def test_pinch_distance_norm():
    # po normalizacji wrist(0)=(0,0), middle_mcp(9)=(0,1) -> skala 1
    pts = [(0.0, 0.0, 0.0)] * 21
    pts[9] = (0.0, 1.0, 0.0)
    thumb_tip = 4
    index_tip = 8
    pts[thumb_tip] = (0.0, 0.0, 0.0)
    pts[index_tip] = (0.6, 0.8, 0.0)  # dystans 1.0 po normalizacji
    norm = normalize_landmarks(make_frame(pts))
    d = pinch_distance_norm(norm, thumb_tip, index_tip)
    assert pytest.approx(d, rel=1e-6) == 1.0
