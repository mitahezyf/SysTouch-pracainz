import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from types import SimpleNamespace

from app.gesture_engine.utils.geometry import distance, angle_between


# prosty obiekt punktu z atrybutami x, y, z
def make_point(x, y, z=0):
    return SimpleNamespace(x=x, y=y, z=z)


# dystans poziomy miedzy (0,0) a (3,0) wynosi 3
def test_distance_horizontal():
    p1 = make_point(0, 0)
    p2 = make_point(3, 0)
    assert distance(p1, p2) == 3


# trojkat prostokatny (0,0) - (3,4) ma dlugosc 5
def test_distance_diagonal():
    p1 = make_point(0, 0)
    p2 = make_point(3, 4)
    assert distance(p1, p2) == 5


# punkty w jednej linii (p1 - p2 - p3) -> kat powinien byc 180
def test_angle_between_straight_line():
    p1 = make_point(0, 0, 0)
    p2 = make_point(1, 0, 0)
    p3 = make_point(2, 0, 0)
    assert angle_between(p1, p2, p3) == pytest.approx(180)


# wektory prostopadle -> kat powinien byc 90
def test_angle_between_perpendicular():
    p1 = make_point(1, 1, 0)
    p2 = make_point(1, 0, 0)
    p3 = make_point(2, 0, 0)
    assert angle_between(p1, p2, p3) == pytest.approx(90)


# jeden z wektorow ma dlugosc 0 -> kat powinien byc 0 - edge case
def test_angle_between_zero_length_vector():
    p1 = make_point(1, 1, 1)
    p2 = make_point(1, 1, 1)
    p3 = make_point(2, 2, 2)
    assert angle_between(p1, p2, p3) == 0
