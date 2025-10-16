from typing import Any, Dict, List

import pytest

from app.gesture_engine.core.gesture_matcher import StaticGestureMatcher


@pytest.fixture
def gestures_minimal() -> List[Dict[str, Any]]:
    # dwa gesty statyczne z rozna stabilizacja
    return [
        {
            "type": "static",
            "name": "g1",
            "namespace": "test",
            "fingers": {"index": {"state": "extended", "extended_thr": 0.3}},
            "predicates": {"pinch": {"enabled": False}},
            "stabilization": {"smooth": 1, "entry": 0.7, "exit": 0.9, "min_hold": 1},
            "action": {"type": "none", "params": {}},
        },
        {
            "type": "static",
            "name": "g2",
            "namespace": "test",
            "fingers": {"index": {"state": "extended", "extended_thr": 0.3}},
            "predicates": {"pinch": {"enabled": True, "entry": 0.2}},
            "stabilization": {"smooth": 1, "entry": 0.5, "exit": 0.4, "min_hold": 1},
            "action": {"type": "none", "params": {}},
        },
    ]


def test_static_matcher_activates_and_deactivates(monkeypatch, gestures_minimal):
    # ustaw scoring palcow: thumb/index/middle/ring/pinky dla 2 gestow (razem 10 wywolan)
    scores_seq = [0.5, 0.0, 0.5, 0.5, 0.5] * 2
    scores_seq2 = [0.5, 1.0, 0.5, 0.5, 0.5] * 2

    def curl_side_effect(*_args):
        return scores_seq.pop(0)

    def curl_side_effect2(*_args):
        return scores_seq2.pop(0)

    # pinch nieuzywany w g1, dla g2 wlaczony: ustaw bardzo maly dystans by spelnic wejsciowy prog
    pd_values = [0.0, 0.0, 1.0, 1.0]
    monkeypatch.setattr(
        "app.gesture_engine.core.gesture_matcher.pinch_distance_norm",
        lambda *_: pd_values.pop(0),
    )

    m = StaticGestureMatcher(gestures_minimal)

    # pierwsze wywolanie: aktywacja przynajmniej jednego gestu
    monkeypatch.setattr(
        "app.gesture_engine.core.gesture_matcher.curl_score", curl_side_effect
    )
    out1 = m.update([(0.0, 0.0, 0.0)] * 21)
    assert out1 is not None and out1["name"] in {"g1", "g2"}

    # drugie wywolanie: modyfikacja sredniej; dopuszczamy brak wyniku lub g2
    monkeypatch.setattr(
        "app.gesture_engine.core.gesture_matcher.curl_score", curl_side_effect2
    )
    out2 = m.update([(0.0, 0.0, 0.0)] * 21)
    assert out2 is None or out2["name"] == "g2"


def test_static_matcher_best_of_two(monkeypatch):
    # definiujemy spec palcow tak, aby pierwszy gest mial niski score, a drugi wysoki
    gestures = [
        {
            "type": "static",
            "name": "weak",
            "namespace": "t",
            "fingers": {"index": {"state": "curled", "curled_thr": 0.8}},
            "predicates": {"pinch": {"enabled": False}},
            "stabilization": {"smooth": 1, "entry": 0.1, "exit": 0.0, "min_hold": 1},
            "action": {"type": "none", "params": {}},
        },
        {
            "type": "static",
            "name": "strong",
            "namespace": "t",
            "fingers": {"index": {"state": "curled", "curled_thr": 0.8}},
            "predicates": {"pinch": {"enabled": False}},
            "stabilization": {"smooth": 1, "entry": 0.1, "exit": 0.0, "min_hold": 1},
            "action": {"type": "none", "params": {}},
        },
    ]

    # slaby gest: 5x 0.2 -> score 0; mocny gest: 5x 0.9 -> score 1
    seq: List[float] = [0.2] * 5 + [0.9] * 5

    def curl_seq(*_):
        return seq.pop(0)

    monkeypatch.setattr("app.gesture_engine.core.gesture_matcher.curl_score", curl_seq)

    m = StaticGestureMatcher(gestures)
    out = m.update([(0.0, 0.0, 0.0)] * 21)
    assert out is not None and out["name"] == "strong"
