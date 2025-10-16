import pytest

from app.gesture_engine.core.gesture_schema import (
    normalize_gesture_def,
    normalize_sequence_def,
    normalize_static_def,
)


def test_normalize_static_def_happy():
    d = {
        "type": "static",
        "name": "ok",
        "namespace": "test",
        # celowo minimalnie: fingers/predicates/action domyslne
    }
    out = normalize_static_def(d)
    assert out["type"] == "static"
    assert out["name"] == "ok"
    assert out["namespace"] == "test"
    assert "fingers" in out and set(out["fingers"].keys()) == {
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    }


def test_normalize_sequence_def_happy():
    d = {
        "type": "sequence",
        "name": "seq",
        "namespace": "test",
        "states": [
            {"name": "s1", "conditions": {"x": 1}, "min": 1, "max": 2},
            {"name": "s2", "conditions": {"y": 2}},
        ],
    }
    out = normalize_sequence_def(d)
    assert out["type"] == "sequence"
    assert out["name"] == "seq"
    assert out["namespace"] == "test"
    assert isinstance(out["states"], list) and len(out["states"]) == 2


def test_normalize_gesture_def_invalid_type():
    with pytest.raises(ValueError):
        normalize_gesture_def({"type": "unknown"})


def test_static_missing_name_raises():
    with pytest.raises(ValueError):
        normalize_static_def({"type": "static", "namespace": "x"})


def test_sequence_missing_states_raises():
    with pytest.raises(ValueError):
        normalize_sequence_def({"type": "sequence", "name": "a", "namespace": "b"})
