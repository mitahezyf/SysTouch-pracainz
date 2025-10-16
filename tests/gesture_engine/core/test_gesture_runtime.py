from typing import Any, Dict, List, Optional, Tuple

from app.gesture_engine.core.gesture_runtime import GestureRuntime


class DummyMatcher:
    def __init__(self, result: Optional[Dict[str, Any]]):
        self._result = result

    def update(self, _frame: List[Tuple[float, float, float]]):
        return self._result


def test_runtime_prefers_higher_priority_over_confidence(monkeypatch):
    # unikamy IO: podmieniamy load_gestures na pusta liste
    monkeypatch.setattr(
        "app.gesture_engine.core.gesture_runtime.load_gestures", lambda *_: []
    )
    rt = GestureRuntime(paths=[])
    # static: priority 10, confidence 0.9
    rt.static = DummyMatcher(
        {
            "name": "s",
            "namespace": "t",
            "action": {"type": "none", "params": {}},
            "confidence": 0.9,
            "priority": 10,
            "type": "static",
        }
    )
    # sequence: priority 20, confidence 0.5
    rt.seq = DummyMatcher(
        {
            "name": "q",
            "namespace": "t",
            "action": {"type": "none", "params": {}},
            "confidence": 0.5,
            "priority": 20,
            "type": "sequence",
        }
    )

    out = rt.update([(0.0, 0.0, 0.0)] * 21)
    assert out is not None and out["name"] == "q"


def test_runtime_fallback_when_one_none(monkeypatch):
    monkeypatch.setattr(
        "app.gesture_engine.core.gesture_runtime.load_gestures", lambda *_: []
    )
    rt = GestureRuntime(paths=[])
    rt.static = DummyMatcher(None)
    rt.seq = DummyMatcher(
        {
            "name": "q",
            "namespace": "t",
            "action": {"type": "none", "params": {}},
            "confidence": 0.3,
            "priority": 1,
            "type": "sequence",
        }
    )
    out = rt.update([(0.0, 0.0, 0.0)] * 21)
    assert out is not None and out["name"] == "q"


def test_runtime_confidence_break_tie(monkeypatch):
    monkeypatch.setattr(
        "app.gesture_engine.core.gesture_runtime.load_gestures", lambda *_: []
    )
    rt = GestureRuntime(paths=[])
    rt.static = DummyMatcher(
        {
            "name": "s",
            "namespace": "t",
            "action": {"type": "none", "params": {}},
            "confidence": 0.6,
            "priority": 5,
            "type": "static",
        }
    )
    rt.seq = DummyMatcher(
        {
            "name": "q",
            "namespace": "t",
            "action": {"type": "none", "params": {}},
            "confidence": 0.7,
            "priority": 5,
            "type": "sequence",
        }
    )
    out = rt.update([(0.0, 0.0, 0.0)] * 21)
    assert out is not None and out["name"] == "q"
