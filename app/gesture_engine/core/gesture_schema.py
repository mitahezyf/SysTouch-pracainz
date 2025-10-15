# schemat i walidacja definicji gestow json
# prosty, bez zewnetrznych zaleznosci

from __future__ import annotations

from typing import Any, Dict, List

# stale domyslne
DEFAULT_EXTENDED_THR = 0.3
DEFAULT_CURLED_THR = 0.6
DEFAULT_SMOOTH = 5
DEFAULT_ENTRY = 0.7
DEFAULT_EXIT = 0.6
DEFAULT_MIN_HOLD = 3

VALID_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
VALID_TYPES = ["static", "sequence"]


def _ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _norm_finger_def(fd: Dict[str, Any]) -> Dict[str, Any]:
    # normalizacja definicji palca (state + progi)
    state = fd.get("state", "any")
    _ensure(state in ("extended", "curled", "any"), "finger state invalid")
    return {
        "state": state,
        "extended_thr": float(fd.get("extended_thr", DEFAULT_EXTENDED_THR)),
        "curled_thr": float(fd.get("curled_thr", DEFAULT_CURLED_THR)),
    }


def _norm_stabilization(stab: Dict[str, Any] | None) -> Dict[str, Any]:
    stab = stab or {}
    return {
        "smooth": int(stab.get("smooth", DEFAULT_SMOOTH)),
        "entry": float(stab.get("entry", DEFAULT_ENTRY)),
        "exit": float(stab.get("exit", DEFAULT_EXIT)),
        "min_hold": int(stab.get("min_hold", DEFAULT_MIN_HOLD)),
    }


def _norm_predicates(p: Dict[str, Any] | None) -> Dict[str, Any]:
    p = p or {}
    pinch = p.get("pinch", {})
    orientation = p.get("orientation", {})
    return {
        "pinch": {
            "enabled": bool(pinch.get("enabled", False)),
            "entry": float(pinch.get("entry", 0.12)),
            "exit": float(pinch.get("exit", 0.15)),
        },
        "orientation": {
            "palm_up": bool(orientation.get("palm_up", False)),
        },
    }


def normalize_static_def(d: Dict[str, Any]) -> Dict[str, Any]:
    # normalizacja i walidacja gestu statycznego
    _ensure(d.get("type") == "static", "type must be static")
    _ensure(isinstance(d.get("name"), str) and d["name"], "name required")
    _ensure(
        isinstance(d.get("namespace"), str) and d["namespace"], "namespace required"
    )
    fingers = d.get("fingers", {})
    _ensure(isinstance(fingers, dict), "fingers must be dict")
    nf: Dict[str, Any] = {}
    for f in VALID_FINGERS:
        nf[f] = _norm_finger_def(fingers.get(f, {}))
    stab = _norm_stabilization(d.get("stabilization"))
    preds = _norm_predicates(d.get("predicates"))
    priority = int(d.get("priority", 50))
    action = d.get("action", {"type": "none", "params": {}})
    _ensure(isinstance(action, dict) and "type" in action, "action invalid")
    return {
        "type": "static",
        "name": d["name"],
        "namespace": d["namespace"],
        "fingers": nf,
        "stabilization": stab,
        "predicates": preds,
        "priority": priority,
        "action": action,
    }


def normalize_sequence_def(d: Dict[str, Any]) -> Dict[str, Any]:
    # normalizacja i walidacja gestu sekwencyjnego (szkic)
    _ensure(d.get("type") == "sequence", "type must be sequence")
    _ensure(isinstance(d.get("name"), str) and d["name"], "name required")
    _ensure(
        isinstance(d.get("namespace"), str) and d["namespace"], "namespace required"
    )
    states = d.get("states")
    # jawna walidacja, aby mypy widziało listę
    if not isinstance(states, list) or len(states) == 0:
        raise ValueError("states required")
    norm_states: List[Dict[str, Any]] = []
    for s in states:
        _ensure(
            "conditions" in s and isinstance(s["conditions"], dict),
            "state conditions required",
        )
        norm_states.append(
            {
                "name": s.get("name", "state"),
                "conditions": s["conditions"],
                "min": int(s.get("min", 1)),
                "max": int(s.get("max", 999999)),
            }
        )
    timeout = int(d.get("timeout_frames", 30))
    stab = _norm_stabilization(d.get("stabilization"))
    priority = int(d.get("priority", 60))
    action = d.get("action", {"type": "none", "params": {}})
    _ensure(isinstance(action, dict) and "type" in action, "action invalid")
    return {
        "type": "sequence",
        "name": d["name"],
        "namespace": d["namespace"],
        "states": norm_states,
        "timeout_frames": timeout,
        "stabilization": stab,
        "priority": priority,
        "action": action,
    }


def normalize_gesture_def(d: Dict[str, Any]) -> Dict[str, Any]:
    # wybor sciezki normalizacji wedlug typu
    t = d.get("type")
    _ensure(t in VALID_TYPES, "unknown type")
    if t == "static":
        return normalize_static_def(d)
    return normalize_sequence_def(d)
