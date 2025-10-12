# matcher gestow z json: statyczne + szkic sekwencji
# prosta implementacja: score per gest, smoothing, histereza, min_hold

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..utils.hand_features import normalize_landmarks, curl_score, pinch_distance_norm

# indeksy mediapipe
WRIST = 0
THUMB = {
    "mcp": 2,
    "pip": 3,
    "dip": 3,
    "tip": 4,
}  # kciuk ma ip zamiast pip/dip, uzyjemy ip jako oba
INDEX = {"mcp": 5, "pip": 6, "dip": 7, "tip": 8}
MIDDLE = {"mcp": 9, "pip": 10, "dip": 11, "tip": 12}
RING = {"mcp": 13, "pip": 14, "dip": 15, "tip": 16}
PINKY = {"mcp": 17, "pip": 18, "dip": 19, "tip": 20}
FINGERS_IDX = {
    "thumb": THUMB,
    "index": INDEX,
    "middle": MIDDLE,
    "ring": RING,
    "pinky": PINKY,
}


class _StabState:
    # stan stabilizacji dla jednego gestu
    def __init__(self, smooth: int, entry: float, exit: float, min_hold: int) -> None:
        self.smooth = max(1, smooth)
        self.entry = entry
        self.exit = exit
        self.min_hold = max(1, min_hold)
        self.buf: Deque[float] = deque(maxlen=self.smooth)
        self.active = False
        self.hold = 0
        self.last_score = 0.0

    def update(self, score: float) -> Tuple[bool, float]:
        self.buf.append(score)
        avg = sum(self.buf) / len(self.buf)
        self.last_score = avg
        if not self.active:
            if avg >= self.entry:
                self.active = True
                self.hold = 1
        else:
            self.hold += 1
            if avg < self.exit and self.hold >= self.min_hold:
                self.active = False
                self.hold = 0
        return self.active, avg


class StaticGestureMatcher:
    # dopasowanie gestow statycznych; zwraca najlepszy aktywny gest
    def __init__(self, gestures: List[Dict[str, Any]]) -> None:
        self.items: List[Dict[str, Any]] = [
            g for g in gestures if g.get("type") == "static"
        ]
        # stan per gest
        self.state: List[_StabState] = [
            _StabState(
                smooth=int(g["stabilization"]["smooth"]),
                entry=float(g["stabilization"]["entry"]),
                exit=float(g["stabilization"]["exit"]),
                min_hold=int(g["stabilization"]["min_hold"]),
            )
            for g in self.items
        ]

    def _score_finger(self, curl: float, spec: Dict[str, Any]) -> float:
        state = spec.get("state", "any")
        if state == "any":
            return 1.0
        if state == "extended":
            return 1.0 if curl < float(spec.get("extended_thr", 0.3)) else 0.0
        if state == "curled":
            return 1.0 if curl > float(spec.get("curled_thr", 0.6)) else 0.0
        return 0.0

    def _raw_score(
        self, frame: List[Tuple[float, float, float]], g: Dict[str, Any]
    ) -> float:
        # oblicza score 0..1 na bazie curl palcow i pinch
        # normalizacja
        nf = normalize_landmarks(frame)
        # curl palcow
        curls: Dict[str, float] = {}
        for name, idxs in FINGERS_IDX.items():
            curls[name] = curl_score(
                nf, idxs["mcp"], idxs["pip"], idxs["dip"], idxs["tip"]
            )
        # score palcow
        fspec = g["fingers"]
        finger_scores = [
            self._score_finger(curls[f], fspec.get(f, {})) for f in FINGERS_IDX.keys()
        ]
        # pinch
        pp = g["predicates"]["pinch"]
        pinch_ok = True
        if pp.get("enabled", False):
            pd = pinch_distance_norm(
                nf, THUMB["tip"], INDEX["tip"]
            )  # po normalizacji to juz skala
            pinch_ok = pd < float(pp.get("entry", 0.12))
        pinch_score = 1.0 if pinch_ok else 0.0
        # orientacja: na razie pomijamy lub zawsze true
        # laczny score: srednia z palcow i pinch (jesli wlaczony)
        parts = finger_scores
        if pp.get("enabled", False):
            parts.append(pinch_score)
        if not parts:
            return 0.0
        return sum(parts) / len(parts)

    def update(
        self, frame: List[Tuple[float, float, float]]
    ) -> Optional[Dict[str, Any]]:
        # aktualizuje stan wszystkich gestow i zwraca najlepszy aktywny
        best: Optional[Tuple[int, Dict[str, Any], float]] = None
        for i, g in enumerate(self.items):
            score = self._raw_score(frame, g)
            active, avg = self.state[i].update(score)
            if active:
                if not best or avg > best[2]:
                    best = (i, g, avg)
        if best:
            g = best[1]
            return {
                "name": g["name"],
                "namespace": g["namespace"],
                "action": g["action"],
                "confidence": float(best[2]),
                "priority": int(g.get("priority", 0)),
                "type": "static",
            }
        return None


class SequenceGestureMatcher:
    # szkic FSM dla sekwencji; implementacja minimalna (TODO)
    def __init__(self, gestures: List[Dict[str, Any]]) -> None:
        self.items: List[Dict[str, Any]] = [
            g for g in gestures if g.get("type") == "sequence"
        ]
        # TODO: stan per gest (indeks stanu, licznik klatek, timeout)

    def update(
        self, frame: List[Tuple[float, float, float]]
    ) -> Optional[Dict[str, Any]]:
        # TODO: dopasowanie sekwencji; na razie zwracamy None
        return None
