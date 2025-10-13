# runtime gestow: ladowanie json, matchery i decyzja
# bezpieczny szkic; nie podpinamy jeszcze do istniejacej petli

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .gesture_loader import load_gestures
from .gesture_matcher import SequenceGestureMatcher, StaticGestureMatcher


class GestureRuntime:
    # prosty runtime gestow z json
    def __init__(self, paths: List[str]) -> None:
        self.defs: List[Dict[str, Any]] = load_gestures(paths)
        self.static = StaticGestureMatcher(self.defs)
        self.seq = SequenceGestureMatcher(self.defs)

    def update(
        self, frame: List[Tuple[float, float, float]]
    ) -> Optional[Dict[str, Any]]:
        # wybiera najlepszy gest z matchera statycznego i sekwencji
        s = self.static.update(frame)
        q = self.seq.update(frame)
        if s and q:
            # wybor po priorytecie lub confidence
            if int(q.get("priority", 0)) > int(s.get("priority", 0)):
                return q
            if float(q.get("confidence", 0.0)) > float(s.get("confidence", 0.0)):
                return q
            return s
        return s or q
