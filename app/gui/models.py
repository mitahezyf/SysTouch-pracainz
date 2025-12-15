from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(slots=True)
class GestureResult:
    # wynik rozpoznania gestu
    # name nazwa gestu lub None; confidence pewnosc [0.0, 1.0]
    name: Optional[str]
    confidence: float = 0.0


@dataclass(slots=True)
class SingleHandResult:
    # wynik dla pojedynczej reki uzywany do obslugi wielu rak jednoczesnie
    index: int
    name: Optional[str]
    confidence: float
    landmarks: Any
    handedness: Optional[str] = None
