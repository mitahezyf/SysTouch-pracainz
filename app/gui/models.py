from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class GestureResult:
    """Wynik rozpoznania gestu.

    - name: nazwa gestu lub None
    - confidence: pewnosc detekcji w zakresie [0.0, 1.0]
    """

    name: Optional[str]
    confidence: float = 0.0
