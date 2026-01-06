from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopping:
    patience: int = 15
    delta: float = 0.005
    verbose: bool = True

    best_loss: Optional[float] = None
    counter: int = 0

    def __call__(self, val_loss: float) -> bool:
        """Zwraca True jesli nalezy przerwac trening (brak poprawy)."""
        if self.best_loss is None:
            self.best_loss = float(val_loss)
            self.counter = 0
            return False

        if float(val_loss) < (self.best_loss - self.delta):
            self.best_loss = float(val_loss)
            self.counter = 0
            return False

        self.counter += 1
        if self.verbose:
            logger.info(
                "[EarlyStopping] brak poprawy (%d/%d), val_loss=%.6f",
                self.counter,
                self.patience,
                val_loss,
            )

        return self.counter >= self.patience
