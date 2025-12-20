class EarlyStopping:
    """implementuje wczesne zatrzymanie treningu gdy val_loss przestaje sie poprawiac.

    Args:
        patience: ile epok czekac bez poprawy przed zakonczeniem
        delta: minimalny wzrost uznawany za poprawe
        verbose: czy logowac informacje
    """

    def __init__(self, patience: int = 15, delta: float = 0.005, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """sprawdza czy nalezy zatrzymac trening.

        Args:
            val_loss: aktualna wartosc validation loss

        Returns:
            True jesli nalezy zatrzymac trening, False w przeciwnym wypadku
        """
        score = -val_loss  # wyzszy score = lepiej

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            return False

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                from app.gesture_engine.logger import logger

                logger.info(
                    "EarlyStopping counter: %d / %d (val_loss=%.4f, best=%.4f)",
                    self.counter,
                    self.patience,
                    val_loss,
                    self.val_loss_min,
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

        return self.early_stop
