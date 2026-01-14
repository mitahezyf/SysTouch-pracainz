"""
Wątek QThread do uruchamiania treningu modelu PJM z callbackami postępu.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class TrainingThread(QThread):
    """Wątek do treningu modelu w tle z raportowaniem postępu."""

    # Sygnały
    progress_updated = Signal(int, int, str)  # current, total, message
    training_finished = Signal(dict)  # results
    training_error = Signal(str)  # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def run(self):
        """Uruchamia pipeline treningu z callbackami."""
        try:
            from app.sign_language.training_runner import run_training_pipeline

            results = run_training_pipeline(progress_callback=self._on_progress)

            if not self._cancelled:
                self.training_finished.emit(results)
        except Exception as e:
            if not self._cancelled:
                self.training_error.emit(str(e))

    def _on_progress(self, current: int, total: int, message: str):
        """Callback dla postępu - emituje sygnał do GUI."""
        if not self._cancelled:
            self.progress_updated.emit(current, total, message)

    def cancel(self):
        """Anuluje trening (best effort)."""
        self._cancelled = True
