"""No-op normalizer for PJM translator.

Keeps landmark coordinates unchanged; present to satisfy optional imports in GUI.
"""

from __future__ import annotations

import numpy as np


def normalize(landmarks: np.ndarray) -> np.ndarray:
    """Zwraca landmarki bez zmian (no-op)."""
    return np.asarray(landmarks)


class HandNormalizer:
    """No-op klasa zgodna z interfejsem normalizera."""

    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        return normalize(landmarks)


class MediaPipeNormalizer(HandNormalizer):
    """No-op normalizer kompatybilny z GUI (MediaPipeNormalizer)."""

    def normalize(
        self, landmarks: np.ndarray, handedness: str | None = None
    ) -> np.ndarray:
        return normalize(landmarks)
