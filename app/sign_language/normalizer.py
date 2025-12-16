# normalizacja landmarkow MediaPipe do formatu 63D (PyTorch backend)
from typing import List, Tuple

import numpy as np
import torch

from app.gesture_engine.logger import logger


class MediaPipeNormalizer:
    """
    Normalizuje landmarki MediaPipe (21 punktow x,y,z) do wektora 63D.

    Normalizacja hand-centric:
    - srodek: nadgarstek (punkt 0)
    - skala: odleglosc nadgarstek-srodek dloni (punkt 9)
    - output: [(x-wrist_x)/scale, (y-wrist_y)/scale, (z-wrist_z)/scale] dla kazdego punktu
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def normalize(
        self, landmarks: np.ndarray | List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Normalizuje landmarki do wektora 63D.

        Args:
            landmarks: lista/array 21 punktow (x,y,z) z MediaPipe

        Returns:
            wektor 63D znormalizowany wzgledem nadgarstka i skali dloni
        """
        # konwersja do tensora
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks, dtype=np.float32)
        elif not isinstance(landmarks, np.ndarray):
            raise TypeError(
                f"Oczekiwano list lub np.ndarray, otrzymano {type(landmarks)}"
            )

        if landmarks.shape != (21, 3):
            raise ValueError(
                f"Oczekiwano ksztaltu (21, 3), otrzymano {landmarks.shape}"
            )

        landmarks_t = torch.from_numpy(landmarks).to(self.device)

        # punkt bazowy: nadgarstek (indeks 0)
        wrist = landmarks_t[0]

        # skala: odleglosc nadgarstek -> srodek dloni (punkt 9: middle_finger_mcp)
        middle_mcp = landmarks_t[9]
        scale = torch.norm(middle_mcp - wrist)

        # zabezpieczenie przed dzieleniem przez zero
        if scale < 1e-6:
            logger.warning("Skala dloni bliska zeru, uzywam fallback scale=1.0")
            scale = torch.tensor(1.0, device=self.device)

        # normalizacja: (punkt - wrist) / scale
        normalized = (landmarks_t - wrist) / scale

        # sploszczenie do 63D
        result = normalized.reshape(-1).cpu().numpy()

        # walidacja wynikow
        if np.isnan(result).any() or np.isinf(result).any():
            logger.error("Normalizacja wygenerowala NaN lub Inf, zwracam wektor zerowy")
            return np.zeros(63, dtype=np.float32)

        return result.astype(np.float32)

    def normalize_batch(self, landmarks_batch: np.ndarray) -> np.ndarray:
        """
        Normalizuje batch landmarkow.

        Args:
            landmarks_batch: array [N, 21, 3]

        Returns:
            array [N, 63] znormalizowanych wektorow
        """
        if landmarks_batch.ndim != 3 or landmarks_batch.shape[1:] != (21, 3):
            raise ValueError(
                f"Oczekiwano ksztaltu (N, 21, 3), otrzymano {landmarks_batch.shape}"
            )

        batch_t = torch.from_numpy(landmarks_batch).to(self.device)

        # punkt bazowy: nadgarstek (indeks 0) dla kazdej probki
        wrist = batch_t[:, 0:1, :]  # [N, 1, 3]

        # skala: odleglosc nadgarstek -> middle_mcp (punkt 9)
        middle_mcp = batch_t[:, 9:10, :]  # [N, 1, 3]
        scale = torch.norm(middle_mcp - wrist, dim=2, keepdim=True)  # [N, 1, 1]

        # zabezpieczenie przed dzieleniem przez zero
        scale = torch.where(scale < 1e-6, torch.tensor(1.0, device=self.device), scale)

        # normalizacja
        normalized = (batch_t - wrist) / scale  # [N, 21, 3]

        # sploszczenie do [N, 63]
        result = normalized.reshape(len(batch_t), -1).cpu().numpy()

        # walidacja
        if np.isnan(result).any() or np.isinf(result).any():
            logger.warning(
                "Batch normalizacja zawiera NaN/Inf, zastepuje zerami w dotknietych wierszach"
            )
            invalid_mask = np.isnan(result).any(axis=1) | np.isinf(result).any(axis=1)
            result[invalid_mask] = 0.0

        return result.astype(np.float32)


# zapewnia kompatybilnosc z API gesture_trainer.normalizer
def normalize_landmarks(
    landmarks: List[Tuple[float, float, float]] | np.ndarray
) -> np.ndarray:
    """
    Funkcja kompatybilna z app.gesture_trainer.normalizer.normalize_landmarks.

    Args:
        landmarks: lista/array 21 punktow (x,y,z)

    Returns:
        wektor 63D znormalizowany
    """
    normalizer = MediaPipeNormalizer()
    return normalizer.normalize(landmarks)


__all__ = ["MediaPipeNormalizer", "normalize_landmarks"]
