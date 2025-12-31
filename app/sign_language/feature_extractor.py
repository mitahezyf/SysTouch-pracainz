import numpy as np
import torch


class FeatureExtractor:
    """
    Ekstraktor cech dla modelu PJM.
    Przekształca surowe landmarki (21x3) w bogaty wektor cech:
    - Znormalizowane współrzędne (relatywne do nadgarstka, skalowane)
    - Kąty zgięcia palców
    - Odległości opuszków
    - Kąt obrotu dłoni (roll)
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        # Indeksy landmarków
        self.WRIST = 0
        self.THUMB_CMC, self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP = 1, 2, 3, 4
        self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP = 5, 6, 7, 8
        self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP = (
            9,
            10,
            11,
            12,
        )
        self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP = 13, 14, 15, 16
        self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP = 17, 18, 19, 20

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Główna metoda ekstrakcji.
        Args:
            landmarks: np.ndarray kształtu (21, 3)
        Returns:
            np.ndarray: wektor cech (float32)
        """
        if landmarks.shape != (21, 3):
            raise ValueError(f"Oczekiwano (21, 3), otrzymano {landmarks.shape}")

        # 1. Normalizacja podstawowa (Centrowanie i Skalowanie)
        wrist = landmarks[self.WRIST]
        middle_mcp = landmarks[self.MIDDLE_MCP]

        scale_raw = np.linalg.norm(middle_mcp - wrist)
        scale: float = float(scale_raw) if scale_raw >= 1e-6 else 1e-6

        # Relatywne współrzędne (63 cechy)
        relative_coords = (landmarks - wrist) / scale
        flat_coords = relative_coords.flatten()

        # 2. Kąty zgięcia (Joint Angles)
        # Funkcja pomocnicza do kątów
        def get_angle(idx1, idx2, idx3):
            p1 = landmarks[idx1]
            p2 = landmarks[idx2]
            p3 = landmarks[idx3]

            v1 = p1 - p2
            v2 = p3 - p2

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            cosine = np.dot(v1, v2) / (norm1 * norm2)
            cosine = np.clip(cosine, -1.0, 1.0)
            return np.degrees(np.arccos(cosine))

        angles = []
        # Kciuk
        angles.append(get_angle(self.WRIST, self.THUMB_CMC, self.THUMB_MCP))
        angles.append(get_angle(self.THUMB_CMC, self.THUMB_MCP, self.THUMB_IP))
        angles.append(get_angle(self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP))

        # Pozostałe palce (MCP, PIP, DIP)
        for mcp, pip, dip, tip in [
            (self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP),
            (self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP),
            (self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP),
            (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP),
        ]:
            angles.append(get_angle(self.WRIST, mcp, pip))
            angles.append(get_angle(mcp, pip, dip))
            angles.append(get_angle(pip, dip, tip))

        # Kąty rozwarcia (Abduction) - między sąsiednimi MCP
        angles.append(get_angle(self.INDEX_MCP, self.WRIST, self.MIDDLE_MCP))
        angles.append(get_angle(self.MIDDLE_MCP, self.WRIST, self.RING_MCP))
        angles.append(get_angle(self.RING_MCP, self.WRIST, self.PINKY_MCP))
        angles.append(get_angle(self.THUMB_CMC, self.WRIST, self.INDEX_MCP))

        # 3. Znormalizowane odległości opuszków od nadgarstka
        distances = []
        for tip in [
            self.THUMB_TIP,
            self.INDEX_TIP,
            self.MIDDLE_TIP,
            self.RING_TIP,
            self.PINKY_TIP,
        ]:
            dist = float(np.linalg.norm(landmarks[tip] - wrist) / scale)
            distances.append(dist)

        # 4. Hand Roll (obrot dloni)
        # Kat wektora wrist->middle_mcp wzgledem osi X (w plaszczyznie XY kamery)
        # To pomaga modelowi zrozumiec orientacje dloni
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        roll_rad = float(np.arctan2(dy, dx))
        roll_deg = float(np.degrees(roll_rad))

        # Normalizacja roll do zakresu [-1, 1] (dzielenie przez 180)
        roll_norm = roll_deg / 180.0

        # Złożenie wektora cech
        # 63 coords + 15 finger angles + 4 spread angles + 5 distances + 1 roll
        # Total: 63 + 19 + 5 + 1 = 88 cech
        features = np.concatenate(
            [
                flat_coords,
                np.array(angles) / 180.0,  # Normalizacja kątów do [0, 1] (z grubsza)
                np.array(distances),
                np.array([roll_norm]),
            ]
        )

        return np.asarray(features.astype(np.float32), dtype=np.float32)

    def extract_batch(self, landmarks_batch: np.ndarray) -> np.ndarray:
        """
        Przetwarza batch landmarków.
        Args:
            landmarks_batch: (N, 21, 3)
        Returns:
            (N, n_features)
        """
        results = []
        for lms in landmarks_batch:
            results.append(self.extract(lms))
        return np.array(results)
