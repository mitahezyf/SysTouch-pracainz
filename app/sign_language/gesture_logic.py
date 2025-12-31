# state machine dla gestow dynamicznych PJM (J, Z, sekwencje CH/RZ)
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.gesture_engine.config import DEBUG_MODE
from app.gesture_engine.logger import logger


@dataclass
class GestureFrame:
    """ramka z historia klatek gestu - przechowuje ksztalt i landmarki"""

    letter: str  # rozpoznany statyczny ksztalt dloni (np. "I" dla J)
    confidence: float  # pewnosc klasyfikatora
    landmarks: np.ndarray  # landmarki (21, 3) dla analizy ruchu
    timestamp: float  # czas w sekundach


@dataclass
class GestureResult:
    """wynik finalny po analizie statycznej i dynamicznej"""

    name: str  # finalna litera (np. "J" po wykryciu ruchu)
    confidence: float  # pewnosc
    gesture_type: str  # "static", "dynamic", "sequence"
    base_shape: Optional[str] = None  # bazowy ksztalt dla dynamicznych (np. I->J)


class GestureManager:
    """
    Menedzer logiki gestow - warstwa 2 architektury.

    Zadania:
    - wygladzanie szumu z klasyfikatora (bufor klatek)
    - wykrywanie ruchu dla liter dynamicznych (J, Z)
    - wykrywanie sekwencji dla dwuznakow (CH, RZ)

    Args:
        buffer_size: rozmiar bufora klatek (historia)
        motion_threshold: prog ruchu nadgarstka dla gestow dynamicznych
        sequence_max_gap_ms: max przerwa miedzy literami w sekwencji (ms)
        gesture_types: mapa litera -> typ ("static"/"dynamic")
        sequences: mapa dwuznak -> lista komponentow (np. "CH" -> ["C", "H"])
    """

    def __init__(
        self,
        buffer_size: int = 30,
        motion_threshold: float = 0.05,
        sequence_max_gap_ms: int = 1500,
        gesture_types: Optional[dict[str, str]] = None,
        sequences: Optional[dict[str, list[str]]] = None,
    ):
        self.buffer_size = buffer_size
        self.motion_threshold = motion_threshold
        self.sequence_max_gap_ms = sequence_max_gap_ms

        # domyslnie wszystkie statyczne, J i Z dynamiczne
        self.gesture_types = gesture_types or {chr(65 + i): "static" for i in range(26)}
        self.gesture_types.update({"J": "dynamic", "Z": "dynamic"})

        self.sequences = sequences or {"CH": ["C", "H"], "RZ": ["R", "Z"]}

        # bufor klatek (history sliding window)
        self.frame_buffer: deque[GestureFrame] = deque(maxlen=buffer_size)

        # stan sekwencji
        self.sequence_buffer: list[tuple[str, float]] = []  # [(litera, timestamp)]
        self.last_letter: Optional[str] = None
        self.last_letter_time: Optional[float] = None

        logger.info(
            "GestureManager: buffer=%d, motion_thresh=%.3f, seq_gap=%dms, dynamic=%s",
            buffer_size,
            motion_threshold,
            sequence_max_gap_ms,
            [k for k, v in self.gesture_types.items() if v == "dynamic"],
        )

    def reset(self) -> None:
        """resetuje stan menedzera (czysci bufory)"""
        self.frame_buffer.clear()
        self.sequence_buffer.clear()
        self.last_letter = None
        self.last_letter_time = None
        logger.debug("GestureManager zresetowany")

    def process(
        self,
        static_letter: str,
        confidence: float,
        landmarks: np.ndarray,
    ) -> Optional[GestureResult]:
        """
        przetwarza pojedyncza klatke - glowna metoda menedzera.

        Args:
            static_letter: rozpoznany statyczny ksztalt z klasyfikatora
            confidence: pewnosc klasyfikatora
            landmarks: landmarki (21, 3)

        Returns:
            GestureResult lub None jesli brak stabilnego rozpoznania
        """
        if landmarks.shape != (21, 3):
            logger.warning(
                "Nieprawidlowy ksztalt landmarkow: %s (oczekiwano (21,3))",
                landmarks.shape,
            )
            return None

        current_time = time.time()

        # dodaj klatke do bufora
        frame = GestureFrame(
            letter=static_letter,
            confidence=confidence,
            landmarks=landmarks.copy(),
            timestamp=current_time,
        )
        self.frame_buffer.append(frame)

        # czekaj az bufor sie zapelni (choc czesc - min 5 klatek)
        if len(self.frame_buffer) < min(5, self.buffer_size):
            return None

        # sprawdz typ gestu
        gesture_type = self.gesture_types.get(static_letter, "static")

        if gesture_type == "dynamic":
            # analiza ruchu dla J/Z
            result = self._detect_motion_gesture(static_letter, confidence)
            if result:
                self._update_sequence_buffer(result.name, current_time)
                return result
            # jesli brak wykrytego ruchu, zwroc bazowy ksztalt jako static
            return GestureResult(
                name=static_letter,
                confidence=confidence,
                gesture_type="static",
            )

        # statyczny gest
        result = GestureResult(
            name=static_letter,
            confidence=confidence,
            gesture_type="static",
        )

        # sprawdz sekwencje (dwuznaki)
        self._update_sequence_buffer(static_letter, current_time)
        sequence_result = self._detect_sequence()
        if sequence_result:
            return sequence_result

        return result

    def _detect_motion_gesture(
        self, base_letter: str, confidence: float
    ) -> Optional[GestureResult]:
        """
        wykrywa ruch dla gestow dynamicznych (J, Z).

        Logika:
        - J: ksztalt "I" + ruch nadgarstka w dol/bok (luk)
        - Z: ksztalt wlaczajacy palec wskazujacy + ruch w ksztalcie zygzaka

        Uproszczenie v1: wykrywamy dowolny ruch powyzej progu
        """
        if len(self.frame_buffer) < 10:
            return None

        # pobierz ostatnie N klatek
        recent_frames = list(self.frame_buffer)[-10:]

        # sprawdz czy wszystkie maja ten sam bazowy ksztalt (stabilnosc)
        if not all(f.letter == base_letter for f in recent_frames):
            return None

        # oblicz przemieszczenie nadgarstka (punkt 0)
        wrists = [f.landmarks[0] for f in recent_frames]
        deltas = []
        for i in range(1, len(wrists)):
            delta = np.linalg.norm(wrists[i] - wrists[i - 1])
            deltas.append(delta)

        avg_motion = np.mean(deltas) if deltas else 0.0

        # wykryj ruch
        if avg_motion > self.motion_threshold:
            if DEBUG_MODE:
                logger.debug(
                    "Wykryto ruch dla %s: avg_motion=%.4f > %.4f",
                    base_letter,
                    avg_motion,
                    self.motion_threshold,
                )
            return GestureResult(
                name=base_letter,  # J lub Z
                confidence=confidence,
                gesture_type="dynamic",
                base_shape=base_letter,
            )

        return None

    def _update_sequence_buffer(self, letter: str, timestamp: float) -> None:
        """aktualizuje bufor sekwencji - rejestruje nowa litere"""
        # ignoruj powtorzenia tej samej litery
        if letter == self.last_letter:
            return

        # dodaj do bufora sekwencji
        self.sequence_buffer.append((letter, timestamp))
        self.last_letter = letter
        self.last_letter_time = timestamp

        # usun stare wpisy z bufora sekwencji
        cutoff_time = timestamp - (self.sequence_max_gap_ms / 1000.0)
        self.sequence_buffer = [
            (letter, t) for letter, t in self.sequence_buffer if t >= cutoff_time
        ]

    def _detect_sequence(self) -> Optional[GestureResult]:
        """
        wykrywa sekwencje dwuznakow (CH, RZ).

        Returns:
            GestureResult dla dwuznaku lub None
        """
        if len(self.sequence_buffer) < 2:
            return None

        # pobierz ostatnie 2 litery
        recent = [letter for letter, _ in self.sequence_buffer[-2:]]

        # sprawdz czy pasuja do zdefiniowanych sekwencji
        for seq_name, components in self.sequences.items():
            if recent == components:
                # wykryto sekwencje!
                # oblicz srednia pewnosc (z ostatnich 2 klatek)
                last_frames = list(self.frame_buffer)[-10:]
                if last_frames:
                    avg_conf = float(np.mean([f.confidence for f in last_frames]))
                else:
                    avg_conf = 0.7

                logger.info(
                    "Wykryto sekwencje: %s (%s -> %s)",
                    seq_name,
                    components[0],
                    components[1],
                )

                # wyczysc bufor sekwencji po wykryciu
                self.sequence_buffer.clear()

                return GestureResult(
                    name=seq_name,
                    confidence=avg_conf,
                    gesture_type="sequence",
                    base_shape=" -> ".join(components),
                )

        return None

    def get_state(self) -> dict:
        """zwraca aktualny stan dla diagnostyki"""
        return {
            "buffer_fill": len(self.frame_buffer),
            "buffer_size": self.buffer_size,
            "sequence_buffer": [letter for letter, _ in self.sequence_buffer],
            "last_letter": self.last_letter,
        }
