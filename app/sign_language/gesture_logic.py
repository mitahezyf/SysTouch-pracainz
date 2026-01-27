"""
GestureManager - warstwa 2 logiki gestów dynamicznych dla translatora PJM.

Moduł ten implementuje gate/stabilizer dla etykiet oznaczonych jako "dynamic" w pjm.json.
NIE generuje nowych etykiet z sekwencji - działa jako filtr stabilizacyjny dla predykcji MLP.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.gesture_engine.logger import logger


@dataclass
class GestureResult:
    """Wynik wykrycia gestu dynamicznego po walidacji i stabilizacji."""

    name: str  # etykieta gestu (np. "A+")
    confidence: float  # pewność wykrycia (0-1)
    gesture_type: str  # zawsze "dynamic" dla wyników z GestureManager


class GestureManager:
    """
    Menedżer gestów dynamicznych - gate/stabilizer dla etykiet "dynamic" z pjm.json.

    GestureManager NIE generuje nowych etykiet. Działa jako warstwa walidacji i stabilizacji
    dla predykcji MLP, które są oznaczone jako "dynamic" w gesture_types.

    Logika:
    - Jeśli predykcja MLP to etykieta "dynamic": zastosuj ostrzejsze progi i stabilizację
    - Jeśli predykcja MLP to etykieta "static": przepuść normalnie (zwróć None)
    - Dynamiczny gest jest zatwierdzany dopiero po spełnieniu warunków stabilności:
      * pred_conf >= dynamic_entry
      * utrzymuje się przez stable_frames kolejnych wywołań
      * utrzymuje się przez dynamic_hold_ms (histereza)

    Parametry:
        gesture_types: mapowanie etykiet -> typ ("static"/"dynamic") z pjm.json
        dynamic_entry: próg confidence dla wejścia w gest dynamiczny (domyślnie 0.75)
        dynamic_exit: próg confidence dla wyjścia z gestu dynamicznego (domyślnie 0.55)
        dynamic_hold_ms: minimalny czas trzymania gestu dynamicznego w ms (domyślnie 600)
        stable_frames: liczba kolejnych stabilnych klatek wymaganych do zatwierdzenia (domyślnie 3)
        buffer_size: rozmiar bufora dla historii predykcji (domyślnie 10)
        motion_gate: czy wymagać ruchu dla gestów dynamicznych (domyślnie False)
        motion_threshold: minimalna wartość ruchu jeśli motion_gate=True (domyślnie 0.0)
    """

    def __init__(
        self,
        gesture_types: dict[str, str],
        dynamic_entry: float = 0.75,
        dynamic_exit: float = 0.55,
        dynamic_hold_ms: int = 600,
        stable_frames: int = 3,
        buffer_size: int = 10,
        motion_gate: bool = False,
        motion_threshold: float = 0.0,
    ):
        self.gesture_types = gesture_types
        self.dynamic_entry = dynamic_entry
        self.dynamic_exit = dynamic_exit
        self.dynamic_hold_ms = dynamic_hold_ms
        self.stable_frames = stable_frames
        self.motion_gate = motion_gate
        self.motion_threshold = motion_threshold

        # bufor historii predykcji (label, confidence, timestamp_ms)
        self.prediction_buffer: deque = deque(maxlen=buffer_size)

        # stan aktualnego gestu dynamicznego
        self.current_dynamic: Optional[str] = None
        self.current_dynamic_confidence: float = 0.0
        self.dynamic_start_time_ms: Optional[int] = None
        self.stable_count: int = 0  # licznik kolejnych stabilnych klatek

        # bufor landmarków dla motion gate
        self.landmarks_buffer: deque = deque(maxlen=5)

        logger.debug(
            "GestureManager init: entry=%.2f exit=%.2f hold=%dms stable=%d motion=%s",
            dynamic_entry,
            dynamic_exit,
            dynamic_hold_ms,
            stable_frames,
            motion_gate,
        )

    def process(
        self,
        pred_label: str,
        pred_conf: float,
        landmarks21: Optional[np.ndarray],
        now_ms: int,
    ) -> Optional[GestureResult]:
        """
        Przetwarza predykcję z MLP i zwraca GestureResult jeśli gest dynamiczny jest stabilny.

        Args:
            pred_label: etykieta przewidziana przez MLP (np. "A+")
            pred_conf: confidence predykcji MLP (0-1)
            landmarks21: opcjonalne landmarki (21, 3) dla motion gate
            now_ms: aktualny czas w ms (dla kontroli histerezy)

        Returns:
            GestureResult jeśli gest dynamiczny spełnia warunki stabilności, None w przeciwnym razie
        """
        # dodaj do bufora historii
        self.prediction_buffer.append((pred_label, pred_conf, now_ms))

        # sprawdź typ gestu
        gesture_type = self.gesture_types.get(pred_label, "static")

        # jeśli predykcja to statyczna etykieta, GestureManager nie ingeruje
        if gesture_type != "dynamic":
            # jeśli mieliśmy aktywny dynamiczny gest, a teraz przyszła statyczna - wyzeruj
            if self.current_dynamic is not None:
                logger.debug(
                    "GestureManager: statyczna etykieta '%s', zerowanie stanu dynamicznego '%s'",
                    pred_label,
                    self.current_dynamic,
                )
                self._reset_dynamic_state()
            return None

        # od tego momentu pred_label jest "dynamic"

        # opcjonalnie: sprawdź motion gate
        if self.motion_gate and landmarks21 is not None:
            motion_detected = self._check_motion(landmarks21)
            if not motion_detected:
                logger.debug(
                    "GestureManager: motion gate nie spełniony dla '%s'", pred_label
                )
                self._reset_dynamic_state()
                return None

        # sprawdź warunki wejścia/utrzymania
        if self.current_dynamic is None:
            # brak aktywnego gestu dynamicznego - próba wejścia
            if pred_conf >= self.dynamic_entry:
                # sprawdź stabilność: czy ta sama etykieta powtarza się stable_frames razy?
                if self._is_stable(pred_label, self.stable_frames):
                    # zatwierdź wejście
                    self.current_dynamic = pred_label
                    self.current_dynamic_confidence = pred_conf
                    self.dynamic_start_time_ms = now_ms
                    self.stable_count = 0
                    logger.debug(
                        "GestureManager: zatwierdzono dynamiczny gest '%s' (conf=%.2f)",
                        pred_label,
                        pred_conf,
                    )
                    return GestureResult(
                        name=pred_label, confidence=pred_conf, gesture_type="dynamic"
                    )
                else:
                    # jeszcze nie stabilne
                    logger.debug(
                        "GestureManager: '%s' nie osiągnęło stable_frames=%d",
                        pred_label,
                        self.stable_frames,
                    )
                    return None
            else:
                # confidence za niskie
                return None
        else:
            # mamy aktywny gest dynamiczny
            assert self.dynamic_start_time_ms is not None
            time_held_ms = now_ms - self.dynamic_start_time_ms

            # NAJPIERW: sprawdź czy confidence aktualnego gestu nie spadło poniżej exit
            if self.current_dynamic_confidence < self.dynamic_exit:
                logger.debug(
                    "GestureManager: wyjście z '%s' (conf=%.2f < %.2f)",
                    self.current_dynamic,
                    self.current_dynamic_confidence,
                    self.dynamic_exit,
                )
                self._reset_dynamic_state()
                return None

            if pred_label == self.current_dynamic:
                # ta sama etykieta - aktualizuj confidence
                self.current_dynamic_confidence = pred_conf
                self.stable_count += 1

                # sprawdź czy confidence nie spadło poniżej exit
                if self.current_dynamic_confidence < self.dynamic_exit:
                    logger.debug(
                        "GestureManager: wyjście z '%s' (conf=%.2f < %.2f)",
                        self.current_dynamic,
                        self.current_dynamic_confidence,
                        self.dynamic_exit,
                    )
                    self._reset_dynamic_state()
                    return None

                # zwróć wynik (gest aktywny)
                return GestureResult(
                    name=self.current_dynamic,
                    confidence=self.current_dynamic_confidence,
                    gesture_type="dynamic",
                )
            else:
                # inna etykieta dynamiczna
                # sprawdź histerezę: czy min_hold_ms minęło?
                if time_held_ms < self.dynamic_hold_ms:
                    # za szybka zmiana - ignoruj, trzymaj aktualny
                    logger.debug(
                        "GestureManager: ignorowanie zmiany '%s'->'%s' (hold=%dms < %dms)",
                        self.current_dynamic,
                        pred_label,
                        time_held_ms,
                        self.dynamic_hold_ms,
                    )
                    return GestureResult(
                        name=self.current_dynamic,
                        confidence=self.current_dynamic_confidence,
                        gesture_type="dynamic",
                    )
                elif pred_conf >= self.dynamic_entry:
                    # nowa etykieta z wystarczającym confidence i po upływie hold_ms - zmień
                    logger.debug(
                        "GestureManager: zmiana dynamiczna '%s'->'%s' (conf=%.2f)",
                        self.current_dynamic,
                        pred_label,
                        pred_conf,
                    )
                    self.current_dynamic = pred_label
                    self.current_dynamic_confidence = pred_conf
                    self.dynamic_start_time_ms = now_ms
                    self.stable_count = 0
                    return GestureResult(
                        name=pred_label, confidence=pred_conf, gesture_type="dynamic"
                    )
                else:
                    # nowy label ma za niski conf i nie wyszliśmy przez exit
                    # trzymaj aktualny gest
                    return GestureResult(
                        name=self.current_dynamic,
                        confidence=self.current_dynamic_confidence,
                        gesture_type="dynamic",
                    )

    def _is_stable(self, label: str, count: int) -> bool:
        """
        Sprawdza czy etykieta powtarza się w ostatnich 'count' predykcjach.

        Args:
            label: etykieta do sprawdzenia
            count: liczba ostatnich predykcji do sprawdzenia

        Returns:
            True jeśli ostatnie 'count' predykcji to ta sama etykieta
        """
        if len(self.prediction_buffer) < count:
            return False

        # weź ostatnie 'count' elementów
        recent = list(self.prediction_buffer)[-count:]
        return all(lbl == label for lbl, _, _ in recent)

    def _check_motion(self, landmarks21: np.ndarray) -> bool:
        """
        Sprawdza czy wykryto ruch na podstawie landmarków (prosty motion gate).

        Args:
            landmarks21: landmarki (21, 3)

        Returns:
            True jeśli wykryto ruch >= motion_threshold
        """
        if landmarks21.shape != (21, 3):
            return False

        self.landmarks_buffer.append(landmarks21.copy())

        if len(self.landmarks_buffer) < 2:
            # za mało danych do analizy ruchu
            return True  # przepuść domyślnie

        # oblicz średnie przemieszczenie index_tip (punkt 8) między klatkami
        index_tip_id = 8
        prev_landmarks = self.landmarks_buffer[-2]
        curr_landmarks = self.landmarks_buffer[-1]

        prev_tip = prev_landmarks[index_tip_id]
        curr_tip = curr_landmarks[index_tip_id]

        displacement = float(np.linalg.norm(curr_tip - prev_tip))

        motion_detected = displacement >= self.motion_threshold
        if not motion_detected:
            logger.debug(
                "GestureManager: motion gate - displacement=%.4f < threshold=%.4f",
                displacement,
                self.motion_threshold,
            )

        return motion_detected

    def _reset_dynamic_state(self) -> None:
        """Resetuje stan aktywnego gestu dynamicznego."""
        self.current_dynamic = None
        self.current_dynamic_confidence = 0.0
        self.dynamic_start_time_ms = None
        self.stable_count = 0

    def reset(self) -> None:
        """Resetuje pełny stan menedżera (bufor, aktualny gest)."""
        self.prediction_buffer.clear()
        self.landmarks_buffer.clear()
        self._reset_dynamic_state()
        logger.debug("GestureManager: pełny reset")


__all__ = ["GestureManager", "GestureResult"]
