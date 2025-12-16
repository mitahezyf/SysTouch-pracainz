import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from app.gesture_engine.logger import logger
from app.sign_language.model import SignLanguageMLP

# sciezki absolutne bazujace na lokalizacji tego pliku
_BASE_DIR = Path(__file__).parent
_DEFAULT_MODEL_PATH = str(_BASE_DIR / "models" / "pjm_model.pth")
_DEFAULT_CLASSES_PATH = str(_BASE_DIR / "models" / "classes.npy")


class SignTranslator:
    """
    Translator liter PJM z buforem, smoothingiem i histereza.

    Parametry stabilizacji:
    - buffer_size: rozmiar bufora klatek (T=7)
    - min_hold_ms: minimalny czas trzymania litery przed zmiana
    - confidence_entry: prog confidence do wejscia w nowa litere
    - confidence_exit: prog confidence do opuszczenia aktualnej litery
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        classes_path: Optional[str] = None,
        buffer_size: int = 7,
        min_hold_ms: int = 400,
        confidence_entry: float = 0.7,
        confidence_exit: float = 0.5,
        max_history: int = 500,
    ):
        # uzywaj sciezek absolutnych jako domyslnych
        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH
        if classes_path is None:
            classes_path = _DEFAULT_CLASSES_PATH

        self.device = torch.device("cpu")  # CPU wystarczy
        self.buffer_size = buffer_size
        self.min_hold_ms = min_hold_ms
        self.confidence_entry = confidence_entry
        self.confidence_exit = confidence_exit

        # wczytanie klas - jawna obsluga bledu
        try:
            self.classes = np.load(classes_path)
        except FileNotFoundError as e:  # pragma: no cover
            raise FileNotFoundError(f"Brak pliku klas: {classes_path}") from e
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Nie mozna wczytac klas z: {classes_path}: {e}") from e

        # Wczytanie state_dict aby dynamicznie dopasowac hidden_size (testy moga miec inne niz domyslne 128)
        try:
            state_dict = torch.load(model_path, map_location=self.device)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Brak pliku modelu: {model_path}") from e
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Nie mozna wczytac modelu z: {model_path}: {e}") from e

        # proba inferencji hidden_size z pierwszej warstwy
        hidden_size = None
        w0 = state_dict.get("network.0.weight")
        if w0 is not None and hasattr(w0, "shape") and len(w0.shape) == 2:
            hidden_size = int(w0.shape[0])
        if hidden_size is None:
            hidden_size = 128  # fallback gdy nie znaleziono

        self.model = SignLanguageMLP(
            input_size=63, hidden_size=hidden_size, num_classes=len(self.classes)
        )

        # Proba strict load, a gdy brak kluczy -> strict=False + ostrzezenie
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # ponowna proba z strict=False aby dopuscic brak czesci wag (fallback)
            missing_sig = (
                "Missing key(s)" in str(e)
                or "Unexpected key(s)" in str(e)
                or "size mismatch" in str(e)
            )
            if missing_sig:
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise
        self.model.eval()

        # bufor klatek (wektor 63D)
        self.frame_buffer: deque = deque(maxlen=buffer_size)

        # stan translatora
        self.current_letter: Optional[str] = None
        self.current_confidence: float = 0.0
        self.letter_start_time: Optional[float] = None

        # licznik statystyk wykrytych liter (dziala w tle)
        self.letter_stats: Counter = Counter()  # liczba wykryc kazdej litery
        self.total_detections: int = 0  # calkowita liczba stabilnych wykryc
        self.session_start_time: float = time.time()  # czas rozpoczecia sesji
        self.last_confirmed_letter: Optional[str] = None  # ostatnia potwierdzona litera

        # historia wykrytych liter (sekwencja do wyswietlenia w UI)
        self.letter_history: list[str] = []
        self.max_history: int = max_history

        logger.info(
            "SignTranslator zainicjalizowany: buffer=%d, min_hold=%dms, conf_entry=%.2f, conf_exit=%.2f, max_history=%d",
            buffer_size,
            min_hold_ms,
            confidence_entry,
            confidence_exit,
            max_history,
        )

    def reset(self, keep_stats: bool = False) -> None:
        """
        Resetuje stan translatora (bufor, aktualna litere).

        Args:
            keep_stats: jesli True, zachowuje statystyki liter (liczniki), ale czysci historie
        """
        self.frame_buffer.clear()
        self.current_letter = None
        self.current_confidence = 0.0
        self.letter_start_time = None
        self.last_confirmed_letter = None
        self.letter_history.clear()  # historia zawsze czyszczona

        if not keep_stats:
            self.letter_stats.clear()
            self.total_detections = 0
            self.session_start_time = time.time()

        logger.debug("SignTranslator zresetowany (keep_stats=%s)", keep_stats)

    def process_frame(self, normalized_landmarks: Sequence[float]) -> Optional[str]:
        """
        Przetwarza pojedyncza klatke i zwraca stabilna litere.

        Args:
            normalized_landmarks: wektor 63D znormalizowanych landmarkow

        Returns:
            aktualna stabilna litera lub None jesli brak stabilizacji
        """
        if len(normalized_landmarks) != 63:
            logger.warning(
                "Oczekiwano wektora 63D, otrzymano %d", len(normalized_landmarks)
            )
            return self.current_letter

        # dodaj do bufora
        self.frame_buffer.append(np.array(normalized_landmarks, dtype=np.float32))

        # czekaj az bufor sie zapelni
        if len(self.frame_buffer) < self.buffer_size:
            return self.current_letter

        # smoothing: mediana z bufora (redukcja szumu)
        smoothed = np.median(np.array(self.frame_buffer), axis=0)

        # predykcja na wygladzonym wektorze
        input_tensor = torch.tensor(smoothed, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        predicted_letter = self.classes[predicted_idx.item()]
        predicted_conf = confidence.item()

        # logika histerezy i min_hold
        current_time = time.time()

        if self.current_letter is None:
            # brak aktualnej litery - wejscie w nowa jesli przekroczy prog entry
            if predicted_conf >= self.confidence_entry:
                self.current_letter = predicted_letter
                self.current_confidence = predicted_conf
                self.letter_start_time = current_time
                self._register_detection(predicted_letter)  # zlicz w statystykach
                logger.debug(
                    "Nowa litera: %s (conf=%.2f)", predicted_letter, predicted_conf
                )
        else:
            # mamy aktualna litere
            # letter_start_time jest zawsze ustawione gdy current_letter nie jest None
            assert self.letter_start_time is not None
            time_held = (current_time - self.letter_start_time) * 1000  # ms

            if predicted_letter == self.current_letter:
                # ta sama litera - aktualizuj confidence
                self.current_confidence = predicted_conf
            else:
                # inna litera
                if time_held < self.min_hold_ms:
                    # za krotko trzymane - ignoruj zmiane
                    pass
                elif predicted_conf >= self.confidence_entry:
                    # nowa litera z wystarczajacym confidence - zmien
                    self.current_letter = predicted_letter
                    self.current_confidence = predicted_conf
                    self.letter_start_time = current_time
                    self._register_detection(predicted_letter)  # zlicz w statystykach
                    logger.debug(
                        "Zmiana litery: %s (conf=%.2f)",
                        predicted_letter,
                        predicted_conf,
                    )
                elif self.current_confidence < self.confidence_exit:
                    # aktualna litera spadla ponizej exit - wyjdz ze stanu
                    logger.debug(
                        "Wyjscie z litery %s (conf=%.2f < %.2f)",
                        self.current_letter,
                        self.current_confidence,
                        self.confidence_exit,
                    )
                    self.current_letter = None
                    self.current_confidence = 0.0
                    self.letter_start_time = None

        return self.current_letter

    def _register_detection(self, letter: str) -> None:
        """
        Rejestruje wykrycie litery w statystykach (wewnetrzna metoda).

        Args:
            letter: wykryta litera
        """
        # zlicz tylko jesli to inna litera niz ostatnia potwierdzona
        # (unikamy wielokrotnego liczenia tej samej litery)
        if letter != self.last_confirmed_letter:
            self.letter_stats[letter] += 1
            self.total_detections += 1
            self.last_confirmed_letter = letter

            # dodaj do historii
            self.letter_history.append(letter)
            # ogranicz dlugosc historii
            if len(self.letter_history) > self.max_history:
                self.letter_history.pop(0)

            logger.debug(
                "[stats] Zarejestrowano: %s (total=%d)", letter, self.total_detections
            )

    def get_state(self) -> dict:
        """Zwraca aktualny stan translatora (dla diagnostyki/GUI)."""
        session_duration_s = time.time() - self.session_start_time

        return {
            "current_letter": self.current_letter,
            "confidence": self.current_confidence,
            "buffer_fill": len(self.frame_buffer),
            "buffer_size": self.buffer_size,
            "time_held_ms": (
                (time.time() - self.letter_start_time) * 1000
                if self.letter_start_time
                else 0
            ),
            "total_detections": self.total_detections,
            "session_duration_s": session_duration_s,
            "detections_per_minute": (
                (self.total_detections / session_duration_s * 60)
                if session_duration_s > 0
                else 0.0
            ),
            "unique_letters": len(self.letter_stats),
        }

    def get_statistics(self) -> dict:
        """
        Zwraca pelne statystyki sesji (do exportu/analizy).

        Returns:
            slownik ze statystykami: liczby wykryc, czasy, rozklady
        """
        session_duration_s = time.time() - self.session_start_time

        return {
            "total_detections": self.total_detections,
            "session_duration_s": session_duration_s,
            "detections_per_minute": (
                (self.total_detections / session_duration_s * 60)
                if session_duration_s > 0
                else 0.0
            ),
            "letter_counts": dict(self.letter_stats),
            "most_common": self.letter_stats.most_common(10),
            "unique_letters": len(self.letter_stats),
            "session_start": self.session_start_time,
        }

    # kompatybilnosc z istniejacym API (stara metoda predict)
    def predict(self, normalized_landmarks: Sequence[float]) -> Optional[str]:
        """Alias dla process_frame (kompatybilnosc wsteczna)."""
        return self.process_frame(normalized_landmarks)

    def get_history(self, format_groups: bool = True) -> str:
        """
        Zwraca historie wykrytych liter jako string.

        Args:
            format_groups: jesli True, dodaje spacje co 5 liter dla czytelnosci

        Returns:
            string z historia liter
        """
        if not format_groups:
            return "".join(self.letter_history)

        # grupuj po 5 znakow ze spacjami
        history_str = "".join(self.letter_history)
        grouped = " ".join(
            [history_str[i : i + 5] for i in range(0, len(history_str), 5)]
        )
        return grouped

    def clear_history(self) -> None:
        """Czysci historie wykrytych liter."""
        self.letter_history.clear()
        logger.debug("[history] Historia wyczyszczona")
