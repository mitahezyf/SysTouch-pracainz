import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from app.gesture_engine.config import DEBUG_MODE
from app.gesture_engine.logger import logger
from app.sign_language.features import FeatureExtractor
from app.sign_language.model import SignLanguageMLP

# sciezki absolutne bazujace na lokalizacji tego pliku
_BASE_DIR = Path(__file__).parent
_DEFAULT_MODEL_PATH = str(_BASE_DIR / "models" / "pjm_model.pth")
_DEFAULT_CLASSES_PATH = str(_BASE_DIR / "models" / "classes.npy")
_DEFAULT_META_PATH = str(_BASE_DIR / "models" / "model_meta.json")
_DEFAULT_LABELS_PATH = str(_BASE_DIR / "labels" / "pjm.json")
_FALLBACK_LEGACY_META_PATH = str(_BASE_DIR / "models" / "pjm_model.json")

# stale dla 3-klatkowego systemu PJM
BLOCK_SIZE = 63  # cechy na klatke
NUM_BLOCKS = 3  # poczatek, srodek, koniec gestu
SEQUENCE_INPUT_SIZE = BLOCK_SIZE * NUM_BLOCKS  # 189 cech


class SignTranslator:
    """
    Translator liter PJM z buforem 3-klatkowym dla gestow ruchomych.

    PJM wymaga 3 klatek (poczatek, srodek, koniec) do klasyfikacji gestu.
    Translator zbiera sekwencje klatek i laczy je w wektor 189D.

    Parametry stabilizacji:
    - sequence_buffer_size: ile sekwencji 3-klatkowych trzymac w buforze
    - min_hold_ms: minimalny czas trzymania litery przed zmiana
    - confidence_entry: prog confidence do wejscia w nowa litere
    - confidence_exit: prog confidence do opuszczenia aktualnej litery
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        classes_path: Optional[str] = None,
        buffer_size: int = 5,  # ile sekwencji 3-klatkowych w buforze
        min_hold_ms: int = 400,
        confidence_entry: float = 0.7,
        confidence_exit: float = 0.5,
        max_history: int = 500,
        enable_dynamic_gestures: bool = True,
        frames_per_sequence: int = 3,  # klatki na sekwencje (poczatek, srodek, koniec)
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
        self.enable_dynamic_gestures = enable_dynamic_gestures
        self.frames_per_sequence = frames_per_sequence

        # bufor na pojedyncze klatki (zbiera 3 klatki przed predykcja)
        self.frame_collector: deque = deque(maxlen=frames_per_sequence)

        # Inicjalizacja ekstraktora cech (mirror lewej wlaczony domyslnie, bez skali)
        from app.sign_language.features import FeatureConfig

        self.feature_extractor = FeatureExtractor(FeatureConfig())

        # wczytanie klas - jawna obsluga bledu
        try:
            self.classes = np.load(classes_path)
        except FileNotFoundError as e:  # pragma: no cover
            raise FileNotFoundError(f"Brak pliku klas: {classes_path}") from e
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Nie mozna wczytac klas z: {classes_path}: {e}") from e

        # wczytanie metadanych modelu (jesli istnieja) - info o usuniętych cechach
        self.zero_var_indices = np.array([], dtype=int)
        self.model_input_size = (
            SEQUENCE_INPUT_SIZE  # domyslny rozmiar 189 (3 bloki x 63)
        )
        meta_path = Path(model_path).parent / "model_meta.json"

        def _load_meta(path: Path) -> None:
            try:
                with open(path, "r") as f:
                    model_meta = json.load(f)

                self.model_input_size = int(
                    model_meta.get("input_size", SEQUENCE_INPUT_SIZE)
                )
                zero_var_list = model_meta.get("zero_var_indices", [])
                if zero_var_list:
                    self.zero_var_indices = np.array(zero_var_list, dtype=int)
                    logger.debug(
                        "Zaladowano info o %d usunietych cechach",
                        len(self.zero_var_indices),
                    )
            except Exception as e:
                logger.warning("Nie mozna wczytac metadanych modelu: %s", e)

        if meta_path.exists():
            _load_meta(meta_path)
        else:
            legacy_meta = Path(_FALLBACK_LEGACY_META_PATH)
            if legacy_meta.exists():
                logger.info("Uzywam fallback meta z %s", legacy_meta)
                _load_meta(legacy_meta)

        # wczytanie metadanych gestow (typy, sekwencje) dla GestureManager
        gesture_types: dict[str, str] = {}
        sequences: dict[str, list[str]] = {}

        labels_path = Path(_DEFAULT_LABELS_PATH)
        if labels_path.exists():
            try:
                has_bom = False
                try:
                    has_bom = labels_path.read_bytes().startswith(b"\xef\xbb\xbf")
                except Exception:
                    has_bom = False

                with open(labels_path, "r", encoding="utf-8-sig") as f:
                    labels_config = json.load(f)
                    gesture_types = labels_config.get("gesture_types", {})
                    sequences = labels_config.get("sequences", {})
                    logger.debug(
                        "Zaladowano metadane gestow: %d typow, %d sekwencji",
                        len(gesture_types),
                        len(sequences),
                    )
                    if has_bom:
                        logger.warning("Wykryto BOM w %s (utf-8-sig)", labels_path)
            except Exception as e:
                logger.warning("Nie mozna wczytac metadanych gestow: %s", e)

        # Wczytanie state_dict aby dynamicznie dopasowac hidden_size (testy moga miec inne niz domyslne 128)
        try:
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Brak pliku modelu: {model_path}") from e
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Nie mozna wczytac modelu z: {model_path}: {e}") from e

        # proba inferencji hidden_size i input_size z pierwszej warstwy
        hidden_size = None
        w0 = state_dict.get("network.0.weight")
        if w0 is not None and hasattr(w0, "shape") and len(w0.shape) == 2:
            hidden_size = int(w0.shape[0])
            # sprawdz input_size z wag (shape[1] to liczba wejsc)
            inferred_input_size = int(w0.shape[1])
            if inferred_input_size != self.model_input_size:
                logger.warning(
                    "Input size z metadanych (%d) != input size z wag (%d), uzywam wag",
                    self.model_input_size,
                    inferred_input_size,
                )
                self.model_input_size = inferred_input_size

        if hidden_size is None:
            hidden_size = 256  # fallback gdy nie znaleziono (bylo 128)

        self.model = SignLanguageMLP(
            input_size=self.model_input_size,
            hidden_size=hidden_size,
            num_classes=len(self.classes),
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

        # bufor klatek (wektor cech)
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

        # inicjalizacja GestureManager (warstwa 2 - logika dynamiczna)
        # UWAGA: modul gesture_logic.py zostal usuniety - funkcjonalnosc wylaczona
        self.gesture_manager: Optional[object] = None
        # kod wylaczony - GestureManager nie istnieje
        logger.info("GestureManager wylaczony (modul nie istnieje)")

        logger.info(
            "SignTranslator zainicjalizowany: buffer=%d, min_hold=%dms, conf_entry=%.2f, conf_exit=%.2f, max_history=%d, input_size=%d",
            buffer_size,
            min_hold_ms,
            confidence_entry,
            confidence_exit,
            max_history,
            self.model_input_size,
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
        self.frame_collector.clear()  # czysc bufor klatek

        if not keep_stats:
            self.letter_stats.clear()
            self.total_detections = 0
            self.session_start_time = time.time()

        # resetuj GestureManager
        if self.gesture_manager:
            self.gesture_manager.reset()  # type: ignore[attr-defined]

        logger.debug("SignTranslator zresetowany (keep_stats=%s)", keep_stats)

    def process_landmarks(
        self, landmarks: np.ndarray, handedness: str | None = None, debug: bool = False
    ) -> Optional[str]:
        """
        Przetwarza surowe landmarki (21x3), ekstrahuje cechy i zbiera sekwencje 3-klatkowa.

        PJM wymaga 3 klatek (poczatek, srodek, koniec) do klasyfikacji gestu.
        Metoda zbiera klatki i wykonuje predykcje gdy ma kompletna sekwencje.

        Args:
            landmarks: np.ndarray (21, 3)
            handedness: "Left" lub "Right"
            debug: czy logowac debug info

        Returns:
            stabilna litera lub None (jesli jeszcze nie zebrano 3 klatek)
        """
        try:
            # wyciagnij 63 cechy z pojedynczej klatki
            features = self.feature_extractor.extract(landmarks, handedness=handedness)

            # dodaj do bufora klatek
            self.frame_collector.append(np.array(features, dtype=np.float32))

            # jesli nie mamy jeszcze 3 klatek, zwroc aktualna litere (lub None)
            if len(self.frame_collector) < self.frames_per_sequence:
                return self.current_letter

            # polacz 3 klatki w jeden wektor 189D
            sequence_features = np.concatenate(list(self.frame_collector), axis=0)

            # warstwa 1: klasyfikator statyczny (PyTorch) na 189D
            static_result = self.process_frame(sequence_features)

            if debug:
                try:
                    vec = np.asarray(sequence_features)
                    logger.debug(
                        "[translator-debug] hand=%s seq_feat:min=%.4f max=%.4f mean=%.4f len=%d",
                        handedness,
                        float(vec.min()),
                        float(vec.max()),
                        float(vec.mean()),
                        len(vec),
                    )
                except Exception:
                    pass

            # warstwa 2: logika dynamiczna (GestureManager)
            if self.gesture_manager and static_result:
                # przekaz do menedzera gestow
                gesture_result = self.gesture_manager.process(  # type: ignore[attr-defined]
                    static_letter=static_result,
                    confidence=self.current_confidence,
                    landmarks=landmarks,
                )

                if gesture_result:
                    # jesli menedzer zwrocil inny wynik (np. sekwencje), uzyj go
                    if gesture_result.name != static_result:
                        logger.debug(
                            "GestureManager nadpisal: %s -> %s (type=%s)",
                            static_result,
                            gesture_result.name,
                            gesture_result.gesture_type,
                        )
                        # aktualizuj statystyki dla nowej litery
                        self._register_detection(gesture_result.name)
                        return str(gesture_result.name)

            return static_result

        except Exception as e:
            logger.error(f"Blad przetwarzania landmarkow: {e}")
            return self.current_letter

    def process_frame(
        self, normalized_landmarks: Sequence[float] | np.ndarray
    ) -> Optional[str]:
        """
        Przetwarza pojedyncza klatke (wektor cech) i zwraca stabilna litere.

        Args:
            normalized_landmarks: wektor cech (dlugosc zgodna z model_input_size)

        Returns:
            aktualna stabilna litera lub None jesli brak stabilizacji
        """
        if len(normalized_landmarks) != self.model_input_size:
            raise ValueError(
                f"niepoprawny rozmiar wektora: {len(normalized_landmarks)} (oczekiwano {self.model_input_size})"
            )

        if np.isnan(normalized_landmarks).any() or np.isinf(normalized_landmarks).any():
            raise ValueError("wektor cech zawiera NaN/Inf")

        # dodaj do bufora
        self.frame_buffer.append(np.array(normalized_landmarks, dtype=np.float32))

        # czekaj az bufor sie zapelni
        if len(self.frame_buffer) < self.buffer_size:
            return self.current_letter

        # smoothing: mediana z bufora (redukcja szumu)
        smoothed = np.median(np.array(self.frame_buffer), axis=0)

        # usun cechy z zerowa wariancja (jesli model byl trenowany bez nich)
        if len(self.zero_var_indices) > 0:
            mask = np.ones(len(smoothed), dtype=bool)
            mask[self.zero_var_indices] = False
            smoothed = smoothed[mask]

        # predykcja na wygladzonym wektorze
        if smoothed.shape[0] != self.model_input_size:
            raise ValueError(
                f"smoothed wektor ma ksztalt {smoothed.shape[0]}, oczekiwano {self.model_input_size}"
            )
        if np.isnan(smoothed).any() or np.isinf(smoothed).any():
            raise ValueError("smoothed wektor zawiera NaN/Inf")

        input_tensor = torch.tensor(smoothed, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        if DEBUG_MODE:
            try:
                top_conf, top_idx = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
                top_pairs = [
                    f"{self.classes[int(i)]}:{float(c):.3f}"
                    for c, i in zip(top_conf[0].tolist(), top_idx[0].tolist())
                ]
                logger.debug("[translator-debug-top5] %s", ", ".join(top_pairs))
            except Exception:
                pass

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
        if DEBUG_MODE:
            logger.debug("[history] Historia wyczyszczona")
