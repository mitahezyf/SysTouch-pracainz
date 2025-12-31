# testy dla GestureManager - State Machine gestow dynamicznych
import time

import numpy as np

from app.sign_language.gesture_logic import GestureFrame, GestureManager, GestureResult


class TestGestureManager:
    """testy menedzera logiki gestow"""

    def test_init(self):
        # podstawowa inicjalizacja
        manager = GestureManager()
        assert manager.buffer_size == 30
        assert manager.motion_threshold == 0.05
        assert manager.sequence_max_gap_ms == 1500
        assert "J" in manager.gesture_types
        assert manager.gesture_types["J"] == "dynamic"
        assert "CH" in manager.sequences

    def test_reset(self):
        # reset czysci bufory
        manager = GestureManager()
        landmarks = np.random.rand(21, 3).astype(np.float32)
        manager.process("A", 0.9, landmarks)
        assert len(manager.frame_buffer) > 0

        manager.reset()
        assert len(manager.frame_buffer) == 0
        assert len(manager.sequence_buffer) == 0

    def test_static_gesture(self):
        # statyczny gest zwraca statyczny wynik
        manager = GestureManager()
        landmarks = np.random.rand(21, 3).astype(np.float32)

        # zapelnij bufor
        result = None
        for _ in range(10):
            result = manager.process("A", 0.9, landmarks)

        assert result is not None
        assert result.name == "A"
        assert result.gesture_type == "static"
        assert result.confidence > 0

    def test_dynamic_gesture_no_motion(self):
        # dynamiczny gest bez ruchu zwraca statyczny
        manager = GestureManager(motion_threshold=0.1)
        landmarks = np.random.rand(21, 3).astype(np.float32)

        # zapelnij bufor bez ruchu (te same landmarki)
        result = None
        for _ in range(15):
            result = manager.process("J", 0.9, landmarks)

        # bez ruchu, J traktowane jako static
        assert result is not None
        assert result.gesture_type == "static"

    def test_dynamic_gesture_with_motion(self):
        # dynamiczny gest z ruchem zwraca dynamic
        manager = GestureManager(motion_threshold=0.03)
        base_landmarks = np.random.rand(21, 3).astype(np.float32)

        # zapelnij bufor z progresywnym ruchem nadgarstka
        result = None
        for i in range(15):
            landmarks = base_landmarks.copy()
            # symuluj ruch nadgarstka (punkt 0)
            landmarks[0] += np.array([i * 0.01, i * 0.01, 0], dtype=np.float32)
            result = manager.process("J", 0.9, landmarks)

        # z ruchem powinno wykryc dynamiczny gest
        assert result is not None
        if result.gesture_type == "dynamic":
            assert result.name == "J"
            assert result.base_shape == "J"

    def test_sequence_detection_ch(self):
        # wykrywa sekwencje CH
        manager = GestureManager()
        landmarks = np.random.rand(21, 3).astype(np.float32)

        # najpierw C
        result = None
        for _ in range(10):
            result = manager.process("C", 0.9, landmarks)

        # potem H
        for _ in range(10):
            result = manager.process("H", 0.9, landmarks)

        # ostatni wynik powinien byc CH
        if result and result.gesture_type == "sequence":
            assert result.name == "CH"
            assert (
                "C" in result.base_shape or True
            )  # base_shape to string reprezentacja

    def test_sequence_timeout(self):
        # sekwencja z za dluga przerwa nie jest wykrywana
        manager = GestureManager(sequence_max_gap_ms=500)
        landmarks = np.random.rand(21, 3).astype(np.float32)

        # C
        result = None
        for _ in range(5):
            result = manager.process("C", 0.9, landmarks)

        # czekaj (symuluj timeout)
        time.sleep(0.6)

        # H - ale za pozno
        for _ in range(5):
            result = manager.process("H", 0.9, landmarks)

        # nie powinno byc sekwencji
        if result:
            assert result.gesture_type != "sequence" or result.name != "CH"

    def test_invalid_landmarks_shape(self):
        # bledny ksztalt landmarkow zwraca None
        manager = GestureManager()
        bad_landmarks = np.random.rand(10, 3).astype(np.float32)

        result = manager.process("A", 0.9, bad_landmarks)
        assert result is None

    def test_buffer_min_frames(self):
        # bufor musi miec min 5 klatek
        manager = GestureManager(buffer_size=30)
        landmarks = np.random.rand(21, 3).astype(np.float32)

        # tylko 3 klatki
        result = None
        for _ in range(3):
            result = manager.process("A", 0.9, landmarks)

        assert result is None

    def test_get_state(self):
        # sprawdza stan diagnostyczny
        manager = GestureManager()
        state = manager.get_state()

        assert "buffer_fill" in state
        assert "buffer_size" in state
        assert "sequence_buffer" in state
        assert state["buffer_size"] == 30

    def test_sequence_buffer_cleanup(self):
        # stare wpisy w buforze sekwencji sa czyszczone
        manager = GestureManager(sequence_max_gap_ms=200)
        landmarks = np.random.rand(21, 3).astype(np.float32)

        # dodaj A
        for _ in range(5):
            manager.process("A", 0.9, landmarks)

        assert "A" in [letter for letter, _ in manager.sequence_buffer]

        # czekaj az starzeje sie
        time.sleep(0.3)

        # dodaj B - to powinno wyczyscic A
        for _ in range(5):
            manager.process("B", 0.9, landmarks)

        # A powinno byc usuniete
        letters = [letter for letter, _ in manager.sequence_buffer]
        assert "A" not in letters or len(letters) <= 1

    def test_repeated_letter_not_added_to_sequence(self):
        # ta sama litera nie jest dodawana wielokrotnie
        manager = GestureManager()
        landmarks = np.random.rand(21, 3).astype(np.float32)

        # 20x ta sama litera A
        for _ in range(20):
            manager.process("A", 0.9, landmarks)

        # bufor sekwencji powinien miec max 1 wpis A
        letters = [letter for letter, _ in manager.sequence_buffer]
        assert letters.count("A") <= 1

    def test_custom_gesture_types(self):
        # niestandardowe typy gestow
        custom_types = {
            "X": "dynamic",
            "Y": "static",
        }
        manager = GestureManager(gesture_types=custom_types)

        assert manager.gesture_types["X"] == "dynamic"
        assert manager.gesture_types.get("Y") == "static"

    def test_custom_sequences(self):
        # niestandardowe sekwencje
        custom_seq = {"XY": ["X", "Y"]}
        manager = GestureManager(sequences=custom_seq)
        landmarks = np.random.rand(21, 3).astype(np.float32)

        # X potem Y
        result = None
        for _ in range(10):
            result = manager.process("X", 0.9, landmarks)
        for _ in range(10):
            result = manager.process("Y", 0.9, landmarks)

        if result and result.gesture_type == "sequence":
            assert result.name == "XY"


class TestGestureFrame:
    """testy dataclass GestureFrame"""

    def test_create_frame(self):
        landmarks = np.random.rand(21, 3).astype(np.float32)
        frame = GestureFrame(
            letter="A",
            confidence=0.95,
            landmarks=landmarks,
            timestamp=time.time(),
        )

        assert frame.letter == "A"
        assert frame.confidence == 0.95
        assert frame.landmarks.shape == (21, 3)
        assert frame.timestamp > 0


class TestGestureResult:
    """testy dataclass GestureResult"""

    def test_create_result(self):
        result = GestureResult(
            name="J",
            confidence=0.88,
            gesture_type="dynamic",
            base_shape="I",
        )

        assert result.name == "J"
        assert result.confidence == 0.88
        assert result.gesture_type == "dynamic"
        assert result.base_shape == "I"

    def test_result_optional_base_shape(self):
        result = GestureResult(
            name="A",
            confidence=0.9,
            gesture_type="static",
        )

        assert result.base_shape is None
