"""
Testy jednostkowe dla modulu GestureManager (gesture_logic.py).
"""

import numpy as np
import pytest

from app.sign_language.gesture_logic import GestureManager, GestureResult


@pytest.fixture
def gesture_types_sample():
    """Przykladowe typy gestow z pjm.json."""
    return {
        "A": "static",
        "A+": "dynamic",
        "B": "static",
        "C": "static",
        "C+": "dynamic",
        "D": "dynamic",
        "E": "static",
        "F": "dynamic",
    }


@pytest.fixture
def gesture_manager(gesture_types_sample):
    """Domyslny GestureManager z testowymi parametrami."""
    return GestureManager(
        gesture_types=gesture_types_sample,
        dynamic_entry=0.75,
        dynamic_exit=0.55,
        dynamic_hold_ms=600,
        stable_frames=3,
        buffer_size=10,
        motion_gate=False,
        motion_threshold=0.0,
    )


def test_gesture_manager_initialization(gesture_types_sample):
    """Test inicjalizacji GestureManager z metadanymi z pjm.json."""
    gm = GestureManager(
        gesture_types=gesture_types_sample,
        dynamic_entry=0.8,
        dynamic_exit=0.6,
        dynamic_hold_ms=500,
        stable_frames=2,
    )

    assert gm.dynamic_entry == 0.8
    assert gm.dynamic_exit == 0.6
    assert gm.dynamic_hold_ms == 500
    assert gm.stable_frames == 2
    assert gm.current_dynamic is None
    assert len(gm.prediction_buffer) == 0


def test_static_gesture_returns_none(gesture_manager):
    """Test: statyczna etykieta nie wywoluje ingerencji GestureManager."""
    # statyczna etykieta "A" z wysokim confidence
    result = gesture_manager.process(
        pred_label="A", pred_conf=0.95, landmarks21=None, now_ms=1000
    )

    # GestureManager nie ingeruje dla statycznych
    assert result is None
    assert gesture_manager.current_dynamic is None


def test_dynamic_gate_requires_entry_threshold(gesture_manager):
    """Test: gest dynamiczny wymaga przekroczenia progu entry."""
    # dynamiczna etykieta "A+" ale za niskie confidence
    result = gesture_manager.process(
        pred_label="A+", pred_conf=0.70, landmarks21=None, now_ms=1000
    )

    # za niskie confidence - brak akceptacji
    assert result is None
    assert gesture_manager.current_dynamic is None

    # teraz z wystarczajacym confidence, ale jeszcze nie stable_frames
    result = gesture_manager.process(
        pred_label="A+", pred_conf=0.80, landmarks21=None, now_ms=1100
    )
    assert result is None  # jeszcze brak stabilnosci


def test_dynamic_gate_requires_stable_frames(gesture_manager):
    """Test: gest dynamiczny wymaga stable_frames kolejnych stałych predykcji."""
    # symuluj 3 kolejne predykcje "A+" (stable_frames=3)
    # pierwsza
    result = gesture_manager.process(
        pred_label="A+", pred_conf=0.80, landmarks21=None, now_ms=1000
    )
    assert result is None  # jeszcze nie stable

    # druga
    result = gesture_manager.process(
        pred_label="A+", pred_conf=0.82, landmarks21=None, now_ms=1100
    )
    assert result is None  # jeszcze nie stable

    # trzecia - teraz powinna zatwierdzyc
    result = gesture_manager.process(
        pred_label="A+", pred_conf=0.85, landmarks21=None, now_ms=1200
    )
    assert result is not None
    assert isinstance(result, GestureResult)
    assert result.name == "A+"
    assert result.gesture_type == "dynamic"
    assert result.confidence == 0.85
    assert gesture_manager.current_dynamic == "A+"


def test_dynamic_hysteresis_hold_ms(gesture_manager):
    """Test: histereza - dynamiczny gest nie zmienia sie przed uplywem hold_ms."""
    # zatwierdz gest "A+"
    for i in range(3):
        gesture_manager.process(
            pred_label="A+", pred_conf=0.85, landmarks21=None, now_ms=1000 + i * 100
        )

    # sprawdz ze "A+" jest aktywny
    assert gesture_manager.current_dynamic == "A+"

    # teraz probuj zmienic na "C+" przed uplywem hold_ms (600ms)
    result = gesture_manager.process(
        pred_label="C+", pred_conf=0.90, landmarks21=None, now_ms=1400
    )

    # powinna zostac "A+" (ignorowanie zmiany, bo 1400-1200=200ms < 600ms)
    assert result is not None
    assert result.name == "A+"  # wciaz A+, nie C+
    assert gesture_manager.current_dynamic == "A+"

    # po uplywie hold_ms (600ms od start_time=1200) zmiana jest mozliwa
    result = gesture_manager.process(
        pred_label="C+", pred_conf=0.90, landmarks21=None, now_ms=1900
    )

    # teraz powinna sie zmienic (1900-1200=700ms >= 600ms)
    assert result is not None
    assert result.name == "C+"
    assert gesture_manager.current_dynamic == "C+"


def test_dynamic_exit_on_low_confidence(gesture_manager):
    """Test: wyjscie z gestu dynamicznego gdy confidence spada ponizej exit."""
    # zatwierdz gest "D"
    for i in range(3):
        gesture_manager.process(
            pred_label="D", pred_conf=0.85, landmarks21=None, now_ms=1000 + i * 100
        )

    assert gesture_manager.current_dynamic == "D"

    # utrzymuj "D" ale stopniowo obnizaj confidence
    result = gesture_manager.process(
        pred_label="D", pred_conf=0.70, landmarks21=None, now_ms=1300
    )
    assert result is not None
    assert result.name == "D"  # wciaz aktywny (powyżej exit=0.55)

    result = gesture_manager.process(
        pred_label="D", pred_conf=0.60, landmarks21=None, now_ms=1400
    )
    assert result is not None
    assert result.name == "D"  # wciaz aktywny

    # teraz inna etykieta z niskim confidence gdy aktualny D ma niskie conf
    # aktualizuj wewnetrzny confidence "D" na bardzo niski
    gesture_manager.current_dynamic_confidence = 0.50  # < exit (0.55)

    result = gesture_manager.process(
        pred_label="F", pred_conf=0.80, landmarks21=None, now_ms=1500
    )

    # powinno wyjsc ze stanu "D" (bo D.conf < exit) i nie zatwierdzic "F" (brak stable_frames)
    assert result is None
    assert gesture_manager.current_dynamic is None


def test_reset_clears_state(gesture_manager):
    """Test: reset() czysci caly stan GestureManager."""
    # zatwierdz gest "A+"
    for i in range(3):
        gesture_manager.process(
            pred_label="A+", pred_conf=0.85, landmarks21=None, now_ms=1000 + i * 100
        )

    assert gesture_manager.current_dynamic == "A+"
    assert len(gesture_manager.prediction_buffer) > 0

    # resetuj
    gesture_manager.reset()

    assert gesture_manager.current_dynamic is None
    assert gesture_manager.current_dynamic_confidence == 0.0
    assert gesture_manager.dynamic_start_time_ms is None
    assert gesture_manager.stable_count == 0
    assert len(gesture_manager.prediction_buffer) == 0
    assert len(gesture_manager.landmarks_buffer) == 0


def test_motion_gate_disabled_by_default(gesture_types_sample):
    """Test: motion_gate=False przepuszcza wszystkie gesty bez analizy ruchu."""
    gm = GestureManager(
        gesture_types=gesture_types_sample,
        motion_gate=False,  # wylaczony
        motion_threshold=100.0,  # wysoki prog, ale motion_gate=False
    )

    # zatwierdz gest bez landmarkow (motion_gate nie dziala)
    for i in range(3):
        _ = gm.process(
            pred_label="A+", pred_conf=0.85, landmarks21=None, now_ms=1000 + i * 100
        )

    # powinno sie udac pomimo braku landmarkow
    assert gm.current_dynamic == "A+"


def test_motion_gate_enabled_requires_motion(gesture_types_sample):
    """Test: motion_gate=True wymaga wykrycia ruchu w landmarkach."""
    gm = GestureManager(
        gesture_types=gesture_types_sample,
        motion_gate=True,
        motion_threshold=0.01,  # niski prog dla testu
        stable_frames=2,
    )

    # przygotuj landmarki bez ruchu (identyczne)
    static_landmarks = np.random.randn(21, 3).astype(np.float32)

    # pierwsza klatka (brak poprzedniej - przepusc)
    result = gm.process(
        pred_label="A+",
        pred_conf=0.85,
        landmarks21=static_landmarks.copy(),
        now_ms=1000,
    )

    # druga klatka (identyczne landmarki - brak ruchu)
    result = gm.process(
        pred_label="A+",
        pred_conf=0.85,
        landmarks21=static_landmarks.copy(),
        now_ms=1100,
    )

    # powinno odrzucic z powodu braku ruchu
    assert result is None
    assert gm.current_dynamic is None

    # teraz z ruchem (zmien index_tip)
    moving_landmarks = static_landmarks.copy()
    moving_landmarks[8] += np.array([0.1, 0.1, 0.1])  # przemieszczenie index_tip

    result = gm.process(
        pred_label="A+", pred_conf=0.85, landmarks21=moving_landmarks, now_ms=1200
    )

    # teraz jest ruch - powinno przepuscic (ale jeszcze moze nie stable_frames)
    # kontynuuj z ruchem
    moving_landmarks2 = moving_landmarks.copy()
    moving_landmarks2[8] += np.array([0.1, 0.1, 0.1])

    result = gm.process(
        pred_label="A+", pred_conf=0.85, landmarks21=moving_landmarks2, now_ms=1300
    )

    # po 2 klatkach z ruchem (stable_frames=2) powinno zatwierdzic
    assert result is not None
    assert result.name == "A+"


def test_mixed_static_dynamic_sequence(gesture_manager):
    """Test: sekwencja statyczne -> dynamiczne -> statyczne."""
    # statyczna "A"
    result = gesture_manager.process(
        pred_label="A", pred_conf=0.90, landmarks21=None, now_ms=1000
    )
    assert result is None  # statyczna nie wywoluje ingerencji

    # przejscie na dynamiczna "A+"
    for i in range(3):
        result = gesture_manager.process(
            pred_label="A+", pred_conf=0.85, landmarks21=None, now_ms=1100 + i * 100
        )

    # A+ powinna byc zatwierdzona
    assert result is not None
    assert result.name == "A+"
    assert gesture_manager.current_dynamic == "A+"

    # powrot do statycznej "B" - powinno wyzerowac stan dynamiczny
    result = gesture_manager.process(
        pred_label="B", pred_conf=0.90, landmarks21=None, now_ms=1500
    )

    assert result is None  # statyczna nie wywoluje ingerencji
    assert gesture_manager.current_dynamic is None  # stan wyzerowany
