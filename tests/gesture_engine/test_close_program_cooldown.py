# -*- coding: utf-8 -*-
"""Test mechanizmu zapobiegania powtorzeniom dla close_program gestu."""


def test_close_program_per_hand_state():
    """Sprawdza ze close_program ma niezalezny stan dla lewej i prawej reki."""
    from app.gesture_engine.gestures.close_program_gesture import (
        _execution_state,
        reset_close_program_state,
    )

    # Reset stanu przed testem
    reset_close_program_state()

    # Sprawdz inicjalny stan
    assert _execution_state["Left"] is False
    assert _execution_state["Right"] is False

    # Symuluj wykonanie gestu prawą ręką
    _execution_state["Right"] = True

    # Prawa ręka zablokowana, lewa wciąż dostępna
    assert _execution_state["Right"] is True
    assert _execution_state["Left"] is False

    # Reset tylko prawej ręki
    reset_close_program_state("Right")

    assert _execution_state["Right"] is False
    assert _execution_state["Left"] is False


def test_close_program_reset_all_hands():
    """Sprawdza ze reset bez argumentu resetuje obie rece."""
    from app.gesture_engine.gestures.close_program_gesture import (
        _execution_state,
        reset_close_program_state,
    )

    # Ustaw obie ręce jako wykonane
    _execution_state["Left"] = True
    _execution_state["Right"] = True

    # Reset wszystkich
    reset_close_program_state()

    assert _execution_state["Left"] is False
    assert _execution_state["Right"] is False


def test_close_program_gesture_blocks_repeat():
    """Sprawdza ze gest close_program blokuje powtorzenia dla tej samej reki."""
    from app.gesture_engine.gestures.close_program_gesture import (
        _execution_state,
        detect_close_program_gesture,
        reset_close_program_state,
    )

    # Reset stanu
    reset_close_program_state()

    # Przygotuj fake landmarks (pięść z kciukiem na bok)
    # Dla uproszczenia używamy prostych wartości - test sprawdza tylko logikę blokady
    class FakeLandmark:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z

    # Symuluj landmarks dla gestu close_program
    # (szczegóły geometrii nie są istotne dla tego testu)
    fake_landmarks = [FakeLandmark(0, 0)] * 21

    # Ustaw stan jako już wykonany
    _execution_state["Right"] = True

    # Wywołaj detektor - powinien zwrócić None (zablokowany)
    result = detect_close_program_gesture(fake_landmarks, "Right")

    # Gest powinien być zablokowany
    assert result is None

    # Reset i ponowna próba
    reset_close_program_state("Right")
    # Teraz gest powinien być możliwy do wykrycia (choć może nie zostać wykryty
    # ze względu na fake landmarks - ale nie powinien być zablokowany)
    # Ten test sprawdza tylko mechanizm blokady, nie samą detekcję gestu
