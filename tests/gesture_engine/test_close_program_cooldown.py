# -*- coding: utf-8 -*-
"""Test cooldown mechanizmu dla close_program gestu."""


def test_close_program_cooldown_prevents_repeated_execution():
    """Sprawdza ze close_program wykonuje sie tylko raz dopoki gest jest trzymany."""
    from app.gesture_engine.actions.close_program_action import (
        _close_program_state,
        handle_close_program,
        reset_close_program_cooldown,
    )

    # Reset stanu przed testem
    reset_close_program_cooldown()

    # Symuluj landmarks i frame_shape (nie sa uzywane w logice cooldown)
    fake_landmarks = None
    fake_frame_shape = (480, 640, 3)

    # Pierwsze wywolanie - powinno wykonac akcje
    initial_state = _close_program_state.copy()
    assert not initial_state["last_executed"]
    assert not initial_state["cooldown_active"]

    handle_close_program(fake_landmarks, fake_frame_shape)

    # Po pierwszym wywolaniu cooldown powinien byc aktywny
    assert _close_program_state["last_executed"]
    assert _close_program_state["cooldown_active"]

    # Drugie wywolanie (gest wciaz trzymany) - NIE powinno wykonac akcji
    handle_close_program(fake_landmarks, fake_frame_shape)

    # Stan powinien pozostac bez zmian
    assert _close_program_state["last_executed"]
    assert _close_program_state["cooldown_active"]

    # Reset cooldown (symulacja zakonczenia gestu)
    reset_close_program_cooldown()

    # Po resecie powinno byc mozliwe ponowne wykonanie
    assert not _close_program_state["last_executed"]
    assert not _close_program_state["cooldown_active"]

    # Trzecie wywolanie (nowy gest) - powinno wykonac akcje
    handle_close_program(fake_landmarks, fake_frame_shape)

    assert _close_program_state["last_executed"]
    assert _close_program_state["cooldown_active"]


def test_close_program_reset_via_hooks():
    """Sprawdza ze hooks.py poprawnie resetuje cooldown close_program."""
    from app.gesture_engine.actions.close_program_action import (
        _close_program_state,
        reset_close_program_cooldown,
    )
    from app.gesture_engine.core.hooks import handle_gesture_start_hook

    # Reset stanu
    reset_close_program_cooldown()

    # Ustaw cooldown jako aktywny (symulacja wykonanej akcji)
    _close_program_state["last_executed"] = True
    _close_program_state["cooldown_active"] = True

    # Symuluj zmiane gestu z close_program na None (zakonczenie gestu)
    # Najpierw ustaw last_gesture_name na close_program
    from app.gesture_engine.core import hooks

    hooks.last_gesture_name = "close_program"

    # Wywolaj hook z gesture_name=None (zakonczenie gestu)
    handle_gesture_start_hook(None, None, None)

    # Cooldown powinien byc zresetowany
    assert not _close_program_state["last_executed"]
    assert not _close_program_state["cooldown_active"]
