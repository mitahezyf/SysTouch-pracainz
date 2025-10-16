from app.gesture_engine.actions import click_action
from app.gesture_engine.core import hooks


def test_reset_hooks_state_clears_click_and_volume():
    # ustawia stan aktywnego clicka i volume, oraz last_gesture_name
    setattr(click_action.handle_click, "active", True)
    click_action.click_state["start_time"] = 123.0
    click_action.click_state["holding"] = True
    click_action.click_state["mouse_down"] = True
    click_action.click_state["click_sent"] = True
    click_action.click_state["was_active"] = True

    hooks.last_gesture_name = "volume"
    if hooks.volume_state is not None:
        hooks.volume_state["phase"] = "adjusting"
        hooks.volume_state["_extend_start"] = 1.0

    # wywoluje reset
    hooks.reset_hooks_state()

    # weryfikuje, ze klik nie jest aktywny, a click_state wyzerowane
    assert getattr(click_action.handle_click, "active") is False
    assert click_action.click_state["start_time"] is None
    assert click_action.click_state["holding"] is False
    assert click_action.click_state["mouse_down"] is False
    assert click_action.click_state["click_sent"] is False
    assert click_action.click_state["was_active"] is False

    # last_gesture_name wyzerowany
    assert hooks.last_gesture_name is None

    # volume w idle (jesli dostepne)
    if hooks.volume_state is not None:
        assert hooks.volume_state["phase"] == "idle"
        assert hooks.volume_state["_extend_start"] is None
