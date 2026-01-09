from app.gesture_engine.actions import click_action
from app.gesture_engine.core import hooks


def test_reset_hooks_state_clears_click_and_volume():
    # ustawia stan aktywnego clicka i volume, oraz last_gesture_name
    setattr(click_action.handle_click, "active", True)
    # Nowa struktura click_state
    click_action.click_state["gesture_start"] = 123.0
    click_action.click_state["mouse_down_active"] = True
    click_action.click_state["click_executed"] = True

    hooks.last_gesture_name = "volume"
    if hooks.volume_state is not None:
        hooks.volume_state["phase"] = "adjusting"
        hooks.volume_state["_extend_start"] = 1.0

    # wywoluje reset
    hooks.reset_hooks_state()

    # weryfikuje, ze klik nie jest aktywny, a click_state wyzerowane
    # Po reset_hooks_state -> release_click() -> click_state zresetowany
    assert click_action.click_state["gesture_start"] is None
    assert click_action.click_state["mouse_down_active"] is False
    assert click_action.click_state["click_executed"] is False

    # last_gesture_name wyzerowany
    assert hooks.last_gesture_name is None

    # volume w idle (jesli dostepne)
    if hooks.volume_state is not None:
        assert hooks.volume_state["phase"] == "idle"
        assert hooks.volume_state["_extend_start"] is None
