# todo dokonczyc
from app.actions.click_action import handle_click
from app.actions.move_mouse_action import handle_move_mouse
from app.actions.scroll_action import handle_scroll
from app.core.hooks import register_gesture_start_hook

gesture_handlers = {
    "click": handle_click,
    "move_mouse": handle_move_mouse,
    "scroll": handle_scroll,
}


def test_scroll_hook(landmarks, frame_shape):
    print("scroll hook wywolany")


register_gesture_start_hook("scroll", test_scroll_hook)
