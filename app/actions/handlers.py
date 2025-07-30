# todo dokonczyc
from .click_action import handle_click
from .hooks import register_gesture_start_hook
from .move_mouse_action import handle_move_mouse
from .scroll_action import handle_scroll

gesture_handlers = {
    "click": handle_click,
    "move_mouse": handle_move_mouse,
    "scroll": handle_scroll,
}


def test_scroll_hook(landmarks, frame_shape):
    print("scroll hook wywolany")


register_gesture_start_hook("scroll", test_scroll_hook)
