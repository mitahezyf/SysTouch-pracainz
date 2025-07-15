from .click_action import handle_click
from .move_mouse_action import handle_move_mouse
from .scroll_action import handle_scroll

gesture_handlers = {
    "click": handle_click,
    "move_mouse": handle_move_mouse,
    "scroll": handle_scroll,
}