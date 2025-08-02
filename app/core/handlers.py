from app.actions.click_action import handle_click
from app.actions.close_program_action import handle_close_program
from app.actions.move_mouse_action import handle_move_mouse
from app.actions.scroll_action import handle_scroll
from app.actions.volume_action import handle_volume


gesture_handlers = {
    "click": handle_click,
    "move_mouse": handle_move_mouse,
    "scroll": handle_scroll,
    "volume": handle_volume,
    "close_program": handle_close_program,
}
