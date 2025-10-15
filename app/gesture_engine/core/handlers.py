from typing import Any, Callable, Dict

from app.gesture_engine.actions.click_action import handle_click
from app.gesture_engine.actions.close_program_action import handle_close_program
from app.gesture_engine.actions.move_mouse_action import handle_move_mouse
from app.gesture_engine.actions.scroll_action import handle_scroll
from app.gesture_engine.actions.volume_action import handle_volume

# Mapowanie nazw gestów na funkcje-obslugiwacze
# Kontrakt: handler(landmarks, frame_shape) -> None
gesture_handlers: Dict[str, Callable[[Any, Any], None]] = {
    "click": handle_click,
    "move_mouse": handle_move_mouse,
    "scroll": handle_scroll,
    "volume": handle_volume,
    "close_program": handle_close_program,
}
