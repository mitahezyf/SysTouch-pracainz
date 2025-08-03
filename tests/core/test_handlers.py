import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# importujemy slownik mapujacy gesty na funkcje
from app.core.handlers import gesture_handlers


# czy w slowniku sa wszystkie wymagane gesty
def test_gesture_handlers_keys():
    expected_keys = {"click", "move_mouse", "scroll", "volume", "close_program"}
    assert set(gesture_handlers.keys()) == expected_keys


# czy kazdy handler w slowniku jest funkcja - callable
def test_gesture_handlers_values_are_callable():
    for key, handler in gesture_handlers.items():
        assert callable(handler), f"Handler for '{key}' is not callable"
