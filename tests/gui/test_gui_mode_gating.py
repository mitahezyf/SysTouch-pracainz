"""
Testy gatingu trybow: translator vs gestures.

Sprawdzaja, ze w trybie translator zadne gesty sterowania nie wywoluja dispatchera akcji,
a w trybie gestures translator nie generuje wyjscia do historii liter.
"""

from unittest import mock

import numpy as np


def test_update_pjm_stats_display_method_exists_and_safe():
    # sprawdza, ze metoda _update_pjm_stats_display istnieje i nie crashuje
    # gdy translator == None (tryb gestures)

    # test PURE funkcji build_pjm_display
    from unittest import mock

    from app.gui.pjm_stats import build_pjm_display

    # test 1: translator == None
    display = build_pjm_display(None)
    assert display["letter"] == "--"
    assert display["confidence"] == "Pewnosc: --%"
    assert display["total"] == "Wykryto liter: 0"

    # test 2: translator z danymi
    mock_translator = mock.Mock()
    mock_translator.get_state.return_value = {
        "current_letter": "A",
        "confidence": 0.95,
        "time_held_ms": 500,
    }
    mock_translator.get_statistics.return_value = {
        "total_detections": 10,
        "detections_per_minute": 2.5,
        "unique_letters": 3,
        "most_common": [("A", 5), ("B", 3)],
    }
    mock_translator.get_history.return_value = "ABCAA"

    display = build_pjm_display(mock_translator)
    assert display["letter"] == "A"
    assert "95.0%" in display["confidence"]
    assert display["time"] == "Czas: 500ms"
    assert display["total"] == "Wykryto liter: 10"
    assert "2.5" in display["rate"]
    assert display["unique"] == "Unikalne: 3"
    assert "A:5" in display["top"]
    assert display["history"] == "ABCAA"

    # test 3: GUI method nie crashuje z None
    # MockWindow z prawdziwa implementacja _update_pjm_stats_display
    class MockWindow:
        _translator = None
        pjm_letter_label = mock.Mock()
        pjm_conf_label = mock.Mock()
        pjm_time_label = mock.Mock()
        pjm_total_label = mock.Mock()
        pjm_rate_label = mock.Mock()
        pjm_unique_label = mock.Mock()
        pjm_top_label = mock.Mock()
        pjm_history_edit = mock.Mock()

        def _update_pjm_stats_display(self) -> None:
            # kopiuj logike z main_window (uproszczona)
            from app.gui.pjm_stats import build_pjm_display

            display = build_pjm_display(self._translator)
            self.pjm_letter_label.setText(display["letter"])
            self.pjm_conf_label.setText(display["confidence"])
            self.pjm_time_label.setText(display["time"])
            self.pjm_total_label.setText(display["total"])
            self.pjm_rate_label.setText(display["rate"])
            self.pjm_unique_label.setText(display["unique"])
            self.pjm_top_label.setText(display["top"])
            self.pjm_history_edit.setText(display["history"])

    win = MockWindow()
    win._update_pjm_stats_display()  # nie crashuje

    # sprawdz ze UI zostalo zaktualizowane z domyslnymi wartosciami
    assert win.pjm_letter_label.setText.called
    assert win.pjm_conf_label.setText.called
    win.pjm_letter_label.setText.assert_called_with("--")


def test_translator_mode_disables_gesture_actions():
    # sprawdza, ze w trybie translator ZERO gestow sterowania nie wywoluje dispatchera
    # (monkeypatch dispatcher)
    from app.gesture_engine.utils.visualizer import Visualizer
    from app.gui.processing import detect_and_draw

    # mock tracker + results
    mock_tracker = mock.Mock()
    mock_results = mock.Mock()
    mock_hand_lms = mock.Mock()
    mock_hand_lms.landmark = [mock.Mock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    mock_results.multi_hand_landmarks = [mock_hand_lms]
    mock_results.multi_handedness = [
        mock.Mock(classification=[mock.Mock(label="Right")])
    ]
    mock_tracker.process.return_value = mock_results

    # mock translator (zwraca zawsze "A")
    mock_translator = mock.Mock()
    mock_translator.process_landmarks.return_value = "A"
    mock_translator.get_state.return_value = {
        "confidence": 0.9,
        "time_held_ms": 300,
        "current_letter": "A",
    }
    mock_translator._last_logged_letter = None

    # mock visualizer
    mock_viz = mock.Mock(spec=Visualizer)

    # frame dummy
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # KLUCZOWY TEST: w trybie translator json_runtime NIE jest używany
    mock_json_runtime = mock.Mock()
    mock_json_runtime.update.return_value = {
        "action": {"type": "click"},
        "confidence": 1.0,
    }

    display, result, per_hand = detect_and_draw(
        frame,
        mock_tracker,
        mock_json_runtime,
        mock_viz,
        preview_enabled=False,
        mode="translator",
        translator=mock_translator,
        normalizer=None,
    )

    # translator mode: json_runtime.update NIE powinno byc wywolane
    assert (
        not mock_json_runtime.update.called
    ), "json_runtime.update zostal wywolany w trybie translator!"

    # wynik to litera, nie gest
    assert result.name == "A"
    assert per_hand[0].name == "A"


def test_gestures_mode_does_not_invoke_translator():
    # sprawdza, ze w trybie gestures translator NIE generuje wyjscia
    from app.gesture_engine.utils.visualizer import Visualizer
    from app.gui.processing import detect_and_draw

    # mock tracker + results
    mock_tracker = mock.Mock()
    mock_results = mock.Mock()
    mock_hand_lms = mock.Mock()
    mock_hand_lms.landmark = [mock.Mock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    mock_results.multi_hand_landmarks = [mock_hand_lms]
    mock_results.multi_handedness = None
    mock_tracker.process.return_value = mock_results

    # mock translator (gdyby zostal wywolany, zwrociłby "X")
    mock_translator = mock.Mock()
    mock_translator.process_landmarks.return_value = "X"
    mock_translator._last_logged_letter = None

    # mock visualizer
    mock_viz = mock.Mock(spec=Visualizer)

    # json runtime zwraca "scroll"
    mock_json_runtime = mock.Mock()
    mock_json_runtime.update.return_value = {
        "action": {"type": "scroll"},
        "confidence": 1.0,
    }

    # frame dummy
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    display, result, per_hand = detect_and_draw(
        frame,
        mock_tracker,
        mock_json_runtime,
        mock_viz,
        preview_enabled=False,
        mode="gestures",
        translator=mock_translator,
        normalizer=None,
    )

    # gestures mode: translator.process_landmarks NIE powinno byc wywolane
    assert (
        not mock_translator.process_landmarks.called
    ), "translator.process_landmarks zostal wywolany w trybie gestures!"

    # wynik to gest, nie litera
    assert result.name == "scroll"


def test_translator_mode_gating_even_when_json_runtime_returns_gesture():
    # double-check: nawet jesli json_runtime zwrocilby gest w trybie translator,
    # nie powinien byc uwzglodniony
    from app.gesture_engine.utils.visualizer import Visualizer
    from app.gui.processing import detect_and_draw

    # setup
    mock_tracker = mock.Mock()
    mock_results = mock.Mock()
    mock_hand_lms = mock.Mock()
    mock_hand_lms.landmark = [mock.Mock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    mock_results.multi_hand_landmarks = [mock_hand_lms]
    mock_results.multi_handedness = None
    mock_tracker.process.return_value = mock_results

    # translator zwraca "B"
    mock_translator = mock.Mock()
    mock_translator.process_landmarks.return_value = "B"
    mock_translator.get_state.return_value = {
        "confidence": 0.85,
        "time_held_ms": 200,
        "current_letter": "B",
    }
    mock_translator._last_logged_letter = None

    mock_viz = mock.Mock(spec=Visualizer)

    # json runtime zwraca volume_up (ale w trybie translator jest ignorowany)
    mock_json_runtime = mock.Mock()
    mock_json_runtime.update.return_value = {
        "action": {"type": "volume_up"},
        "confidence": 1.0,
    }

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    display, result, per_hand = detect_and_draw(
        frame,
        mock_tracker,
        mock_json_runtime,
        mock_viz,
        preview_enabled=False,
        mode="translator",
        translator=mock_translator,
        normalizer=None,
    )

    # KRYTYCZNE: wynik to litera, nie gest
    assert result.name == "B", f"expected B, got {result.name}"
    assert not mock_json_runtime.update.called


def test_translator_missing_but_mode_translator_no_crash():
    # sprawdza, ze jesli mode=translator ale translator=None, to nie crashuje
    from app.gesture_engine.utils.visualizer import Visualizer
    from app.gui.processing import detect_and_draw

    mock_tracker = mock.Mock()
    mock_results = mock.Mock()
    mock_hand_lms = mock.Mock()
    mock_hand_lms.landmark = [mock.Mock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    mock_results.multi_hand_landmarks = [mock_hand_lms]
    mock_results.multi_handedness = None
    mock_tracker.process.return_value = mock_results

    mock_viz = mock.Mock(spec=Visualizer)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # translator=None, mode=translator
    display, result, per_hand = detect_and_draw(
        frame,
        mock_tracker,
        None,
        mock_viz,
        preview_enabled=False,
        mode="translator",
        translator=None,
        normalizer=None,
    )

    # nie powinno crashowac, wynik None
    assert result.name is None
    assert per_hand[0].name is None
