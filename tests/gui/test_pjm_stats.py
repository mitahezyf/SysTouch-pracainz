# test czystej funkcji build_pjm_display bez zaleznosci od Qt
from unittest import mock

from app.gui.pjm_stats import build_pjm_display


def test_build_pjm_display_with_none_translator():
    # sprawdza ze funkcja zwraca domyslne wartosci gdy translator == None
    display = build_pjm_display(None)

    assert display["letter"] == "--"
    assert display["confidence"] == "Pewnosc: --%"
    assert display["time"] == "Czas: 0ms"
    assert display["total"] == "Wykryto liter: 0"
    assert display["rate"] == "Wykryc/min: 0.0"
    assert display["unique"] == "Unikalne: 0"
    assert display["top"] == "Top 5: --"
    assert display["history"] == ""


def test_build_pjm_display_with_active_translator():
    # sprawdza ze funkcja poprawnie mapuje dane z translatora
    mock_translator = mock.Mock()
    mock_translator.get_state.return_value = {
        "current_letter": "B",
        "confidence": 0.87,
        "time_held_ms": 1200,
    }
    mock_translator.get_statistics.return_value = {
        "total_detections": 42,
        "detections_per_minute": 5.3,
        "unique_letters": 12,
        "most_common": [("B", 10), ("A", 8), ("C", 7), ("D", 5), ("E", 3)],
    }
    mock_translator.get_history.return_value = "ABCBDEB"

    display = build_pjm_display(mock_translator)

    assert display["letter"] == "B"
    assert "87.0%" in display["confidence"]
    assert display["time"] == "Czas: 1200ms"
    assert display["total"] == "Wykryto liter: 42"
    assert "5.3" in display["rate"]
    assert display["unique"] == "Unikalne: 12"
    assert "B:10" in display["top"]
    assert "A:8" in display["top"]
    assert display["history"] == "ABCBDEB"


def test_build_pjm_display_with_unknown_letter():
    # sprawdza ze funkcja wyswietla domyslne wartosci gdy current_letter == "unknown"
    mock_translator = mock.Mock()
    mock_translator.get_state.return_value = {
        "current_letter": "unknown",
        "confidence": 0.4,
        "time_held_ms": 100,
    }
    mock_translator.get_statistics.return_value = {
        "total_detections": 0,
        "detections_per_minute": 0.0,
        "unique_letters": 0,
        "most_common": [],
    }
    mock_translator.get_history.return_value = ""

    display = build_pjm_display(mock_translator)

    assert display["letter"] == "--"
    assert display["confidence"] == "Pewnosc: --%"
    assert display["time"] == "Czas: 100ms"
    assert display["total"] == "Wykryto liter: 0"
    assert display["top"] == "Top 5: --"
    assert display["history"] == ""


def test_build_pjm_display_with_partial_most_common():
    # sprawdza ze funkcja dziala z mniej niz 5 elementami w most_common
    mock_translator = mock.Mock()
    mock_translator.get_state.return_value = {
        "current_letter": "X",
        "confidence": 0.92,
        "time_held_ms": 500,
    }
    mock_translator.get_statistics.return_value = {
        "total_detections": 3,
        "detections_per_minute": 1.5,
        "unique_letters": 2,
        "most_common": [("X", 2), ("Y", 1)],
    }
    mock_translator.get_history.return_value = "XYX"

    display = build_pjm_display(mock_translator)

    assert display["letter"] == "X"
    assert "92.0%" in display["confidence"]
    assert "X:2" in display["top"]
    assert "Y:1" in display["top"]
    assert display["history"] == "XYX"
