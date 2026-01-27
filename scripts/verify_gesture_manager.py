"""
Quick verification script for GestureManager integration.
Wykonaj: python scripts/verify_gesture_manager.py
"""

import sys
from pathlib import Path

# dodaj główny katalog do ścieżki
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.sign_language.gesture_logic import GestureManager


def test_basic_functionality():
    """Podstawowy test funkcjonalności GestureManager."""
    print("=== Test GestureManager ===")

    gesture_types = {
        "A": "static",
        "A+": "dynamic",
        "B": "static",
        "C+": "dynamic",
    }

    gm = GestureManager(
        gesture_types=gesture_types,
        dynamic_entry=0.75,
        dynamic_exit=0.55,
        dynamic_hold_ms=600,
        stable_frames=3,
    )

    print("✓ GestureManager utworzony")

    # Test 1: statyczna etykieta nie wywołuje ingerencji
    result = gm.process(pred_label="A", pred_conf=0.95, landmarks21=None, now_ms=1000)
    assert result is None, "Statyczna etykieta powinna zwrócić None"
    print("✓ Test 1 passed: statyczne etykiety nie wywołują ingerencji")

    # Test 2: dynamiczna etykieta z niskim confidence
    result = gm.process(pred_label="A+", pred_conf=0.70, landmarks21=None, now_ms=1100)
    assert result is None, "Za niskie confidence powinno zwrócić None"
    print("✓ Test 2 passed: niskie confidence odrzucone")

    # Test 3: budowanie stabilności (stable_frames=3)
    result = gm.process(pred_label="A+", pred_conf=0.80, landmarks21=None, now_ms=1200)
    assert result is None, "Pierwsza klatka - brak stable_frames"

    result = gm.process(pred_label="A+", pred_conf=0.82, landmarks21=None, now_ms=1300)
    assert result is None, "Druga klatka - brak stable_frames"

    result = gm.process(pred_label="A+", pred_conf=0.85, landmarks21=None, now_ms=1400)
    assert result is not None, "Trzecia klatka - powinno zatwierdzić"
    assert result.name == "A+", f"Oczekiwano A+, otrzymano {result.name}"
    assert result.gesture_type == "dynamic"
    print("✓ Test 3 passed: stable_frames działa poprawnie")

    # Test 4: histereza - zmiana przed upływem hold_ms
    result = gm.process(pred_label="C+", pred_conf=0.90, landmarks21=None, now_ms=1600)
    # 1600 - 1400 = 200ms < 600ms (hold_ms) -> powinna zostać A+
    assert result.name == "A+", f"Histereza: oczekiwano A+, otrzymano {result.name}"
    print("✓ Test 4 passed: histereza działa - zmiana zablokowana przed hold_ms")

    # Test 5: reset
    gm.reset()
    assert gm.current_dynamic is None
    assert len(gm.prediction_buffer) == 0
    print("✓ Test 5 passed: reset() działa poprawnie")

    print("\n=== Wszystkie testy GestureManager PASSED ===\n")


def test_translator_integration():
    """Test integracji z translatorem."""
    print("=== Test Integracja Translator + GestureManager ===")

    import json
    import os
    import tempfile

    import numpy as np
    import torch

    from app.sign_language.model import SignLanguageMLP
    from app.sign_language.translator import SignTranslator

    with tempfile.TemporaryDirectory() as td:
        # przygotuj pliki
        model_path = os.path.join(td, "model.pth")
        classes_path = os.path.join(td, "classes.npy")
        labels_dir = os.path.join(td, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        labels_path = os.path.join(labels_dir, "pjm.json")

        # klasy
        classes = np.array(["A", "A+", "B"])
        np.save(classes_path, classes)

        # labels JSON
        labels_config = {
            "classes": ["A", "A+", "B"],
            "gesture_types": {"A": "static", "A+": "dynamic", "B": "static"},
            "sequences": {},
        }
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_config, f)

        # model 189->3 (3 klatki x 63 cechy)
        model = SignLanguageMLP(input_size=189, hidden_size=32, num_classes=3)
        torch.save(model.state_dict(), model_path)

        # translator z enable_dynamic_gestures=True
        translator = SignTranslator(
            model_path=model_path,
            classes_path=classes_path,
            buffer_size=3,
            confidence_entry=0.01,
            enable_dynamic_gestures=True,
            frames_per_sequence=3,
        )

        print("✓ Translator utworzony z enable_dynamic_gestures=True")
        assert (
            translator.gesture_manager is not None
        ), "GestureManager powinien być zainicjalizowany"
        print("✓ GestureManager zainicjalizowany w translatorze")

        # translator z enable_dynamic_gestures=False
        translator_disabled = SignTranslator(
            model_path=model_path,
            classes_path=classes_path,
            buffer_size=3,
            confidence_entry=0.01,
            enable_dynamic_gestures=False,
            frames_per_sequence=3,
        )

        assert (
            translator_disabled.gesture_manager is None
        ), "GestureManager powinien być None"
        print("✓ GestureManager wyłączony gdy enable_dynamic_gestures=False")

        # test reset
        translator.reset()
        if translator.gesture_manager:
            assert len(translator.gesture_manager.prediction_buffer) == 0
        print("✓ Reset translatora resetuje GestureManager")

    print("\n=== Wszystkie testy integracji PASSED ===\n")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_translator_integration()
        print("=" * 50)
        print("SUKCES: Wszystkie testy weryfikacyjne przeszły ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
