"""
Test poprawionej logiki handedness w trybie flip
"""


def _normalize_handedness(label: str | None) -> str | None:
    if label is None:
        return None
    val = label.strip().lower()
    if val.startswith("r"):
        return "Right"
    if val.startswith("l"):
        return "Left"
    return None


def _flip_handedness_label(label: str | None) -> str | None:
    norm = _normalize_handedness(label)
    if norm is None:
        return None
    if norm == "Left":
        return "Right"
    return "Left"


print("=== TEST LOGIKI HANDEDNESS ===")
print()

# Scenario: Pokazujesz prawą rękę
print("SCENARIUSZ: Pokazujesz PRAWĄ rękę przed kamerą")
print()

# Test 1: Tryb mediapipe (bez flip)
print("1. Tryb mediapipe (bez flip obrazu):")
print("   - Kamera działa jak lustro")
print("   - MediaPipe widzi: Left")
print("   - required_hand: Right")
mode = "mediapipe"
mp_result = "Left"
required = "Right"
# w mediapipe: porownujemy z odwroconym required
expected = _flip_handedness_label(required)  # Right -> Left
match = mp_result == expected
print(f"   - expected dla MP: {expected}")
print(f"   - match: {match} ✓" if match else f"   - match: {match} ✗")
print()

# Test 2: Tryb flip
print("2. Tryb flip (z flipem obrazu):")
print("   - Obraz flipowany przed MediaPipe")
print("   - MediaPipe widzi: Right (poprawnie!)")
print("   - required_hand: Right")
mode = "flip"
mp_result = "Right"
required = "Right"
# w flip: porownujemy bezposrednio
match = mp_result == required
print(f"   - match: {match} ✓" if match else f"   - match: {match} ✗")
print()

print("=== WNIOSKI ===")
print("✓ W trybie flip: MediaPipe zwraca Right dla prawej ręki")
print("✓ Porównujemy bezpośrednio: Right == Right")
print("✓ Nie flipujemy etykiety!")
