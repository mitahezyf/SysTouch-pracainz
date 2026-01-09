"""
Test logiki flip handedness
"""

# Symulacja:
# 1. Pokażesz prawą rękę przed kamerą
# 2. Flip obrazu: cv2.flip(frame, 1) - odbicie lustrzane w poziomie
# 3. Po flipie: prawa ręka jest TERAZ po lewej stronie obrazu
# 4. MediaPipe: widzi rękę po lewej stronie → oznacza jako "Left"
# 5. Flip etykiety: Left → Right
# 6. required_hand = Right
# 7. Porównanie: Right == Right → PASUJE!

# ALE! Może MediaPipe rozpoznaje handedness nie po pozycji, ale po orientacji dłoni?

# Test 1: Bez flip
# - Prawa ręka przed kamerą
# - MediaPipe widzi normalny obraz
# - MediaPipe oznacza jako... Left? (bo kamera działa jak lustro)

# Test 2: Z flip
# - Prawa ręka przed kamerą
# - Po flip: obraz jest odbity
# - MediaPipe widzi odbity obraz
# - MediaPipe oznacza jako... Right? (bo flip cofnął efekt lustra kamery)

print("Test logiki:")
print("1. Kamera bez flip: prawa ręka → MediaPipe widzi Left (efekt lustra)")
print("2. Kamera z flip: prawa ręka → flip → MediaPipe widzi Right")
print("")
print("Więc w trybie flip:")
print("- MediaPipe zwraca: Right")
print("- Flip etykiety: Right → Left")
print("- required_hand: Right")
print("- Porównanie: Left != Right → ODRZUCONE!")
print("")
print("To jest błąd! Nie powinniśmy flipować etykiety gdy używamy flip mode!")
