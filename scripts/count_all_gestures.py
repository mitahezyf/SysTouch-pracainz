"""Zlicza wszystkie nagrane gesty w data/collected"""

from collections import Counter
from pathlib import Path

collected = Path("data/collected")
gestures: Counter[str] = Counter()

for session in collected.iterdir():
    if not session.is_dir():
        continue
    features_dir = session / "features"
    if not features_dir.exists():
        continue

    for csv_file in features_dir.glob("*.csv"):
        gesture_name = csv_file.name.split("_")[0]
        gestures[gesture_name] += 1

print("Nagrane gesty (liczba plikow CSV):\n")
for gesture, count in sorted(gestures.items()):
    print(f"{gesture:>4}: {count:>3} plikow")

print(f"\nRazem gestow: {len(gestures)}")
print(f"Razem plikow: {sum(gestures.values())}")
