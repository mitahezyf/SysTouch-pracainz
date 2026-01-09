"""Analizuje wszystkie pliki CSV w ostatniej sesji"""

import csv
import sys
from pathlib import Path

collected = Path("data/collected")
sessions = sorted(collected.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

if not sessions:
    print("Brak sesji")
    sys.exit(1)

last_session = sessions[0]
features_dir = last_session / "features"

print(f"Sesja: {last_session.name}")
print("Analiza plikow CSV:\n")

csv_files = sorted(features_dir.glob("*.csv"))

total_clips = 0
total_frames = 0
total_with_hand = 0
total_with_features = 0

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_clips += 1
    frames = len(rows)
    with_hand = sum(1 for r in rows if r.get("has_hand") == "1")
    with_features = sum(1 for r in rows if r.get("feat_0", "").strip() != "")

    total_frames += frames
    total_with_hand += with_hand
    total_with_features += with_features

    hand_pct = round(with_hand / frames * 100) if frames > 0 else 0
    feat_pct = round(with_features / frames * 100) if frames > 0 else 0

    status = "✓ OK" if hand_pct >= 70 else ("⚠️ MALO" if hand_pct > 0 else "✗ BRAK")

    print(
        f"{csv_file.name[:40]:40} | has_hand: {with_hand:3}/{frames:3} ({hand_pct:3}%) | features: {feat_pct:3}% | {status}"
    )

print(f"\n{'='*100}")
print("PODSUMOWANIE:")
print(f"  Wszystkich klipow: {total_clips}")
print(f"  Wszystkich klatek: {total_frames}")
print(
    f"  Klatek z has_hand=1: {total_with_hand} ({round(total_with_hand/total_frames*100)}%)"
)
print(
    f"  Klatek z features: {total_with_features} ({round(total_with_features/total_frames*100)}%)"
)

if total_with_hand == total_with_features and total_with_features > 0:
    print("\n✅ SUKCES: Wszystkie klatki z has_hand=1 maja wypelnione features!")
elif total_with_features == 0:
    print("\n❌ PROBLEM: Zadna klatka nie ma features!")
else:
    print("\n⚠️ UWAGA: Niektore klatki z has_hand=1 nie maja features")
