"""
Analiza czy zebrane CSV nadaja sie do trenowania modelu PJM
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

collected = Path("data/collected")
sessions = sorted(collected.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

if not sessions:
    print("[ERROR] Brak sesji - zbierz dane najpierw!")
    sys.exit(1)

print("=" * 80)
print("ANALIZA DATASETU DO TRENINGU MODELU PJM")
print("=" * 80)
print()

# agreguj wszystkie dane
all_data: dict[str, dict[str, Any]] = defaultdict(
    lambda: {"clips": [], "frames": 0, "frames_ok": 0}
)
total_clips = 0
total_frames = 0
total_frames_ok = 0

print("[FOLDER] Sesje:")
for session in sessions[:10]:  # ostatnie 10 sesji
    features_dir = session / "features"
    if not features_dir.exists():
        continue

    # czytaj session.json
    session_json = session / "session.json"
    if session_json.exists():
        with open(session_json) as f:
            meta = json.load(f)
        mode = meta.get("handedness_mode", "mediapipe")
        mirror = meta.get("mirror_left", True)
        required = meta.get("require_handedness", "")
    else:
        mode = "?"
        mirror = "?"
        required = "?"

    csv_files = list(features_dir.glob("*.csv"))
    session_clips = len(csv_files)
    session_frames = 0
    session_frames_ok = 0

    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        label = rows[0]["label"] if rows else "?"
        frames = len(rows)
        frames_ok = sum(
            1
            for r in rows
            if r.get("has_hand") == "1" and r.get("feat_0", "").strip() != ""
        )

        session_frames += frames
        session_frames_ok += frames_ok

        all_data[label]["clips"].append(csv_file.name)
        all_data[label]["frames"] += frames
        all_data[label]["frames_ok"] += frames_ok

    total_clips += session_clips
    total_frames += session_frames
    total_frames_ok += session_frames_ok

    if session_clips > 0:
        pct = (
            round(session_frames_ok / session_frames * 100) if session_frames > 0 else 0
        )
        print(
            f"  {session.name[:16]:16} | clips: {session_clips:3} | frames: {session_frames_ok:4}/{session_frames:4} ({pct:3}%) | mode: {mode:10} | req: {required:5}"
        )

print()
print("=" * 80)
print("PODSUMOWANIE PER LITERA:")
print("=" * 80)
print()

labels_sorted = sorted(all_data.keys())

for label in labels_sorted:
    data = all_data[label]
    clips_count = len(data["clips"])
    frames_ok = data["frames_ok"]
    frames_total = data["frames"]
    pct = round(frames_ok / frames_total * 100) if frames_total > 0 else 0

    # ocena czy wystarczy do treningu
    if clips_count >= 30:
        status = "[OK] DOSKONALE"
    elif clips_count >= 20:
        status = "[OK] DOBRE"
    elif clips_count >= 10:
        status = "[WARNING] MALO (min 20)"
    else:
        status = "[ERROR] ZA MALO (min 20)"

    print(
        f"{label}: clips={clips_count:3} | frames_ok={frames_ok:5} ({pct:3}%) | {status}"
    )

print()
print("=" * 80)
print("OGÓLNE STATYSTYKI:")
print("=" * 80)
print()
print(f"  Liczba liter: {len(all_data)}")
print(f"  Wszystkich klipów: {total_clips}")
print(f"  Wszystkich klatek: {total_frames}")
print(
    f"  Klatek z features: {total_frames_ok} ({round(total_frames_ok/total_frames*100) if total_frames > 0 else 0}%)"
)
print()

# wymagania do treningu
print("=" * 80)
print("WYMAGANIA DO TRENINGU:")
print("=" * 80)
print()
print("MINIMUM (dla eksperymentu):")
print("  - 10-20 klipów na literę")
print("  - 1 osoba")
print("  - ~200-400 klatek na literę")
print()
print("ZALECANE (dla dobrego modelu):")
print("  - 30-50 klipów na literę")
print("  - 2+ osoby (różne dłonie)")
print("  - ~600-1000 klatek na literę")
print()

# ocena
print("=" * 80)
print("OCENA DATASETU:")
print("=" * 80)
print()

if len(all_data) < 10:
    print(
        "[ERROR] Za mało liter (masz {}, potrzebujesz min 10-15 dla sensownego modelu)".format(
            len(all_data)
        )
    )
    can_train = False
elif any(len(data["clips"]) < 10 for data in all_data.values()):
    print("[WARNING] Niektóre litery mają za mało clipów (min 10 dla eksperymentu)")
    can_train = True
    print("[OK] Możesz spróbować wytrenować, ale jakość będzie niska")
elif any(len(data["clips"]) < 20 for data in all_data.values()):
    print("[WARNING] Niektóre litery mają mało clipów (zalecane 20+)")
    can_train = True
    print("[OK] Możesz wytrenować przyzwoity model")
else:
    print("[OK] Dataset wygląda dobrze!")
    can_train = True
    print("[OK] Możesz wytrenować dobry model")

print()

if total_frames_ok < 1000:
    print(
        "[WARNING] Mało klatek total (masz {}, zalecane 2000+)".format(total_frames_ok)
    )
    print("   Zbierz więcej danych dla lepszej jakości")
elif total_frames_ok < 5000:
    print(
        "[OK] Rozsądna liczba klatek ({}), model powinien się nauczyć podstaw".format(
            total_frames_ok
        )
    )
else:
    print("[OK] Dużo klatek ({}), model powinien być dobry!".format(total_frames_ok))

print()
print("=" * 80)
print("CZY MOŻESZ ZASTĄPIĆ DANE Z KAGGLE?")
print("=" * 80)
print()

if can_train and len(all_data) >= 15 and total_frames_ok >= 2000:
    print("[OK] TAK - masz wystarczająco danych aby całkowicie zastąpić Kaggle")
    print("   Twój model będzie:")
    print("   • Wytrenowany tylko na Twoich gestach")
    print("   • Lepiej rozpoznawał Ciebie i Werkę")
    print("   • Bardziej dostosowany do Twojego środowiska (światło, tło, kamera)")
    print()
    print("   Uruchom trening:")
    print(
        "   python tools\\train_model.py --vectors=TWOJ_VECTORS.csv --output=models/moj_model.pt"
    )
elif can_train and len(all_data) >= 10:
    print("[WARNING] CZĘŚCIOWO - masz dane, ale lepiej zbierz więcej")
    print(f"   Masz {len(all_data)} liter, zalecane 20-26 (pełny alfabet PJM)")
    print(f"   Masz {total_frames_ok} klatek, zalecane 2000+ dla solidnego modelu")
    print()
    print("   Co możesz zrobić:")
    print("   1. Zbierz pozostałe litery ({}->26)".format(len(all_data)))
    print("   2. Dodaj więcej powtórzeń dla istniejących liter (30-50 klipów/litera)")
    print("   3. Spróbuj wytrenować z tym co masz (będzie działać, ale gorzej)")
else:
    print("[ERROR] NIE - zbierz więcej danych")
    print(f"   Masz tylko {len(all_data)} liter, potrzebujesz min 10-15")
    print(f"   Masz tylko {total_clips} klipów total")
    print()
    print("   Musisz zebrać:")
    print("   • Min 10-15 liter")
    print("   • Min 10-20 klipów na literę")
    print("   • Min 1000-2000 klatek total")

print()
print("=" * 80)

if can_train:
    print()
    print("[NEXT] NASTĘPNE KROKI:")
    print()
    print("1. Zbierz więcej liter (jeśli < 20):")
    print("   .\\ZBIERAJ_GESTY_PRAWA_REKA.bat")
    print()
    print("2. Skonsoliduj dane do jednego CSV:")
    print("   python tools\\consolidate_dataset.py")
    print()
    print("3. Wytrenuj model:")
    print("   python tools\\train_model.py --vectors=data/consolidated/vectors.csv")
    print()
