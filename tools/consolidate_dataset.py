"""
Konsoliduje wszystkie zebrane sesje do jednego pliku vectors.csv gotowego do treningu
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    collected = Path("data/collected")
    output_dir = Path("data/consolidated")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "vectors.csv"

    print("=" * 80)
    print("KONSOLIDACJA DATASETU")
    print("=" * 80)
    print()

    sessions = sorted(collected.glob("*"), key=lambda p: p.stat().st_mtime)

    if not sessions:
        print("❌ Brak sesji w data/collected")
        return 1

    # agreguj dane
    all_rows = []
    stats = defaultdict(lambda: {"clips": 0, "frames": 0})

    print("Przetwarzam sesje:")
    for session in sessions:
        features_dir = session / "features"
        if not features_dir.exists():
            continue

        # czytaj session.json
        session_json = session / "session.json"
        if session_json.exists():
            with open(session_json) as f:
                meta = json.load(f)
            mode = meta.get("handedness_mode", "mediapipe")
            required = meta.get("require_handedness", "")
        else:
            mode = "?"
            required = "?"

        csv_files = list(features_dir.glob("*.csv"))
        session_clips = 0
        session_frames = 0

        for csv_file in csv_files:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                continue

            label = rows[0]["label"]

            # filtruj tylko klatki z has_hand=1 i features
            valid_rows = [
                r
                for r in rows
                if r.get("has_hand") == "1" and r.get("feat_0", "").strip() != ""
            ]

            if valid_rows:
                # mapuj kolumny aby były kompatybilne z train_model.py
                for row in valid_rows:
                    # wyciagnij performer z clip_id (format: A_Krzysiek_1_timestamp)
                    clip_id = row.get("clip_id", "")
                    parts = clip_id.split("_")
                    performer = parts[1] if len(parts) > 1 else "unknown"

                    # dodaj user_id (z clip_id) i sign_label (z label)
                    row["user_id"] = performer
                    row["sign_label"] = row["label"]

                all_rows.extend(valid_rows)
                stats[label]["clips"] += 1
                stats[label]["frames"] += len(valid_rows)
                session_clips += 1
                session_frames += len(valid_rows)

        if session_clips > 0:
            print(
                f"  {session.name[:20]:20} | clips: {session_clips:3} | frames: {session_frames:4} | mode: {mode:10}"
            )

    if not all_rows:
        print()
        print("❌ Brak danych do konsolidacji!")
        print("   Upewnij się że masz zebrane sesje z wypełnionymi features")
        return 1

    print()
    print("=" * 80)
    print("STATYSTYKI:")
    print("=" * 80)
    print()

    labels_sorted = sorted(stats.keys())
    for label in labels_sorted:
        clips = stats[label]["clips"]
        frames = stats[label]["frames"]
        print(f"  {label}: clips={clips:3} | frames={frames:5}")

    print()
    print(f"  Total liter: {len(stats)}")
    print(f"  Total klipów: {sum(s['clips'] for s in stats.values())}")
    print(f"  Total klatek: {len(all_rows)}")

    # zapisz do CSV
    print()
    print(f"Zapisuję do: {output_csv}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"✓ Zapisano {len(all_rows)} klatek")

    # zapisz metadata
    meta_file = output_dir / "metadata.json"
    metadata = {
        "created": "2026-01-07",
        "total_labels": len(stats),
        "total_clips": sum(s["clips"] for s in stats.values()),
        "total_frames": len(all_rows),
        "labels": {label: dict(s) for label, s in stats.items()},
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Metadata: {meta_file}")

    print()
    print("=" * 80)
    print("✅ KONSOLIDACJA ZAKOŃCZONA")
    print("=" * 80)
    print()
    print("Dataset gotowy do treningu:")
    print(f"  {output_csv}")
    print()

    if len(stats) < 10:
        print("⚠️ Uwaga: Masz tylko {} liter, zalecane 15-26".format(len(stats)))
        print("   Model będzie słabszy, ale możesz spróbować wytrenować")

    print()
    print("Następny krok - trening:")
    print(f"  python tools\\train_model.py --vectors={output_csv}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
