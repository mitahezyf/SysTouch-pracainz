"""
Diagnostyka: sprawdza czy MediaPipe wykrywa reke w nagranych klipach
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.gesture_engine.detector.hand_tracker import HandTracker


def check_video(video_path: Path) -> dict:
    """Sprawdza ile klatek zawiera wykryta reke"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "nie mozna otworzyc video"}

    tracker = HandTracker(max_num_hands=1)

    total = 0
    detected = 0
    left_count = 0
    right_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total += 1

        # test bez flip
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.process(rgb)

        if results and getattr(results, "multi_hand_landmarks", None):
            detected += 1
            if getattr(results, "multi_handedness", None):
                label = results.multi_handedness[0].classification[0].label
                if label == "Left":
                    left_count += 1
                elif label == "Right":
                    right_count += 1

        # test z flip
        flipped = cv2.flip(frame, 1)
        rgb_flip = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        results_flip = tracker.process(rgb_flip)

        has_flip = results_flip and getattr(results_flip, "multi_hand_landmarks", None)

        if detected == 0 and has_flip and total <= 5:
            print(f"  Klatka {total}: bez flip=NO, z flip=YES <- problem!")

    cap.release()

    return {
        "total": total,
        "detected_no_flip": detected,
        "detected_pct": round(detected / total * 100, 1) if total > 0 else 0,
        "left": left_count,
        "right": right_count,
    }


if __name__ == "__main__":
    # sprawdz ostatnia sesje
    collected_dir = Path("data/collected")
    sessions = sorted(
        collected_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not sessions:
        print("Brak sesji")
        sys.exit(1)

    last_session = sessions[0]
    clips_dir = last_session / "clips"

    print(f"Sesja: {last_session.name}")
    print(f"Sprawdzam klipy w: {clips_dir}\n")

    videos = sorted(clips_dir.glob("*.mp4"))[:5]  # pierwsze 5

    for video in videos:
        print(f"{video.name}:")
        stats = check_video(video)

        if "error" in stats:
            print(f"  ERROR: {stats['error']}")
        else:
            print(f"  Klatek total: {stats['total']}")
            print(
                f"  Wykryto (bez flip): {stats['detected_no_flip']} ({stats['detected_pct']}%)"
            )
            print(f"  Left: {stats['left']}, Right: {stats['right']}")

            if stats["detected_pct"] < 50:
                print("  ⚠️ PROBLEM: Malo wykrytych klatek!")
            else:
                print("  ✓ OK")

        print()
