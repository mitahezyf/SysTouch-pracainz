import argparse
import csv
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from app.gesture_engine.utils.video_capture import ThreadedCapture
from app.sign_language.translator import SignTranslator

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zbiera runtime wektory PJM z kamery do CSV"
    )
    parser.add_argument("--out", type=Path, default=Path("runtime_vectors.csv"))
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--max", type=int, default=250)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # nie wymusza env; debug w GUI steruje set_debug_mode
    translator = SignTranslator(debug_mode=False)

    cap = ThreadedCapture()
    try:
        from app.gesture_engine.detector.hand_tracker import HandTracker

        tracker = HandTracker()
    except Exception as e:
        raise RuntimeError(f"nie mozna zainicjalizowac HandTracker: {e}")

    deadline = time.time() + float(args.seconds)
    rows: list[list[float]] = []

    # naglowek zgodny z runtime (63D)
    feat = translator._landmarks_to_vectors(np.zeros((21, 3), dtype=np.float32), None)
    header = ["frame_idx", "handedness", "pred", "conf"] + [
        f"f{i}" for i in range(int(feat.shape[0]))
    ]

    frame_idx = 0
    while time.time() < deadline and len(rows) < int(args.max):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker.process(rgb)
        results = tracker.get_results() if hasattr(tracker, "get_results") else None

        frame_idx += 1

        if not results or not getattr(results, "multi_hand_landmarks", None):
            if frame_idx % 30 == 0:
                logger.info("[capture] frame=%d hand_detected=False", frame_idx)
            continue

        # bierze pierwsza dlon
        hand_landmarks = results.multi_hand_landmarks[0]
        lms_np = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
        )

        handed_label = None
        try:
            if (
                getattr(results, "multi_handedness", None)
                and len(results.multi_handedness) > 0
            ):
                handed_label = results.multi_handedness[0].classification[0].label
        except Exception:
            handed_label = None

        feat = translator._landmarks_to_vectors(lms_np, handed_label)
        pred = translator.process_landmarks(lms_np, handedness=handed_label)
        conf = float(translator._last_confidence)

        row = [float(frame_idx), str(handed_label or ""), str(pred or ""), conf] + [
            float(x) for x in feat.tolist()
        ]
        rows.append(row)

        if len(rows) % 25 == 0:
            logger.info(
                "[capture] saved=%d last_pred=%s conf=%.3f", len(rows), pred, conf
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info("[capture] zapisano %d rekordow do %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
