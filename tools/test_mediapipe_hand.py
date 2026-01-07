# szybki test - co MediaPipe widzi dla twojej reki
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from app.gesture_engine.detector.hand_tracker import HandTracker

print("Nacisnij 'q' aby wyjsc")
print("Pokaz PRAWA reke przed kamera i sprawdz co MediaPipe raportuje")
print()

tracker = HandTracker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = tracker.process(rgb)

    if results and results.multi_hand_landmarks:
        hand_lm = results.multi_hand_landmarks[0]

        # handedness
        handedness = "UNKNOWN"
        hand_score = 0.0
        if results.multi_handedness:
            classification = results.multi_handedness[0].classification[0]
            handedness = classification.label
            hand_score = classification.score

        # landmarki - sprawdz pozycje
        lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
        wrist = lms[0]
        index_tip = lms[8]

        # czy palec jest nad nadgarstkiem (Y mniejsze = wyzej na ekranie)?
        finger_up = index_tip[1] < wrist[1]

        info = f"MediaPipe mowi: {handedness} ({hand_score:.0%})"
        info2 = f"Palec {'NAD' if finger_up else 'POD'} nadgarstkiem"

        cv2.putText(
            display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.putText(
            display, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )

        print(
            f"\r{info} | {info2} | wrist_y={wrist[1]:.3f}, index_y={index_tip[1]:.3f}",
            end="",
        )

        # rysuj landmarki
        h, w = display.shape[:2]
        for i, lm in enumerate(hand_lm.landmark):
            x, y = int((1 - lm.x) * w), int(lm.y * h)
            color = (0, 255, 0) if i in [0, 8] else (0, 0, 255)
            cv2.circle(display, (x, y), 4, color, -1)
    else:
        cv2.putText(
            display,
            "Nie wykryto dloni",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Test MediaPipe", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
