# diagnostyka na zywo z UI - pokazuje landmarki i predykcje
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

from app.gesture_engine.detector.hand_tracker import HandTracker
from app.sign_language.features import FeatureConfig, FeatureExtractor
from app.sign_language.translator import SignTranslator

print("=" * 60)
print("DIAGNOSTYKA NA ZYWO - pokazuje landmarki i predykcje")
print("Nacisnij 'q' aby wyjsc")
print("=" * 60)

# init
translator = SignTranslator(
    buffer_size=3,
    min_hold_ms=100,
    confidence_entry=0.3,  # niski prog dla diagnostyki
)
tracker = HandTracker()
extractor = FeatureExtractor(FeatureConfig(mirror_left=True))
classes = list(translator.classes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nPokazuj gesty przed kamera...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # flip dla naturalnego podgladu (lustro)
    display = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = tracker.process(rgb)

    prediction_text = "Nie wykryto dloni"
    confidence_text = ""
    handedness_text = ""

    if results and results.multi_hand_landmarks:
        hand_lm = results.multi_hand_landmarks[0]

        # handedness
        handedness = None
        if results.multi_handedness:
            classification = results.multi_handedness[0].classification[0]
            handedness = classification.label
            hand_score = classification.score
            handedness_text = f"MediaPipe: {handedness} ({hand_score:.0%})"

        # landmarki do numpy
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32
        )

        # ekstrakcja cech (63D)
        features = extractor.extract(landmarks, handedness=handedness)

        # sekwencja 189D (powtorz 3x dla uproszczenia)
        seq_features = np.tile(features, 3)

        # predykcja
        input_tensor = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = translator.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            top_conf, top_idx = torch.topk(probs, k=5, dim=1)

        # top-5
        top1_letter = classes[int(top_idx[0, 0].item())]
        top1_conf = top_conf[0, 0].item()

        prediction_text = f"Predykcja: {top1_letter}"
        confidence_text = f"Confidence: {top1_conf:.0%}"

        # top-5 jako string
        top5_parts = []
        for i in range(5):
            letter = classes[int(top_idx[0, i].item())]
            conf = top_conf[0, i].item()
            top5_parts.append(f"{letter}:{conf:.0%}")
        top5_text = "Top-5: " + ", ".join(top5_parts)

        # rysuj landmarki na FLIP obrazie
        h, w = display.shape[:2]
        for i, lm in enumerate(hand_lm.landmark):
            # flip x dla display
            x = int((1 - lm.x) * w)
            y = int(lm.y * h)

            # rozne kolory dla roznych palcow
            if i == 0:  # wrist
                color = (0, 255, 0)  # zielony
                size = 8
            elif i in [4, 8, 12, 16, 20]:  # czubki palcow
                color = (0, 0, 255)  # czerwony
                size = 6
            else:
                color = (255, 0, 0)  # niebieski
                size = 4

            cv2.circle(display, (x, y), size, color, -1)

        # polaczenia miedzy punktami
        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # kciuk
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),  # wskazujacy
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),  # srodkowy
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),  # serdeczny
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),  # maly
            (5, 9),
            (9, 13),
            (13, 17),  # podstawy palcow
        ]
        for start, end in connections:
            x1 = int((1 - hand_lm.landmark[start].x) * w)
            y1 = int(hand_lm.landmark[start].y * h)
            x2 = int((1 - hand_lm.landmark[end].x) * w)
            y2 = int(hand_lm.landmark[end].y * h)
            cv2.line(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # wyswietl top-5 na dole
        cv2.putText(
            display,
            top5_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # duzy tekst predykcji
    cv2.putText(
        display,
        prediction_text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )

    if confidence_text:
        cv2.putText(
            display,
            confidence_text,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    if handedness_text:
        cv2.putText(
            display,
            handedness_text,
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    cv2.imshow("PJM Diagnostyka - nacisnij 'q' aby wyjsc", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nZakonczono.")
