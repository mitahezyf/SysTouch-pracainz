# diagnostyka na zywo - pokazuje co model widzi z kamery
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
print("DIAGNOSTYKA NA ZYWO - TRANSLATOR PJM")
print("Nacisnij 'q' aby wyjsc, 's' aby zapisac probke")
print("=" * 60)

# init
translator = SignTranslator(
    buffer_size=3,
    min_hold_ms=200,
    confidence_entry=0.5,
)
tracker = HandTracker()
extractor = FeatureExtractor(FeatureConfig(mirror_left=False))

# wczytaj dane treningowe dla porownania
train_data = np.load("app/sign_language/data/processed/train.npz", allow_pickle=True)
X_train = train_data["X"]
train_mean = X_train.mean(axis=0)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

saved_samples = []

print("\nPokazuj gesty przed kamera...")
print("Model widzi top-5 predykcji z confidence\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # flip dla naturalnego podgladu
    display = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = tracker.process(rgb)

    if results and results.multi_hand_landmarks:
        hand_lm = results.multi_hand_landmarks[0]

        # pobierz handedness
        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        # landmarki do numpy
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32
        )

        # ekstrakcja cech (63D)
        features = extractor.extract(landmarks, handedness=handedness)

        # ZMIANA: powtorz ta sama klatke 3 razy (189D)
        seq_features = np.tile(features, 3)

        # porownaj z danymi treningowymi
        diff = np.abs(seq_features - train_mean)
        max_diff = diff.max()
        mean_diff = diff.mean()

        # predykcja
        input_tensor = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = translator.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            top_conf, top_idx = torch.topk(probs, k=5, dim=1)

        # wypisz top-5
        classes = translator.classes
        top5_str = ", ".join(
            [
                f"{classes[i]}:{c:.0%}"
                for c, i in zip(top_conf[0].tolist(), top_idx[0].tolist())
            ]
        )

        # wyswietl na obrazie
        cv2.putText(
            display,
            f"Hand: {handedness}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display,
            f"Top: {top5_str}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            display,
            f"Diff: mean={mean_diff:.2f}, max={max_diff:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        # wypisz w konsoli
        print(
            f"\rHand={handedness:5s} | {top5_str:50s} | diff_mean={mean_diff:.3f}",
            end="",
        )

        # rysuj landmarki
        h, w = display.shape[:2]
        for lm in hand_lm.landmark:
            x, y = int((1 - lm.x) * w), int(lm.y * h)
            cv2.circle(display, (x, y), 3, (0, 255, 0), -1)
    else:
        cv2.putText(
            display,
            "Nie wykryto dloni",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.imshow("PJM Diagnostyka", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        saved_samples.append(seq_features.copy())
        print(f"\n[SAVED] Probka {len(saved_samples)}")

cap.release()
cv2.destroyAllWindows()

if saved_samples:
    np.save("runtime_samples.npy", np.array(saved_samples))
    print(f"\nZapisano {len(saved_samples)} probek do runtime_samples.npy")
