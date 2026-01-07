# test bez gui - zbiera dane z kamery i wypisuje
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import cv2
import numpy as np

from app.gesture_engine.detector.hand_tracker import HandTracker
from app.sign_language.features import (
    FeatureConfig,
    FeatureExtractor,
)

print("Zbieranie danych z kamery przez 5 sekund...")
print("Pokaz gest A (zacisnięta pięść, paznokcie do przodu)")
print()

tracker = HandTracker()
extractor = FeatureExtractor(FeatureConfig(mirror_left=True))
cap = cv2.VideoCapture(0)

samples = []
start = time.time()

while time.time() - start < 5:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = tracker.process(rgb)

    if results and results.multi_hand_landmarks:
        hand_lm = results.multi_hand_landmarks[0]

        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        lms = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32
        )
        feat = extractor.extract(lms, handedness=handedness)

        samples.append({"handedness": handedness, "features": feat, "landmarks": lms})

        print(f"\rZebrano {len(samples)} probek, hand={handedness}", end="")

cap.release()
print(f"\n\nZebrano {len(samples)} probek")

if samples:
    # analiza
    feats = np.array([s["features"] for s in samples])
    hands = [s["handedness"] for s in samples]

    print(f"\nHandedness: {set(hands)}")
    print(f"Features shape: {feats.shape}")
    print(f"Features mean: {feats.mean():.4f}")
    print(f"Features std: {feats.std():.4f}")
    print(f"Features range: [{feats.min():.4f}, {feats.max():.4f}]")

    # porownaj z danymi treningowymi
    import json

    train = np.load("app/sign_language/data/processed/train.npz", allow_pickle=True)
    X_train = train["X"]
    y_train = train["y"]
    meta = json.loads(str(train["meta"]))
    classes = meta["classes"]

    # srednia cech dla A w treningu (pierwszy blok = 63 cechy)
    a_idx = classes.index("A")
    a_samples = X_train[y_train == a_idx][:, :63]  # pierwszy blok
    a_mean = a_samples.mean(axis=0)

    # porownaj z zebranymi
    runtime_mean = feats.mean(axis=0)
    diff = np.abs(runtime_mean - a_mean)

    print("\nPorownanie z A z datasetu:")
    print(f"  Max roznica: {diff.max():.4f}")
    print(f"  Mean roznica: {diff.mean():.4f}")

    # ktore cechy sa najbardziej rozne?
    worst_idx = np.argsort(diff)[-5:]
    print(f"  Najbardziej rozne cechy: {worst_idx}")
    for idx in worst_idx:
        # dekoduj co to za cecha
        if idx < 3:
            axis = ["x", "y", "z"][idx]
            desc = f"hand_normal {axis}"
        else:
            bone = (idx - 3) // 3
            axis = ["x", "y", "z"][(idx - 3) % 3]
            desc = f"bone_{bone} {axis}"
        print(
            f"    cecha {idx} ({desc}): runtime={runtime_mean[idx]:.3f}, A_train={a_mean[idx]:.3f}"
        )

    # sprawdz co model przewiduje
    import torch

    from app.sign_language.translator import SignTranslator

    translator = SignTranslator()

    # utworz sekwencje 189D (3x ta sama klatka dla uproszczenia)
    seq = np.tile(runtime_mean, 3)
    input_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = translator.model(input_t)
        probs = torch.softmax(out, dim=1)
        top_conf, top_idx = torch.topk(probs, k=5, dim=1)

    print("\nPredykcja modelu na zebranych danych:")
    for c, i in zip(top_conf[0].tolist(), top_idx[0].tolist()):
        print(f"  {classes[i]}: {c:.1%}")
