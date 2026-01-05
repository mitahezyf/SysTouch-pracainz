import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from app.sign_language.model import SignLanguageMLP

FIXTURE = Path("tests/fixtures/golden_abc.json")


@pytest.mark.parametrize("min_conf", [0.60])  # mozesz podniesc na 0.75+
def test_golden_abc_predictions(min_conf: float):
    if not FIXTURE.exists():
        pytest.skip("Brak golden set fixture (tests/fixtures/golden_abc.json)")

    ckpt_path = os.environ.get("SIGN_MODEL_CKPT", "models/sign_language_best.pt")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        pytest.skip(f"Brak checkpointu modelu: {ckpt_path} (ustaw SIGN_MODEL_CKPT)")

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    samples = payload["samples"]

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    classes = checkpoint["metadata"]["classes"]
    {c: i for i, c in enumerate(classes)}

    model = SignLanguageMLP(input_size=63, hidden_size=256, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    bad = []
    with torch.no_grad():
        for s in samples:
            y_true = s["label"]
            x = torch.tensor(np.array(s["x"], dtype=np.float32)).unsqueeze(0)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            pred_idx = int(np.argmax(probs))
            y_pred = classes[pred_idx]
            conf = float(probs[pred_idx])

            if y_pred != y_true or conf < min_conf:
                bad.append((y_true, y_pred, conf))

    # Jak cos pada, dostaniesz konkretne pomylki
    assert not bad, f"Zle predykcje ({len(bad)}): np. {bad[:10]}"
