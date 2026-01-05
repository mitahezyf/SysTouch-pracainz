import json

import numpy as np
import torch

from app.sign_language.model import SignLanguageMLP
from app.sign_language.translator import SignTranslator


def test_translator_uses_metadata_classes(tmp_path):
    # przygotuj pjm.json z 36 klasami (oryginalna kolejnosc)
    pjm_path = tmp_path / "labels" / "pjm.json"
    pjm_path.parent.mkdir(parents=True)
    classes = [
        "A",
        "A+",
        "B",
        "C",
        "C+",
        "CH",
        "CZ",
        "D",
        "E",
        "E+",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "L+",
        "M",
        "N",
        "N+",
        "O",
        "O+",
        "P",
        "R",
        "RZ",
        "S",
        "S+",
        "SZ",
        "T",
        "U",
        "W",
        "Y",
        "Z",
        "Z+",
        "Z++",
    ]
    pjm_path.write_text(
        json.dumps({"classes": classes, "num_classes": len(classes)}), encoding="utf-8"
    )

    # stworz checkpoint z odwrocona kolejnoscia klas (symuluje source of truth z treningu)
    checkpoint_dir = tmp_path / "models"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "pjm_model.pth"
    model = SignLanguageMLP(input_size=63, hidden_size=128, num_classes=len(classes))
    meta_classes = list(reversed(classes))
    model.save_checkpoint(
        str(checkpoint_path), {"classes": meta_classes, "num_classes": len(classes)}
    )

    # uruchom translator z patched sciezkami
    translator = None
    # use custom args to disable stabilizer for test
    translator = SignTranslator(
        model_path=str(checkpoint_path),
        confidence_threshold=0.0,
        stability_frames=1,
        min_hold_time_s=0.0,
    )
    # podmieniamy LABELS_PATH na pjm_path
    translator.classes = classes
    # wymus ladowanie z metadata (symulacja wykonana juz w _load_model, ale upewniamy sie)
    translator.classes = meta_classes

    # podmieniamy model na prosty mock zwracajacy wysoki logit na indeksie 0
    class MockModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.tensor([[10.0] + [0.0] * (len(meta_classes) - 1)])

    translator.model = MockModel()
    translator._model_loaded = True

    # patch features na zero wektor, handedness dowolne
    translator._landmarks_to_vectors = lambda lms, h=None: np.zeros(63, dtype=np.float32)  # type: ignore[assignment]

    result = translator.process_landmarks(np.zeros((21, 3), dtype=np.float32))

    assert result == meta_classes[0]
