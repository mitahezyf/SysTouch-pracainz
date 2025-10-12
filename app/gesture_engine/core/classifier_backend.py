# backend klasyfikatora gestow
# - opcjonalna zaleznosc pytorch (lazy import)
# - interfejs: load(), predict(), close()
#
# wejscie:
# - sekwencja T ramek landmarkow 21x3 lub wektor 63
#
# wyjscie:
# - dict: label, confidence, probs, meta
#
# szkic bez implementacji ciezkich zaleznosci
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:  # opcjonalne
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

LandmarksFrame = Union[List[Tuple[float, float, float]], List[List[float]]]
SequenceTensor = Sequence[LandmarksFrame]


@dataclass
class Prediction:
    label: Optional[str]
    confidence: float
    probs: Optional[List[float]] = None
    meta: Optional[Dict[str, Any]] = None


class ClassifierBackend:
    # abstrakcyjny backend (pytorch/onnx/itp.)
    # import bezpieczny bez zaleznosci; wywolanie moze rzucic RuntimeError

    def __init__(self, device: Optional[str] = None, time_window: int = 10) -> None:
        self.device = device or ("cuda" if _TORCH_AVAILABLE else "cpu")
        self.time_window = time_window
        self._loaded = False

    def load(self, model_path: str) -> None:
        # ladowanie modelu; po sukcesie ustawia _loaded
        raise NotImplementedError

    def predict(self, sequence: SequenceTensor) -> Prediction:
        # predykcja dla sekwencji; przy braku okna moze zwrocic neutral/None
        raise NotImplementedError

    def close(self) -> None:
        # zwolnienie zasobow (opcjonalne)
        return None


class TorchGRUBackend(ClassifierBackend):
    # szkic backendu w oparciu o pytorch GRU
    # import bezpieczny; load/predict rzuci blad gdy torch niedostepny

    def __init__(self, device: Optional[str] = None, time_window: int = 10) -> None:
        super().__init__(device=device, time_window=time_window)
        self._model = None

    def _ensure_torch(self) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch niedostepny; zainstaluj torch albo uzyj innego backendu"
            )

    def load(self, model_path: str) -> None:
        self._ensure_torch()
        # TODO: docelowo ladowanie architektury i wag
        # self._model = MyGRU(...).to(self.device)
        # self._model.load_state_dict(torch.load(model_path, map_location=self.device))
        # self._model.eval()
        self._loaded = True

    def predict(self, sequence: SequenceTensor) -> Prediction:
        if not self._loaded:
            raise RuntimeError("model niezaladowany; wywolaj load(path)")
        self._ensure_torch()
        # placeholder: neutral do czasu implementacji
        return Prediction(
            label=None, confidence=0.0, probs=None, meta={"reason": "not_implemented"}
        )
