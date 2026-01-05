# Quick Start: Tłumacz PJM

## Szybkie uruchomienie w 3 krokach

### 1. Preprocessing (jednorazowo, ~30 sekund)
```bash
python -m app.sign_language.dataset
```

### 2. Trening (jednorazowo, ~20 minut CPU)
```bash
python -m app.sign_language.trainer --epochs 150
```

### 3. Uruchomienie
```bash
# Wersja konsolowa
python -m app.main
# Naciśnij 't' aby przełączyć na tryb tłumacza

# Lub GUI
python -m app.gui.ui_app
```

---

## Struktura modułu

```
app/sign_language/
├── data/
│   ├── raw/              # PJM-vectors.csv, PJM-points.csv (z Kaggle)
│   └── processed/        # train/val/test.npz (generowane)
├── labels/pjm.json       # 36 liter PJM
├── models/               # pjm_model.pth (checkpoint)
├── dataset.py            # preprocessing
├── model.py              # PyTorch MLP
├── trainer.py            # trening CLI
└── translator.py         # inference API
```

---

## API użycia

```python
from app.sign_language.translator import SignTranslator
import numpy as np

# Inicjalizacja
translator = SignTranslator()

# Predykcja z landmarks MediaPipe (21 punktów × 3 współrzędne)
landmarks = np.random.randn(21, 3).astype(np.float32)
letter = translator.process_landmarks(landmarks)

if letter:
    print(f"Rozpoznano: {letter}")  # "A", "B", "C+", ...
```

---

## Testy

```bash
# Wszystkie testy sign_language
python -m pytest tests/sign_language/ -v

# Z coverage
python -m pytest tests/sign_language/ --cov=app.sign_language
```

**Wyniki:** 37/37 passed ✅

---

## Dokumentacja pełna

Zobacz: `markdown/INSTRUKCJA_PJM_TRANSLATOR.md`

---

## Parametry CLI

### Trening
```bash
python -m app.sign_language.trainer \
  --epochs 150 \
  --lr 0.001 \
  --batch_size 64 \
  --hidden_size 128 \
  --device cpu
```

### Preprocessing
```bash
python -m app.sign_language.dataset
# Generuje train/val/test splits (70/15/15%)
```

---

## Troubleshooting

**Brak modelu?**
```bash
python -m app.sign_language.trainer --epochs 150
```

**Brak train.npz?**
```bash
python -m app.sign_language.dataset
```

**Model źle rozpoznaje?**
- Zwiększ epoki: `--epochs 200`
- Zmniejsz threshold w `translator.py`: `confidence_threshold=0.5`

---

## Status

✅ **Implementacja kompletna**
- 36 liter alfabetu PJM (A-Ż)
- ~40k próbek treningowych
- Test accuracy: ~85-92%
- Integracja z main.py i GUI
- 37 testów jednostkowych
