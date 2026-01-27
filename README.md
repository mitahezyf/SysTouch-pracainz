# SystemTouch

[![CI](https://img.shields.io/github/actions/workflow/status/mitahezyf/SysTouch-pracainz/ci.yml?branch=main&style=for-the-badge&logo=github)](https://github.com/mitahezyf/SysTouch-pracainz/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/mitahezyf/SysTouch-pracainz?branch=main&style=for-the-badge&logo=codecov)](https://codecov.io/gh/mitahezyf/SysTouch-pracainz)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/mitahezyf/SysTouch-pracainz/codeql.yml?branch=main&label=CodeQL&style=for-the-badge&logo=github)](https://github.com/mitahezyf/SysTouch-pracainz/actions/workflows/codeql.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=for-the-badge&logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Nowoczesne sterowanie komputerem za pomoca gestow dloni wykrywanych w kamerze oraz **tlumacz polskiego jezyka migowego (PJM)** z rozpoznawaniem 36 liter alfabetu. Projekt dziala lokalnie na Windows i wykorzystuje MediaPipe, OpenCV oraz PyTorch do detekcji dloni i rozpoznawania gestow i znakow PJM.

## Wymagania
- System: Windows 10/11
- Python: 3.12
- Kamera: min. 720p, 30 FPS

## Zaleznosci
Glowne biblioteki uzywane przez projekt:
- mediapipe - sledzenie dloni/landmarkow
- opencv-python-headless (lub opencv-python) - przetwarzanie obrazu; w CLI okno podgladu przez imshow
- numpy - operacje numeryczne
- pytorch - model MLP dla PJM (~40k probek, 85-92% accuracy)
- scikit-learn - metryki i preprocessing
- Tylko Windows: pycaw, comtypes, pywin32, PyAutoGUI (do glosnosci, okien, myszy)
- Dev/GUI: PySide6 (GUI), pytest, ruff, mypy, black, bandit, pre-commit

Zobacz `requirements.txt` i `requirements-dev.txt` dla kompletnej listy i wersji. Jesli nie potrzebujesz okna podgladu OpenCV, mozesz uzyc wariantu `opencv-python-headless`.

## Szybki start

### CLI - Sterowanie gestami
Punkt wejsciowy CLI: `app/main.py`.
```bash
python -m app.main
# lub
python app\main.py
```
- **Wyjscie**: nacisnij ESC w oknie podgladu
- **Przelaczenie trybu PJM**: nacisnij **'t'** (STEROWANIE ↔ TLUMACZ PJM)

Domyslnie program otwiera okno podgladu (OpenCV HighGUI). Jesli srodowisko nie wspiera GUI OpenCV, aplikacja automatycznie przechodzi w tryb headless. Mozesz tez wylaczyc okno recznie (`SHOW_WINDOW = False` w `app/gesture_engine/config.py`).

### GUI - Zalecane (PySide6)
```bash
# Instalacja zaleznosci GUI
python -m pip install -r requirements-dev.txt

# Uruchomienie (opcja 1 - zalecane)
python start.py

# Uruchomienie (opcja 2)
python -m app.gui.ui_app
```

## GUI - Funkcje

### Sterowanie Podstawowe
- ✅ **Wybor kamery** - automatyczne skanowanie co 3s (kamera fizyczna i wirtualna, np. OBS)
- ✅ **Start/Stop** przetwarzania
- ✅ **Wykonuj akcje** - przelacznik aktywacji akcji systemowych (domyslnie wyl.)
- ✅ **Pokaz podglad** - wlacz/wylacz overlay z landmarkami i gestami
- ✅ **Podglad gestu** - aktualnie rozpoznany gest/litera
- ✅ **Metryki** - FPS i FrameTime

### Modul PJM (Jezyk Migowy)
- ✅ **Tryb PJM** - przelacznik: sterowanie gestami ↔ tlumacz alfabetu PJM
- ✅ **Nagraj probke PJM** - okno dialogowe z wyborem litery (A-Z)
  - Wykonaj gest przed kamera, nacisnij **SPACJE** aby zapisac klatke
  - Format: `data/collected/{timestamp}/{LITERA}_{index}.npy`
- ✅ **Pokaz probki** - otwiera folder `data/collected/` w Explorerze Windows
- ✅ **Wytrenuj model** - automatyczna konsolidacja danych + trening PyTorch MLP
  - Pasek postepu z callbackami epok
  - Okno wynikow z dokladnoscia per-klasa
- ✅ **Panel statystyk PJM** (gdy tryb PJM aktywny):
  - Historia rozpoznanych liter (tabela)
  - Eksport historii do CSV
  - Przycisk "Przeładuj model PJM"
  - Czyszczenie historii

### Wizualizacja
- Podglad wideo z kamery (640×480, mirrored selfie view)
- Rysowanie landmarkow MediaPipe (21 punktow na dlon)
- Overlay z nazwa rozpoznanego gestu lub litery PJM
- Status trybu: **"TRYB: STEROWANIE"** lub **"TRYB: TLUMACZ (PJM)"** (lewy dolny rog)

### Ikona Aplikacji
Ikona okna i paska zadan Windows: `SysTouchIco.jpg` (w katalogu glownym projektu).

## Tlumacz Jezyka Migowego (PJM)

Projekt zawiera w pelni funkcjonalny modul rozpoznawania alfabetu polskiego jezyka migowego (PJM).

### Funkcje
- **36 liter alfabetu PJM** (A-Z)
- Model: PyTorch MLP (256 hidden units)
- Dataset: ~40k probek treningowych
- Dokladnosc: **85-92%** (test accuracy)
- Integracja z CLI (klawisz 't') i GUI (przelacznik "Tryb PJM")

### Przygotowanie modelu (jednorazowo)

Jesli model nie jest wytrenowany, wykonaj nastepujace kroki:

```bash
# 1. Preprocessing (jednorazowo, ~30 sekund)
python -m app.sign_language.dataset

# 2. Trening (jednorazowo, ~20 minut CPU)
python -m app.sign_language.trainer --epochs 150
```

### Uzycie w Aplikacji

#### CLI - Przelaczenie trybu
```bash
python -m app.main
# W oknie podgladu nacisnij 't' aby przelaczyc tryb
# "TRYB: STEROWANIE" → "TRYB: TLUMACZ (PJM)"
```

#### GUI - Przelacznik trybu
```bash
python start.py
# Uzyj przelacznika "Tryb PJM" w interfejsie
```

### Nagrywanie Wlasnych Probek (GUI)

1. Uruchom GUI: `python start.py`
2. Kliknij **"Nagraj probke PJM"**
3. Wybierz litere (A-Z) z listy rozwijanej
4. Wykonaj gest przed kamera
5. Nacisnij **SPACJE** aby zapisac klatke
6. Probki zapisuja sie w `data/collected/{timestamp}/`

**Format**: `{LITERA}_{index}.npy` (landmarks MediaPipe 21×3 float32)

#### Przeglądanie nagranych probek
Kliknij **"Pokaz probki"** w GUI - otworzy sie folder `data/collected/` w Explorerze.

### Trening Modelu (GUI)

1. Nagraj probki dla wybranych liter (min. 10-20 probek na litere)
2. Kliknij **"Wytrenuj model"**
3. Automatyczna konsolidacja: `data/collected/` → `data/consolidated/` → `app/sign_language/data/processed/`
4. Trening modelu z paskiem postepu
5. Wyniki: okno dialogowe z dokladnoscia per-klasa

**Alternatywnie (CLI)**:
```bash
# 1. Konsolidacja
python -m app.sign_language.dataset

# 2. Trening (parametry opcjonalne)
python -m app.sign_language.trainer --epochs 150 --lr 0.001 --batch_size 64 --hidden_size 256
```

### Statystyki i Eksport (GUI)

Gdy tryb PJM jest aktywny:
- **Historia rozpoznanych liter** (tabela w GUI)
- **Eksport do CSV**: przycisk "Eksportuj statystyki"
- **Czyszczenie historii**: przycisk "Wyczysc"
- **Przeladowanie modelu**: przycisk "Przeladuj model PJM"

### Struktura modulu sign_language

```
app/sign_language/
├── data/
│   ├── raw/              # PJM-vectors.csv, PJM-points.csv (z Kaggle)
│   └── processed/        # train.npz, val.npz, test.npz (70/15/15%)
├── labels/pjm.json       # 36 liter PJM
├── models/pjm_model.pth  # Wytrenowany model PyTorch
├── dataset.py            # Preprocessing i konsolidacja
├── trainer.py            # Trening CLI
├── training_runner.py    # Pipeline GUI (konsolidacja + trening + callbacks)
├── translator.py         # API inferencji (uzywane w main.py)
├── gesture_logic.py      # GestureManager (gesty dynamiczne)
├── features.py           # Ekstrakcja cech z landmarks
├── normalizer.py         # Normalizacja wspolrzednych
└── README.md             # Quick start guide

data/
├── collected/            # Probki nagrane przez GUI
│   └── {timestamp}/      # Sesja nagrywania (format: 20260127_133045_123456)
│       ├── A_0.npy         # landmarks dla litery A (21×3 float32)
│       ├── A_1.npy
│       ├── B_0.npy
│       └── ...
└── consolidated/         # Po konsolidacji
    ├── vectors.csv       # Wszystkie probki + labels
    └── metadata.json     # Klasy, liczba probek
```

### API (programatyczne uzycie)

```python
from app.sign_language.translator import SignTranslator
import numpy as np

# Inicjalizacja
translator = SignTranslator()

# Predykcja z landmarks MediaPipe (21 punktow × 3 wspolrzedne)
landmarks = np.array([[x, y, z], ... ])  # shape: (21, 3)
letter = translator.process_landmarks(landmarks, handedness="Right")

if letter:
    print(f"Rozpoznano: {letter}")  # np. "A", "B", "C+"
```

### Konfiguracja translatora

Parametry w `translator.py` (`SignTranslator.__init__()`):
- `confidence_threshold` - prog pewnosci predykcji (domyslnie 0.7)
- `enable_dynamic_gestures` - gesty dynamiczne (domyslnie True)
- `dynamic_entry_conf` - prog wejscia dla gestow dynamicznych (0.85)
- `dynamic_exit_conf` - prog wyjscia (0.5)

### Troubleshooting PJM

**Brak modelu?**
```bash
python -m app.sign_language.trainer --epochs 150
```

**Brak train.npz?**
```bash
python -m app.sign_language.dataset
```

**Model zle rozpoznaje?**
- Zwieksz epoki: `--epochs 200`
- Nagraj wiecej probek przez GUI (min. 20 na litere)
- Zmniejsz threshold: `confidence_threshold=0.5` w konstruktorze `SignTranslator`

**GUI nie pokazuje statystyk PJM?**
- Upewnij sie, ze tryb PJM jest wlaczony (przelacznik w GUI)
- Model musi byc zaladowany (`app/sign_language/models/pjm_model.pth`)

**Testy modulu PJM**:
```bash
# Wszystkie testy sign_language (37 testow)
python -m pytest tests/sign_language/ -v

# Z coverage
python -m pytest tests/sign_language/ --cov=app.sign_language
```

## Workflow Zbierania Danych i Treningu

### 1. Nagrywanie Probek (GUI)

```bash
python start.py
```

1. Kliknij **"Nagraj probke PJM"**
2. Wybierz litere z listy (A-Z)
3. Wykonaj gest przed kamera (pokazuje podglad na zywo)
4. **SPACJA** - zapisz klatke (landmarks 21×3 jako .npy)
5. Probki w: `data/collected/{timestamp}/`

### 2. Struktura Danych

```
data/
├── collected/                       # Probki surowe (GUI)
│   ├── 20260127_133045_123456/      # Timestamp sesji
│   │   ├── A_0.npy                  # landmarks (21×3 float32)
│   │   ├── A_1.npy
│   │   ├── B_0.npy
│   │   └── ...
│   └── 20260128_101530_987654/
│       └── ...
├── consolidated/                    # Po konsolidacji (GUI lub dataset.py)
│   ├── vectors.csv                  # Wszystkie probki + labels
│   └── metadata.json                # Klasy, liczba probek

app/sign_language/data/
└── processed/                       # Po preprocessing (dataset.py)
    ├── train.npz                    # 70% probek
    ├── val.npz                      # 15% probek
    └── test.npz                     # 15% probek
```

### 3. Konsolidacja

**Automatyczna** (przez GUI):
- Przycisk **"Wytrenuj model"** automatycznie konsoliduje dane z `data/collected/`

**Reczna** (CLI):
```bash
python -m app.sign_language.dataset
# Wczytuje data/collected/*/*.npy
# Generuje: app/sign_language/data/processed/{train,val,test}.npz
```

### 4. Trening

**GUI** (zalecane):
```bash
python start.py
→ Kliknij "Wytrenuj model"
```

**CLI**:
```bash
python -m app.sign_language.trainer \
  --epochs 150 \
  --lr 0.001 \
  --batch_size 64 \
  --hidden_size 256 \
  --device cpu
```

**Wynik**: `app/sign_language/models/pjm_model.pth`

### 5. Uzycie Modelu

Model automatycznie ladowany przez:
- `app.main` (CLI - klawisz 't' przelacza tryb)
- `app.gui.ui_app` (GUI - przelacznik "Tryb PJM")

## Konfiguracja
Plik: `app/gesture_engine/config.py`. Kluczowe opcje:
- Kamera i obraz:
  - `CAMERA_INDEX` (int lub "video=<NAZWA>")
  - `CAPTURE_WIDTH`, `CAPTURE_HEIGHT` (np. 1280x720)
  - `DISPLAY_WIDTH`, `DISPLAY_HEIGHT` (np. 640x480)
  - `TARGET_CAMERA_FPS`, `PROCESSING_MAX_FPS`
  - `SHOW_WINDOW` (wlacza/wylacza okno imshow w CLI)
  - wymuszenia/opt: `CAMERA_SET_BUFFERSIZE`, `CAMERA_BUFFERSIZE`, `CAMERA_FORCE_MJPG`
- Logowanie i overlay: `LOG_LEVEL`, `DEBUG_MODE`, `SHOW_FPS`, `SHOW_DELAY`
- Progi gestow: `CLICK_THRESHOLD`, `HOLD_THRESHOLD`, `SCROLL_*`, `VOLUME_THRESHOLD`, `GESTURE_CONFIDENCE_THRESHOLD`
- Gesty JSON (opcjonalnie): `USE_JSON_GESTURES`, `JSON_GESTURE_PATHS`

## Gesty Systemowe i Akcje
Mapowanie (zob. `app/gesture_engine/core/handlers.py`):
- `click` - klik/mouseDown/mouseUp (PyAutoGUI)
- `move_mouse` - poruszanie kursorem w watku (smoothing, deadzone)
- `scroll` - przewijanie (Windows user32)
- `volume` - regulacja glosnosci (Pycaw)
- `close_program` - zamykanie aktywnego okna (pywin32)

Detekcja i wizualizacja:
- Detekcja landmarkow: `app/gesture_engine/detector/hand_tracker.py`
- Ladowanie i wykrywanie gestow: `app/gesture_engine/detector/gesture_detector.py`
- Rysowanie overlay: `app/gesture_engine/utils/visualizer.py`
- Przechwytywanie kamery w watku: `app/gesture_engine/utils/video_capture.py`
- Logi: `reports/logs/app.log` (rotacja), poziom wg `LOG_LEVEL`

Gesty z JSON (opcjonalnie):
- Ustaw `USE_JSON_GESTURES = True` i podaj katalogi w `JSON_GESTURE_PATHS`.
- Runtime: `app/gesture_engine/core/gesture_runtime.py` (+ `gesture_loader.py`, `gesture_matcher.py`).
- Wykryte akcje z JSON mapuja sie na istniejace handlery po nazwie akcji (`action.type`).

## Trener gestow (eksperymentalny)
Moduly: `app/gesture_trainer/`
- `calibrator.py` - zapisuje skale dloni do `app/gesture_trainer/data/calibration.json`
- `normalizer.py` - przelicza 21 punktow (x,y,z) na wektor 63D wzgledem nadgarstka i skali
- `recorder.py` - zapisuje probki do `app/gesture_trainer/data/raw_landmarks.json`
- `classifier.py` - KNN (K=3), zapis modelu do `app/gesture_trainer/data/trained_model.pkl`
- `manager.py` - mapowanie gest->akcja w `app/gesture_trainer/data/gesture_action_map.json`

Zbieranie probek (tryb pomocniczy):
- `python -m app.train_gesture`
  - [s] zapisuje aktualna klatke z landmarkami do pliku `.npy`
  - [q] konczy

Uwaga: kolektor `.npy` sluzy do szybkich eksperymentow. Docelowy przeplyw trenera korzysta z `recorder.py` (JSON) i `classifier.py`.

## Testy i kontrole lokalne
- Testy: `python -m pytest -q`
- Testy PJM: `python -m pytest tests/sign_language/ -v` (37 testow)
- Lint: `ruff check .`
- Typy: `mypy .`
- Security: `bandit -r app/ -ll`

Raporty coverage w CI laduja do `reports/coverage.xml` i Codecov.

## CI/CD
- CI: **Windows-latest**, Python 3.12, pytest+coverage, ruff, mypy, bandit, pre-commit, pip-audit
- Coverage: automatyczny upload do Codecov (wymaga CODECOV_TOKEN w GitHub Secrets)
- CodeQL: analiza bezpieczenstwa na push/PR
- Bandit: skanowanie kodu pod katem problemow security (13 Low issues - akceptowalne)

Badge u gory pokazuja status CI, pokrycie testami i security na galezi `main`.

## Struktura repo (skrot)
- `start.py` - punkt wejsciowy GUI (wrapper dla `app.gui.ui_app`)
- `app/` - kod aplikacji
  - `main.py` - punkt wejsciowy CLI (OpenCV imshow)
  - `gui/ui_app.py` - punkt wejsciowy GUI (PySide6)
  - `gesture_engine/` - silnik gestow
    - `actions/` - akcje systemowe (mysz, scroll, glosnosc, zamknij)
    - `detector/` - tracker (MediaPipe), loader detektorow, wykrywanie
    - `core/` - runtime JSON, hooki, handlery
    - `utils/` - visualizer, video_capture, performance, geometry, landmarks
    - `gestures/`, `gestures_json/` - detektory i definicje gestow
  - `gesture_trainer/` - kalibracja, normalizacja, rejestracja probek, klasyfikacja
  - `sign_language/` - modul tłumacza PJM (36 liter, PyTorch MLP)
    - `translator.py` - API inferencji
    - `trainer.py` - trening modelu
    - `dataset.py` - preprocessing i konsolidacja
    - `gesture_logic.py` - GestureManager (gesty dynamiczne)
  - `gui/` - okna, obsluga przetwarzania, modele dla UI
- `data/` - dane uzytkownika (probki PJM)
  - `collected/` - nagrane probki
  - `consolidated/` - skonsolidowane dane
- `tests/` - testy jednostkowe (pytest)
- `diagramy/` - diagramy UML i architektury (Mermaid)
- `reports/` - logi aplikacji i raporty testow
- `SysTouchIco.jpg` - ikona aplikacji

## Troubleshooting (Windows)
- Kamera nie otwiera sie:
  - sprawdz `CAMERA_INDEX` lub uzyj formatu `"video=<NAZWA>"`
  - sprobuj innych backendow (DirectShow/MSMF) lub odznacz w systemie blokady uprawnien kamery
- Brak okna podgladu:
  - OpenCV moze nie wspierac GUI; aplikacja przejdzie w tryb headless
  - ustaw `SHOW_WINDOW = False` aby wymusic tryb bez okna
- Brak MediaPipe / OpenCV:
  - zainstaluj zaleznosci z `requirements.txt`
- GUI (PySide6) nie startuje:
  - zainstaluj `requirements-dev.txt`, uruchom `python start.py` lub `python -m app.gui.ui_app`
- Akcje systemowe nie dzialaja:
  - sprawdz zaleznosci Windows-only (pycaw, comtypes, pywin32, PyAutoGUI)
  - upewnij sie, ze okno docelowe ma focus (scroll/close)
- Model PJM nie laduje sie:
  - sprawdz czy istnieje `app/sign_language/models/pjm_model.pth`
  - jesli nie, uruchom trening: `python -m app.sign_language.trainer --epochs 150`

---

## Specyfikacja Funkcjonalna

### Funkcje Glowne
1. **Sterowanie gestami** - kontrola myszy, scroll, glosnosc, zamykanie okien
2. **Tlumacz PJM** - rozpoznawanie 36 liter alfabetu polskiego jezyka migowego
3. **Zbieranie danych** - nagrywanie probek PJM przez GUI
4. **Trening modeli** - automatyczny pipeline konsolidacji + treningu
5. **Wizualizacja** - podglad kamery z overlay landmarkow i gestow

### Technologia
- MediaPipe dziala bez flipu wejscia (spojne z treningiem PJM)
- Overlay rysowany na klatce zmirrorowanej (selfie view)
- ACTIONS (mysz/glosnosc) mapowane w trybie selfie: lustro osi X dla move_mouse i intuicyjny znak roll dla volume
- Model PJM: PyTorch MLP, ~40k probek, 85-92% accuracy
