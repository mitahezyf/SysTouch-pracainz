# SystemTouch

[![CI](https://img.shields.io/github/actions/workflow/status/mitahezyf/SysTouch-pracainz/ci.yml?branch=main&style=for-the-badge&logo=github)](https://github.com/mitahezyf/SysTouch-pracainz/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/mitahezyf/SysTouch-pracainz?branch=main&style=for-the-badge&logo=codecov)](https://codecov.io/gh/mitahezyf/SysTouch-pracainz)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white) 
[![CodeQL](https://img.shields.io/github/actions/workflow/status/mitahezyf/SysTouch-pracainz/codeql.yml?branch=main&label=CodeQL&style=for-the-badge&logo=github)](https://github.com/mitahezyf/SysTouch-pracainz/actions/workflows/codeql.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=for-the-badge&logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


Nowoczesne sterowanie komputerem za pomoca gestow dloni wykrywanych w kamerze. Projekt dziala lokalnie na Windows i wykorzystuje MediaPipe oraz OpenCV do detekcji dloni, a nastepnie mapuje rozpoznane gesty na akcje systemowe.

- Dokumentacja pipeline: docs/GESTURE_PIPELINE.md
- Mapowanie gestow -> akcje i MVP GUI: docs/GESTURE_MAPPING_AND_GUI.md
- Projektowy TODO (checklista, komentarze): docs/PROJECT_TODO.todo
- Diagram przypadkow uzycia: UseCaseDiagram1.png

## Wymagania
- System: Windows 10/11
- Python: 3.12
- Kamera: min. 720p, 30 FPS

## Zaleznosci
Glowne biblioteki uzywane przez projekt:
- mediapipe - sledzenie dloni/landmarkow
- opencv-python-headless (lub opencv-python) - przetwarzanie obrazu; w CLI okno podgladu przez imshow
- numpy - operacje numeryczne
- scikit-learn - prosty klasyfikator w module trenera
- Tylko Windows: pycaw, comtypes, pywin32, PyAutoGUI (do glosnosci, okien, myszy)
- Dev/GUI: PySide6 (MVP GUI), pytest, ruff, mypy, black, bandit, pre-commit

Zobacz `requirements.txt` i `requirements-dev.txt` dla kompletnej listy i wersji. Jesli nie potrzebujesz okna podgladu OpenCV, mozesz uzyc wariantu `opencv-python-headless`.

## Szybki start (CLI)
Punkt wejsciowy CLI: `app/main.py`.
- Uruchom:
  - `python -m app.main`
  - lub: `python app\main.py`
- Wyjscie: w oknie podgladu nacisnij ESC.

Domyslnie program otwiera okno podgladu (OpenCV HighGUI). Jesli srodowisko nie wspiera GUI OpenCV, aplikacja automatycznie przechodzi w tryb headless. Mozesz tez wylaczyc okno recznie (`SHOW_WINDOW = False` w `app/gesture_engine/config.py`).

## GUI (PySide6)
Dostepne jest lekkie GUI oparte o PySide6, korzystajace z tego samego silnika gestow.

- Instalacja (GUI):
  - `python -m pip install -r requirements-dev.txt`
- Uruchom GUI:
  - `python -m app.gui.ui_app`
- Funkcje MVP:
  - wybor kamery (skanowanie okresowe)
  - start/stop przetwarzania
  - przelacznik „Wykonuj akcje” (bezpiecznie domyslnie wyl.)
  - przelacznik „Pokaz podglad” (rysowanie overlay tylko, gdy wlaczone)
  - podglad gestu, FPS i FrameTime

Uwagi:
- GUI wymaga PySide6; jesli uzywasz `opencv-python-headless`, GUI nadal dziala (podglad rysuje Visualizer na ramce), ale do okien OpenCV potrzebny bylby pelny `opencv-python`.

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

## Gesty i akcje
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
- Lint: `ruff check .`
- Typy: `mypy .`

Raporty coverage w CI laduja do `reports/coverage.xml`.

## CI/CD
- CI: Windows, Python 3.12, pytest+coverage, artefakty JUnit i coverage, Codecov
- CodeQL: analiza bezpieczenstwa na push/PR

Badge u gory pokazuja status CI i pokrycie na galezi `main`.

## Struktura repo (skrot)
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
  - `gui/` - okna, obsluga przetwarzania, modele dla UI
- `tests/` - testy jednostkowe (pytest)
- `docs/` - dokumenty projektowe
- `reports/` - logi aplikacji i raporty testow

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
  - zainstaluj `requirements-dev.txt`, uruchom `python -m app.gui.ui_app`
- Akcje systemowe nie dzialaja:
  - sprawdz zaleznosci Windows-only (pycaw, comtypes, pywin32, PyAutoGUI)
  - upewnij sie, ze okno docelowe ma focus (scroll/close)

---

### Specyfikacja i przypadki uzycia (opis funkcjonalny)

- Sterowanie komputerem za pomoca predefiniowanych gestow
- Rozpoznawanie alfabetu oraz podstawowych slow w jezyku migowym (planowane)
- Definiowanie wlasnych gestow i przypisywanie akcji (moduly trenera)
- Zmiana widocznosci podgladu i prostych opcji w GUI

Diagram przypadkow uzycia: `UseCaseDiagram1.png`
