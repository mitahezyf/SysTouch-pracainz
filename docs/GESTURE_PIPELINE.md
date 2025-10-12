# Plan i pipeline rozpoznawania gestów (MediaPipe + PyTorch + sekwencje)

Data: 2025-10-13
Status: draft v1
Zakres: Windows 10, Python 3.12, MediaPipe (landmarki dłoni), PyTorch (model sekwencyjny), UI do „wyklikiwania” gestów, rozszerzalność o gesty użytkownika.

Spis treści
- 1. Cele i kryteria sukcesu
- 2. Architektura wysokiego poziomu
- 3. Reprezentacje danych i normalizacja
- 4. Gesty statyczne „wyklikiwane” (szablony reguł)
- 5. Gesty dynamiczne (sekwencje/FSM)
- 6. Backend ML (PyTorch, sekwencyjny)
- 7. Gesty użytkownika (few‑shot/prototypy) i rozszerzalność
- 8. Konfiguracja i kontrakty interfejsów
- 9. Pipeline treningowy (dane, augmentacje, metryki)
- 10. Przepływ runtime (end‑to‑end)
- 11. CI/CD i dystrybucja wag
- 12. Ryzyka i mitigacje
- 13. Roadmap (fazy wdrożenia)
- 14. Słowniczek

---

1. Cele i kryteria sukcesu
- Stabilne, niskolatencyjne rozpoznawanie gestów dłoni w czasie rzeczywistym.
- Łatwe dodawanie nowych gestów, w tym własnych (bez pełnego treningu).
- Przewidywalność statycznych pozy (wyklikiwane szablony) i odporność dla ruchów (sekwencje/ML).
- Docelowa latencja: < 50 ms end‑to‑end (MediaPipe + logika + ML) na PC.

2. Architektura wysokiego poziomu
- Detekcja i śledzenie dłoni: MediaPipe → landmarki 21×(x,y,z) per klatka.
- Warstwa cech dłoni:
  - curl (zgięcie) per palec, splay (rozstaw), pinche, orientacja dłoni, handedness.
  - odległości/doty znormalizowane skalą dłoni.
- Dopasowanie gestów:
  - Szablony statyczne („wyklikiwane”) z tolerancją i histerezą.
  - Sekwencje/FSM dla ruchów.
  - Backend ML (PyTorch, GRU) dla trudnych wzorców sekwencyjnych; późna fuzja wyników.
- Mapowanie gest → akcja: konfiguracja z priorytetami i parametrami.

3. Reprezentacje danych i normalizacja
- Landmarki per klatka: [21, 3] w układzie kamery. Bufor sekwencji: [T, 21, 3].
- Normalizacja (hand‑centric):
  - Zero‑centering względem WRIST.
  - Skala: dziel przez rozmiar dłoni (np. dystans WRIST→MIDDLE_MCP lub średnia do MCP).
  - Mirroring: mapuj lewą dłoń do kanonicznej (prawej), by uniezależnić progi.
  - Oś palca: kierunek od MCP do TIP (do projekcji „wyżej/niżej” w układzie dłoni).
- Z (głębokość): ostrożnie; można zmniejszyć wagę lub pominąć w wersji v1.

4. Gesty statyczne „wyklikiwane” (szablony reguł)
- Atomy:
  - curl per palec: extended/curled/any z progami i strefą martwą.
  - pinch: THUMB_TIP↔INDEX_TIP odległość znormalizowana + histereza.
  - porównania „wyżej/niżej” jako projekcje w układzie dłoni (nie w screen‑space y!).
  - orientacja dłoni (palm up/down), splay (rozstaw palców), handedness.
- Stabilizacja: smoothing 3–5 klatek, entry_threshold, exit_threshold, min_hold_frames.
- Przykładowe domyślne progi:
  - curl: extended < 0.3, curled > 0.6; deadzone 0.3–0.6.
  - pinch distance (znormalizowana): entry 0.12, exit 0.15.
- Przykładowy JSON gestu statycznego (schemat poglądowy):
  {
    "name": "pinch_click",
    "namespace": "builtin",
    "type": "static",
    "fingers": {
      "thumb": { "state": "any" },
      "index": { "state": "extended", "extended_thr": 0.3, "curled_thr": 0.6 },
      "middle": { "state": "any" },
      "ring": { "state": "curled" },
      "pinky": { "state": "curled" }
    },
    "predicates": {
      "pinch": { "enabled": true, "entry": 0.12, "exit": 0.15 },
      "orientation": { "palm_up": false }
    },
    "stabilization": { "smooth": 5, "entry": 0.7, "exit": 0.6, "min_hold": 3 },
    "action": { "type": "click", "params": {} },
    "priority": 50
  }

5. Gesty dynamiczne (sekwencje/FSM)
- Definicja jako scenariusz stanów na bazie atomów z pkt. 4.
- Każdy etap ma min/max klatek i warunki przejścia.
- Przykład (klik pinch):
  - Open (index extended, pinch off) → min 3 kl.
  - PinchOn (pinch on) → 1–6 kl.
  - Open (release) → min 2 kl. w horyzoncie ≤ 0.5 s.
- Przykładowy JSON gestu sekwencyjnego (zarys):
  {
    "name": "pinch_click_seq",
    "namespace": "builtin",
    "type": "sequence",
    "states": [
      { "name": "open", "conditions": { "index": "extended", "pinch": false }, "min": 3 },
      { "name": "pinch_on", "conditions": { "pinch": true }, "min": 1, "max": 6 },
      { "name": "open", "conditions": { "pinch": false }, "min": 2 }
    ],
    "timeout_frames": 15,
    "stabilization": { "smooth": 5, "entry": 0.7, "exit": 0.6, "min_hold": 1 },
    "action": { "type": "click", "params": {} },
    "priority": 60
  }

6. Backend ML (PyTorch, sekwencyjny)
- Wejście: [T, 63] (21×3 po normalizacji; opcjonalnie bez z); T domyślnie 10.
- Architektura v1: GRU (hidden 128) → MLP → logits (N klas builtin). Dropout 0.1.
- Wyjście inferencji: { class_id, confidence, probs[N] }.
- Stabilizacja: smoothing 5 predykcji, histereza 0.7/0.6, min_hold 3.
- Rola: trudne sekwencje; late‑fusion z szablonami (bierzemy wynik pewniejszy > progu).
- ONNX (opcjonalnie v2) do lżejszej integracji i CI.

7. Gesty użytkownika (few‑shot/prototypy) i rozszerzalność
- Embedding z warstwy przedostatniej modelu (np. 64–128D).
- Dodanie klasy bez treningu: prototypy = średnie embeddingi z 10–30 sekwencji.
- Decyzja: nearest prototype (kosinus/Euklides) + próg odrzucania i klasa „unknown”.
- Tryb szybki fine‑tune: retrenowanie samego headu na zebranych próbkach (sekundy/minuty).
- Przechowywanie:
  - builtin: w repo/artefaktach (readonly),
  - user: %APPDATA%/YourApp/gestures/ (JSON + .npz prototypów).

8. Konfiguracja i kontrakty interfejsów
- Konfiguracja (np. `gesture_engine/config.py`):
  - ścieżka do modelu `.pt`, device=auto, T, progi i smoothing, histereza, min_hold.
  - ścieżki do rejestrów gestów (builtin/user), polityka priorytetów.
- Kontrakt backenda klasyfikacji:
  - predict(sequence[T,21,3] lub [T,63]) → { label, confidence, probs, meta }.
  - Błędy: brak dłoni/niepełne landmarki → None/neutral.
- Rejestr gestów:
  - lista obiektów (statyczne/sekwencyjne/ML), każdy z: name, namespace, type, conditions/states, stabilization, action, priority.

9. Pipeline treningowy (dane, augmentacje, metryki)
- Zbieranie danych: `gesture_trainer/recorder` (rozszerzyć o sekwencje). Min. 500 sekwencji/klasa builtin.
- Augmentacje w przestrzeni landmarków: skalowanie ±10%, małe rotacje, szum Gauss, sporadyczny drop punktu.
- Trening:
  - Loss: CrossEntropy.
  - Opt: Adam lr=1e‑3, weight_decay=1e‑4.
  - Batch: ~128, Epoki: 30–60, EarlyStopping: patience 7.
  - Walidacja: 10–20% danych; seedy dla powtarzalności.
- Metryki: accuracy/F1 per klasa, confusion matrix, latency inferencji.

10. Przepływ runtime (end‑to‑end)
1) Kamera → MediaPipe → landmarki [21,3].
2) Normalizacja (center, scale, mirror) + wyliczenie cech (curl, pinch, splay, orientacja).
3) Aktualizacja bufora ring [T] i smoothing.
4) Matcher szablonów statycznych → score.
5) FSM sekwencyjny → dopasowanie ruchu.
6) Backend ML (opcjonalnie) → predykcja klasy.
7) Late‑fusion + histereza + min_hold → finalny gest.
8) Mapowanie do akcji + wykonanie (z parametrami).

11. CI/CD i dystrybucja wag
- Wagi modeli poza repo (Git LFS/Release). Pobieranie przy starcie lub cache lokalny.
- CI: import backenda opcjonalny; testy z mockiem; onnxruntime (opcjonalnie) do smoke‑testu.
- Raporty: testy jednostkowe dla normalizacji, matchera, FSM; benchmark latencji lokalnie.

12. Ryzyka i mitigacje
- Niestabilny Z → zmniejsz wagę/wyłącz; skup się na 2D + projekcje w układzie dłoni.
- Rotacja/pozycja dłoni → normalizacja hand‑centric + progi zależne od skali dłoni.
- Drganie predykcji → smoothing, histereza, min_hold, priorytety.
- Kolizje gestów → rejestr priorytetów i reguły rozstrzygania; klasa „unknown”.
- Instalacja Torch w CI → opcjonalne importy, CPU‑only lub ONNX w testach.

13. Roadmap (fazy wdrożenia)
- Faza 1: Normalizacja i matcher szablonów statycznych + UI „wyklikiwane”.
- Faza 2: FSM sekwencyjny dla ruchów (klik/pinch, scroll, volume) + stabilizacja.
- Faza 3: Backend PyTorch (GRU) dla trudnych wzorców + late‑fusion + metryki runtime.
- Faza 4: Few‑shot prototypy dla gestów użytkownika + kreator nagrań.
- Faza 5: ONNX export, optymalizacje i rozszerzenia (np. TensorRT opcjonalnie).

14. Słowniczek
- curl: stopień zgięcia palca (0..1, 0 = prosty, 1 = zgięty).
- splay: rozstaw palców względem siebie.
- FSM: automat stanów skończonych; modeluje sekwencje.
- late‑fusion: łączenie wyników wielu metod na etapie decyzji końcowej.
- few‑shot/prototypy: dodanie klasy na bazie kilku przykładów przez porównanie embeddingów.
