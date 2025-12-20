# Dataset PJM - struktura

## Katalogi

### `raw/`
Miejsce na surowe dane z Kaggle.

**Obsługiwane pliki (wersja 2.0 - multi-dataset):**
- `PJM-vectors.csv` - dataset z Kaggle zawierajacy znormalizowane wektory 63D dla liter PJM
- `PJM-points.csv` - dataset z surowych landmarkow (21 punktow x 3 wspolrzedne), normalizowanych automatycznie
- `PJM-images.csv` - pominiety (tylko metadane pikseli, nie cechy)

**Połączenie datasetów:**
Preprocessing automatycznie laczy vectors + points -> 2x wiecej probek treningowych (~40k zamiast ~20k)

Format oczekiwany PJM-vectors.csv:
```
label,p0_x,p0_y,p0_z,p1_x,p1_y,p1_z,...,p20_x,p20_y,p20_z
A,0.123,0.456,0.789,...
B,0.234,0.567,0.890,...
```

Format oczekiwany PJM-points.csv:
```
sign_label,vector_hand_1_x,vector_hand_1_y,vector_hand_1_z,point_1_1,...,point_1_63
A,0.123,0.456,0.789,5.585,170.316,...
```

### `processed/`
Automatycznie generowane pliki .npz po przetworzeniu datasetu:
- `train.npz` - zbior treningowy (70%)
- `val.npz` - zbior walidacyjny (15%)
- `test.npz` - zbior testowy (15%)

Kazdy plik zawiera:
- `X` - macierz cech [N, 63]
- `y` - wektor etykiet [N]
- `meta` - metadane (klasy, rozmiar, wersja)

## Instrukcje

1. Pobierz `PJM-vectors.csv` i `PJM-points.csv` z Kaggle
2. Umiec w katalogu `raw/`
3. Uruchom: `python -m app.sign_language.dataset` (przygotuje train/val/test z obu plikow)
4. Uruchom: `python -m app.sign_language.trainer --epochs 150` (wytrenuje model na rozszerzonym datasecie)

**Zalecane parametry treningu:**
- Epoki: 150 (wiecej danych = dłuższy trening)
- Learning rate: 0.001 (domyslny)
- Hidden size: 128 (domyslny)

**Oczekiwane wyniki:**
- Test Accuracy: ~80% (zamiast ~73% na starym datasecie)
- Train set: ~28,000 probek
- Val set: ~6,000 probek
- Test set: ~6,000 probek

## Struktura pliku PJM-vectors.csv

Plik powinien zawierac:
- 1 kolumna: `label` (litera A-Z)
- 63 kolumny: `p0_x, p0_y, p0_z, ..., p20_x, p20_y, p20_z` (wspolrzedne 21 punktow dloni)

Normalizacja: wspolrzedne sa juz znormalizowane wzgledem nadgarstka i skali dloni.
