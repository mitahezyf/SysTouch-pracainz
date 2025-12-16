# Dataset PJM - struktura

## Katalogi

### `raw/`
Miejsce na surowe dane z Kaggle.

**Wymagany plik:**
- `PJM-vectors.csv` - dataset z Kaggle zawierajacy znormalizowane wektory 63D dla liter PJM

Format oczekiwany:
```
label,p0_x,p0_y,p0_z,p1_x,p1_y,p1_z,...,p20_x,p20_y,p20_z
A,0.123,0.456,0.789,...
B,0.234,0.567,0.890,...
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

1. Pobierz `PJM-vectors.csv` z Kaggle
2. Umiec w katalogu `raw/`
3. Uruchom: `python -m app.sign_language.dataset` (przygotuje train/val/test)
4. Uruchom: `python -m app.sign_language.trainer` (wytrenuje model)

## Struktura pliku PJM-vectors.csv

Plik powinien zawierac:
- 1 kolumna: `label` (litera A-Z)
- 63 kolumny: `p0_x, p0_y, p0_z, ..., p20_x, p20_y, p20_z` (wspolrzedne 21 punktow dloni)

Normalizacja: wspolrzedne sa juz znormalizowane wzgledem nadgarstka i skali dloni.
