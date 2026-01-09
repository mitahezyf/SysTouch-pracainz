@echo off
REM Skrypt do zbierania gestow DYNAMICZNYCH z 4 osobami
REM Optymalizowany dla gestow ruchowych (ą, ć, ę, ł, ń, ó, ś, ź, ż)

echo ========================================
echo ZBIERANIE GESTOW DYNAMICZNYCH - 4 OSOBY
echo ========================================
echo.
echo Ten skrypt zbierze gesty RUCHOWE (ą, ć, ę, ł, ń, ó, ś, ź, ż)
echo.
echo WAZNE INSTRUKCJE:
echo - Kazdy gest POWTARZAJ w petli 4-5 razy
echo - Kazdy klip trwa 5 sekund
echo - Uzywaj TYLKO prawej reki
echo - Trzymaj reke w kadrze caly czas
echo.
echo Przyklad dla gestu "ą":
echo   1. Pokaz "a"
echo   2. Przekrec nadgarstek
echo   3. Wroc do pozycji
echo   4. POWTORZ 4-5 razy plynnie w ciagu 5s
echo.

set /p CONFIRM="Czy wszyscy 4 sa gotowi? (t/n): "
if /i not "%CONFIRM%"=="t" (
    echo Anulowano.
    pause
    exit /b 0
)

echo.
set /p NAMES="Podaj 4 imiona oddzielone przecinkiem (np. Jan,Kasia,Piotr,Ania): "
if "%NAMES%"=="" set NAMES=Osoba1,Osoba2,Osoba3,Osoba4

echo.
echo Nagrywam dla: %NAMES%
echo.
echo Szacowany czas: ~20-30 minut (9 gestow x 4 osoby x 5 powtorzen)
echo.

pause

python tools\collect_dataset.py ^
  --labels=ą,ć,ę,ł,ń,ó,ś,ź,ż ^
  --repeats=5 ^
  --clip-seconds=5 ^
  --handedness-mode=flip ^
  --require-handedness=Right ^
  --interactive ^
  --show-landmarks ^
  --mirror-left=false ^
  --performers=%NAMES%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BLAD podczas zbierania!
    pause
    exit /b 1
)

echo.
echo ========================================
echo GOTOWE!
echo ========================================
echo.
echo Zebrano:
echo - 9 gestow dynamicznych (ą, ć, ę, ł, ń, ó, ś, ź, ż)
echo - 4 osoby: %NAMES%
echo - 5 powtorzen na osobe
echo - 180 klipow total (9 x 4 x 5)
echo - ~4500 klatek (~500 na znak)
echo.
echo Nastepny krok:
echo   python tools\consolidate_dataset.py
echo   python tools\train_model.py --vectors=data/consolidated/vectors.csv
echo.
pause
