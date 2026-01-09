@echo off
REM Skrypt do zbierania gestow DYNAMICZNYCH (A+, C+, CH, CZ, RZ, SZ) - prawa reka

echo ========================================
echo ZBIERANIE GESTOW DYNAMICZNYCH - PRAWA REKA
echo ========================================
echo.
echo Ten skrypt zbiera gesty dynamiczne (z ruchem).
echo Wykonuj gest naturalnie przez caly czas nagrywania (5s).
echo.
echo Przyklad:
echo - A+ (a nosowe): zacznij od A i przekrec dlon
echo - C+ (c kreska): zacznij od C i pociagnij w dol
echo - CH: zacznij od C i zrob ruch H
echo.

set /p LABELS="Podaj litery (przecinek, np. A+,C+,CH): "
set /p REPEATS="Ile powtorzen na osobe (domyslnie 5): "
if "%REPEATS%"=="" set REPEATS=5

set /p CLIP_SECONDS="Dlugosc klipu w sekundach (domyslnie 5): "
if "%CLIP_SECONDS%"=="" set CLIP_SECONDS=5

echo.
echo Uruchamiam zbieranie z parametrami:
echo - Litery: %LABELS%
echo - Powtorzenia: %REPEATS%
echo - Dlugosc klipu: %CLIP_SECONDS%s
echo - Tryb: flip (prawa reka)
echo - Wymagana reka: Right
echo.

python tools\collect_dataset.py ^
  --labels=%LABELS% ^
  --repeats=%REPEATS% ^
  --clip-seconds=%CLIP_SECONDS% ^
  --handedness-mode=flip ^
  --require-handedness=Right ^
  --interactive ^
  --show-landmarks ^
  --mirror-left=false

echo.
echo ========================================
echo GOTOWE!
echo ========================================
echo.
echo Nastepny krok: uruchom konsolidacje danych
echo   python -m tools.consolidate_collected
echo.
pause
