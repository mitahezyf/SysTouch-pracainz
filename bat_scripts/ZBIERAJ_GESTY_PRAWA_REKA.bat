@echo off
REM Skrypt do zbierania gestow gdy zawsze uzywasz prawej reki
REM Uzywaj tego zamiast uruchamiac collect_dataset.py recznie

echo ========================================
echo ZBIERANIE GESTOW - PRAWA REKA
echo ========================================
echo.
echo Ten skrypt poprawnie rozpozna Twoja prawa reke.
echo.

set /p LABELS="Podaj litery (przecinek): "
set /p REPEATS="Ile powtorzen na osobe (domyslnie 3): "
if "%REPEATS%"=="" set REPEATS=3

echo.
echo Uruchamiam zbieranie z parametrami:
echo - Litery: %LABELS%
echo - Powtorzenia: %REPEATS%
echo - Tryb: flip (prawa reka)
echo - Wymagana reka: Right
echo.

python tools\collect_dataset.py ^
  --labels=%LABELS% ^
  --repeats=%REPEATS% ^
  --handedness-mode=flip ^
  --require-handedness=Right ^
  --interactive ^
  --show-landmarks ^
  --mirror-left=false

echo.
echo ========================================
echo GOTOWE!
echo ========================================
pause
