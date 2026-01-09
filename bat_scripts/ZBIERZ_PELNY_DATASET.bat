@echo off
REM Szybkie zbieranie WSZYSTKICH liter PJM (A-Z bez D,F,G,H,J,K,Q,V,X,Z)
REM Litery PJM: A,B,C,E,I,L,M,N,O,P,R,S,T,U,W,Y

echo ========================================
echo ZBIERANIE PELNEGO DATASETU PJM
echo ========================================
echo.
echo Ten skrypt zbierze WSZYSTKIE litery PJM
echo Potrzebne: ~30-40 minut (2 osoby x 16 liter x 3 powtorzenia)
echo.

set /p CONFIRM="Czy jestes gotowy? (t/n): "
if /i not "%CONFIRM%"=="t" (
    echo Anulowano.
    pause
    exit /b 0
)

echo.
echo ========================================
echo ETAP 1/3: Litery A-I
echo ========================================
echo.

python tools\collect_dataset.py ^
  --labels=A,B,C,E,I ^
  --repeats=3 ^
  --handedness-mode=flip ^
  --require-handedness=Right ^
  --interactive ^
  --show-landmarks ^
  --mirror-left=false ^
  --clip-seconds=3

if %ERRORLEVEL% NEQ 0 (
    echo BLAD w etapie 1!
    pause
    exit /b 1
)

echo.
echo ========================================
echo ETAP 2/3: Litery L-P
echo ========================================
echo.

python tools\collect_dataset.py ^
  --labels=L,M,N,O,P ^
  --repeats=3 ^
  --handedness-mode=flip ^
  --require-handedness=Right ^
  --interactive ^
  --show-landmarks ^
  --mirror-left=false ^
  --clip-seconds=3

if %ERRORLEVEL% NEQ 0 (
    echo BLAD w etapie 2!
    pause
    exit /b 1
)

echo.
echo ========================================
echo ETAP 3/3: Litery R-Y
echo ========================================
echo.

python tools\collect_dataset.py ^
  --labels=R,S,T,U,W,Y ^
  --repeats=3 ^
  --handedness-mode=flip ^
  --require-handedness=Right ^
  --interactive ^
  --show-landmarks ^
  --mirror-left=false ^
  --clip-seconds=3

if %ERRORLEVEL% NEQ 0 (
    echo BLAD w etapie 3!
    pause
    exit /b 1
)

echo.
echo ========================================
echo WSZYSTKIE LITERY ZEBRANE!
echo ========================================
echo.
echo Sprawdzam dataset...
python check_dataset_ready.py

echo.
echo Konsoliduje dane...
python tools\consolidate_dataset.py

echo.
echo ========================================
echo GOTOWE!
echo ========================================
echo.
echo Dataset gotowy do treningu.
echo Nastepny krok:
echo   python tools\train_model.py --vectors=data/consolidated/vectors.csv
echo.
pause
