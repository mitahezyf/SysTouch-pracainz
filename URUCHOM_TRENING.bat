@echo off
REM Pelny trening modelu PJM ze wszystkimi probkami
REM Wystarczy kliknac dwukrotnie ten plik lub uruchomic z cmd

echo ========================================
echo TRENING MODELU PJM - START
echo ========================================
echo.

REM aktywuj venv (opcjonalne - moze byc juz aktywne)
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM uruchom trening z domyslnymi parametrami (200 epok)
python -m app.sign_language.trainer --epochs 200 --lr 0.001 --batch_size 64 --hidden_size 128

echo.
echo ========================================
echo TRENING ZAKONCZONY
echo ========================================
echo Model zapisany w: app\sign_language\models\pjm_model.pth
echo.
pause
