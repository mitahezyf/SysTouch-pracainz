# Instrukcja naprawy PowerShell Execution Policy

## Problem
PowerShell blokuje uruchamianie skryptów (w tym aktywację venv).

## Rozwiązanie

### Krok 1: Otwórz PowerShell jako Administrator
1. Naciśnij `Win + X`
2. Wybierz "Windows PowerShell (Administrator)" lub "Terminal (Administrator)"

### Krok 2: Zmień Execution Policy
Wykonaj w terminalu administratora:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Potwierdź naciskając `Y` (Yes)

### Krok 3: Sprawdź czy działa
Zamknij terminal administratora i wróć do normalnego terminala w VS Code.
Spróbuj ponownie:

```powershell
.\.venv\Scripts\Activate.ps1
```

Powinno zadziałać! ✓

---

## Co to robi?
- `RemoteSigned` - pozwala na uruchamianie lokalnych skryptów, ale wymaga podpisu dla skryptów pobranych z internetu
- `CurrentUser` - zmienia ustawienia tylko dla Twojego konta (nie wymaga uprawnień administratora dla całego systemu)

## Alternatywa (jeśli nie masz uprawnień administratora)
Możesz obejść to dla pojedynczej sesji:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

Ale to będzie działać tylko w bieżącej sesji terminala.
