# Skrypty pomocnicze

Ten katalog zawiera skrypty naprawcze i narzedzia pomocnicze projektu.

## Skrypty naprawcze PyTorch

### fix_torch.ps1
Szybka naprawa problemu z PyTorch DLL (c10.dll).

**Uruchomienie:**
```powershell
.\scripts\fix_torch.ps1
```

**Co robi:**
1. Odinstalowuje torch, torchvision, torchaudio
2. Czysci cache pip
3. Instaluje torch==2.4.1 CPU z oficjalnego repo
4. Testuje import torch i SignTranslator

### reinstall_torch.ps1
Pelna reinstalacja PyTorch z diagnostyka.

**Uruchomienie:**
```powershell
.\scripts\reinstall_torch.ps1
```

**Co robi:**
- Sprawdza istnienie venv
- Odinstalowuje stara wersje torch
- Instaluje torch==2.4.1 CPU
- Testuje import z diagnostyka bledow
- Podaje rozwiazania w przypadku problemow

### quick_fix_torch.bat
Szybka naprawa PyTorch (wersja batch).

**Uruchomienie:**
```cmd
.\scripts\quick_fix_torch.bat
```

## Skrypty testowe

### test_fix.bat
Test weryfikujacy naprawe projektu.

### test_pjm_ready.ps1
Test gotowosci modulu PJM (Polski Jezyk Migowy).

## Narzedzia pomocnicze

### unify_comments.py
Skrypt ujednolicajacy komentarze w projekcie - usuwa znaki diakrytyczne z komentarzy.

**Uruchomienie:**
```bash
python scripts\unify_comments.py
```

**Co robi:**
- Przeszukuje wszystkie pliki `.py` w `app/` i `tests/`
- Zamienia polskie znaki (ą, ć, ę, ł, ń, ó, ś, ź, ż) na ASCII
- Zachowuje formatowanie kodu
- Nie modyfikuje stringow, tylko komentarze

## Uruchamianie z glownego katalogu

Wszystkie skrypty mozna uruchomic z glownego katalogu projektu:

```powershell
# PowerShell
.\scripts\fix_torch.ps1
.\scripts\reinstall_torch.ps1
.\scripts\test_pjm_ready.ps1

# CMD
.\scripts\quick_fix_torch.bat
.\scripts\test_fix.bat
```

## Uwagi

- Skrypty `.ps1` wymagaja PowerShell
- Skrypty `.bat` dzialaja w CMD i PowerShell
- Wszystkie skrypty naprawcze PyTorch instaluja wersje 2.4.1 CPU
