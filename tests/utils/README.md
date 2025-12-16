# Testy pomocnicze i narzędzia diagnostyczne

Ten katalog zawiera testy pomocnicze i skrypty diagnostyczne, które nie są regularnymi testami jednostkowymi uruchamianymi przez pytest.

## Pliki

### test_attributeerror_fix.py
Test weryfikujący naprawę AttributeError w `main_window.py`.

**Uruchomienie:**
```bash
python tests/utils/test_attributeerror_fix.py
```

**Cel:** Sprawdzenie czy translator PJM jest poprawnie inicjalizowany i nie powoduje AttributeError przy przełączaniu trybu.

### test_torch_import.py
Test importu PyTorch po reinstalacji.

**Uruchomienie:**
```bash
python tests/utils/test_torch_import.py
```

**Cel:** Weryfikacja czy PyTorch został poprawnie zainstalowany i może być zaimportowany.

## Uwagi

- Te testy NIE są uruchamiane automatycznie przez pytest
- Służą do manualnej weryfikacji i diagnostyki
- Są pomocne przy troubleshootingu problemów z instalacją i konfiguracją
