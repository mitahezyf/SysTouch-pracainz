Debugging by tests

Cel
- Te testy maja dawac jednoznaczne odpowiedzi: czy problem jest w danych CSV, w cechach (kolejnosc/flip), w przeciekach, w splicie po userach, w mapowaniu labeli, czy w smoothingu realtime.

Konfiguracja sciezek
- CSV z wektorami:
  - domyslnie: app/sign_language/data/raw/PJM-vectors.csv
  - override: ustaw zmienna srodowiskowa PJM_VECTORS_CSV
- Model joblib:
  - domyslnie: app/sign_language/models/pjm_model.joblib
  - override: ustaw zmienna srodowiskowa PJM_MODEL_JOBLIB

Trening modelu (gdy testy krzycza ze brak artefaktu)
- uruchom:
  - python tools\train_model.py --vectors app\sign_language\data\raw\PJM-vectors.csv --out app\sign_language\models\pjm_model.joblib

Uruchomienie testow
- python -m pytest -q

Jak czytac typowe fail'e
- "Brak PJM-vectors.csv" - plik nie jest w repo lub testy nie moga go znalezc; ustaw PJM_VECTORS_CSV
- "W CSV nadal sa zabronione etykiety" - w danych nadal sa klasy dni/por roku; musza byc usuniete
- "metadane w feature_cols" - do X trafilo user_id/lux_value/sign_label; napraw builder X
- "split miesza userow" - train i test maja te same user_id; split musi byc GroupShuffleSplit/GroupKFold z groups=user_id
- "artefakt nie zawiera kluczy" - zapis modelu jest niekompletny i runtime nie ma kontraktu (feature_cols/preprocess_config)
- "Kolejnosc nazw cech rozna" - runtime buduje cechy w innej kolejnosci niz model byl trenowany; napraw runtime_feature_names() / builder
- "pipeline.predict_proba rzuca wyjatek" - shape/typy X nie pasuja do tego co pipeline oczekuje; zwykle problem w feature_cols albo runtime builderze
