# SystemTouch

## CI/CD
- CI (ci.yml): lint (ruff/black/bandit/pre-commit) na Ubuntu i testy na Windows (Python 3.11/3.12) z raportami JUnit i coverage; cache pip, concurrency, ręczny trigger i harmonogram co tydzień.
- CodeQL (codeql.yml): analiza bezpieczeństwa dla Pythona na push/PR i w harmonogramie.
- Release (release.yml): automatyczny GitHub Release po wypchnięciu taga `v*.*.*` z dołączonym archiwum repo.
- Dependabot: cotygodniowe PR z aktualizacjami pip i GitHub Actions.
- Coverage: wyniki publikowane do Codecov; jeśli repo jest prywatne, dodaj secret `CODECOV_TOKEN` w Settings → Secrets and variables → Actions.

# Program komputerowy SystemTouch
## Wstępnie planowane biblioteki
- OpenCV — wstępne przetwarzanie obrazu
- Mediapipe — śledzenie pozycji punktów
- PyAutoGUI/pynput — symulacja zdarzeń klawiatury i myszy
- TensorFlow/PyTorch — tworzenie własnego modelu rozpoznawania gestów za pomocąuczenia maszynowego
- PyQt6 — tworzene własnego interfejsu graficznego


## Wymagania sprzętowe
- Kamerka minimum 30 FPS o rozdzielczości minimum 720p

## Opis funkcjonalności
- Sterowanie komputerem za pomocą predefiniowanych gestów
- Rozpoznawanie alfabetu oraz podstawowych słów w języku migowym
- Możliwość definiowania własnych gestów
- Zmiana motywów graficznych oraz widoczności programu/nakładki podczas korzystania z komputera
- prowadzenie statystyk częstotliwości występowania poszczególnych gestów

## Aktorzy
Jedyny aktor przewidziany jest dla użytkonika, program przewidziany jest na rozwiązania lokalne więc nie potrzeba nikogo o zarządzania


## Przykładowe gesty
- lupa — oddalenie od siebie kciuka i palca wskazującego
- przewijanie w dół — przesunięcie ręki i gest "idź sobie"
- zmiana głośności — przytrzymanie złączonego kciuka i palca serdecznego następnie przeliczanie ich odległości na skalę głośności
- klik — złączone palce i gest puknięcia
- powrót — machnięcie dłoni w lewo

<div style="page-break-after: always;"></div>

## Interakcja z programem
Program w czasie rzeczywistym (określenie względne w zależności od mocy obliczeniowej maszyny na której został uruchomiony)
reaguje na dane wsadowe w postaci gestów i wykonuje przypisane do nich akcje, po wykonaniu specjalnego gestu przełączany jest tryb obsługi gestów oraz rozpoznawania języka migowego.
Po włączeniu specjalnego trybu uproszczonego program śledzi położenie specjalnie przeznaczonego do tego przedmiotu i na zasadzie rozpoznawania geometrii oraz gestów (w wersji prototypowej czarny ołówek z zieloną gumką).

<div style="page-break-after: always;"></div>

## Diagram przypadków użycia

![UseCaseDiagram1.png](UseCaseDiagram1.png)

<div style="page-break-after: always;"></div>


### Przypadki użycia
| Nazwa:                          | Sterowanie komputerem                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Krótki opis:                    | Sterowanie komputerem przy pomocy gestów wykonywanych jedną lub obiema rękami oraz gestów rozpoznawanych przez analizę kształtów geometrycznych.                                                                                                                                                                                                                                                                                                                                     |
| Warunki wstępne:                | Użytkownik musi mieć skonfigurowane urządzenie do rejestrowania gestów (np. kamera) oraz aktywny system rozpoznawania gestów.                                                                                                                                                                                                                                                                                                                                                        |
| Warunki końcowe:                | System poprawnie interpretuje gesty użytkownika jako polecenia i wykonuje odpowiadające im akcje w systemie operacyjnym.                                                                                                                                                                                                                                                                                                                                                             |
| Główny przepływ zdarzeń:        | 1. Użytkownik inicjuje sesję sterowania gestami. <br/>2. System uruchamia moduł rozpoznawania gestów. <br/>3. Użytkownik wykonuje gesty jednoręczne lub oburęczne. <br/>4. System rozpoznaje gest jako jedno z dostępnych poleceń (np. kliknięcie, przewijanie). <br/>5. Użytkownik wykonuje gest geometryczny (np. koło, linia). <br/>6. System interpretuje kształt i wykonuje przypisaną akcję. <br/>7. System informuje użytkownika o wykonaniu akcji (np. dźwiękiem lub ikoną). |
| Alternatywne przepływy zdarzeń: | 3a. System nie rozpoznaje gestu: wyświetlana jest informacja o błędzie i użytkownik może spróbować ponownie. <br/>4a. Gest jest nieprawidłowy lub niewłaściwie wykonany – brak akcji, system prosi o powtórzenie.                                                                                                                                                                                                                                                                    |
| Specjalne wymagania:            | 1. Rozpoznanie gestu nie może trwać dłużej niż 1 sekundę.<br/> 2.System powinien działać w czasie rzeczywistym z opóźnieniem nie większym niż 0.5 sekundy od wykonania gestu do reakcji.<br/> 3. Dopuszczalne środowisko pracy: minimalne oświetlenie 150 luksów.                                                                                                                                                                                                                    |

<div style="page-break-after: always;"></div>

| Nazwa:                          | Gesty obsługi dłońmi                                                                                                                                                                      |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                                                                                                                |
| Krótki opis:                    | Umożliwienie użytkownikowi wykonywania poleceń za pomocą gestów jednej lub dwóch dłoni.                                                                                                   |
| Warunki wstępne:                | System musi mieć uruchomione rozpoznawanie gestów i dostęp do kamery.                                                                                                                     |
| Warunki końcowe:                | Gesty są poprawnie zinterpretowane i wywołują odpowiednie akcje.                                                                                                                          |
| Główny przepływ zdarzeń:        | 1. Użytkownik wykonuje gest dłonią.<br/>2. System analizuje układ i ruch dłoni.<br/>3. Gest zostaje zaklasyfikowany jako jedno- lub oburęczny.<br/>4. System wykonuje odpowiednią akcję.  |
| Alternatywne przepływy zdarzeń: | 2a. Ręka niewidoczna – system zgłasza brak widoczności dłoni.<br/>3a. Niezidentyfikowany gest – użytkownik proszony o powtórzenie.                                                        |
| Specjalne wymagania:            | 1. Minimalna rozdzielczość kamery: 720p.<br/>2. Opóźnienie maksymalne: 0.3 sekundy.                                                                                                       |



| Nazwa:                          | Gesty oburącz                                                                                                                                                          |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                                                                                             |
| Krótki opis:                    | Rozpoznawanie i interpretacja gestów wykonywanych obiema rękami.                                                                                                       |
| Warunki wstępne:                | Obie ręce muszą być widoczne dla systemu.                                                                                                                              |
| Warunki końcowe:                | System poprawnie wykonuje przypisaną akcję do gestu oburęcznego.                                                                                                       |
| Główny przepływ zdarzeń:        | 1. Użytkownik unosi obie ręce i wykonuje gest.<br/>2. System rozpoznaje układ dłoni oraz synchronizację ruchów.<br/>3. Wykonywana jest przypisana operacja systemowa.  |
| Alternatywne przepływy zdarzeń: | 1a. Jedna ręka niewidoczna – system prosi o poprawne wykonanie gestu.                                                                                                  |
| Specjalne wymagania:            | 1. Wymagana synchronizacja rąk z dokładnością do 0.2 sekundy.                                                                                                          |

<div style="page-break-after: always;"></div>

| Nazwa:                          | Gesty jednorącz                                                                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                                                                           |
| Krótki opis:                    | Wykonywanie akcji za pomocą gestów jednej dłoni.                                                                                                     |
| Warunki wstępne:                | Widoczność jednej dłoni w kadrze.                                                                                                                    |
| Warunki końcowe:                | System wykonuje przypisaną operację.                                                                                                                 |
| Główny przepływ zdarzeń:        | 1. Użytkownik wykonuje gest jednoręczny.<br/>2. System analizuje jego kształt, kierunek i dynamikę.<br/>3. Rozpoznany gest powoduje określoną akcję. |
| Alternatywne przepływy zdarzeń: | 2a. Niezrozumiały gest – prośba o powtórzenie.                                                                                                       |
| Specjalne wymagania:            | Ręka nie może być zasłonięta ani rozmyta.                                                                                                            |



| Nazwa:                          | Gesty obsługi za pomocą śledzenia punktu i wykrywania kształtów geometrycznych                                                                                                                    |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                                                                                                                        |
| Krótki opis:                    | Rozpoznawanie gestów przez śledzenie ruchu punktu (np. palca) i kształtów (np. koła, trójkąta).                                                                                                   |
| Warunki wstępne:                | Aktywny tryb śledzenia punktów.                                                                                                                                                                   |
| Warunki końcowe:                | Zidentyfikowany kształt powoduje wykonanie akcji.                                                                                                                                                 |
| Główny przepływ zdarzeń:        | 1. Użytkownik wykonuje rysunek w powietrzu (np. okrąg).<br/>2. System śledzi trajektorię ruchu.<br/>3. Kształt zostaje rozpoznany i zinterpretowany jako polecenie.<br/>4. Wykonywana jest akcja. |
| Alternatywne przepływy zdarzeń: | 2a. Kształt zbyt nieczytelny – komunikat błędu.                                                                                                                                                   |
| Specjalne wymagania:            | Dopuszczalna tolerancja rozpoznania kształtu: ±10%.                                                                                                                                               |

<div style="page-break-after: always;"></div>

| Nazwa:                          | Tłumacz migowego                                                                                                                                          |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                                                                                |
| Krótki opis:                    | Moduł tłumaczący gesty języka migowego na tekst lub mowę.                                                                                                 |
| Warunki wstępne:                | Włączony tryb tłumacza migowego.                                                                                                                          |
| Warunki końcowe:                | Gesty przetłumaczone na tekst lub komunikaty dźwiękowe.                                                                                                   |
| Główny przepływ zdarzeń:        | 1. Użytkownik wykonuje gest w języku migowym.<br/>2. System analizuje i identyfikuje gest.<br/>3. Odpowiedni tekst zostaje wyświetlony lub wypowiedziany. |
| Alternatywne przepływy zdarzeń: | 2a. Brak rozpoznania gestu – użytkownik informowany o błędzie.                                                                                            |
| Specjalne wymagania:            | Obsługa co najmniej 500 najczęściej używanych gestów.                                                                                                     |



| Nazwa:                          | Rozpoznawanie migów pojedynczych słów                                                                        |
|---------------------------------|--------------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                                   |
| Krótki opis:                    | Tłumaczenie pojedynczych gestów na konkretne słowa.                                                          |
| Warunki wstępne:                | Tryb tłumaczenia migów aktywny.                                                                              |
| Warunki końcowe:                | Każdy gest zostaje przypisany do słowa.                                                                      |
| Główny przepływ zdarzeń:        | 1. Użytkownik wykonuje gest odpowiadający konkretnemu słowu.<br/>2. System wyświetla tłumaczenie na ekranie. |
| Alternatywne przepływy zdarzeń: | 1a. Gest nieznany – komunikat błędu.                                                                         |
| Specjalne wymagania:            | Słownik co najmniej 300 słów.                                                                                |

<div style="page-break-after: always;"></div>

| Nazwa:                          | Rozpoznawanie migów alfabetu                                                                    |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                      |
| Krótki opis:                    | System rozpoznaje litery alfabetu migowego.                                                     |
| Warunki wstępne:                | Aktywny tryb alfabetu.                                                                          |
| Warunki końcowe:                | System wyświetla literę odpowiadającą gestowi.                                                  |
| Główny przepływ zdarzeń:        | 1. Użytkownik pokazuje literę w języku migowym.<br/>2. System rozpoznaje ją i dodaje do tekstu. |
| Alternatywne przepływy zdarzeń: | 1a. Nierozpoznana litera – informacja zwrotna o błędzie.                                        |
| Specjalne wymagania:            | Obsługa pełnego alfabetu (26 liter).                                                            |



| Nazwa:                          | Ustawienia                                                                                             |
|---------------------------------|--------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                             |
| Krótki opis:                    | Dostęp do konfiguracji i personalizacji systemu.                                                       |
| Warunki wstępne:                | Użytkownik uruchamia panel ustawień.                                                                   |
| Warunki końcowe:                | Zmiany są zapisane i aktywne.                                                                          |
| Główny przepływ zdarzeń:        | 1. Użytkownik otwiera ustawienia.<br/>2. Przegląda dostępne opcje konfiguracji.<br/>3. Dokonuje zmian. |
| Alternatywne przepływy zdarzeń: |                                                                                                        |
| Specjalne wymagania:            | Panel musi być dostępny w każdej chwili.                                                               |

<div style="page-break-after: always;"></div>

| Nazwa:                          | Zmiana motywów graficznych                                                                                                 |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                                                 |
| Krótki opis:                    | Zmiana wyglądu interfejsu użytkownika.                                                                                     |
| Warunki wstępne:                | Użytkownik musi mieć dostęp do ustawień.                                                                                   |
| Warunki końcowe:                | Motyw zostaje zmieniony i zapisany.                                                                                        |
| Główny przepływ zdarzeń:        | 1. Użytkownik wchodzi do sekcji „Motywy”.<br/>2. Wybiera preferowany motyw.<br/>3. System natychmiast stosuje nowy wygląd. |
| Alternatywne przepływy zdarzeń: |                                                                                                                            |
| Specjalne wymagania:            | Obsługa przynajmniej motywu jasnego i ciemnego.                                                                            |



| Nazwa:                          | Dodawanie personalizowanych gestów                                                                                                              |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Aktorzy:                        | Użytkownik                                                                                                                                      |
| Krótki opis:                    | Użytkownik może definiować własne gesty i przypisywać im funkcje.                                                                               |
| Warunki wstępne:                | Aktywny panel personalizacji.                                                                                                                   |
| Warunki końcowe:                | Nowy gest jest rozpoznawany przez system i aktywny.                                                                                             |
| Główny przepływ zdarzeń:        | 1. Użytkownik otwiera panel dodawania gestu.<br/>2. Wykonuje gest próbny.<br/>3. Przypisuje mu określoną funkcję.<br/>4. Zapisuje konfigurację. |
| Alternatywne przepływy zdarzeń: | 2a. Gest zbyt podobny do istniejącego – system prosi o inny.                                                                                    |
| Specjalne wymagania:            | Możliwość zapisania co najmniej 20 gestów użytkownika.                                                                                          |

<div style="page-break-after: always;"></div>




[//]: # (by Pomaranski)
