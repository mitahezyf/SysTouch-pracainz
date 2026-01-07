# Test manualny - sprawdzenie czy features sa wypelniane

# krok 1: uruchom zbieranie dla jednej litery, 1 powtorzenie
# python tools\collect_dataset.py --labels=TEST --repeats=1 --handedness-mode=flip --require-handedness=Right --interactive --show-landmarks --clip-seconds=2

# krok 2: znajdz ostatnia sesje
$lastSession = Get-ChildItem -Path "data\collected" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1

Write-Host "Ostatnia sesja: $($lastSession.Name)"
Write-Host ""

# krok 3: znajdz pierwszy plik CSV w tej sesji
$csvFile = Get-ChildItem -Path "$($lastSession.FullName)\features" -Filter "*.csv" | Select-Object -First 1

if ($csvFile) {
    Write-Host "Sprawdzam plik: $($csvFile.Name)"
    Write-Host ""

    # pokaz pierwsze 3 linie (naglowek + 2 klatki)
    Write-Host "=== Pierwsze 3 linie ==="
    Get-Content $csvFile.FullName | Select-Object -First 3 | ForEach-Object {
        # skroc linie dla czytelnosci
        if ($_.Length -gt 200) {
            $_.Substring(0, 200) + "..."
        } else {
            $_
        }
    }
    Write-Host ""

    # policz klatki z has_hand=1 i niepuste features
    Write-Host "=== Statystyki ==="
    $content = Get-Content $csvFile.FullName
    $totalLines = $content.Count - 1  # minus naglowek

    if ($totalLines -gt 0) {
        # zlicz has_hand=1 (kolumna 7)
        $hasHandLines = ($content | Select-Object -Skip 1 | ForEach-Object {
            $cols = $_ -split ","
            if ($cols.Count -gt 6 -and $cols[6] -eq "1") { 1 } else { 0 }
        } | Measure-Object -Sum).Sum

        # zlicz niepuste feat_0 (kolumna po ostatnim lm_20_z, czyli 71)
        $nonEmptyFeatures = ($content | Select-Object -Skip 1 | ForEach-Object {
            $cols = $_ -split ","
            if ($cols.Count -gt 71 -and $cols[71] -ne "") { 1 } else { 0 }
        } | Measure-Object -Sum).Sum

        Write-Host "Wszystkich klatek: $totalLines"
        Write-Host "Klatek z has_hand=1: $hasHandLines ($([math]::Round($hasHandLines/$totalLines*100, 1))%)"
        Write-Host "Klatek z features: $nonEmptyFeatures ($([math]::Round($nonEmptyFeatures/$totalLines*100, 1))%)"
        Write-Host ""

        if ($nonEmptyFeatures -eq 0) {
            Write-Host "BLAD: Features sa puste!" -ForegroundColor Red
            Write-Host "Sprawdz czy uzywasz --handedness-mode=flip i --require-handedness=Right" -ForegroundColor Yellow
        } elseif ($hasHandLines/$totalLines -lt 0.7) {
            Write-Host "OSTRZEZENIE: Malo klatek z wykryta reka (<70%)" -ForegroundColor Yellow
        } else {
            Write-Host "OK: Features sa wypelnione!" -ForegroundColor Green
        }
    }
} else {
    Write-Host "Brak plikow CSV w ostatniej sesji" -ForegroundColor Red
}
