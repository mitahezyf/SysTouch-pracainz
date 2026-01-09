# Skrypt do przenoszenia plików .md z root do markdown/
# Uruchom: .\cleanup_markdown.ps1

$rootPath = $PSScriptRoot
$markdownDir = Join-Path $rootPath "markdown"

# Wyklucz README.md - powinien zostać w root
$excludeFiles = @("README.md")

# Znajdź wszystkie pliki .md w root (nie w podkatalogach)
Get-ChildItem -Path $rootPath -Filter "*.md" -File | Where-Object {
    $excludeFiles -notcontains $_.Name
} | ForEach-Object {
    Write-Host "Przenoszę: $($_.Name) -> markdown/" -ForegroundColor Green
    Move-Item -Path $_.FullName -Destination $markdownDir -Force
}

Write-Host "`nGotowe! Wszystkie pliki .md (oprócz README.md) przeniesione do markdown/" -ForegroundColor Cyan
