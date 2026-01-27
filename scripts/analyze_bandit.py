#!/usr/bin/env python3
"""Analiza raportu Bandit JSON"""
import json

with open("reports/bandit-complete-scan.json", "r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data["results"])
app_issues = [r for r in data["results"] if "\\app\\" in r["filename"]]
test_issues = [r for r in data["results"] if "\\tests\\" in r["filename"]]
external = total - len(app_issues) - len(test_issues)

print("=" * 60)
print("ANALIZA PELNEGO SKANU BANDIT")
print("=" * 60)
print()
print(f"Twoj kod (app/):       {len(app_issues):,}")
print(f"Testy (tests/):        {len(test_issues):,}")
print(f"Biblioteki zewnetrzne: {external:,}")
print(f"RAZEM:                 {total:,}")
print()

metrics = data["metrics"]["_totals"]
print("Severity breakdown (caly scan):")
all_high = sum(1 for r in data["results"] if r["issue_severity"] == "HIGH")
all_med = sum(1 for r in data["results"] if r["issue_severity"] == "MEDIUM")
all_low = sum(1 for r in data["results"] if r["issue_severity"] == "LOW")
print(f"  HIGH:   {all_high:,}")
print(f"  MEDIUM: {all_med:,}")
print(f"  LOW:    {all_low:,}")
print()

# Analiza app/ tylko
if app_issues:
    print("=" * 60)
    print("TYLKO TWOJ KOD (app/):")
    print("=" * 60)
    by_severity = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    by_test: dict[str, int] = {}

    for issue in app_issues:
        sev = issue["issue_severity"]
        by_severity[sev] = by_severity.get(sev, 0) + 1
        test_id = issue["test_id"]
        by_test[test_id] = by_test.get(test_id, 0) + 1

    print("\nSeverity w app/:")
    print(f"  HIGH:   {by_severity.get('HIGH', 0)}")
    print(f"  MEDIUM: {by_severity.get('MEDIUM', 0)}")
    print(f"  LOW:    {by_severity.get('LOW', 0)}")

    print("\nTop problemy w app/:")
    for test_id, count in sorted(by_test.items(), key=lambda x: -x[1])[:10]:
        print(f"  {test_id}: {count}")

    # Pokaz HIGH issues z app/
    high_in_app = [i for i in app_issues if i["issue_severity"] == "HIGH"]
    if high_in_app:
        print(f"\n!!! UWAGA: {len(high_in_app)} HIGH severity w app/ !!!")
        for issue in high_in_app[:5]:
            print(f"\n  - {issue['test_id']}: {issue['issue_text']}")
            print(f"    Plik: {issue['filename'].split('SysTouch')[-1]}")
            print(f"    Linia: {issue['line_number']}")

    # Pokaz MEDIUM issues z app/
    med_in_app = [i for i in app_issues if i["issue_severity"] == "MEDIUM"]
    if med_in_app:
        print(f"\n{'='*60}")
        print(f"MEDIUM severity w app/ ({len(med_in_app)}):")
        print(f"{'='*60}")
        for issue in med_in_app:
            print(f"\n  - {issue['test_id']}: {issue['issue_text']}")
            print(f"    Plik: {issue['filename'].split('inzynierka')[-1]}")
            print(f"    Linia: {issue['line_number']}")
            print(f"    Kod: {issue.get('code', 'N/A')[:80]}")
    else:
        print("\n[OK] Brak MEDIUM severity w app/")
