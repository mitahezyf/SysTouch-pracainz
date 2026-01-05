"""Analiza coverage z pliku XML."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

xml_path = Path("reports/coverage.xml")
if not xml_path.exists():
    print("Brak pliku coverage.xml")
    exit(1)

tree = ET.parse(xml_path)  # nosec B314
root = tree.getroot()

# zbierz dane per plik
files_data: list[dict[str, Any]] = []
for cls in root.findall(".//class"):
    filename = cls.get("filename", "")
    line_rate = float(cls.get("line-rate", 0))

    # policz linie z hits
    lines = cls.findall(".//line")
    total_lines = len(lines)
    covered_lines = sum(1 for line in lines if int(line.get("hits", 0)) > 0)

    if total_lines > 0:  # pomijamy puste pliki
        files_data.append(
            {
                "file": filename,
                "coverage": line_rate * 100,
                "lines_covered": covered_lines,
                "lines_total": total_lines,
            }
        )

# sortuj po coverage
files_data.sort(key=lambda x: x["coverage"])

print("\n=== PLIKI Z NAJNIZSZYM COVERAGE ===\n")
for item in files_data[:20]:
    print(
        f"{item['coverage']:5.1f}% - {item['file']} ({item['lines_covered']}/{item['lines_total']})"
    )

print("\n=== STATYSTYKI KATEGORII ===\n")
categories: dict[str, list[float]] = {
    "GUI": [],
    "Sign Language": [],
    "Gesture Engine": [],
    "Core": [],
    "Utils": [],
    "Tests/Scripts": [],
}

for item in files_data:
    f = str(item["file"])
    cov = float(item["coverage"])

    if "gui" in f:
        categories["GUI"].append(cov)
    elif "sign_language" in f:
        categories["Sign Language"].append(cov)
    elif "gesture_engine" in f:
        categories["Gesture Engine"].append(cov)
    elif "main.py" in f or "train_gesture.py" in f:
        categories["Core"].append(cov)
    elif "utils" in f or "detector" in f:
        categories["Utils"].append(cov)
    else:
        categories["Tests/Scripts"].append(cov)

for cat, covs in categories.items():
    if covs:
        avg = sum(covs) / len(covs)
        print(f"{cat:20s}: {avg:5.1f}% srednia ({len(covs)} plikow)")

# calkowity coverage
total_lines = sum(int(item["lines_total"]) for item in files_data)
covered_lines = sum(int(item["lines_covered"]) for item in files_data)
total_cov = (covered_lines / total_lines * 100) if total_lines > 0 else 0

print("\n=== CALKOWITY COVERAGE ===")
print(f"Linie pokryte: {covered_lines}/{total_lines}")
print(f"Coverage: {total_cov:.1f}%")
