#!/usr/bin/env python3
# skrypt analizujacy projekt - szuka plikow powyzej 120 linii, testow poza tests/ i md poza markdown/

from pathlib import Path
from typing import List, Tuple


def count_lines(filepath: Path) -> int:
    """zwraca liczbe linii w pliku"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return len(f.readlines())
    except Exception:
        return 0


def find_large_files(root: Path, threshold: int = 120) -> List[Tuple[Path, int]]:
    """znajduje pliki py powyzej threshold linii"""
    large_files = []
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        lines = count_lines(py_file)
        if lines > threshold:
            large_files.append((py_file, lines))
    return sorted(large_files, key=lambda x: -x[1])


def find_misplaced_tests(root: Path) -> List[Path]:
    """znajduje pliki testow poza katalogiem tests/"""
    misplaced = []
    for py_file in root.rglob("test_*.py"):
        if "__pycache__" in str(py_file):
            continue
        # sprawdza czy plik jest poza tests/
        if "tests" not in py_file.parts:
            misplaced.append(py_file)
    return misplaced


def find_misplaced_markdown(root: Path) -> List[Path]:
    """znajduje pliki md poza katalogiem markdown/"""
    misplaced = []
    for md_file in root.rglob("*.md"):
        if ".git" in str(md_file):
            continue
        # sprawdza czy plik jest w root lub w podkatalogach (nie w markdown/)
        relative = md_file.relative_to(root)
        if relative.parts[0] != "markdown" and len(relative.parts) > 1:
            # plik jest w podkatalogu ale nie w markdown/
            misplaced.append(md_file)
        elif len(relative.parts) == 1 and relative.name not in ["README.md"]:
            # plik jest w root ale nie jest README.md
            misplaced.append(md_file)
    return misplaced


def main():
    root = Path(__file__).parent.parent

    print("=" * 70)
    print("ANALIZA PROJEKTU")
    print("=" * 70)

    # pliki powyzej 120 linii
    print("\n1. PLIKI PYTHON POWYZEJ 120 LINII:")
    print("-" * 70)
    large_files = find_large_files(root, threshold=120)
    if large_files:
        for filepath, lines in large_files:
            rel_path = filepath.relative_to(root)
            print(f"  {rel_path}: {lines} linii")
    else:
        print("  Brak plikow powyzej 120 linii")

    # testy poza tests/
    print("\n2. PLIKI TESTOW POZA KATALOGIEM tests/:")
    print("-" * 70)
    misplaced_tests = find_misplaced_tests(root)
    if misplaced_tests:
        for filepath in misplaced_tests:
            rel_path = filepath.relative_to(root)
            print(f"  {rel_path}")
    else:
        print("  Brak plikow testow poza tests/")

    # markdown poza markdown/
    print("\n3. PLIKI MARKDOWN POZA KATALOGIEM markdown/:")
    print("-" * 70)
    misplaced_md = find_misplaced_markdown(root)
    if misplaced_md:
        for filepath in misplaced_md:
            rel_path = filepath.relative_to(root)
            print(f"  {rel_path}")
    else:
        print("  Brak plikow markdown poza markdown/")

    print("\n" + "=" * 70)
    print("PODSUMOWANIE:")
    print(f"  - Duze pliki: {len(large_files)}")
    print(f"  - Testy do przeniesienia: {len(misplaced_tests)}")
    print(f"  - Markdown do przeniesienia: {len(misplaced_md)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
