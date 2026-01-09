#!/usr/bin/env python3
"""
Skrypt ujednolicajacy komentarze w projekcie
Usuwa znaki diakrytyczne z komentarzy w plikach Python
Data: 2025-12-16
"""
import sys
from pathlib import Path


def unify_comments_in_file(filepath: Path) -> bool:
    """
    Ujednolica komentarze w pojedynczym pliku Python.
    Zwraca True jesli plik zostal zmodyfikowany.
    """
    # Mapowanie znakow diakrytycznych na ASCII
    replacements = {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ź": "Z",
        "Ż": "Z",
    }

    try:
        content = filepath.read_text(encoding="utf-8")
        original_content = content
        lines = content.split("\n")
        modified_lines = []

        for line in lines:
            if "#" in line:
                # Znajdz pozycje pierwszego komentarza (poza stringami)
                in_string = False
                string_char = None
                comment_start = -1

                for i, char in enumerate(line):
                    if char in ('"', "'") and (i == 0 or line[i - 1] != "\\"):
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                    elif char == "#" and not in_string:
                        comment_start = i
                        break

                if comment_start >= 0:
                    before_comment = line[:comment_start]
                    comment = line[comment_start:]

                    # Zamien znaki diakrytyczne tylko w komentarzu
                    for old, new in replacements.items():
                        comment = comment.replace(old, new)

                    modified_lines.append(before_comment + comment)
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)

        new_content = "\n".join(modified_lines)

        if new_content != original_content:
            filepath.write_text(new_content, encoding="utf-8")
            return True

        return False

    except Exception as e:
        print(f"  ERR {filepath.name}: {e}", file=sys.stderr)
        return False


def main():
    """Glowna funkcja"""
    project_root = Path(__file__).parent

    print("=" * 70)
    print("UJEDNOLICANIE KOMENTARZY W PROJEKCIE")
    print("=" * 70)
    print()
    print("Wyszukiwanie plikow Python...")
    print()

    # Znajdz wszystkie pliki Python
    py_files = []
    py_files.extend(project_root.glob("app/**/*.py"))
    py_files.extend(project_root.glob("tests/**/*.py"))

    # Filtruj pliki (pomijaj venv, pycache)
    py_files = [
        f for f in py_files if ".venv" not in str(f) and "__pycache__" not in str(f)
    ]

    print(f"Znaleziono {len(py_files)} plikow Python")
    print()
    print("Przetwarzanie...")
    print()

    modified_count = 0

    for py_file in sorted(py_files):
        rel_path = py_file.relative_to(project_root)

        if unify_comments_in_file(py_file):
            print(f"  OK  {rel_path}")
            modified_count += 1
        else:
            # Nie drukuj "---" dla kazdego pliku, tylko zlicz
            pass

    print()
    print("=" * 70)
    print(f"ZAKONCZONE: {modified_count} plikow zmodyfikowano")
    print("=" * 70)
    print()

    if modified_count > 0:
        print("Zmodyfikowane pliki:")
        print(f"  - {modified_count} plikow z usuniętymi znakami diakrytycznymi")
        print()
        print("Komentarze sa teraz ujednolicone (bez ą, ć, ę, ł, ń, ó, ś, ź, ż)")
    else:
        print("Wszystkie pliki juz byly ujednolicone!")

    print()


if __name__ == "__main__":
    main()
