#!/usr/bin/env python3
"""
Skrypt do czyszczenia historii Git z niepotrzebnych plików
Używa BFG Repo-Cleaner (szybszy niż git filter-branch)
"""
import subprocess
import sys
from pathlib import Path

# Pliki do usunięcia z historii Git
FILES_TO_REMOVE = [
    # Pliki tymczasowe/testowe które były w root
    "systemSpec",
    "run_gui_with_logging.py",
    "test_pycaw_quick.py",
    "junit.xml",
    ".coveragerc",
    "ruff.toml",
    "copy.txt",
    # Pliki w reports/ (generowane)
    "reports/junit.xml",
    # Stara dokumentacja która została przeniesiona
    "docs/GESTURE_PIPELINE.md",
    "app/sign_language/data/README.md",
    # Stare pliki które zostały zreorganizowane
    "actions/handlers.py",
    "actions/move_mouse_action.py",
    "actions/scroll_action.py",
    "config.py",
    "app/logger.py",
    "detector/__init__.py",
    "detector/gesture_detector.py",
    "detector/hand_tracker.py",
    "gestures/__init__.py",
    "gestures/click_gesture.py",
    "gestures/close_program_gesture.py",
]

# Katalogi do całkowitego usunięcia
DIRS_TO_REMOVE = [
    "KOSZ/",  # Jeśli był kiedyś commitowany
    "markdown/",  # Dokumentacja pomocnicza
    "reports/",  # Raporty
    "logs_debug/",  # Logi
]


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Uruchamia komendę i zwraca kod, stdout, stderr"""
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return result.returncode, result.stdout, result.stderr


def check_git_status():
    """Sprawdza czy są niezacommitowane zmiany"""
    code, stdout, _ = run_command(["git", "status", "--porcelain"])
    return len(stdout.strip()) > 0


def main():
    print("=" * 70)
    print("CZYSZCZENIE HISTORII GIT")
    print("=" * 70)
    print()

    # Sprawdź czy jesteśmy w repo Git
    if not Path(".git").exists():
        print("[ERROR] Błąd: To nie jest katalog Git!")
        return 1

    # Sprawdź niezacommitowane zmiany
    if check_git_status():
        print("[WARNING]  UWAGA: Masz niezacommitowane zmiany!")
        print()
        print("Opcje:")
        print("1. Commit obecne zmiany i kontynuuj czyszczenie")
        print("2. Stash zmiany i kontynuuj")
        print("3. Przerwij")
        print()
        choice = input("Wybór (1/2/3): ").strip()

        if choice == "1":
            print("\n[COMMIT] Commitowanie zmian...")
            run_command(["git", "add", "-A"])
            code, _, _ = run_command(
                ["git", "commit", "-m", "chore: przed czyszczeniem historii Git"]
            )
            if code != 0:
                print("[ERROR] Błąd podczas commit!")
                return 1
            print("[OK] Zmiany zacommitowane")

        elif choice == "2":
            print("\n[STASH] Stashowanie zmian...")
            code, _, _ = run_command(
                ["git", "stash", "push", "-u", "-m", "Przed czyszczeniem historii"]
            )
            if code != 0:
                print("[ERROR] Błąd podczas stash!")
                return 1
            print("[OK] Zmiany schowane w stash")

        else:
            print("[ERROR] Przerwano przez użytkownika")
            return 0

    print("\n" + "=" * 70)
    print("METODA: git filter-branch")
    print("=" * 70)
    print()
    print(f"Plików do usunięcia: {len(FILES_TO_REMOVE)}")
    print(f"Katalogów do usunięcia: {len(DIRS_TO_REMOVE)}")
    print()

    # Pokaż próbkę plików
    print("Przykładowe pliki do usunięcia:")
    for f in FILES_TO_REMOVE[:10]:
        print(f"  - {f}")
    if len(FILES_TO_REMOVE) > 10:
        print(f"  ... i {len(FILES_TO_REMOVE) - 10} więcej")
    print()

    response = input("Czy kontynuować? (tak/nie): ").strip().lower()
    if response not in ["tak", "t", "yes", "y"]:
        print("[ERROR] Anulowano")
        return 0

    print("\n[CLEANUP] Czyszczenie historii Git...")
    print("To może potrwać kilka minut...\n")

    # Przygotuj listę plików do git rm
    files_str = " ".join(FILES_TO_REMOVE + DIRS_TO_REMOVE)

    # Użyj git filter-branch
    cmd = [
        "git",
        "filter-branch",
        "--force",
        "--index-filter",
        f"git rm -r --cached --ignore-unmatch {files_str}",
        "--prune-empty",
        "--tag-name-filter",
        "cat",
        "--",
        "--all",
    ]

    code, stdout, stderr = run_command(cmd)

    if code != 0:
        print(f"[ERROR] Błąd podczas czyszczenia!\n{stderr}")
        return 1

    print("[OK] Historia Git wyczyszczona!")
    print()

    # Cleanup refs
    print("[CLEANUP] Czyszczenie refs...")
    run_command(["git", "for-each-ref", "--format=%(refname)", "refs/original/"])
    run_command(["git", "update-ref", "-d", "refs/original/refs/heads/main"])

    # Expire reflog
    print("[CLEANUP] Czyszczenie reflog...")
    run_command(["git", "reflog", "expire", "--expire=now", "--all"])

    # Garbage collect
    print("[CLEANUP] Garbage collection...")
    run_command(["git", "gc", "--prune=now", "--aggressive"])

    print("\n" + "=" * 70)
    print("[OK] ZAKOŃCZONO!")
    print("=" * 70)
    print()
    print("[STATS] Statystyki repo:")

    # Pokaż rozmiar
    code, stdout, _ = run_command(["git", "count-objects", "-vH"])
    print(stdout)

    print("\n[WARNING]  WAŻNE:")
    print("1. Historia Git została przepisana - wszystkie commity mają nowe hash")
    print("2. Jeśli już pushowałeś na GitHub, musisz zrobić force push:")
    print("   git push origin --force --all")
    print("3. Inne osoby muszą sklonować repo na nowo (git clone)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
