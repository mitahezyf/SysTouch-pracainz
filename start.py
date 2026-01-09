"""Punkt wejścia do aplikacji GUI."""

import sys
from pathlib import Path

# Dodaj katalog główny projektu do ścieżki, aby importy działały poprawnie
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.gui.ui_app import main

if __name__ == "__main__":
    main()
