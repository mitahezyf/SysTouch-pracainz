from __future__ import annotations

import importlib
import sys


def main() -> None:
    """Uruchamia aplikacje GUI oparta o PySide6.

    - sprawdza dostepnosc PySide6
    - tworzy glowne okno i uruchamia petle zdarzen Qt
    """
    try:
        qtw = importlib.import_module("PySide6.QtWidgets")
    except Exception as e:  # pragma: no cover - srodowiska bez PySide6
        raise RuntimeError(
            "PySide6 nie jest zainstalowane. Zainstaluj pakiet 'PySide6' aby uruchomic GUI."
        ) from e

    QApplication = qtw.QApplication
    from app.gui.window import create_main_window

    app = QApplication(sys.argv)
    win = create_main_window()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
