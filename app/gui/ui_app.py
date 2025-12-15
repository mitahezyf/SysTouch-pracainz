from __future__ import annotations

import importlib
import sys


def main() -> None:
    # uruchamia aplikacje gui oparta o pyside6
    # sprawdza dostepnosc pyside6 tworzy glowne okno i uruchamia petle zdarzen qt
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
