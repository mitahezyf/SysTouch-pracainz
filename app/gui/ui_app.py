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

    # Ustaw Windows AppUserModelID aby odróżnić aplikację od Pythona
    try:
        import ctypes

        myappid = "mitahezyf.systouch.gui.1.0"  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass  # nie krytyczne jeśli się nie uda

    # Ustaw ikonę aplikacji dla całej aplikacji (taskbar, Start menu)
    from pathlib import Path

    QIcon = importlib.import_module("PySide6.QtGui").QIcon
    icon_path = Path(__file__).resolve().parent.parent / "SysTouchIco.jpg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    win = create_main_window()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
