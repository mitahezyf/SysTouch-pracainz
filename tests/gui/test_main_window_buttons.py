import importlib
import sys
import types

import pytest


def test_main_window_has_record_and_train_buttons(monkeypatch):
    # pomin test jezeli brak PySide6
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    # zapewnia singleton QApplication
    if QApplication.instance() is None:
        QApplication([])

    gui_main = importlib.import_module("app.gui.main_window")
    MainWindow = getattr(gui_main, "MainWindow")
    win = MainWindow()
    assert hasattr(win, "record_btn")
    assert hasattr(win, "train_btn")

    called = {"record": False, "train": False}

    class DummyPopen:
        def __init__(self, cmd, cwd=None):  # noqa: D401
            if cmd and isinstance(cmd, (list, tuple)):
                tail = cmd[-1]
                if "recorder" in tail:
                    called["record"] = True
                if "trainer" in tail:
                    called["train"] = True

    # tworzy sztuczny modul subprocess z DummyPopen
    fake_subprocess = types.ModuleType("subprocess")
    fake_subprocess.Popen = DummyPopen  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

    win.on_record_sign_language()
    win.on_train_sign_language()

    assert called["record"] is True
    assert called["train"] is True

    # zamyka okno bez wygaszania aplikacji
    win.close()
