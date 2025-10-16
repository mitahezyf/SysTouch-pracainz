import app.gesture_engine.utils.pycaw_controller as pc


def test_set_get_volume_windows_interface_exception(monkeypatch):
    # symuluje Windows i wyjatek w _get_volume_interface
    monkeypatch.setattr(pc.sys, "platform", "win32")
    monkeypatch.setattr(
        pc, "_get_volume_interface", lambda: (_ for _ in ()).throw(Exception("boom"))
    )

    # set powinien byc no-op bez wyjatku
    pc.set_system_volume(50)
    # get powinien zwrocic 0 przy wyjatku interfejsu
    assert pc.get_system_volume() == 0
