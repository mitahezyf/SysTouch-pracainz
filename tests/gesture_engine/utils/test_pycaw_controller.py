import app.gesture_engine.utils.pycaw_controller as pc


def test_set_and_get_volume_non_windows(monkeypatch):
    # symulacja nie-Windows
    monkeypatch.setattr(pc.sys, "platform", "linux")

    # nie powinno rzucac
    pc.set_system_volume(50)
    assert pc.get_system_volume() == 0


def test_set_and_get_volume_windows_stubbed(monkeypatch):
    # symulacja Windows z zamockowanym interfejsem pycaw
    monkeypatch.setattr(pc.sys, "platform", "win32")

    class DummyVolume:
        def __init__(self):
            self.scalar = 0.0

        def SetMasterVolumeLevelScalar(self, value, _):
            # przechowuje ustawiony poziom
            self.scalar = float(value)

        def GetMasterVolumeLevelScalar(self):
            return self.scalar

    dummy = DummyVolume()
    monkeypatch.setattr(pc, "_get_volume_interface", lambda: dummy)

    # set klampuje do [0,1]
    pc.set_system_volume(120)
    assert dummy.scalar == 1.0
    assert pc.get_system_volume() == 100

    pc.set_system_volume(-10)
    assert dummy.scalar == 0.0
    assert pc.get_system_volume() == 0


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
