"""fixtures dla testow actions - mockuje pycaw_controller aby uniknac access violation COM."""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_pycaw_controller(monkeypatch):
    """mockuje set_system_volume i poke_volume_osd, aby testy nie uzywaly prawdziwego COM"""
    if sys.platform == "win32":
        # mockuj funkcje importowane w volume_action
        mock_set = MagicMock()
        mock_poke = MagicMock()
        monkeypatch.setattr(
            "app.gesture_engine.actions.volume_action.set_system_volume", mock_set
        )
        monkeypatch.setattr(
            "app.gesture_engine.actions.volume_action.poke_volume_osd", mock_poke
        )
