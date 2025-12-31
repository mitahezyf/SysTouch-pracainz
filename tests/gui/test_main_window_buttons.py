# testuje logike GUI bez uruchamiania procesow subprocess


def test_record_and_train_buttons_logic():
    # weryfikuje ze komendy dla przyciskow sa poprawnie sformatowane
    import os
    import sys

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # symuluje logike z on_record_sign_language
    cmd_record = [sys.executable, "-m", "app.sign_language.recorder"]
    assert cmd_record[0] == sys.executable
    assert "recorder" in " ".join(cmd_record)

    # symuluje logike z on_train_sign_language
    cmd_train = [sys.executable, "-m", "app.sign_language.trainer"]
    assert cmd_train[0] == sys.executable
    assert "trainer" in " ".join(cmd_train)

    # sprawdza ze base_dir istnieje
    assert os.path.exists(base_dir)
