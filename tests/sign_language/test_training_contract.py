# test kontraktowy: sprawdza czy trening jest dostepny z GUI i CLI
from pathlib import Path


def test_training_entrypoint_exists_and_importable():
    # sprawdza czy narzedzie do trenowania istnieje i jest dostepne
    train_path = Path("tools/train_model.py")
    assert (
        train_path.exists()
    ), "brak tools/train_model.py - GUI nie moze uruchomic treningu"

    # importowalnosc
    try:
        import tools.train_model
    except Exception as exc:
        raise AssertionError(f"tools.train_model nie jest importowalny: {exc}")

    # sprawdza ze main() istnieje
    assert hasattr(tools.train_model, "main"), "tools.train_model nie ma funkcji main()"


def test_gui_does_not_reference_nonexistent_train_path():
    # sprawdza ze GUI nie wskazuje na nieistniejacy plik treningu
    # (nie moze byc hardcoded app/gui/tools/train_model.py)
    from pathlib import Path

    gui_main_window = Path("app/gui/main_window.py")
    assert gui_main_window.exists(), "brak app/gui/main_window.py"

    content = gui_main_window.read_text(encoding="utf-8")

    # sprawdza ze nie ma zlej sciezki app\\gui\\tools\\train_model.py
    forbidden = "app/gui/tools/train_model.py"
    forbidden_win = "app\\gui\\tools\\train_model.py"

    assert forbidden not in content, f"GUI zawiera nieistniejaca sciezke: {forbidden}"
    assert (
        forbidden_win not in content
    ), f"GUI zawiera nieistniejaca sciezke: {forbidden_win}"

    # trening teraz przez app.sign_language.trainer - nie wymagamy tools/train_model.py
