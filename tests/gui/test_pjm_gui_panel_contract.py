from pathlib import Path


def test_pjm_panel_files_exist() -> None:
    # minimalny test kontraktu: pliki panelu musza istniec
    assert Path("app/gui/pjm_tools.py").exists(), "brak app/gui/pjm_tools.py"
    assert Path("tools/train_model.py").exists(), "brak tools/train_model.py"


def test_pjm_gui_has_no_feature_mode_dropdown() -> None:
    # sprawdza, ze UX nie pokazuje wyboru 63/189
    content = Path("app/gui/ui_components.py").read_text(encoding="utf-8")
    assert "Feature mode" not in content
    assert "Full 189D" not in content
    assert "Runtime-compatible 63D" not in content


def test_default_vectors_csv_path_exists() -> None:
    # domyslna sciezka datasetu ma byc stabilna w repo
    default_csv = Path("app/sign_language/data/raw/PJM-vectors.csv")
    assert default_csv.exists(), f"brak datasetu: {default_csv}"
