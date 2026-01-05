import os
from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    # repo root to katalog nadrzedny tests/
    return Path(__file__).resolve().parents[1]


def _find_vectors_csv(repo_root: Path) -> Path | None:
    # szuka PJM-vectors.csv w repo, zeby testy dzialaly mimo zmian sciezek
    # priorytet: standardowa sciezka aplikacji
    preferred = repo_root / "app" / "sign_language" / "data" / "raw" / "PJM-vectors.csv"
    if preferred.exists():
        return preferred

    matches = list(repo_root.rglob("PJM-vectors.csv"))
    return matches[0] if matches else None


def _find_model_joblib(repo_root: Path) -> Path | None:
    preferred = repo_root / "app" / "sign_language" / "models" / "pjm_model.joblib"
    if preferred.exists():
        return preferred
    matches = list(repo_root.rglob("pjm_model.joblib"))
    return matches[0] if matches else None


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _find_repo_root()


@pytest.fixture(scope="session")
def vectors_csv_path(repo_root: Path) -> Path:
    # pozwala nadpisac sciezke z env dla CI
    env = os.environ.get("PJM_VECTORS_CSV")
    if env:
        p = Path(env)
        if not p.exists():
            raise AssertionError(f"PJM_VECTORS_CSV wskazuje nieistniejacy plik: {p}")
        return p

    found = _find_vectors_csv(repo_root)
    if found is None:
        raise AssertionError(
            "Brak PJM-vectors.csv w repo. "
            "Ustaw env PJM_VECTORS_CSV lub dodaj plik do app/sign_language/data/raw/"
        )
    return found


@pytest.fixture(scope="session")
def model_joblib_path(repo_root: Path) -> Path:
    env = os.environ.get("PJM_MODEL_JOBLIB")
    if env:
        p = Path(env)
        if not p.exists():
            raise AssertionError(f"PJM_MODEL_JOBLIB wskazuje nieistniejacy plik: {p}")
        return p

    found = _find_model_joblib(repo_root)
    # model moze nie istniec - testy modelu beda failowac z instrukcja treningu
    return found or (
        repo_root / "app" / "sign_language" / "models" / "pjm_model.joblib"
    )
