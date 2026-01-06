import pathlib

import numpy as np
import pytest

from app.sign_language.features import from_points25
from tools.verify_feature_parity import parse_points, parse_vectors
from tools.verify_mediapipe_reconstruction_parity import (
    build_landmarks21_from_points25,
)

DATA_DIR = pathlib.Path("app/sign_language/data/raw")
POINTS_PATH = DATA_DIR / "PJM-points.csv"
VECTORS_PATH = DATA_DIR / "PJM-vectors.csv"


@pytest.mark.skipif(
    not POINTS_PATH.exists() or not VECTORS_PATH.exists(), reason="brak danych PJM"
)
def test_feature_parity_small_sample():
    n = 50
    tol_hand = 0.08
    tol_bones = 0.005

    with POINTS_PATH.open(newline="", encoding="utf-8") as fp, VECTORS_PATH.open(
        newline="", encoding="utf-8"
    ) as fv:
        points_reader = iter(list(__import__("csv").DictReader(fp)))
        vectors_reader = iter(list(__import__("csv").DictReader(fv)))
        count = 0
        for p_row, v_row in zip(points_reader, vectors_reader):
            if count >= n:
                break
            points25 = parse_points(p_row)
            feat = from_points25(points25)
            vec_expected = parse_vectors(v_row)
            diff = np.abs(feat - vec_expected)
            assert diff[:3].max() <= tol_hand
            assert diff[3:].max() <= tol_bones
            count += 1
        assert count > 0


@pytest.mark.skipif(not POINTS_PATH.exists(), reason="brak danych PJM")
def test_reconstruction_parity_small_sample():
    # test sprawdza parytet rekonstrukcji 21->25 punktow
    # UWAGA: from_mediapipe_landmarks odwraca Y dla danych z kamery,
    # ale dane z PJM nie wymagaja odwrocenia, wiec porownujemy bezposrednio z from_points25
    n = 50
    tol_hand = 0.08
    tol_bones = 0.005

    import csv

    from app.sign_language.features import (
        FeatureConfig,
        _build_points25_from_mediapipe21,
        _features_from_points25,
    )

    with POINTS_PATH.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        count = 0
        for row in reader:
            if count >= n:
                break
            points25 = parse_points(row)
            landmarks21 = build_landmarks21_from_points25(points25)
            feat_gold = from_points25(points25)
            # bez odwrocenia Y - dane PJM sa juz w prawidlowym ukladzie
            pts25_recon = _build_points25_from_mediapipe21(landmarks21)
            feat_mp = _features_from_points25(
                pts25_recon, handedness="Right", cfg=FeatureConfig()
            )
            diff = np.abs(feat_gold - feat_mp)
            assert diff[:3].max() <= tol_hand
            assert diff[3:].max() <= tol_bones
            count += 1
        assert count > 0
