# testy jednostkowe dla normalizera PyTorch
import numpy as np
import pytest

from app.sign_language.normalizer import MediaPipeNormalizer, normalize_landmarks


def test_normalizer_basic():
    # test podstawowej normalizacji 21 punktow
    normalizer = MediaPipeNormalizer()

    # sztuczne landmarki: nadgarstek w (0,0,0), middle_mcp w (1,0,0)
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0, 0, 0]  # nadgarstek
    landmarks[9] = [1, 0, 0]  # middle_mcp - scale=1

    result = normalizer.normalize(landmarks)

    assert result.shape == (63,)
    assert result.dtype == np.float32
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()

    # pierwszy punkt (nadgarstek) powinien byc (0,0,0) po normalizacji
    assert np.allclose(result[0:3], [0, 0, 0], atol=1e-5)


def test_normalizer_batch():
    # test normalizacji batcha
    normalizer = MediaPipeNormalizer()

    # batch 2 probek
    batch = np.zeros((2, 21, 3), dtype=np.float32)
    batch[0, 0] = [0, 0, 0]
    batch[0, 9] = [1, 0, 0]
    batch[1, 0] = [0, 0, 0]
    batch[1, 9] = [2, 0, 0]  # inna skala

    result = normalizer.normalize_batch(batch)

    assert result.shape == (2, 63)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_normalizer_zero_scale():
    # test zabezpieczenia przed dzieleniem przez zero (wszystkie punkty w tym samym miejscu)
    normalizer = MediaPipeNormalizer()

    landmarks = np.zeros((21, 3), dtype=np.float32)  # wszystkie w (0,0,0)

    result = normalizer.normalize(landmarks)

    # powinien zwrocic wektor zerowy lub nie crashowac
    assert result.shape == (63,)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_normalizer_invalid_shape():
    # test walidacji ksztaltu
    normalizer = MediaPipeNormalizer()

    with pytest.raises(ValueError, match="Oczekiwano ksztaltu"):
        normalizer.normalize(np.zeros((10, 3)))  # zla liczba punktow


def test_normalize_landmarks_function():
    # test funkcji kompatybilnej z API gesture_trainer
    landmarks_list = [(0, 0, 0)] + [(i * 0.1, i * 0.1, i * 0.1) for i in range(1, 21)]

    result = normalize_landmarks(landmarks_list)

    assert result.shape == (63,)
    assert result.dtype == np.float32
    assert not np.isnan(result).any()
