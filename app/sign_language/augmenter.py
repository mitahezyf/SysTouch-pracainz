from typing import Tuple

import numpy as np


def rotate_landmarks(landmarks: np.ndarray, angle_deg: float) -> np.ndarray:
    """obraca landmarki o zadany kat wokol centroidu.

    Args:
        landmarks: tablica shape (63,) reprezentujaca 21 punktow (x,y,z)
        angle_deg: kat obrotu w stopniach

    Returns:
        obrocone landmarki o tym samym shape
    """
    original_dtype = landmarks.dtype

    # reshape do (21, 3)
    points = landmarks.reshape(21, 3)

    # centroid (srednia pozycja)
    centroid = np.mean(points, axis=0)

    # przesun do originu
    centered = points - centroid

    # macierz rotacji wokol osi Z
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    # obrot
    rotated = centered @ rotation_matrix.T

    # przesun z powrotem
    result = rotated + centroid

    flattened = result.flatten().astype(original_dtype)
    return np.asarray(flattened, dtype=original_dtype)


def translate_landmarks(landmarks: np.ndarray, shift_ratio: float) -> np.ndarray:
    """przesuwa landmarki o losowy offset.

    Args:
        landmarks: tablica shape (63,)
        shift_ratio: maksymalny stosunek przesuniecia do zakresu danych (np. 0.02 = 2%)

    Returns:
        przesuniete landmarki
    """
    points = landmarks.reshape(21, 3)

    # zakres danych
    data_range = np.ptp(points, axis=0)  # max - min dla kazdego wymiaru

    # losowe przesuniecie
    shift = (np.random.rand(3) * 2 - 1) * data_range * shift_ratio

    shifted = points + shift

    result_flat: np.ndarray = shifted.flatten()
    return result_flat


def scale_landmarks(landmarks: np.ndarray, scale_ratio: float) -> np.ndarray:
    """skaluje landmarki wzgledem centroidu.

    Args:
        landmarks: tablica shape (63,)
        scale_ratio: maksymalny stosunek skalowania (np. 0.1 = ±10%)

    Returns:
        przeskalowane landmarki
    """
    points = landmarks.reshape(21, 3)
    centroid = np.mean(points, axis=0)

    # losowy faktor skalowania
    scale_factor = 1.0 + (np.random.rand() * 2 - 1) * scale_ratio

    # skaluj wzgledem centroidu
    centered = points - centroid
    scaled = centered * scale_factor
    result = scaled + centroid

    flattened_result: np.ndarray = result.flatten()
    return flattened_result


def add_gaussian_noise(landmarks: np.ndarray, noise_std: float) -> np.ndarray:
    """dodaje szum gaussowski do landmarkow.

    Args:
        landmarks: tablica shape (63,)
        noise_std: odchylenie standardowe szumu wzgledem zakresu danych

    Returns:
        landmarki z szumem
    """
    points = landmarks.reshape(21, 3)

    # zakres danych
    data_range = np.ptp(points, axis=0)

    # szum proporcjonalny do zakresu
    noise = np.random.randn(21, 3) * data_range * noise_std

    noisy = points + noise

    flattened = noisy.flatten()
    return np.asarray(flattened, dtype=flattened.dtype)


def augment_sample(
    landmarks: np.ndarray,
    rotation_deg: float = 5.0,
    translation_ratio: float = 0.02,
    scale_ratio: float = 0.1,
    noise_std: float = 0.01,
) -> np.ndarray:
    """stosuje losowa kombinacje transformacji augmentacyjnych.

    Args:
        landmarks: tablica shape (63,) lub (88,) - cechy relatywne
        rotation_deg: maksymalny kat obrotu
        translation_ratio: maksymalny stosunek przesuniecia
        scale_ratio: maksymalny stosunek skalowania
        noise_std: odchylenie standardowe szumu

    Returns:
        augmentowane landmarki
    """
    # dla cech 88D, wyciagamy pierwsze 63 (surowe landmarks)
    # reszta (25 cech) to pochodne: katy, odleglosci, roll
    if len(landmarks) == 88:
        # ekstrahuj surowe landmarks (63D)
        raw_landmarks = landmarks[:63].copy()

        # augmentuj surowe landmarks
        result = raw_landmarks.copy()

        # losowy wybor transformacji (80% szansa na kazda)
        if np.random.rand() > 0.2:
            angle = (np.random.rand() * 2 - 1) * rotation_deg
            result = rotate_landmarks(result, angle)

        if np.random.rand() > 0.2:
            result = translate_landmarks(result, translation_ratio)

        if np.random.rand() > 0.2:
            result = scale_landmarks(result, scale_ratio)

        if np.random.rand() > 0.2:
            result = add_gaussian_noise(result, noise_std)

        # ponownie ekstrahuj cechy 88D z augmentowanych landmarks
        from app.sign_language.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor()
        result_reshaped = result.reshape(21, 3)
        return extractor.extract(result_reshaped)

    elif len(landmarks) == 63:
        # legacy - 63D
        result = landmarks.copy()

        # losowy wybor transformacji (80% szansa na kazda)
        if np.random.rand() > 0.2:
            angle = (np.random.rand() * 2 - 1) * rotation_deg
            result = rotate_landmarks(result, angle)

        if np.random.rand() > 0.2:
            result = translate_landmarks(result, translation_ratio)

        if np.random.rand() > 0.2:
            result = scale_landmarks(result, scale_ratio)

        if np.random.rand() > 0.2:
            result = add_gaussian_noise(result, noise_std)

        return result
    else:
        raise ValueError(
            f"Nieobslugiwany wymiar cech: {len(landmarks)}, oczekiwano 63 lub 88"
        )


def augment_class_samples(
    X: np.ndarray,
    y: np.ndarray,
    class_label: int,
    multiplier: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """augmentuje probki danej klasy.

    Args:
        X: dane wejsciowe shape (N, 63) lub (N, 88)
        y: etykiety shape (N,)
        class_label: indeks klasy do augmentacji
        multiplier: ile razy zwiekszyc liczbe probek tej klasy

    Returns:
        (X_augmented, y_augmented) - oryginalne + nowe probki
    """
    # znajdz probki danej klasy
    class_mask = y == class_label
    class_samples = X[class_mask]

    if len(class_samples) == 0:
        return X, y

    # generuj augmentowane probki
    augmented_samples = []
    for sample in class_samples:
        for _ in range(multiplier - 1):  # -1 bo oryginalna juz jest
            aug_sample = augment_sample(sample)
            augmented_samples.append(aug_sample)

    if len(augmented_samples) == 0:
        return X, y

    # polacz z oryginalnym zbiorem
    X_aug = np.vstack([X, np.array(augmented_samples)])
    y_aug = np.hstack([y, np.full(len(augmented_samples), class_label)])

    return X_aug, y_aug
