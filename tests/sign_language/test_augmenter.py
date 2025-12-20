import numpy as np

from app.sign_language.augmenter import (
    add_gaussian_noise,
    augment_class_samples,
    augment_sample,
    rotate_landmarks,
    scale_landmarks,
    translate_landmarks,
)


def test_rotate_landmarks_preserves_shape():
    # wektor 63D (21 punktow x,y,z)
    landmarks = np.random.rand(63).astype(np.float32)
    rotated = rotate_landmarks(landmarks, 45.0)

    assert rotated.shape == landmarks.shape
    assert rotated.dtype == landmarks.dtype


def test_rotate_landmarks_changes_values():
    landmarks = np.random.rand(63).astype(np.float32)
    rotated = rotate_landmarks(landmarks, 45.0)

    # po obrocie wartosci powinny sie zmienic
    assert not np.allclose(rotated, landmarks)


def test_translate_landmarks_preserves_shape():
    landmarks = np.random.rand(63).astype(np.float32)
    translated = translate_landmarks(landmarks, 0.02)

    assert translated.shape == landmarks.shape


def test_scale_landmarks_preserves_shape():
    landmarks = np.random.rand(63).astype(np.float32)
    scaled = scale_landmarks(landmarks, 0.1)

    assert scaled.shape == landmarks.shape


def test_add_gaussian_noise_preserves_shape():
    landmarks = np.random.rand(63).astype(np.float32)
    noisy = add_gaussian_noise(landmarks, 0.01)

    assert noisy.shape == landmarks.shape


def test_augment_sample_preserves_shape():
    landmarks = np.random.rand(63).astype(np.float32)
    augmented = augment_sample(landmarks)

    assert augmented.shape == landmarks.shape


def test_augment_sample_changes_values():
    np.random.seed(42)  # dla powtarzalnosci
    landmarks = np.random.rand(63).astype(np.float32)
    augmented = augment_sample(landmarks)

    # augmentacja powinna zmienic przynajmniej niektore wartosci
    assert not np.allclose(augmented, landmarks, rtol=1e-3)


def test_augment_class_samples_increases_size():
    X = np.random.rand(100, 63).astype(np.float32)
    y = np.random.randint(0, 5, size=100)

    # augmentuj klase 2 (10x)
    X_aug, y_aug = augment_class_samples(X, y, class_label=2, multiplier=10)

    # rozmiar powinien wzrosnac
    assert len(X_aug) > len(X)
    assert len(y_aug) == len(X_aug)

    # liczba probek klasy 2 powinna wzrosnac 10x
    original_count = np.sum(y == 2)
    augmented_count = np.sum(y_aug == 2)
    assert augmented_count == original_count * 10


def test_augment_class_samples_empty_class():
    X = np.random.rand(100, 63).astype(np.float32)
    y = np.random.randint(0, 5, size=100)

    # augmentuj nieistniejaca klase
    X_aug, y_aug = augment_class_samples(X, y, class_label=99, multiplier=10)

    # rozmiar nie powinien sie zmienic
    assert len(X_aug) == len(X)
    assert len(y_aug) == len(y)


def test_augment_class_samples_preserves_other_classes():
    X = np.random.rand(100, 63).astype(np.float32)
    y = np.random.randint(0, 5, size=100)

    # zlicz probki pozostalych klas
    class_to_augment = 2
    other_classes_count = {i: np.sum(y == i) for i in range(5) if i != class_to_augment}

    X_aug, y_aug = augment_class_samples(
        X, y, class_label=class_to_augment, multiplier=5
    )

    # pozostale klasy powinny miec te sama liczbe probek
    for cls, count in other_classes_count.items():
        assert np.sum(y_aug == cls) == count
