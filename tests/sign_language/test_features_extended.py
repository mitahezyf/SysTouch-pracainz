# -*- coding: utf-8 -*-
"""Additional tests for sign_language features to increase coverage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from app.sign_language.features import (
    FeatureExtractor,
    from_mediapipe_landmarks,
    from_points25,
)


def test_from_mediapipe_landmarks_basic():
    """Test basic feature extraction from MediaPipe landmarks."""
    # Create fake 21x3 landmarks
    landmarks = np.random.rand(21, 3).astype(np.float32)
    features = from_mediapipe_landmarks(landmarks)

    # Should return 63 features (hand normal + 20 bone vectors)
    assert features.shape == (63,)
    assert features.dtype == np.float32


def test_from_mediapipe_landmarks_with_handedness():
    """Test feature extraction with handedness (mirroring for left hand)."""
    landmarks = np.random.rand(21, 3).astype(np.float32)

    features_right = from_mediapipe_landmarks(landmarks, handedness="Right")
    features_left = from_mediapipe_landmarks(landmarks, handedness="Left")

    # Both should have 63 features
    assert features_right.shape == (63,)
    assert features_left.shape == (63,)

    # Left hand features should be mirrored (different from right)
    assert not np.array_equal(features_right, features_left)


def test_from_points25():
    """Test feature extraction from 25-point format."""
    # Create fake 25x3 points
    points25 = np.random.rand(25, 3).astype(np.float32)
    features = from_points25(points25)

    # Should return 63 features
    assert features.shape == (63,)


def test_feature_extractor_single():
    """Test FeatureExtractor for single landmark set."""
    extractor = FeatureExtractor()
    landmarks = np.random.rand(21, 3).astype(np.float32)

    features = extractor.extract(landmarks)
    assert features.shape == (63,)


def test_feature_extractor_batch():
    """Test FeatureExtractor for batch of landmarks."""
    extractor = FeatureExtractor()
    batch = np.random.rand(10, 21, 3).astype(np.float32)

    features = extractor.extract_batch(batch)
    assert features.shape == (10, 63)


def test_feature_extractor_batch_with_handedness():
    """Test FeatureExtractor batch with handedness list."""
    extractor = FeatureExtractor()
    batch = np.random.rand(5, 21, 3).astype(np.float32)
    handedness_list = ["Right", "Left", "Right", "Left", "Right"]

    features = extractor.extract_batch(batch, handedness_list)
    assert features.shape == (5, 63)
