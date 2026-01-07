import numpy as np
import pytest

from app.gesture_trainer.dataset_collector import build_row, csv_header


def test_csv_header_counts() -> None:
    cols = csv_header(include_landmarks=True, include_features=True)
    assert cols[:8] == [
        "session_id",
        "clip_id",
        "label",
        "frame_idx",
        "timestamp_ms",
        "handedness",
        "has_hand",
        "mirror_applied",
    ]
    assert len(cols) == 8 + (21 * 3) + 63


@pytest.mark.parametrize(
    ("include_landmarks", "include_features"),
    [(True, True), (True, False), (False, True)],
)
def test_build_row_shapes(include_landmarks: bool, include_features: bool) -> None:
    lm = np.zeros((21, 3), dtype=np.float32)
    feat = np.zeros((63,), dtype=np.float32)

    row = build_row(
        session_id="s",
        clip_id="c",
        label="A",
        frame_idx=1,
        timestamp_ms=10,
        handedness="Left",
        has_hand=True,
        mirror_applied=True,
        landmarks21=lm,
        features63=feat,
        include_landmarks=include_landmarks,
        include_features=include_features,
    )

    expected = 8
    if include_landmarks:
        expected += 63
    if include_features:
        expected += 63

    assert len(row) == expected


def test_build_row_empty_when_missing_hand() -> None:
    row = build_row(
        session_id="s",
        clip_id="c",
        label="A",
        frame_idx=1,
        timestamp_ms=10,
        handedness=None,
        has_hand=False,
        mirror_applied=False,
        landmarks21=None,
        features63=None,
        include_landmarks=True,
        include_features=True,
    )
    # po bazowych 8 kolumnach powinny byc puste komorki dla danych
    assert row[0:8] == ["s", "c", "A", 1, 10, "", 0, 0]
    assert all(x == "" for x in row[8:])


def test_build_row_rejects_bad_features_shape() -> None:
    with pytest.raises(ValueError, match="niepoprawny ksztalt features"):
        build_row(
            session_id="s",
            clip_id="c",
            label="A",
            frame_idx=1,
            timestamp_ms=10,
            handedness=None,
            has_hand=True,
            mirror_applied=False,
            landmarks21=np.zeros((21, 3), dtype=np.float32),
            features63=np.zeros((62,), dtype=np.float32),
            include_landmarks=True,
            include_features=True,
        )
