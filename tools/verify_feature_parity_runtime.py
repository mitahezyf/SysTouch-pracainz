import argparse
import logging
from pathlib import Path

import joblib
import numpy as np

from app.sign_language.features import from_mediapipe_landmarks
from tools.train_model import runtime_feature_names

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sprawdza parzystosc runtime feature buildera z feature_cols zapisanymi w modelu"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("app/sign_language/models/pjm_model.joblib"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    artifact = joblib.load(args.model)
    feature_cols: list[str] = list(artifact["feature_cols"])

    expected = runtime_feature_names()

    if feature_cols != expected:
        logger.error(
            "feature_cols mismatch: model=%d expected=%d",
            len(feature_cols),
            len(expected),
        )
        for i in range(min(len(feature_cols), len(expected), 20)):
            if feature_cols[i] != expected[i]:
                logger.error(
                    "mismatch at %d: model=%s expected=%s",
                    i,
                    feature_cols[i],
                    expected[i],
                )
        raise SystemExit(2)

    dummy = np.random.randn(21, 3).astype(np.float32)
    feat = from_mediapipe_landmarks(dummy, handedness="Right")

    if feat.shape != (len(feature_cols),):
        logger.error(
            "shape mismatch: runtime=%s model=%d", feat.shape, len(feature_cols)
        )
        raise SystemExit(2)

    logger.info("OK: feature_cols zgodne, runtime shape=%s", feat.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
