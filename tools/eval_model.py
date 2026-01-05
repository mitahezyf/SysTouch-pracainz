import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ewaluacja modelu PJM joblib na csv vector_*"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("app/sign_language/models/pjm_model.joblib"),
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=Path("app/sign_language/data/raw/PJM-vectors.csv"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    artifact = joblib.load(args.model)
    pipe = artifact["pipeline"]
    le = artifact["label_encoder"]
    feature_cols = artifact["feature_cols"]

    df = pd.read_csv(args.vectors)
    df = df.dropna(subset=feature_cols + ["sign_label"])

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["sign_label"].astype(str).to_numpy()
    y = le.transform(y_raw)

    y_pred = pipe.predict(X)

    report = classification_report(
        y, y_pred, target_names=le.classes_.tolist(), zero_division=0
    )
    cm = confusion_matrix(y, y_pred)

    logger.info("\n" + "=" * 60)
    logger.info("classification_report (na calym csv):\n%s", report)
    logger.info("confusion_matrix:\n%s", cm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
