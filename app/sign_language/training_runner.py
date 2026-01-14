"""
Runner dla pipeline treningu (konsolidacja + trening) z callbackami postępu.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from app.gesture_engine.logger import logger


def run_training_pipeline(
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    Uruchamia konsolidację danych + trening modelu z raportowaniem postępu.

    Args:
        progress_callback: funkcja(current, total, message) do raportowania postępu

    Returns:
        dict z wynikami treningu:
        {
            'success': bool,
            'test_accuracy': float,
            'val_accuracy': float,
            'num_classes': int,
            'classes': list[str],
            'train_samples': int,
            'val_samples': int,
            'test_samples': int,
            'per_class_stats': dict  # klasa -> {'samples': int, 'precision': float, ...}
        }
    """
    try:
        # 1. Konsolidacja (0-30%)
        if progress_callback:
            progress_callback(0, 100, "Konsolidacja próbek...")

        logger.info("[Training Pipeline] Rozpoczynam konsolidację danych...")
        base_dir = Path(__file__).resolve().parents[2]

        # Uruchom konsolidację
        consolidate_cmd = [sys.executable, "-m", "app.sign_language.dataset"]
        result = subprocess.run(
            consolidate_cmd, cwd=base_dir, capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Konsolidacja failed: {result.stderr}")

        if progress_callback:
            progress_callback(
                30, 100, "Konsolidacja zakończona, rozpoczynam trening..."
            )

        logger.info("[Training Pipeline] Konsolidacja zakończona, rozpoczynam trening")

        # 2. Trening (30-100%)
        from app.sign_language.trainer import train

        def epoch_callback(epoch: int, total_epochs: int):
            """Callback dla postępu epok."""
            if progress_callback:
                pct = 30 + int((epoch / total_epochs) * 70)
                progress_callback(pct, 100, f"Epoka {epoch}/{total_epochs}")

        # Uruchom trening z callbackiem
        train_results = train(progress_callback=epoch_callback)

        if progress_callback:
            progress_callback(100, 100, "Trening zakończony!")

        # 3. Załaduj dodatkowe statystyki
        from app.sign_language.dataset import load_processed_split

        X_train, y_train, meta_train = load_processed_split("train")
        X_val, y_val, meta_val = load_processed_split("val")
        X_test, y_test, meta_test = load_processed_split("test")

        classes = meta_train["classes"]

        # 4. Oblicz per-class statistics
        # Załaduj wytrenowany model i przetestuj
        import torch
        from sklearn.metrics import classification_report

        from app.sign_language.model import SignLanguageMLP

        model_path = "app/sign_language/models/pjm_model.pth"
        input_dim = X_test.shape[1]
        num_classes = len(classes)

        model = SignLanguageMLP(
            input_size=input_dim, hidden_size=256, num_classes=num_classes
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()

        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, test_predicted = torch.max(test_outputs, 1)

        # Classification report jako dict
        report_dict = classification_report(
            y_test,
            test_predicted.numpy(),
            labels=list(range(num_classes)),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )

        # Per-class stats
        per_class_stats = {}
        for i, cls in enumerate(classes):
            train_count = int(np.sum(y_train == i))
            per_class_stats[cls] = {
                "train_samples": train_count,
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1-score": report_dict[cls]["f1-score"],
                "support": int(report_dict[cls]["support"]),
            }

        # 5. Zwróć wyniki
        results = {
            "success": True,
            "test_accuracy": train_results["accuracy"],
            "val_accuracy": train_results["val_accuracy"],
            "num_classes": num_classes,
            "classes": list(classes),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "per_class_stats": per_class_stats,
        }

        logger.info("[Training Pipeline] Zakończono pomyślnie")
        return results

    except Exception as e:
        logger.error("[Training Pipeline] Błąd: %s", e, exc_info=True)
        raise
