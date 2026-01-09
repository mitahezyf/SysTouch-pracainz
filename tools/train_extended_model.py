# trening modelu PJM z rozszerzonymi cechami i paskiem postepu
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from app.sign_language.model import SignLanguageMLP

# sciezki
BASE_DIR = Path(__file__).parent.parent / "app" / "sign_language"
PROCESSED_DIR = BASE_DIR / "data" / "processed_extended"
MODELS_DIR = BASE_DIR / "models"

# domyslne parametry
DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.001
DEFAULT_HIDDEN = 512  # wiekszy model dla wiekszego inputu
DEFAULT_BATCH = 64


def load_processed_split(split: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Wczytuje przetworzony split (train/val/test)."""
    path = PROCESSED_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku: {path}")

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    meta = json.loads(str(data["meta"]))

    return X, y, meta


def train_extended(
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    hidden_size: int = DEFAULT_HIDDEN,
    batch_size: int = DEFAULT_BATCH,
) -> dict:
    """
    Trenuje model na rozszerzonych cechach (246D).

    Returns:
        slownik z metrykami
    """
    print("=== TRENING MODELU PJM (ROZSZERZONE CECHY) ===")

    # wczytaj dane
    try:
        X_train, y_train, meta = load_processed_split("train")
        X_val, y_val, _ = load_processed_split("val")
        X_test, y_test, _ = load_processed_split("test")
    except FileNotFoundError as e:
        print(f"[BLAD] {e}")
        print("Najpierw uruchom: python tools/process_extended_dataset.py")
        raise

    classes = np.array(meta["classes"])
    num_classes = len(classes)
    input_size = meta["input_size"]

    print(f"Input size: {input_size}D")
    print(f"Klasy: {num_classes}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # tensory
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # model
    model = SignLanguageMLP(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20
    )

    print(f"\nModel: {input_size} -> {hidden_size} -> {num_classes}")
    print(f"Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print()

    # trening z paskiem postepu
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 30

    pbar = tqdm(range(epochs), desc="Trening", unit="epoka")

    for epoch in pbar:
        model.train()

        # mini-batch training
        indices = torch.randperm(len(X_train_t))
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]

            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # walidacja
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = (val_pred == y_val_t).sum().item() / len(y_val_t)

        # aktualizuj scheduler
        scheduler.step(val_acc)

        # aktualizuj pasek postepu
        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "val_acc": f"{val_acc*100:.1f}%",
                "best": f"{best_val_acc*100:.1f}%",
            }
        )

        # zapisz najlepszy model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0

            # zapisz checkpoint
            torch.save(model.state_dict(), MODELS_DIR / "pjm_model_extended.pth")
        else:
            patience_counter += 1

        # early stopping
        if patience_counter >= patience:
            tqdm.write(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
            break

    # wczytaj najlepszy model
    model.load_state_dict(torch.load(MODELS_DIR / "pjm_model_extended.pth"))

    # test
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, test_pred = torch.max(test_outputs, 1)
        test_acc = accuracy_score(y_test_t.numpy(), test_pred.numpy())

    print("\n=== WYNIKI ===")
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # raport klasyfikacji
    print("\nClassification Report:")
    print(
        classification_report(
            y_test_t.numpy(), test_pred.numpy(), target_names=classes, zero_division=0
        )
    )

    # zapisz metadane
    model_meta = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "version": "extended_82x3",
    }

    with open(MODELS_DIR / "model_meta_extended.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    # zapisz klasy
    np.save(MODELS_DIR / "classes_extended.npy", classes)

    print("\nZapisano:")
    print(f"  {MODELS_DIR / 'pjm_model_extended.pth'}")
    print(f"  {MODELS_DIR / 'model_meta_extended.json'}")
    print(f"  {MODELS_DIR / 'classes_extended.npy'}")

    return {
        "val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "best_epoch": best_epoch,
        "num_classes": num_classes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Trening modelu PJM (rozszerzone cechy)"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)

    args = parser.parse_args()

    metrics = train_extended(
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
    )

    print("\n=== ZAKONCZONO ===")
    print(f"Metryki: {metrics}")


if __name__ == "__main__":
    main()
