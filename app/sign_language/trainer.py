from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.gesture_engine.config import DEBUG_MODE
from app.gesture_engine.logger import logger
from app.sign_language.dataset import load_processed_split
from app.sign_language.model import SignLanguageMLP

MODEL_PATH = "app/sign_language/models/pjm_model.pth"
CLASSES_PATH = "app/sign_language/models/classes.npy"


def train(
    model_path: str = MODEL_PATH,
    classes_path: str = CLASSES_PATH,
    epochs: int = 100,
    lr: float = 0.001,
    hidden_size: int = 256,
    batch_size: int = 32,
    augment_low_accuracy: bool = False,
    augment_multiplier: int = 10,
) -> dict:
    """
    Trenuje model MLP na datasecie PJM.

    Args:
        model_path: sciezka zapisu modelu
        classes_path: sciezka zapisu klas
        epochs: liczba epok
        lr: learning rate
        hidden_size: rozmiar warstwy ukrytej
        batch_size: rozmiar batcha (opcjonalnie mini-batch)
        augment_low_accuracy: czy augmentowac litery z niska accuracy (T, H, Y, I)
        augment_multiplier: ile razy zwiekszyc liczbe probek dla augmentowanych liter

    Returns:
        slownik z metrykami: accuracy, loss, num_classes
    """
    logger.info("=== Trening modelu PJM ===")

    # wczytanie danych
    try:
        X_train, y_train, meta_train = load_processed_split("train")
        X_val, y_val, meta_val = load_processed_split("val")
        X_test, y_test, meta_test = load_processed_split("test")
    except FileNotFoundError as e:
        logger.error(
            "Brak przetworzonych danych. Uruchom najpierw: python -m app.sign_language.dataset"
        )
        raise e

    classes = np.array(meta_train["classes"])
    num_classes = len(classes)

    logger.info("Train: %d probek", len(X_train))
    logger.info("Val: %d probek", len(X_val))
    logger.info("Test: %d probek", len(X_test))
    logger.info("Klasy: %s", list(classes))

    # augmentacja dla problematycznych liter (jesli wlaczona)
    if augment_low_accuracy:
        logger.info(
            "Augmentacja liter T, H, Y, I (multiplier=%d)...", augment_multiplier
        )
        from app.sign_language.augmenter import augment_class_samples

        # mapowanie liter na indeksy klas
        low_accuracy_letters = ["T", "H", "Y", "I"]
        for letter in low_accuracy_letters:
            if letter in classes:
                class_idx = int(np.where(classes == letter)[0][0])
                before_count = len(X_train)
                X_train, y_train = augment_class_samples(
                    X_train, y_train, class_idx, augment_multiplier
                )
                after_count = len(X_train)
                logger.info(
                    "Augmentowano klase %s: %d -> %d probek (+%d)",
                    letter,
                    before_count,
                    after_count,
                    after_count - before_count,
                )

        logger.info(
            "Augmentacja zakonczona. Nowy rozmiar train: %d probek", len(X_train)
        )

    # zapisuje klasy
    Path(classes_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(classes_path, classes)

    # konwersja do tensorow
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # model i optymalizator
    input_dim = X_train.shape[1]  # dynamicznie z danych (88D)
    model = SignLanguageMLP(
        input_size=input_dim, hidden_size=hidden_size, num_classes=num_classes
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(
        "Model: input=%d, hidden=%d, output=%d", input_dim, hidden_size, num_classes
    )

    # early stopping
    from app.sign_language.early_stopping import EarlyStopping

    early_stopping = EarlyStopping(patience=15, delta=0.005, verbose=True)

    # trening
    best_val_acc = 0.0
    last_loss = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        # walidacja co kilka epok
        if (epoch + 1) % max(1, epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = (val_predicted == y_val_t).sum().item() / y_val_t.size(0)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            logger.info(
                "Epoch [%d/%d], Loss: %.4f, Val Loss: %.4f, Val Acc: %.2f%%",
                epoch + 1,
                epochs,
                last_loss,
                val_loss.item(),
                val_acc * 100,
            )

            # sprawdz early stopping
            if early_stopping(val_loss.item()):
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    # ewaluuje ostateczny model na zbiorze testowym
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, test_predicted = torch.max(test_outputs, 1)
        test_acc = accuracy_score(y_test_t.numpy(), test_predicted.numpy())

    logger.info("Test Accuracy: %.2f%%", test_acc * 100)

    # raport klasyfikacji - labels zapewnia ze wszystkie klasy sa uwzglednione
    report = classification_report(
        y_test_t.numpy(),
        test_predicted.numpy(),
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    )
    logger.info("Classification Report:\n%s", report)

    # confusion matrix
    cm = confusion_matrix(y_test_t.numpy(), test_predicted.numpy())
    if DEBUG_MODE:
        logger.debug("Confusion Matrix:\n%s", cm)

    # zapis modelu
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info("Model zapisany: %s", model_path)

    # zapis metadanych modelu (input_size dla kompatybilnosci z translatorem)
    import json

    meta_path = Path(model_path).parent / "model_meta.json"
    model_meta = {
        "input_size": input_dim,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "version": "2.0_relative_features",
    }
    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2)
    logger.info("Metadane zapisane: %s", meta_path)

    return {
        "accuracy": float(test_acc),
        "val_accuracy": float(best_val_acc),
        "loss": float(last_loss),
        "num_classes": int(num_classes),
    }


def main():
    # punkt wejsciowy CLI z obsluga argumentow
    import argparse

    parser = argparse.ArgumentParser(description="Trening modelu PJM")
    parser.add_argument(
        "--epochs", type=int, default=100, help="liczba epok (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="rozmiar warstwy ukrytej (default: 256)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="augmentuj litery T, H, Y, I (10x wiecej probek)",
    )
    parser.add_argument(
        "--augment-multiplier",
        type=int,
        default=10,
        help="mnoznik augmentacji dla slabych klas (default: 10)",
    )

    args = parser.parse_args()

    logger.info("Rozpoczynam trening modelu PJM...")
    logger.info(
        "Parametry: epochs=%d, lr=%.4f, hidden_size=%d, augment=%s",
        args.epochs,
        args.lr,
        args.hidden_size,
        args.augment,
    )

    metrics = train(
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        augment_low_accuracy=args.augment,
        augment_multiplier=args.augment_multiplier,
    )
    logger.info("Trening zakonczony. Metryki: %s", metrics)


if __name__ == "__main__":
    main()
