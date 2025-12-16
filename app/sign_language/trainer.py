from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    hidden_size: int = 128,
    batch_size: int = 32,
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
    model = SignLanguageMLP(
        input_size=63, hidden_size=hidden_size, num_classes=num_classes
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info("Model: input=%d, hidden=%d, output=%d", 63, hidden_size, num_classes)

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
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = (val_predicted == y_val_t).sum().item() / y_val_t.size(0)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            logger.info(
                "Epoch [%d/%d], Loss: %.4f, Val Acc: %.2f%%",
                epoch + 1,
                epochs,
                last_loss,
                val_acc * 100,
            )

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
    logger.debug("Confusion Matrix:\n%s", cm)

    # zapis modelu
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info("Model zapisany: %s", model_path)

    return {
        "accuracy": float(test_acc),
        "val_accuracy": float(best_val_acc),
        "loss": float(last_loss),
        "num_classes": int(num_classes),
    }


def main():
    # punkt wejsciowy CLI
    logger.info("Rozpoczynam trening modelu PJM...")
    metrics = train(epochs=100, lr=0.001, hidden_size=128)
    logger.info("Trening zakonczony. Metryki: %s", metrics)


if __name__ == "__main__":
    main()
