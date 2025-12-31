# skrypt naprawczy dla modelu PJM - usuwa cechy z zerowa wariancja i dodaje class weights
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from app.gesture_engine.config import DEBUG_MODE
from app.gesture_engine.logger import logger
from app.sign_language.dataset import load_processed_split
from app.sign_language.model import SignLanguageMLP

MODEL_PATH = "app/sign_language/models/pjm_model.pth"
CLASSES_PATH = "app/sign_language/models/classes.npy"


def identify_zero_variance_features():
    """identyfikuje cechy z zerowa wariancja w zbiorze treningowym"""
    logger.info("=== Identyfikacja cech z zerowa wariancja ===")

    X_train, _, _ = load_processed_split("train")

    # oblicz std dla kazdej cechy
    feature_stds = np.std(X_train, axis=0)
    zero_var_mask = feature_stds < 1e-6
    zero_var_indices = np.where(zero_var_mask)[0]

    logger.info("Liczba cech z zerowa wariancja: %d / 63", len(zero_var_indices))
    logger.info("Indeksy: %s", list(zero_var_indices))

    for idx in zero_var_indices:
        logger.info(
            "  Cecha %d: std=%.8f, mean=%.8f",
            idx,
            feature_stds[idx],
            np.mean(X_train[:, idx]),
        )

    return zero_var_indices


def remove_zero_variance_features(X, zero_var_indices):
    """usuwa cechy z zerowa wariancja z macierzy cech"""
    mask = np.ones(X.shape[1], dtype=bool)
    mask[zero_var_indices] = False
    return X[:, mask]


def compute_class_weights_array(y_train, num_classes):
    """oblicza wagi klas dla imbalanced dataset"""
    logger.info("=== Obliczanie wag klas ===")

    # znajdz tylko klasy ktore rzeczywiscie wystepuja w y_train
    unique_classes = np.unique(y_train)

    # oblicz wagi tylko dla istniejacych klas
    class_weights_partial = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=y_train
    )

    # rozszerz do pelnego wektora (wypelnij brakujace klasy wagami 1.0)
    class_weights = np.ones(num_classes, dtype=np.float32)
    for cls_idx, weight in zip(unique_classes, class_weights_partial):
        class_weights[cls_idx] = weight

    logger.info("Class weights (tylko dla istniejacych klas):")
    for cls_idx in range(num_classes):
        if cls_idx in unique_classes:
            logger.info("  Klasa %d: %.4f", cls_idx, class_weights[cls_idx])
        else:
            if DEBUG_MODE:
                logger.debug("  Klasa %d: brak w datasecie (weight=1.0)", cls_idx)

    return torch.tensor(class_weights, dtype=torch.float32)


def train_fixed_model(
    epochs: int = 150,
    lr: float = 0.001,
    hidden_size: int = 128,
    use_class_weights: bool = True,
    remove_zero_var: bool = True,
):
    """
    trenuje model z poprawkami:
    - usuwa cechy z zerowa wariancja
    - dodaje class weights do loss function
    """
    logger.info("=== Trening poprawionego modelu PJM ===")
    logger.info("Parametry:")
    logger.info("  Epochs: %d", epochs)
    logger.info("  Learning rate: %.4f", lr)
    logger.info("  Hidden size: %d", hidden_size)
    logger.info("  Use class weights: %s", use_class_weights)
    logger.info("  Remove zero variance features: %s", remove_zero_var)

    # wczytaj dane
    X_train, y_train, meta_train = load_processed_split("train")
    X_val, y_val, meta_val = load_processed_split("val")
    X_test, y_test, meta_test = load_processed_split("test")

    classes = np.array(meta_train["classes"])
    num_classes = len(classes)

    logger.info("Train: %d probek", len(X_train))
    logger.info("Val: %d probek", len(X_val))
    logger.info("Test: %d probek", len(X_test))
    logger.info("Klasy: %d", num_classes)

    # identyfikuj i usun cechy z zerowa wariancja
    zero_var_indices = np.array([], dtype=int)
    if remove_zero_var:
        zero_var_indices = identify_zero_variance_features()
        if len(zero_var_indices) > 0:
            logger.info("Usuwam %d cech z zerowa wariancja", len(zero_var_indices))
            X_train = remove_zero_variance_features(X_train, zero_var_indices)
            X_val = remove_zero_variance_features(X_val, zero_var_indices)
            X_test = remove_zero_variance_features(X_test, zero_var_indices)
            logger.info("Nowy rozmiar cech: %d", X_train.shape[1])

    input_size = X_train.shape[1]

    # zapisz klasy
    Path(CLASSES_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.save(CLASSES_PATH, classes)

    # zapisz indeksy usuniętych cech (do późniejszego użycia w translatorze)
    if len(zero_var_indices) > 0:
        zero_var_path = Path(CLASSES_PATH).parent / "zero_var_indices.npy"
        np.save(zero_var_path, zero_var_indices)
        logger.info("Zapisano indeksy usunietych cech: %s", zero_var_path)

    # konwersja do tensorow
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # oblicz class weights
    class_weights_tensor = None
    if use_class_weights:
        class_weights_tensor = compute_class_weights_array(y_train, num_classes)

    # model i optymalizator
    model = SignLanguageMLP(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(
        "Model: input=%d, hidden=%d, output=%d", input_size, hidden_size, num_classes
    )
    if use_class_weights:
        logger.info("Cross-entropy loss z class weights (balanced)")
    else:
        logger.info("Cross-entropy loss bez weights")

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

        # walidacja co 10% epok
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

    # ewaluacja na zbiorze testowym
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, test_predicted = torch.max(test_outputs, 1)
        test_acc = accuracy_score(y_test_t.numpy(), test_predicted.numpy())

    logger.info("Test Accuracy: %.2f%%", test_acc * 100)

    # raport klasyfikacji
    report = classification_report(
        y_test_t.numpy(),
        test_predicted.numpy(),
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    )
    logger.info("Classification Report:\n%s", report)

    # porownaj z poprzednim modelem (jesli istnieje)
    if Path(MODEL_PATH).exists():
        logger.info("\n=== Porownanie z poprzednim modelem ===")
        try:
            old_state_dict = torch.load(
                MODEL_PATH, map_location="cpu", weights_only=False
            )
            old_w0 = old_state_dict.get("network.0.weight")
            old_input_size = 63  # stary model zawsze 63

            if old_w0 is not None and hasattr(old_w0, "shape"):
                old_hidden = int(old_w0.shape[0])
            else:
                old_hidden = 128

            old_model = SignLanguageMLP(
                input_size=old_input_size,
                hidden_size=old_hidden,
                num_classes=num_classes,
            )
            old_model.load_state_dict(old_state_dict)
            old_model.eval()

            # testuj stary model (bez usuniecia cech)
            X_test_old, y_test_old, _ = load_processed_split("test")
            X_test_old_t = torch.tensor(X_test_old, dtype=torch.float32)

            with torch.no_grad():
                old_outputs = old_model(X_test_old_t)
                _, old_predicted = torch.max(old_outputs, 1)
                old_acc = accuracy_score(y_test_old, old_predicted.numpy())

            logger.info("Stary model: %.2f%%", old_acc * 100)
            logger.info("Nowy model: %.2f%%", test_acc * 100)
            logger.info("Poprawa: %.2f%%", (test_acc - old_acc) * 100)

        except Exception as e:
            logger.warning("Nie mozna zaladowac starego modelu: %s", e)

    # zapisz nowy model
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Model zapisany: %s", MODEL_PATH)

    # zapisz metadane modelu (input_size, hidden_size, removed_features)
    model_meta = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "removed_features": len(zero_var_indices),
        "zero_var_indices": (
            zero_var_indices.tolist() if len(zero_var_indices) > 0 else []
        ),
        "use_class_weights": use_class_weights,
    }

    import json

    meta_path = Path(MODEL_PATH).parent / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2)
    logger.info("Metadane modelu zapisane: %s", meta_path)

    return {
        "accuracy": float(test_acc),
        "val_accuracy": float(best_val_acc),
        "loss": float(last_loss),
        "num_classes": int(num_classes),
        "input_size": int(input_size),
        "removed_features": int(len(zero_var_indices)),
    }


def main():
    """punkt wejsciowy CLI"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Naprawia i trenuje model PJM z poprawkami"
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="liczba epok (default: 150)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="rozmiar warstwy ukrytej (default: 128)",
    )
    parser.add_argument(
        "--no-class-weights", action="store_true", help="wylacz class weights"
    )
    parser.add_argument(
        "--keep-zero-var",
        action="store_true",
        help="nie usuwaj cech z zerowa wariancja",
    )

    args = parser.parse_args()

    logger.info("Rozpoczynam trening poprawionego modelu PJM...")
    logger.info(
        "Parametry: epochs=%d, lr=%.4f, hidden_size=%d",
        args.epochs,
        args.lr,
        args.hidden_size,
    )

    metrics = train_fixed_model(
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        use_class_weights=not args.no_class_weights,
        remove_zero_var=not args.keep_zero_var,
    )

    logger.info("Trening zakonczony. Metryki: %s", metrics)


if __name__ == "__main__":
    main()
