# szybki test modelu PJM na danych testowych
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from app.gesture_engine.logger import logger
from app.sign_language.dataset import load_processed_split
from app.sign_language.model import SignLanguageMLP


def quick_test() -> None:
    """Szybki test modelu na zbiorze testowym."""
    logger.info("=== Szybki test modelu PJM ===")

    # wczytaj dane testowe
    try:
        X_test, y_test, meta_test = load_processed_split("test")
    except FileNotFoundError as e:
        logger.error("Brak danych testowych: %s", e)
        return

    classes = np.array(meta_test["classes"])
    logger.info("Dane testowe: %d probek, %d klas", len(X_test), len(classes))

    # wczytaj model
    model_path = "app/sign_language/models/pjm_model.pth"
    classes_path = "app/sign_language/models/classes.npy"

    try:
        model_classes = np.load(classes_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    except FileNotFoundError as e:
        logger.error("Brak pliku modelu: %s", e)
        return

    # zbuduj model
    input_dim = X_test.shape[1]

    # inferencja hidden_size z wag
    w0 = state_dict.get("network.0.weight")
    if w0 is not None:
        hidden_size = int(w0.shape[0])
    else:
        hidden_size = 256  # domyslny

    model = SignLanguageMLP(
        input_size=input_dim,
        hidden_size=hidden_size,
        num_classes=len(model_classes),
    )
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(
        "Model: input=%d, hidden=%d, output=%d",
        input_dim,
        hidden_size,
        len(model_classes),
    )

    # predykcja
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_test_t)
        probs = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, 1)

    predictions_np = predictions.numpy()
    confidences_np = confidences.numpy()

    # metryki
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_test, predictions_np)

    logger.info("Test Accuracy: %.2f%%", acc * 100)
    logger.info("Srednie Confidence: %.3f", confidences_np.mean())
    logger.info("Min Confidence: %.3f", confidences_np.min())
    logger.info("Max Confidence: %.3f", confidences_np.max())

    # raport per-klasa
    report = classification_report(
        y_test,
        predictions_np,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    )
    logger.info("Classification Report:\n%s", report)

    # znajdz najtrudniejsze klasy
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions_np, labels=list(range(len(classes))), zero_division=0
    )

    # sortuj po f1-score (rosnaco)
    class_scores = [(classes[i], f1[i], support[i]) for i in range(len(classes))]
    class_scores.sort(key=lambda x: x[1])

    logger.info("Najslabsze klasy (F1-score):")
    for cls_name, f1_score, supp in class_scores[:5]:
        logger.info("  %s: F1=%.3f (support=%d)", cls_name, f1_score, supp)

    logger.info("Najlepsze klasy (F1-score):")
    for cls_name, f1_score, supp in class_scores[-5:]:
        logger.info("  %s: F1=%.3f (support=%d)", cls_name, f1_score, supp)

    # confusion matrix - pokaz tylko najczesciej mylene
    cm = confusion_matrix(y_test, predictions_np)

    # znajdz najwieksze off-diagonal wartosci (tylko dla klas obecnych w y_test)
    present_classes = np.unique(y_test)
    mistakes = []
    for i in present_classes:
        for j in present_classes:
            if i != j and i < len(cm) and j < len(cm[0]) and cm[i, j] > 0:
                mistakes.append((classes[i], classes[j], cm[i, j]))

    mistakes.sort(key=lambda x: x[2], reverse=True)

    logger.info("Najczestsze pomylki (true -> predicted):")
    for true_cls, pred_cls, count in mistakes[:10]:
        logger.info("  %s -> %s: %d razy", true_cls, pred_cls, count)


if __name__ == "__main__":
    quick_test()
