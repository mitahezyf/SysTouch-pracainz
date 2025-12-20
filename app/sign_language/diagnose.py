# skrypt diagnostyczny PJM - wykrywa problemy w datasecie i modelu
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from app.gesture_engine.logger import logger
from app.sign_language.dataset import load_processed_split
from app.sign_language.model import SignLanguageMLP

MODEL_PATH = Path(__file__).parent / "models" / "pjm_model.pth"
CLASSES_PATH = Path(__file__).parent / "models" / "classes.npy"


def check_data_distribution():
    """sprawdza rozklad klas w przetworzonych danych"""
    logger.info("=== 1. ANALIZA ROZKLADU KLAS ===")

    try:
        X_train, y_train, meta_train = load_processed_split("train")
        X_val, y_val, meta_val = load_processed_split("val")
        X_test, y_test, meta_test = load_processed_split("test")
    except FileNotFoundError as e:
        logger.error("Brak przetworzonych danych: %s", e)
        logger.error("Uruchom: python -m app.sign_language.dataset")
        return False

    classes = meta_train["classes"]
    logger.info("Liczba klas: %d", len(classes))
    logger.info("Klasy: %s", classes)

    # analiza train set
    logger.info("\n--- TRAIN SET (%d probek) ---", len(X_train))
    train_counts = Counter(y_train)
    total_train = len(y_train)

    class_stats = []
    for cls_idx in sorted(train_counts.keys()):
        count = train_counts[cls_idx]
        percentage = (count / total_train) * 100
        cls_name = classes[cls_idx]
        class_stats.append((cls_name, count, percentage))
        logger.info("  %s: %5d probek (%.2f%%)", cls_name, count, percentage)

    # wykryj skrajny imbalance
    counts_only = [count for _, count, _ in class_stats]
    min_count = min(counts_only)
    max_count = max(counts_only)
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    logger.info("\nStatystyki:")
    logger.info("  Min: %d probek", min_count)
    logger.info("  Max: %d probek", max_count)
    logger.info("  Srednia: %.1f probek", np.mean(counts_only))
    logger.info("  Mediana: %.1f probek", np.median(counts_only))
    logger.info("  Imbalance ratio: %.2fx", imbalance_ratio)

    if imbalance_ratio > 5.0:
        logger.warning(
            "UWAGA: Skrajny imbalance klas! Ratio %.2fx > 5.0", imbalance_ratio
        )
        logger.warning("To moze powodowac bias w kierunku najliczniejszej klasy")
        most_common = max(class_stats, key=lambda x: x[1])
        logger.warning(
            "Najliczniejsza klasa: %s (%d probek, %.2f%%)",
            most_common[0],
            most_common[1],
            most_common[2],
        )

    # analiza val i test
    logger.info("\n--- VAL SET (%d probek) ---", len(X_val))
    val_counts = Counter(y_val)
    for cls_idx in sorted(val_counts.keys()):
        count = val_counts[cls_idx]
        cls_name = classes[cls_idx]
        logger.info("  %s: %d probek", cls_name, count)

    logger.info("\n--- TEST SET (%d probek) ---", len(X_test))
    test_counts = Counter(y_test)
    for cls_idx in sorted(test_counts.keys()):
        count = test_counts[cls_idx]
        cls_name = classes[cls_idx]
        logger.info("  %s: %d probek", cls_name, count)

    return True


def check_data_quality():
    """sprawdza jakosc danych - NaN, Inf, duplikaty"""
    logger.info("\n=== 2. ANALIZA JAKOSCI DANYCH ===")

    try:
        X_train, y_train, _ = load_processed_split("train")
    except FileNotFoundError:
        logger.error("Brak danych treningowych")
        return False

    # sprawdz NaN/Inf
    nan_count = np.isnan(X_train).sum()
    inf_count = np.isinf(X_train).sum()

    logger.info("NaN wartosci: %d", nan_count)
    logger.info("Inf wartosci: %d", inf_count)

    if nan_count > 0 or inf_count > 0:
        logger.warning("UWAGA: Dataset zawiera NaN lub Inf!")
        logger.warning("To moze powodowac nieprawidlowe predykcje")

    # sprawdz duplikaty
    unique_samples = np.unique(X_train, axis=0)
    duplicate_count = len(X_train) - len(unique_samples)
    duplicate_percentage = (duplicate_count / len(X_train)) * 100

    logger.info("Unikalne probki: %d / %d", len(unique_samples), len(X_train))
    logger.info("Duplikaty: %d (%.2f%%)", duplicate_count, duplicate_percentage)

    if duplicate_percentage > 10.0:
        logger.warning("UWAGA: >10%% duplikatow w datasecie!")

    # sprawdz statystyki podstawowe
    logger.info("\nStatystyki cech (63D):")
    logger.info("  Mean: %.4f", np.mean(X_train))
    logger.info("  Std: %.4f", np.std(X_train))
    logger.info("  Min: %.4f", np.min(X_train))
    logger.info("  Max: %.4f", np.max(X_train))

    # sprawdz czy wszystkie cechy maja variance
    feature_stds = np.std(X_train, axis=0)
    zero_variance_features = np.sum(feature_stds < 1e-6)

    logger.info("  Cechy z zerowa wariancja: %d / 63", zero_variance_features)

    if zero_variance_features > 0:
        logger.warning(
            "UWAGA: %d cech ma zerowa wariancje (nie niosą informacji)",
            zero_variance_features,
        )

    return True


def check_model_predictions():
    """sprawdza predykcje modelu na testset - confusion matrix i bias"""
    logger.info("\n=== 3. ANALIZA PREDYKCJI MODELU ===")

    # sprawdz czy model istnieje
    if not MODEL_PATH.exists():
        logger.error("Brak modelu: %s", MODEL_PATH)
        logger.error("Uruchom: python -m app.sign_language.trainer")
        return False

    if not CLASSES_PATH.exists():
        logger.error("Brak pliku klas: %s", CLASSES_PATH)
        return False

    # wczytaj dane testowe
    try:
        X_test, y_test, meta_test = load_processed_split("test")
    except FileNotFoundError:
        logger.error("Brak danych testowych")
        return False

    classes = np.load(CLASSES_PATH)
    num_classes = len(classes)

    logger.info("Zaladowano model: %s", MODEL_PATH)
    logger.info("Liczba klas: %d", num_classes)

    # wczytaj metadane modelu (jesli istnieja)
    zero_var_indices = np.array([], dtype=int)
    model_input_size = 63

    meta_path = Path(MODEL_PATH).parent / "model_meta.json"
    if meta_path.exists():
        try:
            import json

            with open(meta_path, "r") as f:
                model_meta = json.load(f)

            model_input_size = model_meta.get("input_size", 63)
            zero_var_list = model_meta.get("zero_var_indices", [])
            if zero_var_list:
                zero_var_indices = np.array(zero_var_list, dtype=int)
                logger.info(
                    "Model ma %d usuniętych cech (input_size=%d)",
                    len(zero_var_indices),
                    model_input_size,
                )
        except Exception as e:
            logger.warning("Nie mozna wczytac metadanych modelu: %s", e)

    # wczytaj model
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    # inferencja hidden_size i input_size z wag
    w0 = state_dict.get("network.0.weight")
    hidden_size = 128
    if w0 is not None:
        hidden_size = int(w0.shape[0])
        inferred_input_size = int(w0.shape[1])
        if inferred_input_size != model_input_size:
            logger.warning(
                "Input size z metadanych (%d) != input size z wag (%d), uzywam wag",
                model_input_size,
                inferred_input_size,
            )
            model_input_size = inferred_input_size

    model = SignLanguageMLP(
        input_size=model_input_size, hidden_size=hidden_size, num_classes=num_classes
    )
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(
        "Model zaladowany (input_size=%d, hidden_size=%d)",
        model_input_size,
        hidden_size,
    )

    # usun cechy z zerowa wariancja (jesli model byl trenowany bez nich)
    X_test_filtered = X_test
    if len(zero_var_indices) > 0:
        mask = np.ones(X_test.shape[1], dtype=bool)
        mask[zero_var_indices] = False
        X_test_filtered = X_test[:, mask]
        logger.info("Usunieto %d cech z X_test przed predykcja", len(zero_var_indices))

    # predykcje na testset
    X_test_t = torch.tensor(X_test_filtered, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_test_t)
        probs = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, 1)

    predictions_np = predictions.numpy()
    confidences_np = confidences.numpy()

    # accuracy
    correct = np.sum(predictions_np == y_test)
    accuracy = correct / len(y_test) * 100

    logger.info("\nTest Accuracy: %.2f%% (%d / %d)", accuracy, correct, len(y_test))

    # rozklad predykcji
    logger.info("\n--- ROZKLAD PREDYKCJI ---")
    pred_counts = Counter(predictions_np)

    pred_stats = []
    for cls_idx in sorted(pred_counts.keys()):
        count = pred_counts[cls_idx]
        percentage = (count / len(predictions_np)) * 100
        cls_name = classes[cls_idx]
        pred_stats.append((cls_name, count, percentage))
        logger.info("  %s: %5d predykcji (%.2f%%)", cls_name, count, percentage)

    # wykryj skrajny bias
    pred_counts_only = [count for _, count, _ in pred_stats]
    max_pred_count = max(pred_counts_only)
    max_pred_percentage = (max_pred_count / len(predictions_np)) * 100

    if max_pred_percentage > 30.0:
        most_predicted = max(pred_stats, key=lambda x: x[1])
        logger.warning(
            "UWAGA: Model ma bias w kierunku klasy %s (%.2f%% predykcji)!",
            most_predicted[0],
            most_predicted[2],
        )
        logger.warning("To sugeruje ze model 'defaultuje' do tej klasy")

    # confusion matrix - pokaz tylko najczestsze bledy
    logger.info("\n--- TOP 10 NAJCZESTSZYCH BLEDOW ---")
    errors = []
    for i in range(len(y_test)):
        if predictions_np[i] != y_test[i]:
            true_cls = classes[y_test[i]]
            pred_cls = classes[predictions_np[i]]
            conf = confidences_np[i]
            errors.append((true_cls, pred_cls, conf))

    error_counts = Counter((true, pred) for true, pred, _ in errors)

    for (true_cls, pred_cls), count in error_counts.most_common(10):
        logger.info("  %s -> %s: %d razy", true_cls, pred_cls, count)

    # srednia confidence
    logger.info("\n--- CONFIDENCE ---")
    logger.info("  Srednia: %.2f%%", np.mean(confidences_np) * 100)
    logger.info("  Mediana: %.2f%%", np.median(confidences_np) * 100)
    logger.info("  Min: %.2f%%", np.min(confidences_np) * 100)
    logger.info("  Max: %.2f%%", np.max(confidences_np) * 100)

    # confidence dla poprawnych vs blednych predykcji
    correct_mask = predictions_np == y_test
    correct_conf = confidences_np[correct_mask]
    incorrect_conf = confidences_np[~correct_mask]

    if len(correct_conf) > 0:
        logger.info("  Confidence (poprawne): %.2f%%", np.mean(correct_conf) * 100)
    if len(incorrect_conf) > 0:
        logger.info("  Confidence (bledne): %.2f%%", np.mean(incorrect_conf) * 100)

    # per-class accuracy
    logger.info("\n--- ACCURACY PER KLASA ---")
    for cls_idx in range(num_classes):
        mask = y_test == cls_idx
        if np.sum(mask) == 0:
            continue

        cls_correct = np.sum(predictions_np[mask] == y_test[mask])
        cls_total = np.sum(mask)
        cls_acc = cls_correct / cls_total * 100
        cls_name = classes[cls_idx]

        logger.info("  %s: %.2f%% (%d / %d)", cls_name, cls_acc, cls_correct, cls_total)

    return True


def check_model_weights():
    """sprawdza czy wagi modelu nie sa dziwne (wszystkie zero, NaN itp)"""
    logger.info("\n=== 4. ANALIZA WAG MODELU ===")

    if not MODEL_PATH.exists():
        logger.error("Brak modelu: %s", MODEL_PATH)
        return False

    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    logger.info("Warstwy modelu:")
    total_params = 0

    for name, param in state_dict.items():
        param_count = param.numel()
        total_params += param_count

        mean = param.mean().item()
        std = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()

        # sprawdz czy sa NaN/Inf
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()

        # sprawdz czy wszystkie zero
        all_zero = torch.all(param == 0).item()

        logger.info("  %s: shape=%s, params=%d", name, tuple(param.shape), param_count)
        logger.info(
            "    mean=%.4f, std=%.4f, min=%.4f, max=%.4f", mean, std, min_val, max_val
        )

        if has_nan:
            logger.warning("    UWAGA: Warstwa zawiera NaN!")
        if has_inf:
            logger.warning("    UWAGA: Warstwa zawiera Inf!")
        if all_zero:
            logger.warning("    UWAGA: Wszystkie wagi to zero!")

    logger.info("\nLaczna liczba parametrow: %d", total_params)

    # sprawdz bias w ostatniej warstwie (output)
    output_bias_key = None
    for key in state_dict.keys():
        if (
            "bias" in key
            and key.startswith("network.")
            and len(state_dict[key].shape) == 1
        ):
            # znajdz ostatni bias (najwiekszy numer)
            output_bias_key = key

    if output_bias_key:
        output_bias = state_dict[output_bias_key]
        logger.info("\n--- BIAS WARSTWY WYJSCIOWEJ ---")
        classes = np.load(CLASSES_PATH)

        bias_values = output_bias.numpy()
        bias_sorted = sorted(enumerate(bias_values), key=lambda x: x[1], reverse=True)

        logger.info("Top 5 najwyzszych biasow:")
        for cls_idx, bias_val in bias_sorted[:5]:
            cls_name = classes[cls_idx]
            logger.info("  %s: %.4f", cls_name, bias_val)

        logger.info("\nTop 5 najnizszych biasow:")
        for cls_idx, bias_val in bias_sorted[-5:]:
            cls_name = classes[cls_idx]
            logger.info("  %s: %.4f", cls_name, bias_val)

        # sprawdz czy jest skrajny bias
        bias_range = bias_values.max() - bias_values.min()
        if bias_range > 5.0:
            logger.warning(
                "UWAGA: Duza roznica w biasach warstwy wyjsciowej (%.2f)", bias_range
            )
            logger.warning("Model moze byc biased w kierunku pewnych klas")

    return True


def main():
    """uruchamia pelna diagnostyke"""
    logger.info("=" * 60)
    logger.info("DIAGNOSTYKA DATASETU I MODELU PJM")
    logger.info("=" * 60)

    # sprawdz wszystkie aspekty
    results = []

    results.append(("Rozklad klas", check_data_distribution()))
    results.append(("Jakosc danych", check_data_quality()))
    results.append(("Predykcje modelu", check_model_predictions()))
    results.append(("Wagi modelu", check_model_weights()))

    # podsumowanie
    logger.info("\n" + "=" * 60)
    logger.info("PODSUMOWANIE")
    logger.info("=" * 60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "OK" if result else "FAIL"
        logger.info("%s: %s", name, status)

    if all_passed:
        logger.info("\nWszystkie testy diagnostyczne przeszly pomyslnie")
        logger.info("Sprawdz powyzsze ostrzezenia (UWAGA) aby zidentyfikowac problemy")
    else:
        logger.error("\nNiektore testy zakonczyly sie bledem")
        logger.error("Sprawdz logi powyzej aby uzyskac szczegoly")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
