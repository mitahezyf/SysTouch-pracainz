import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.sign_language.model import SignLanguageMLP

DATA_FILE = "app/sign_language/data/dataset_pjm.csv"
MODEL_PATH = "app/sign_language/models/pjm_model.pth"
CLASSES_PATH = "app/sign_language/models/classes.npy"


def train(
    data_file: str = DATA_FILE,
    model_path: str = MODEL_PATH,
    classes_path: str = CLASSES_PATH,
    epochs: int = 100,
    lr: float = 0.001,
) -> dict:
    # trenuje model signLanguageMLP
    # zwraca metryki: accuracy, loss, num_classes
    # parametryzacja ulatwia testy (mniejsza liczba epok i sciezki tymczasowe)
    df = pd.read_csv(data_file)
    X = df.iloc[:, 1:].values.astype(np.float32)  # cechy
    y = df.iloc[:, 0].values  # etykiety

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # zapisuje klasy do pliku
    np.save(classes_path, encoder.classes_)

    # dzieli dane na zbior treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

    # tworzy tensory
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # buduje model oraz optymalizator
    model = SignLanguageMLP(
        input_size=X_train_t.shape[1], num_classes=len(encoder.classes_)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    last_loss = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # ewaluacja modelu
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_t).sum().item() / y_test_t.size(0)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    # zapisuje model
    torch.save(model.state_dict(), model_path)
    print("Model saved!")

    return {
        "accuracy": float(accuracy),
        "loss": float(last_loss or 0.0),
        "num_classes": int(len(encoder.classes_)),
    }


if __name__ == "__main__":
    train()
