# trenuje i przewiduje klasyfikator na wektorach landmarkow
import pickle

from sklearn.neighbors import KNeighborsClassifier

MODEL_PATH = __file__.replace("classifier.py", "data/trained_model.pkl")


class GestureClassifier:
    def __init__(self):
        self.model = None

    def train(self, data_dict):
        X, y = [], []
        for gesture, samples in data_dict.items():
            X.extend(samples)
            y.extend([gesture] * len(samples))

        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(X, y)

    def predict(self, vector):
        if not self.model:
            raise ValueError("Model nie został wytrenowany")
        return self.model.predict([vector])[0]

    def save(self):
        if not self.model:
            raise ValueError("Brak modelu do zapisania")
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
