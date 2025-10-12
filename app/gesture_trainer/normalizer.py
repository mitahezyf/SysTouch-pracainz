# normalizuje landmarki do wektora 1d (63 wartosci) przeskalowanego wzgledem dloni
from app.gesture_trainer.calibrator import load_calibration


def normalize_landmarks(landmarks):
    calibration = load_calibration()
    if not calibration:
        raise ValueError("Brak danych kalibracyjnych")

    hand_size = calibration["hand_size"]
    wrist = landmarks[0]  # baza pozycji w ukladzie dloni

    vector = []
    for x, y, z in landmarks:
        dx = (x - wrist[0]) / hand_size
        dy = (y - wrist[1]) / hand_size
        dz = (z - wrist[2]) / hand_size
        vector.extend([dx, dy, dz])

    return vector  # zwraca finalny wektor 63-wymiarowy
