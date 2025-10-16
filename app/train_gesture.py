from pathlib import Path

from app.gesture_engine.detector.hand_tracker import HandTracker
from app.gesture_engine.logger import logger


def main():
    # lokalne importy ciezkich bibliotek
    import cv2
    import mediapipe as mp
    import numpy as np

    # inicjalizacja rysowania dla mediapipe
    _mp_drawing = mp.solutions.drawing_utils
    _mp_hands = mp.solutions.hands

    def draw_hand_landmarks(frame, hand_landmarks):
        # rysuje landmarki i polaczenia dloni na klatce podgladu
        _mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            _mp_hands.HAND_CONNECTIONS,
        )

    # folder do zapisu danych treningowych
    GESTURE_DIR = Path("gesture_trainer/gestures")
    GESTURE_DIR.mkdir(parents=True, exist_ok=True)

    # inicjalizacja trackera
    hand_tracker = HandTracker(max_num_hands=1)

    # nazwa gestu do zebrania
    gesture_name = "nowy_gest"
    save_path = GESTURE_DIR / f"{gesture_name}.npy"

    logger.info(f"=== Zbieranie gestu: '{gesture_name}' ===")
    logger.info("Nacisnij [s] aby zapisac klatke z dlonia")
    logger.info("Nacisnij [q] aby zakonczyc")

    # otwarcie kamery
    cap = cv2.VideoCapture(0)
    saved_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Nie udalo sie pobrac obrazu z kamery")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_tracker.process(frame_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]

            # rysuje podglad z landmarkami
            draw_hand_landmarks(frame, landmarks)

            # konwertuje landmarki do numpy
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            flat_landmark = landmark_array.flatten()
        else:
            flat_landmark = None

        cv2.imshow("Train Gesture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and flat_landmark is not None:
            saved_data.append(flat_landmark)
            logger.info(f"Zapisano klatke ({len(saved_data)})")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved_data:
        import numpy as np  # lokalny import dla zapisu

        np.save(save_path, np.array(saved_data))
        logger.info(f"Zapisano dane gestu do {save_path}")
    else:
        logger.warning("Nie zapisano zadnych danych gestu")


if __name__ == "__main__":
    main()
