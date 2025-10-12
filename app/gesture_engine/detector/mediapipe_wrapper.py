# pobiera snapshot dloni z kamerki i zwraca landmarki albo None
# Bezpieczny import cv2 – aby import modułu nie wywracał się w CI bez OpenCV
try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:  # pragma: no cover

    class _CV2Stub:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        COLOR_BGR2RGB = 4

        class VideoCapture:
            def __init__(self, *_, **__):
                raise ImportError(
                    "cv2 (OpenCV) nie jest zainstalowane – zainstaluj opencv-python(-headless)."
                )

        @staticmethod
        def cvtColor(*_, **__):
            raise ImportError(
                "cv2.cvtColor niedostępne – zainstaluj opencv-python(-headless)."
            )

    cv2 = _CV2Stub()  # type: ignore

from app.gesture_engine.config import CAMERA_INDEX
from app.gesture_engine.config import CAPTURE_HEIGHT
from app.gesture_engine.config import CAPTURE_WIDTH
from app.gesture_engine.detector.hand_tracker import HandTracker

hand_tracker = HandTracker(max_num_hands=1)


def get_hand_landmarks():
    print(">>> [mediapipe_wrapper] get_hand_landmarks() odpalony")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # ustawia rozdzielczosc (moze nie zadzialac na kazdej kamerze)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    if not cap.isOpened():
        print("[mediapipe_wrapper] Kamera nie zostala otwarta!")
        return None
    else:
        print("[mediapipe_wrapper] Kamera otwarta OK")

    ret, frame = cap.read()
    cap.release()

    print(f"[mediapipe_wrapper] ret: {ret}, frame is None: {frame is None}")

    if not ret or frame is None:
        print("[mediapipe_wrapper] Nie udalo sie odczytac klatki z kamery")
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_tracker.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
        print(f"[mediapipe_wrapper] Wykryto {len(landmarks)} punktow")
        return landmarks

    print("[mediapipe_wrapper] Brak wykrytej dloni")
    return None
