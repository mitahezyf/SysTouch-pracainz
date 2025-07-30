import cv2

from app.actions.handlers import gesture_handlers
from app.actions.hooks import handle_gesture_start_hook
from app.config import CAPTURE_HEIGHT
from app.config import CAPTURE_WIDTH
from app.config import DISPLAY_HEIGHT
from app.config import DISPLAY_WIDTH
from app.detector.gesture_detector import detect_gesture
from app.detector.hand_tracker import HandTracker
from app.utils.performance import PerformanceTracker
from app.utils.video_capture import ThreadedCapture
from app.utils.visualizer import Visualizer


# Inicjalizacja komponentów
cap = ThreadedCapture()
tracker = HandTracker()
performance = PerformanceTracker()
visualizer = Visualizer(
    capture_size=(CAPTURE_WIDTH, CAPTURE_HEIGHT),
    display_size=(DISPLAY_WIDTH, DISPLAY_HEIGHT),
)

last_gestures = {}
detected_hands_ids = set()


def get_hand_id(handedness, idx):
    if handedness and idx < len(handedness):
        return handedness[idx].classification[0].label
    return f"hand_{idx}"


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret or frame is None:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_shape = frame.shape
    display_frame = frame.copy()

    results = tracker.process(frame_rgb)
    current_hands_ids = set()

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_id = get_hand_id(results.multi_handedness, i)
            current_hands_ids.add(hand_id)

            gesture_name = None
            confidence = 0.0
            gesture = detect_gesture(hand_landmarks.landmark)

            if gesture:
                gesture_name, confidence = gesture

            # zawsze wywołuj hook – nawet jeśli gest == None (np. gest zniknął)
            handle_gesture_start_hook(
                gesture_name, hand_landmarks.landmark, frame_shape
            )
            last_gestures[hand_id] = gesture_name

            # handler tylko dla rozpoznanego gestu
            if gesture_name:
                handler = gesture_handlers.get(gesture_name)
                if handler:
                    handler(hand_landmarks.landmark, frame_shape)

            label_text = (
                f"{gesture_name}: ({confidence * 100:.1f})" if gesture_name else ""
            )

            visualizer.draw_landmarks(display_frame, hand_landmarks)
            visualizer.draw_hand_box(display_frame, hand_landmarks, label=label_text)

    for missing_id in detected_hands_ids - current_hands_ids:
        last_gestures.pop(missing_id, None)

    detected_hands_ids = current_hands_ids

    performance.update()

    # skalowanie do wyświetlenia + overlay tekstowy
    resized_frame = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    visualizer.draw_fps(resized_frame, performance.fps)
    visualizer.draw_frametime(resized_frame, performance.frametime_ms)

    cv2.imshow("SysTouch", resized_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.stop()
cv2.destroyAllWindows()
