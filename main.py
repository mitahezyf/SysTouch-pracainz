import cv2

from actions.click_action import release_click, get_click_state_name
from actions.handlers import gesture_handlers
from detector.gesture_detector import detect_gesture
from detector.hand_tracker import HandTracker
from utils.performance import PerformanceTracker
from utils.visualizer import Visualizer

#inicjalizacja kamery
cap = cv2.VideoCapture(0)
#inicjalizacja sledzenia dloni`
tracker = HandTracker()
#inicjalizacja obliczen wydajnosci
performance = PerformanceTracker()
#inicjalizacja klasy rysujacej landmarki, obszar, opisy
visualizer = Visualizer()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #odbicie lustrzane i konwersja rgb
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = tracker.process(frame_rgb)

    #rozmiar okienka
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            gesture = detect_gesture(hand.landmark)

            gesture_name = ""
            confidence = 0.0

            if gesture:
                gesture_name, confidence = gesture
                handler = gesture_handlers.get(gesture_name)
                if handler:
                    handler(hand.landmark, (h, w))
            else:
                release_click()


            visualizer.draw_landmarks(frame, hand)

            gesture_label = get_click_state_name() or gesture_name
            label = f"{gesture_name} ({confidence * 100:.1f}%)" if gesture_name else ""
            visualizer.draw_hand_box(frame, hand, label=label)

    #aktualizacja pomiarow wydajnosci
    performance.update()
    visualizer.draw_fps(frame, performance.fps)
    visualizer.draw_frametime(frame, performance.frametime_ms)

    #wyswietlanie okienka z programem
    cv2.imshow("SysTouch", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
