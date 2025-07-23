import cv2
import time

from actions.click_action import release_click, get_click_state_name
from actions.handlers import gesture_handlers
from config import CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT, DISPLAY_WIDTH, DISPLAY_HEIGHT, TARGET_CAMERA_FPS
from detector.gesture_detector import detect_gesture
from detector.hand_tracker import HandTracker
from utils.performance import PerformanceTracker
from utils.visualizer import Visualizer
from utils.video_capture import ThreadedCapture
from config import DRAW_EVERY_N_FRAMES


#inicjalizacja kamery
cap = ThreadedCapture()

#inicjalizacja sledzenia dloni`
tracker = HandTracker()

#inicjalizacja obliczen wydajnosci
performance = PerformanceTracker()

#inicjalizacja klasy rysujacej landmarki, obszar, opisy
visualizer = Visualizer(
    capture_size = (CAPTURE_WIDTH, CAPTURE_HEIGHT),
    display_size = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
)

draw_frame_count = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    #przetwarzaniew wysokiej rozdzielczosci
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = tracker.process(frame_rgb)
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            gesture = detect_gesture(hand.landmark)
            gesture_name = ""
            confidence = 0.0

            if gesture:
                gesture_name, confidence = gesture
                handler = gesture_handlers.get(gesture_name)
                if handler:
                    handler(hand.landmark, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
            else:
                release_click()

            #wizualizacja
            gesture_label = get_click_state_name() or gesture_name
            label_text = f"{gesture_name}: ({confidence * 100:.1f})" if gesture_name else ""

            if draw_frame_count % DRAW_EVERY_N_FRAMES == 0:
                visualizer.draw_landmarks(display_frame, hand)
                visualizer.draw_hand_box(display_frame, hand, label = label_text)
            draw_frame_count += 1


    performance.update()
    visualizer.draw_fps(display_frame, performance.fps)
    visualizer.draw_frametime(display_frame, performance.frametime_ms)

    frame_time = (time.time() - start_time) * 1000
    print(f"Frame time: {frame_time:.2f} ms  =>  FPS: {1000 / frame_time:.1f}")

    cv2.imshow('SysTouch', display_frame)


    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.stop()
cv2.destroyAllWindows()
