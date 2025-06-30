import cv2
import pyautogui
from detector.hand_tracker import HandTracker
from detector.gesture_detector import detect_gesture
from actions.gesture_actions import execute_action

import time

mouse_down = False
tracker = HandTracker()

prev_frame_time = time.time()

while True:
    frame, landmarks_data = tracker.get_hand_landmarks()

    h, w, _ = frame.shape

    for hand_landmarks, hand_label in landmarks_data:
        gesture = detect_gesture(hand_landmarks)
        if not gesture:
            continue

        # MYSZKA – tylko prawa ręka
        if hand_label == "Right" and gesture.name == "move_mouse":
            execute_action(gesture, (h, w))

        # KLIKANIE/PRZYCISK – tylko lewa ręka
        if hand_label == "Left" and gesture.name == "click":
            if gesture.confidence > 0.95 and not mouse_down:
                pyautogui.mouseDown()
                mouse_down = True
            elif gesture.confidence <= 0.95 and mouse_down:
                pyautogui.mouseUp()
                mouse_down = False

        # Ramka i etykieta
        x_list = [lm.x for lm in hand_landmarks.landmark]
        y_list = [lm.y for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_list) * w), int(max(x_list) * w)
        y_min, y_max = int(min(y_list) * h), int(max(y_list) * h)

        label_text = f"{gesture.name.upper()} ({int(gesture.confidence * 100)}%) [{hand_label}]"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #licznik FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
    frametime_ms = (new_frame_time - prev_frame_time) * 1000
    prev_frame_time = new_frame_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(frame, f'Delay: {int(frametime_ms)} ms', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

    cv2.imshow("wykrywacz", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.release()
cv2.destroyAllWindows()
