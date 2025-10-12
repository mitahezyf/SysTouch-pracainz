def main():
    import cv2  # lokalny import, by uniknac bledow przy samym imporcie modułu

    from app.gesture_engine.config import CAPTURE_HEIGHT
    from app.gesture_engine.config import CAPTURE_WIDTH
    from app.gesture_engine.config import DISPLAY_HEIGHT
    from app.gesture_engine.config import DISPLAY_WIDTH
    from app.gesture_engine.core.handlers import gesture_handlers
    from app.gesture_engine.core.hooks import handle_gesture_start_hook
    from app.gesture_engine.detector.gesture_detector import detect_gesture
    from app.gesture_engine.detector.hand_tracker import HandTracker
    from app.gesture_engine.logger import logger
    from app.gesture_engine.utils.performance import PerformanceTracker
    from app.gesture_engine.utils.video_capture import ThreadedCapture
    from app.gesture_engine.utils.visualizer import Visualizer

    # inicjalizacja komponentow
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
            logger.debug("Brak klatki z kamery – pomijam")
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
                    logger.debug(
                        f"[gesture] {hand_id}: {gesture_name} ({confidence:.2f})"
                    )

                handle_gesture_start_hook(
                    gesture_name, hand_landmarks.landmark, frame_shape
                )
                last_gestures[hand_id] = gesture_name

                if gesture_name:
                    handler = gesture_handlers.get(gesture_name)
                    if handler:
                        logger.debug(f"Wywołanie handlera dla gestu: {gesture_name}")
                        handler(hand_landmarks.landmark, frame_shape)

                label_text = (
                    f"{gesture_name}: ({confidence * 100:.1f})" if gesture_name else ""
                )

                visualizer.draw_landmarks(display_frame, hand_landmarks)
                visualizer.draw_hand_box(
                    display_frame, hand_landmarks, label=label_text
                )

        for missing_id in detected_hands_ids - current_hands_ids:
            logger.debug(f"Ręka zniknęła: {missing_id}")
            last_gestures.pop(missing_id, None)

        detected_hands_ids = current_hands_ids

        performance.update()

        resized_frame = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        visualizer.draw_fps(resized_frame, performance.fps)
        visualizer.draw_frametime(resized_frame, performance.frametime_ms)

        # pobiera ostatni gest i confidence jesli istnieje
        gesture_name, confidence = None, 0.0
        if last_gestures:
            # bierze pierwszą rękę (nazwa nieużywana dalej)
            gesture = detect_gesture(results.multi_hand_landmarks[0].landmark)
            if gesture:
                gesture_name, confidence = gesture

        visualizer.draw_current_gesture(resized_frame, gesture_name, confidence)

        cv2.imshow("SysTouch", resized_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            logger.info("Zamknięcie aplikacji przez ESC")
            break

    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
