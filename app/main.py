def main():
    import cv2  # lokalny import, by uniknac bledow przy samym imporcie modulu

    from app.gesture_engine.config import (
        CAPTURE_HEIGHT,
        CAPTURE_WIDTH,
        DISPLAY_HEIGHT,
        DISPLAY_WIDTH,
        SHOW_WINDOW,
    )
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

    # flaga GUI z configu; gdy highgui niedostepne, przelaczymy na headless w locie
    display_enabled = bool(SHOW_WINDOW)

    # gesty json (opcjonalnie)
    from app.gesture_engine.config import JSON_GESTURE_PATHS, USE_JSON_GESTURES

    json_runtime = None
    if USE_JSON_GESTURES:
        try:
            from app.gesture_engine.core.gesture_runtime import GestureRuntime

            json_runtime = GestureRuntime(JSON_GESTURE_PATHS)
            logger.info("[json] runtime gestow json wlaczony")
        except Exception as e:
            logger.warning(f"[json] nie udalo sie uruchomic runtime: {e}")
            json_runtime = None

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
            logger.debug("Brak klatki z kamery - pomijam")
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

                # json runtime (jesli wlaczony) -> preferowany wynik
                if json_runtime is not None:
                    # ramka jako lista (x,y,z)
                    lm = [
                        (lm.x, lm.y, getattr(lm, "z", 0.0))
                        for lm in hand_landmarks.landmark
                    ]
                    try:
                        res = json_runtime.update(lm)
                    except Exception as e:
                        logger.debug(f"[json] blad matchera: {e}")
                        res = None
                    if res:
                        # nazwa gestu do UI to nazwa akcji (mapowanie do handlera)
                        gesture_name = res.get("action", {}).get("type")
                        confidence = float(res.get("confidence", 1.0))

                # fallback do istniejacego detektora
                if gesture_name is None:
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
                        logger.debug(f"Wywolanie handlera dla gestu: {gesture_name}")
                        handler(hand_landmarks.landmark, frame_shape)

                label_text = (
                    f"{gesture_name}: ({confidence * 100:.1f})" if gesture_name else ""
                )

                visualizer.draw_landmarks(display_frame, hand_landmarks)
                visualizer.draw_hand_box(
                    display_frame, hand_landmarks, label=label_text
                )

        for missing_id in detected_hands_ids - current_hands_ids:
            logger.debug(f"Reka zniknela: {missing_id}")
            last_gestures.pop(missing_id, None)

        detected_hands_ids = current_hands_ids

        performance.update()

        resized_frame = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        visualizer.draw_fps(resized_frame, performance.fps)
        visualizer.draw_frametime(resized_frame, performance.frametime_ms)

        # pobiera ostatni gest i confidence jesli istnieje
        gesture_name, confidence = None, 0.0
        if last_gestures:
            # bierze pierwsza reke (nazwa nieuzywana dalej)
            gesture = detect_gesture(results.multi_hand_landmarks[0].landmark)
            if gesture:
                gesture_name, confidence = gesture

        visualizer.draw_current_gesture(resized_frame, gesture_name, confidence)

        if display_enabled:
            try:
                cv2.imshow("SysTouch", resized_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    logger.info("Zamkniecie aplikacji przez ESC")
                    break
            except cv2.error as e:
                logger.warning(
                    "OpenCV GUI niedostepne (imshow). Przechodze w tryb headless. Szczegoly: %s",
                    e,
                )
                display_enabled = False
        else:
            # w trybie headless brak okna i klawiatury; krotki sleep implicit w cap.read/loop
            pass

    cap.stop()
    if display_enabled:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
