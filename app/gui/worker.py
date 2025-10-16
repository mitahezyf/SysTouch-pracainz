from __future__ import annotations

import importlib
import threading
import time
from typing import Any, Protocol, Union, cast

from app.gesture_engine.config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    JSON_GESTURE_PATHS,
    USE_JSON_GESTURES,
)
from app.gesture_engine.core.handlers import gesture_handlers
from app.gesture_engine.core.hooks import handle_gesture_start_hook
from app.gesture_engine.detector.hand_tracker import HandTracker
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.performance import PerformanceTracker
from app.gesture_engine.utils.video_capture import ThreadedCapture
from app.gesture_engine.utils.visualizer import Visualizer
from app.gui.processing import detect_and_draw


class ProcessingWorkerProtocol(Protocol):
    frameReady: Any
    status: Any
    metrics: Any
    gesture: Any
    hands: Any
    startedOK: Any
    stoppedOK: Any

    def __init__(self) -> None: ...
    def configure(
        self,
        camera_index: Union[int, str],
        actions_enabled: bool,
        preview_enabled: bool,
    ) -> None: ...
    def set_actions_enabled(self, enabled: bool) -> None: ...
    def set_preview_enabled(self, enabled: bool) -> None: ...
    def stop(self) -> None: ...
    def isRunning(self) -> bool: ...
    def start(self) -> None: ...
    def wait(self, msecs: int) -> None: ...


def create_processing_worker() -> ProcessingWorkerProtocol:
    """Tworzy i zwraca QThread realizujacy przetwarzanie wideo i rozpoznanie gestow."""
    qtcore = importlib.import_module("PySide6.QtCore")
    qtgui = importlib.import_module("PySide6.QtGui")

    QThread = qtcore.QThread
    Signal = qtcore.Signal
    QImage = qtgui.QImage

    import cv2  # lokalny import, wymagany do konwersji klatek

    class ProcessingWorker(QThread):  # type: ignore[misc, valid-type]
        frameReady = Signal(QImage)
        status = Signal(str)
        metrics = Signal(int, int)  # fps, frametime_ms
        gesture = Signal(object)  # GestureResult
        hands = Signal(object)  # List[SingleHandResult]
        startedOK = Signal()
        stoppedOK = Signal()

        def __init__(self):
            super().__init__()
            self._camera_index: Union[int, str, None] = None
            self._stop_flag = threading.Event()
            self._actions_enabled = False
            self._preview_enabled = True

        def configure(
            self,
            camera_index: Union[int, str],
            actions_enabled: bool,
            preview_enabled: bool,
        ) -> None:
            self._camera_index = camera_index
            self._actions_enabled = bool(actions_enabled)
            self._preview_enabled = bool(preview_enabled)

        def set_actions_enabled(self, enabled: bool) -> None:
            self._actions_enabled = bool(enabled)
            logger.info("[dispatch] actions_enabled=%s", self._actions_enabled)
            # jesli wlaczono akcje, raportuje capabilities aby uzytkownik widzial ewentualne braki
            if self._actions_enabled:
                self._report_capabilities()

        def set_preview_enabled(self, enabled: bool) -> None:
            self._preview_enabled = bool(enabled)

        def stop(self) -> None:
            self._stop_flag.set()

        def _create_json_runtime(self):
            if not USE_JSON_GESTURES:
                return None
            try:
                from app.gesture_engine.core.gesture_runtime import GestureRuntime

                rt = GestureRuntime(JSON_GESTURE_PATHS)
                logger.info("[json] runtime gestow json wlaczony")
                return rt
            except Exception as e:
                logger.warning(f"[json] nie udalo sie uruchomic runtime: {e}")
                return None

        def _report_capabilities(self) -> None:
            # sprawdza zaleznosci akcji (pyautogui/pycaw/pywin32) i emituje status
            try:
                from app.gesture_engine.actions.capabilities import (
                    detect_action_capabilities,
                )

                caps = detect_action_capabilities()
                missing = [name for name, (ok, _msg) in caps.items() if not ok]
                if missing:
                    self.status.emit(
                        "Brak zaleznosci akcji: {}".format(", ".join(missing))
                    )
                    for name, (ok, info) in caps.items():
                        if not ok:
                            logger.warning("[cap] %s: %s", name, info)
                else:
                    self.status.emit("Akcje: wszystkie zaleznosci OK")
            except Exception as e:
                logger.debug("capabilities check error: %s", e)

        def run(self) -> None:  # noqa: C901
            if self._camera_index is None:
                self.status.emit("Brak wybranego indeksu kamery")
                self.stoppedOK.emit()
                return

            try:
                cap = ThreadedCapture(camera_index=self._camera_index)
            except Exception as e:
                self.status.emit(f"Blad kamery: {e}")
                self.stoppedOK.emit()
                return

            tracker = HandTracker()
            performance = PerformanceTracker()
            visualizer = Visualizer(
                capture_size=(CAPTURE_WIDTH, CAPTURE_HEIGHT),
                display_size=(DISPLAY_WIDTH, DISPLAY_HEIGHT),
            )
            json_runtime = self._create_json_runtime()

            self.startedOK.emit()
            # po starcie raportuje capabilities, zeby uzytkownik wiedzial, czy akcje beda dzialac
            self._report_capabilities()

            try:
                while not self._stop_flag.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.005)
                        continue

                    frame = cv2.flip(frame, 1)

                    # przetwarzanie + rysowanie w jednej funkcji
                    display_frame, gesture_res, per_hand = detect_and_draw(
                        frame, tracker, json_runtime, visualizer, self._preview_enabled
                    )

                    # obsluga akcji/hookow per reka
                    if per_hand:
                        for hand in per_hand:
                            # zawsze informuje hook o zmianie gestu (takze None)
                            try:
                                handle_gesture_start_hook(
                                    hand.name,
                                    hand.landmarks,
                                    frame.shape,
                                )
                            except Exception as e:
                                logger.debug(f"[hook] wyjatek: {e}")

                            handler = (
                                gesture_handlers.get(hand.name) if hand.name else None
                            )
                            logger.debug(
                                "[dispatch] hand=%s/%s gesture=%s conf=%.2f actions=%s handler=%s",
                                getattr(hand, "index", None),
                                getattr(hand, "handedness", None),
                                hand.name,
                                getattr(hand, "confidence", 0.0),
                                self._actions_enabled,
                                bool(handler),
                            )

                            if not self._actions_enabled or handler is None:
                                continue

                            try:
                                handler(hand.landmarks, frame.shape)
                            except Exception as e:
                                logger.debug(f"[handler] wyjatek: {e}")

                    # metryki i render
                    performance.update()
                    resized_frame = cv2.resize(
                        cast(Any, display_frame), (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                    )
                    if self._preview_enabled:
                        visualizer.draw_fps(resized_frame, performance.fps)
                        visualizer.draw_frametime(
                            resized_frame, performance.frametime_ms
                        )

                        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape
                        bytes_per_line = ch * w
                        qimg = QImage(
                            rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
                        ).copy()
                        self.frameReady.emit(qimg)

                    self.gesture.emit(gesture_res)
                    self.hands.emit(per_hand)
                    self.metrics.emit(performance.fps, performance.frametime_ms)

                self.status.emit("Zatrzymano przetwarzanie")
            except Exception as e:
                logger.exception("Wyjatek w watku przetwarzania: %s", e)
                self.status.emit(f"Blad watku: {e}")
            finally:
                try:
                    cap.stop()
                except Exception as e:
                    logger.debug("ProcessingWorker.run: cap.stop error: %s", e)
                self.stoppedOK.emit()

    return ProcessingWorker()
