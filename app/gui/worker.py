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
    LOG_PER_FRAME,
    PROCESSING_MAX_FPS,
    RELOAD_DETECTORS_SEC,
    USE_JSON_GESTURES,
)
from app.gesture_engine.core.handlers import gesture_handlers
from app.gesture_engine.core.hooks import handle_gesture_start_hook, reset_hooks_state
from app.gesture_engine.detector.gesture_detector import reload_gesture_detectors
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
            # wewnetrzne stany do logowania zmian gestow
            self._last_best: Union[str, None] = None
            self._last_per_hand: dict[int, Union[str, None]] = {}

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

        def start(self) -> None:
            # przygotowuje stan przed startem watku
            try:
                self._stop_flag.clear()
            except Exception:
                # w razie gdyby Event zostal podmieniony
                self._stop_flag = threading.Event()
            self._last_best = None
            self._last_per_hand = {}
            # reset globalnych hookow i przeladowanie detektorow gestow
            try:
                reset_hooks_state()
            except Exception as e:
                logger.debug("reset_hooks_state error: %s", e)
            try:
                reload_gesture_detectors()
            except Exception as e:
                logger.debug("reload_gesture_detectors error: %s", e)
            super().start()

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
            # po starcie raportuje capabilities, zeby uzytkownik widzial, czy akcje beda dzialac
            self._report_capabilities()

            # licznik do okresowego przeladowania detektorow gestow
            last_reload = time.monotonic()
            failed_reads = 0

            # pacing petli przetwarzania
            target_dt = (
                1.0 / float(PROCESSING_MAX_FPS) if PROCESSING_MAX_FPS > 0 else 0.0
            )

            try:
                while not self._stop_flag.is_set():
                    loop_t0 = time.monotonic()
                    # okresowy hot-reload gestow wg konfiguracji
                    if RELOAD_DETECTORS_SEC and RELOAD_DETECTORS_SEC > 0:
                        now = loop_t0
                        if now - last_reload >= RELOAD_DETECTORS_SEC:
                            try:
                                reload_gesture_detectors()
                            except Exception as e:
                                logger.debug(
                                    "periodic reload_gesture_detectors error: %s", e
                                )
                            last_reload = now

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        failed_reads += 1
                        # po dluzszej serii bledow przerywa, aby GUI moglo automatycznie wznowic
                        if failed_reads >= 120:
                            self.status.emit("Brak klatek z kamery - auto stop")
                            break
                        time.sleep(0.005)
                        # pacing nawet przy braku klatki
                        if target_dt > 0.0:
                            elapsed = time.monotonic() - loop_t0
                            sleep_for = target_dt - elapsed
                            if sleep_for > 0:
                                time.sleep(sleep_for)
                        continue

                    failed_reads = 0

                    frame = cv2.flip(frame, 1)

                    # przetwarzanie + rysowanie w jednej funkcji
                    display_frame, gesture_res, per_hand = detect_and_draw(
                        frame, tracker, json_runtime, visualizer, self._preview_enabled
                    )

                    # loguje zmiane najlepszego gestu (do UI)
                    if gesture_res.name != self._last_best:
                        if self._last_best is None and gesture_res.name:
                            logger.info(
                                "[ui] gesture start: %s (conf=%.2f)",
                                gesture_res.name,
                                gesture_res.confidence or 0.0,
                            )
                        elif self._last_best is not None and gesture_res.name is None:
                            logger.info("[ui] gesture end: %s", self._last_best)
                        else:
                            if LOG_PER_FRAME:
                                logger.debug(
                                    "[ui] gesture change: %s -> %s (conf=%.2f)",
                                    self._last_best,
                                    gesture_res.name,
                                    gesture_res.confidence or 0.0,
                                )
                        self._last_best = gesture_res.name

                    # obsluga akcji/hookow per reka
                    if per_hand:
                        for hand in per_hand:
                            idx = getattr(hand, "index", -1)
                            handed = getattr(hand, "handedness", None)
                            name = getattr(hand, "name", None)
                            conf = getattr(hand, "confidence", 0.0)

                            # loguje tylko start/end na INFO; zmiany na DEBUG
                            last = self._last_per_hand.get(idx)
                            if last != name:
                                if last is None and name is not None:
                                    logger.info(
                                        "[gesture] start hand=%s/%s: %s (conf=%.2f)",
                                        idx,
                                        handed,
                                        name,
                                        conf,
                                    )
                                elif last is not None and name is None:
                                    logger.info(
                                        "[gesture] end hand=%s/%s: %s",
                                        idx,
                                        handed,
                                        last,
                                    )
                                else:
                                    if LOG_PER_FRAME:
                                        logger.debug(
                                            "[gesture] change hand=%s/%s: %s -> %s (conf=%.2f)",
                                            idx,
                                            handed,
                                            last,
                                            name,
                                            conf,
                                        )
                                self._last_per_hand[idx] = name

                                # informuje hook tylko przy zmianie (takze None)
                                try:
                                    handle_gesture_start_hook(
                                        name, hand.landmarks, frame.shape
                                    )
                                except Exception as e:
                                    logger.debug(f"[hook] wyjatek: {e}")

                            handler = (
                                gesture_handlers.get(hand.name) if hand.name else None
                            )
                            if LOG_PER_FRAME:
                                logger.debug(
                                    "[dispatch] hand=%s/%s gesture=%s conf=%.2f actions=%s handler=%s",
                                    getattr(hand, "index", None),
                                    getattr(hand, "handedness", None),
                                    hand.name,
                                    getattr(hand, "confidence", 0.0),
                                    self._actions_enabled,
                                    bool(handler),
                                )

                            # log na DEBUG gdy gest wykryty, ale akcje sa wylaczone lub brak handlera
                            if hand.name and (
                                not self._actions_enabled or handler is None
                            ):
                                if LOG_PER_FRAME:
                                    logger.debug(
                                        "[action] skip '%s' (actions=%s handler=%s)",
                                        hand.name,
                                        self._actions_enabled,
                                        bool(handler),
                                    )

                            if not self._actions_enabled or handler is None:
                                continue

                            try:
                                # loguje wywolanie akcji tylko na DEBUG i tylko gdy wlaczono logi per-klatka
                                if LOG_PER_FRAME:
                                    logger.debug("[action] invoke '%s'", hand.name)
                                handler(hand.landmarks, frame.shape)
                            except Exception as e:
                                logger.debug(f"[handler] wyjatek: {e}")

                    # metryki i render
                    performance.update()

                    if self._preview_enabled:
                        # tylko gdy podglad wlaczony: resize + rysowanie metryk i konwersja do QImage
                        resized_frame = cv2.resize(
                            cast(Any, display_frame), (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                        )
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

                    # pacing petli przetwarzania
                    if target_dt > 0.0:
                        elapsed = time.monotonic() - loop_t0
                        sleep_for = target_dt - elapsed
                        if sleep_for > 0:
                            time.sleep(sleep_for)

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
