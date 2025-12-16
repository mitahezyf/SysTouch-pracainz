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
        mode: str,
    ) -> None: ...
    def set_actions_enabled(self, enabled: bool) -> None: ...
    def set_preview_enabled(self, enabled: bool) -> None: ...
    def stop(self) -> None: ...
    def isRunning(self) -> bool: ...
    def start(self) -> None: ...
    def wait(self, msecs: int) -> None: ...
    def set_mode(
        self, mode: str, translator: Any | None, normalizer: Any | None
    ) -> None: ...


def create_processing_worker() -> ProcessingWorkerProtocol:
    # tworzy i zwraca qThread realizujacy przetwarzanie wideo i rozpoznanie gestow
    qtcore = importlib.import_module("PySide6.QtCore")
    qtgui = importlib.import_module("PySide6.QtGui")

    QThread = qtcore.QThread
    Signal = qtcore.Signal
    QImage = qtgui.QImage

    import cv2  # lokalny import wymagany do konwersji klatek

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
            # przechowuje ostatni globalny gest i per reka
            self._last_best: Union[str, None] = None
            self._last_per_hand: dict[int, Union[str, None]] = {}
            self._mode: str = "gestures"
            self._translator = None
            self._normalizer = None

        def configure(
            self,
            camera_index: Union[int, str],
            actions_enabled: bool,
            preview_enabled: bool,
            mode: str,
        ) -> None:
            self._camera_index = camera_index
            self._actions_enabled = bool(actions_enabled)
            self._preview_enabled = bool(preview_enabled)
            self._mode = mode

        def set_actions_enabled(self, enabled: bool) -> None:
            self._actions_enabled = bool(enabled)
            logger.info("[dispatch] actions_enabled=%s", self._actions_enabled)
            # jesli wlaczono akcje raportuje capabilities aby uzytkownik widzial braki
            if self._actions_enabled:
                self._report_capabilities()

        def set_preview_enabled(self, enabled: bool) -> None:
            self._preview_enabled = bool(enabled)

        def set_mode(
            self, mode: str, translator: Any | None, normalizer: Any | None
        ) -> None:
            # ustawia tryb pracy workera oraz referencje translatora i normalizera (dla trybu translator)
            try:
                self._mode = str(mode) if mode else "gestures"
            except Exception:
                self._mode = "gestures"
            self._translator = translator
            self._normalizer = normalizer
            logger.info(
                "[mode] ustawiono mode=%s translator=%s normalizer=%s",
                self._mode,
                bool(self._translator),
                bool(self._normalizer),
            )
            if self._mode == "translator" and (
                self._translator is None or self._normalizer is None
            ):
                try:
                    self.status.emit("Translator niedostepny - brak modelu")
                except Exception:
                    pass

        def stop(self) -> None:
            self._stop_flag.set()

        def start(self) -> None:
            # przygotowuje stan przed startem watku
            try:
                self._stop_flag.clear()
            except Exception:
                self._stop_flag = threading.Event()
            self._last_best = None
            self._last_per_hand = {}
            # resetuje globalne hooki i przeladowuje detektory gestow
            try:
                reset_hooks_state()
            except Exception as e:
                logger.debug("reset_hooks_state error: %s", e)
            try:
                reload_gesture_detectors()
                # po reloadzie detektorow podmienia wskaznik hooks.volume_state na biezacy z modulu gestures
                try:
                    from app.gesture_engine.core import hooks as _hooks_mod
                    from app.gesture_engine.gestures.volume_gesture import (
                        volume_state as _vs,
                    )

                    _hooks_mod.volume_state = _vs
                except Exception as e2:
                    logger.debug("volume_state rebind after reload error: %s", e2)
            except Exception as e:
                logger.debug("reload_gesture_detectors error: %s", e)
            super().start()

        def _create_json_runtime(self) -> Any | None:
            if not USE_JSON_GESTURES:
                return None
            try:
                from app.gesture_engine.core.gesture_runtime import GestureRuntime

                rt: Any = GestureRuntime(JSON_GESTURE_PATHS)
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
                # automatycznie wlacza sterowanie systemowa glosnoscia, jesli pycaw jest dostepne i akcje sa wlaczone
                try:
                    from app.gesture_engine.gestures.volume_gesture import (
                        volume_state as _vs,
                    )

                    _vs["apply_system"] = bool(
                        caps.get("pycaw", (False, ""))[0]
                    ) and bool(self._actions_enabled)
                except Exception as e:
                    logger.debug("set apply_system failed: %s", e)

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
            # po starcie raportuje capabilities aby uzytkownik widzial czy akcje beda dzialac
            self._report_capabilities()

            last_reload = time.monotonic()
            failed_reads = 0

            target_dt = (
                1.0 / float(PROCESSING_MAX_FPS) if PROCESSING_MAX_FPS > 0 else 0.0
            )

            try:
                while not self._stop_flag.is_set():
                    loop_t0 = time.monotonic()
                    # okresowo hot-reloaduje gesty wg konfiguracji
                    if RELOAD_DETECTORS_SEC and RELOAD_DETECTORS_SEC > 0:
                        now = loop_t0
                        if now - last_reload >= RELOAD_DETECTORS_SEC:
                            try:
                                reload_gesture_detectors()
                                # po reloadzie ponownie podmienia referencje volume_state w hookach
                                try:
                                    from app.gesture_engine.core import (
                                        hooks as _hooks_mod,
                                    )
                                    from app.gesture_engine.gestures.volume_gesture import (
                                        volume_state as _vs,
                                    )

                                    _hooks_mod.volume_state = _vs
                                except Exception as e2:
                                    logger.debug(
                                        "volume_state rebind after periodic reload error: %s",
                                        e2,
                                    )
                            except Exception as e:
                                logger.debug(
                                    "periodic reload_gesture_detectors error: %s", e
                                )
                            last_reload = now

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        failed_reads += 1
                        # po dluzszej serii bledow przerywa aby gui moglo automatycznie wznowic
                        if failed_reads >= 120:
                            self.status.emit("Brak klatek z kamery - auto stop")
                            break
                        time.sleep(0.005)
                        if target_dt > 0.0:
                            elapsed = time.monotonic() - loop_t0
                            sleep_for = target_dt - elapsed
                            if sleep_for > 0:
                                time.sleep(sleep_for)
                        continue

                    failed_reads = 0

                    frame = cv2.flip(frame, 1)

                    display_frame, gesture_res, per_hand = detect_and_draw(
                        frame,
                        tracker,
                        json_runtime,
                        visualizer,
                        self._preview_enabled,
                        mode=self._mode,
                        translator=self._translator,
                        normalizer=self._normalizer,
                    )

                    # loguje zmiane najlepszego gestu (do ui)
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

                    # obsluguje akcje i hooki per reka
                    if per_hand:
                        for hand in per_hand:
                            idx = getattr(hand, "index", -1)
                            handed = getattr(hand, "handedness", None)
                            name = getattr(hand, "name", None)
                            conf = getattr(hand, "confidence", 0.0)

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

                            # pomijaj akcje w trybie translator (tylko rozpoznawanie liter)
                            if self._mode == "translator":
                                if LOG_PER_FRAME:
                                    logger.debug(
                                        "[action] skip '%s' (tryb translator - tylko litery)",
                                        hand.name,
                                    )
                                continue

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
                                if LOG_PER_FRAME:
                                    logger.debug("[action] invoke '%s'", hand.name)
                                handler(hand.landmarks, frame.shape)
                            except Exception as e:
                                logger.debug(f"[handler] wyjatek: {e}")

                    performance.update()

                    if self._preview_enabled:
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

    return cast(ProcessingWorkerProtocol, ProcessingWorker())
