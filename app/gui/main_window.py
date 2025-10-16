from __future__ import annotations

import importlib
from typing import Any, List, Union, cast

from app.gesture_engine.config import CAMERA_MAX_INDEX_SCAN, CAMERA_SCAN_INTERVAL_MS
from app.gesture_engine.logger import logger
from app.gui.camera import discover_camera_sources
from app.gui.styles import DARK_STYLESHEET
from app.gui.ui_components import build_ui
from app.gui.worker import ProcessingWorkerProtocol, create_processing_worker


class MainWindow:  # faktyczna klasa QMainWindow tworzona dynamicznie
    def __new__(cls):  # tworzy instancje realnej klasy QMainWindow z PySide6
        # qtcore nie jest potrzebne na tym etapie
        qtw = importlib.import_module("PySide6.QtWidgets")
        qtgui = importlib.import_module("PySide6.QtGui")
        qtcore = importlib.import_module("PySide6.QtCore")

        QMainWindow = qtw.QMainWindow
        QPixmap = qtgui.QPixmap
        QTimer = qtcore.QTimer

        from app.gesture_engine.config import DISPLAY_HEIGHT, DISPLAY_WIDTH

        class _MainWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("SysTouch GUI")
                self.setMinimumSize(900, 700)
                self.setStyleSheet(DARK_STYLESHEET)

                ui = build_ui(DISPLAY_WIDTH, DISPLAY_HEIGHT)
                self.title_label = ui.title_label
                self.video_label = ui.video_label
                self.camera_combo = ui.camera_combo
                self.refresh_cams_btn = ui.refresh_cams_btn
                self.start_btn = ui.start_btn
                self.stop_btn = ui.stop_btn
                self.exec_actions_chk = ui.exec_actions_chk
                self.preview_chk = ui.preview_chk
                self.status_label = ui.status_label
                self.fps_label = ui.fps_label
                self.gesture_label = ui.gesture_label
                self.left_hand_label = ui.left_hand_label
                self.right_hand_label = ui.right_hand_label
                self.setCentralWidget(ui.central_widget)

                # worker tworzymy dopiero przy starcie, bo QThread nie jest restartowalny
                self.worker: ProcessingWorkerProtocol | None = None
                # lista znanych zrodel kamer (source, name)
                self._known_cams: list[tuple[Union[int, str], str]] = []

                self.start_btn.clicked.connect(self.on_start)
                self.stop_btn.clicked.connect(self.on_stop)
                self.refresh_cams_btn.clicked.connect(self.populate_cameras)
                self.exec_actions_chk.stateChanged.connect(self.on_actions_toggle)
                self.preview_chk.stateChanged.connect(self.on_preview_toggle)
                # restartuje przetwarzanie po zmianie kamery
                self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)

                self.populate_cameras()
                # auto-start jesli kamera dostepna
                self._auto_start_if_possible()

                # timer plug-and-play - odswieza liste kamer i auto-startuje/restartuje
                self._cams_timer = QTimer(self)
                self._cams_timer.setInterval(int(CAMERA_SCAN_INTERVAL_MS))
                self._cams_timer.timeout.connect(self._on_cams_tick)
                self._cams_timer.start()

            # zarzadzanie workerem -------------------------------------------------
            def _connect_worker_signals(self, w: ProcessingWorkerProtocol) -> None:
                w.frameReady.connect(self.on_frame)
                w.status.connect(self.on_status)
                w.metrics.connect(self.on_metrics)
                w.gesture.connect(self.on_gesture)
                w.hands.connect(self.on_hands)
                w.startedOK.connect(self.on_started)
                w.stoppedOK.connect(self.on_stopped)

            def _disconnect_worker_signals(self, w: ProcessingWorkerProtocol) -> None:
                # bezpieczne odpinanie (w Qt disconnect bez arg moze rzucic, wiec w try/except)
                try:
                    w.frameReady.disconnect(self.on_frame)
                except Exception as e:
                    logger.debug(
                        "_disconnect_worker_signals: frameReady disconnect error: %s", e
                    )
                try:
                    w.status.disconnect(self.on_status)
                except Exception as e:
                    logger.debug(
                        "_disconnect_worker_signals: status disconnect error: %s", e
                    )
                try:
                    w.metrics.disconnect(self.on_metrics)
                except Exception as e:
                    logger.debug(
                        "_disconnect_worker_signals: metrics disconnect error: %s", e
                    )
                try:
                    w.gesture.disconnect(self.on_gesture)
                except Exception as e:
                    logger.debug(
                        "_disconnect_worker_signals: gesture disconnect error: %s", e
                    )
                try:
                    w.hands.disconnect(self.on_hands)
                except Exception as e:
                    logger.debug(
                        "_disconnect_worker_signals: hands disconnect error: %s", e
                    )
                try:
                    w.startedOK.disconnect(self.on_started)
                except Exception as e:
                    logger.debug(
                        "_disconnect_worker_signals: startedOK disconnect error: %s", e
                    )
                try:
                    w.stoppedOK.disconnect(self.on_stopped)
                except Exception as e:
                    logger.debug(
                        "_disconnect_worker_signals: stoppedOK disconnect error: %s", e
                    )

            def _destroy_worker(self) -> None:
                if self.worker is None:
                    return
                try:
                    if self.worker.isRunning():
                        self.worker.stop()
                        self.worker.wait(2000)
                except Exception as e:
                    logger.debug("_destroy_worker: stop/wait error: %s", e)
                try:
                    self._disconnect_worker_signals(self.worker)
                except Exception as e:
                    logger.debug("_destroy_worker: disconnect error: %s", e)
                self.worker = None

            def _create_worker(self) -> ProcessingWorkerProtocol:
                w = create_processing_worker()
                self._connect_worker_signals(w)
                return w

            def _restart_with_camera(self, cam_data: Union[int, str]) -> None:
                actions_enabled = self.exec_actions_chk.isChecked()
                preview_enabled = self.preview_chk.isChecked()
                try:
                    self._destroy_worker()
                except Exception as e:
                    logger.debug("_restart_with_camera: destroy error: %s", e)
                self.worker = self._create_worker()
                self.worker.configure(cam_data, actions_enabled, preview_enabled)
                # natychmiast ustawia stan akcji/podgladu przed startem
                try:
                    self.worker.set_actions_enabled(actions_enabled)
                    self.worker.set_preview_enabled(preview_enabled)
                except Exception as e:
                    logger.debug("_restart_with_camera: sync pre-start error: %s", e)
                self.worker.start()
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.camera_combo.setEnabled(False)
                self.refresh_cams_btn.setEnabled(False)

            def _auto_start_if_possible(self) -> None:
                # jesli nie ma uruchomionego workera i jest jakakolwiek kamera - startuje automatycznie
                if self.worker is not None and self.worker.isRunning():
                    return
                if self.camera_combo.count() == 0:
                    return
                cam_data_any: Any = self.camera_combo.currentData()
                cam_data: Union[int, str]
                if isinstance(cam_data_any, int):
                    if cam_data_any < 0:
                        return
                    cam_data = cast(Union[int, str], cam_data_any)
                elif isinstance(cam_data_any, str):
                    cam_data = cast(Union[int, str], cam_data_any)
                else:
                    return
                logger.info("[plug] auto-start z kamera: %s", cam_data)
                self._restart_with_camera(cam_data)

            # UI i akcje ----------------------------------------------------------
            def populate_cameras(self):
                # zapamietuje aktualnie wybrane zrodlo, aby zachowac wybor po odswiezeniu
                current_data = self.camera_combo.currentData()
                current_text = self.camera_combo.currentText()

                self.camera_combo.blockSignals(True)
                self.camera_combo.clear()
                sources = discover_camera_sources(max_index=int(CAMERA_MAX_INDEX_SCAN))
                for source, name in sources:
                    self.camera_combo.addItem(name, source)
                if self.camera_combo.count() == 0:
                    self.camera_combo.addItem("Brak kamer", -1)
                else:
                    # proba przywrocenia poprzedniego wyboru (po data, a nastepnie po tekscie)
                    if current_data is not None:
                        idx = self.camera_combo.findData(current_data)
                        if idx >= 0:
                            self.camera_combo.setCurrentIndex(idx)
                        else:
                            # fallback po tekscie
                            if current_text:
                                idx2 = self.camera_combo.findText(current_text)
                                if idx2 >= 0:
                                    self.camera_combo.setCurrentIndex(idx2)
                self.camera_combo.blockSignals(False)

                cams_text = (
                    ", ".join([name for _, name in sources]) if sources else "brak"
                )
                self.status_label.setText(f"Status: Wykryto kamery: {cams_text}")
                # aktualizuje cache znanych kamer
                self._known_cams = [(cast(Union[int, str], s), n) for s, n in sources]

            def _on_cams_tick(self) -> None:
                # nie skanuje kamer, gdy worker pracuje - unika konfliktu z uchwytem kamery
                if self.worker is not None and self.worker.isRunning():
                    return
                # odswieza liste zrodel bez klikania i uruchamia/zmienia automatycznie w razie zmian
                prev_sources = list(self._known_cams)
                self.populate_cameras()
                new_sources = list(self._known_cams)

                if not prev_sources and new_sources:
                    # pojawila sie kamera -> auto-start
                    self.status_label.setText("Status: Wykryto kamere - auto-start")
                    self._auto_start_if_possible()
                    return

                if prev_sources and not new_sources:
                    # wszystkie kamery zniknely -> zatrzymuje worker
                    self.status_label.setText("Status: Brak kamer - zatrzymano")
                    self._destroy_worker()
                    self.start_btn.setEnabled(True)
                    self.stop_btn.setEnabled(False)
                    self.camera_combo.setEnabled(True)
                    self.refresh_cams_btn.setEnabled(True)
                    return

                # jesli aktualnie wybrane zrodlo zniknelo -> restart z pierwszym dostepnym
                if new_sources and self.worker is not None and self.worker.isRunning():
                    current_data_any: Any = self.camera_combo.currentData()
                    current_is_known = any(
                        current_data_any == s for s, _ in new_sources
                    )
                    if not current_is_known:
                        self.status_label.setText(
                            "Status: Zrodlo zniknelo - auto restart z inna kamera"
                        )
                        self.camera_combo.setCurrentIndex(0)
                        cd_any: Any = self.camera_combo.currentData()
                        if isinstance(cd_any, (int, str)):
                            self._restart_with_camera(cd_any)
                        return

                # jesli worker nie dziala, a kamera jest, sprobuje wystartowac
                if new_sources and (self.worker is None or not self.worker.isRunning()):
                    self._auto_start_if_possible()

            def on_start(self):
                # zawsze tworz nowy watek przetwarzania
                self._destroy_worker()
                cam_data_any: Any = self.camera_combo.currentData()
                cam_data: Union[int, str]
                if isinstance(cam_data_any, int):
                    if cam_data_any < 0:
                        self.status_label.setText("Status: Wybierz prawidlowa kamere")
                        return
                    cam_data = cast(Union[int, str], cam_data_any)
                elif isinstance(cam_data_any, str):
                    cam_data = cast(Union[int, str], cam_data_any)
                else:
                    self.status_label.setText("Status: Wybierz prawidlowa kamere")
                    return
                actions_enabled = self.exec_actions_chk.isChecked()
                preview_enabled = self.preview_chk.isChecked()
                self.worker = self._create_worker()
                self.worker.configure(cam_data, actions_enabled, preview_enabled)
                # natychmiast ustawia stan akcji/podgladu przed startem
                try:
                    self.worker.set_actions_enabled(actions_enabled)
                    self.worker.set_preview_enabled(preview_enabled)
                except Exception as e:
                    logger.debug("on_start: sync pre-start error: %s", e)
                self.worker.start()
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.camera_combo.setEnabled(False)
                self.refresh_cams_btn.setEnabled(False)

            def on_stop(self):
                if self.worker is not None:
                    try:
                        if self.worker.isRunning():
                            self.worker.stop()
                            self.worker.wait(2000)
                    except Exception as e:
                        logger.debug("on_stop: stop/wait error: %s", e)
                    # opcjonalnie od razu niszczy, aby uniknac ponownego startu na tym samym QThread
                    self._destroy_worker()

            def on_camera_changed(self, _index):
                # jesli przetwarzanie trwa, bezpiecznie restartuje watek z nowym zrodlem
                if self.worker is None or not self.worker.isRunning():
                    return
                cd_any: Any = self.camera_combo.currentData()
                if not isinstance(cd_any, (int, str)):
                    return
                cam_data = cast(Union[int, str], cd_any)
                self.status_label.setText(
                    "Status: Zmiana kamery - restart przetwarzania..."
                )
                try:
                    self.worker.stop()
                    self.worker.wait(2000)
                except Exception as e:
                    logger.debug("on_camera_changed: stop/wait error: %s", e)
                try:
                    self._disconnect_worker_signals(self.worker)
                except Exception as e:
                    logger.debug("on_camera_changed: disconnect error: %s", e)
                actions_enabled = self.exec_actions_chk.isChecked()
                preview_enabled = self.preview_chk.isChecked()
                self.worker = self._create_worker()
                self.worker.configure(cam_data, actions_enabled, preview_enabled)
                # natychmiast ustawia stan akcji/podgladu przed startem
                try:
                    self.worker.set_actions_enabled(actions_enabled)
                    self.worker.set_preview_enabled(preview_enabled)
                except Exception as e:
                    logger.debug("on_camera_changed: sync pre-start error: %s", e)
                self.worker.start()
                self.status_label.setText(
                    "Status: Przetwarzanie uruchomione (nowa kamera)"
                )

            def on_actions_toggle(self, state):
                qtcore = importlib.import_module("PySide6.QtCore")
                if self.worker is not None:
                    try:
                        is_checked = state == qtcore.Qt.CheckState.Checked
                    except Exception:
                        is_checked = state == qtcore.Qt.Checked
                    self.worker.set_actions_enabled(is_checked)
                # status UI
                try:
                    checked_now = self.exec_actions_chk.isChecked()
                    self.status_label.setText(
                        f"Status: Akcje {'wlaczone' if checked_now else 'wylaczone'}"
                    )
                except Exception as e:
                    logger.debug("on_actions_toggle: status update error: %s", e)

            def on_preview_toggle(self, state):
                qtcore = importlib.import_module("PySide6.QtCore")
                try:
                    enabled = state == qtcore.Qt.CheckState.Checked
                except Exception:
                    enabled = state == qtcore.Qt.Checked
                if self.worker is not None:
                    self.worker.set_preview_enabled(enabled)
                self.video_label.setVisible(enabled)
                if not enabled:
                    self.video_label.setText("Podglad ukryty")
                else:
                    self.status_label.setText("Status: Podglad wlaczony")

            def on_frame(self, img):
                # QPixmap importowane na gorze przez qtgui
                self.video_label.setPixmap(QPixmap.fromImage(img))

            def on_status(self, text):
                self.status_label.setText(f"Status: {text}")

            def on_metrics(self, fps, frametime_ms):
                self.fps_label.setText(f"FPS: {fps} | FrameTime: {frametime_ms} ms")

            def on_gesture(self, result):
                if result.name:
                    self.gesture_label.setText(
                        f"Gesture (best): {result.name} ({int(result.confidence * 100)}%)"
                    )
                else:
                    self.gesture_label.setText("Gesture (best): None")

            def on_hands(self, per_hand: List[object]):
                # per_hand to lista SingleHandResult z polami: handedness ("Left"/"Right"), name, confidence
                left_txt = "Left: None"
                right_txt = "Right: None"
                try:
                    for h in per_hand:
                        handed = getattr(h, "handedness", None)
                        name = getattr(h, "name", None)
                        conf = getattr(h, "confidence", 0.0)
                        if handed and str(handed).lower().startswith("left"):
                            left_txt = (
                                f"Left: {name} ({int(conf * 100)}%)"
                                if name
                                else "Left: None"
                            )
                        elif handed and str(handed).lower().startswith("right"):
                            right_txt = (
                                f"Right: {name} ({int(conf * 100)}%)"
                                if name
                                else "Right: None"
                            )
                except Exception as e:
                    logger.debug("on_hands parse error: %s", e)
                self.left_hand_label.setText(left_txt)
                self.right_hand_label.setText(right_txt)

            def on_started(self):
                self.status_label.setText("Status: Przetwarzanie uruchomione")
                # wstrzymuje skanowanie kamer podczas pracy (mniej zastrzalow i ostrzezen)
                try:
                    self._cams_timer.stop()
                except Exception as e:
                    logger.debug("on_started: cams_timer.stop error: %s", e)
                # synchronizuje stan akcji i podgladu z checkboxami (zapobiega rozjazdom)
                try:
                    if self.worker is not None:
                        self.worker.set_actions_enabled(
                            self.exec_actions_chk.isChecked()
                        )
                        self.worker.set_preview_enabled(self.preview_chk.isChecked())
                except Exception as e:
                    logger.debug("on_started sync error: %s", e)

            def on_stopped(self):
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.camera_combo.setEnabled(True)
                self.refresh_cams_btn.setEnabled(True)
                # wznawia skanowanie kamer po zakonczeniu przetwarzania
                try:
                    self._cams_timer.start()
                except Exception as e:
                    logger.debug("on_stopped: cams_timer.start error: %s", e)

            def closeEvent(self, event):  # noqa: N802
                try:
                    self._destroy_worker()
                except Exception as e:
                    logger.debug("MainWindow.closeEvent: destroy error: %s", e)
                super().closeEvent(event)

        return _MainWindow()
