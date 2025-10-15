from __future__ import annotations

import importlib

from app.gesture_engine.logger import logger
from app.gui.camera import discover_cameras
from app.gui.styles import DARK_STYLESHEET
from app.gui.ui_components import build_ui
from app.gui.worker import create_processing_worker


class MainWindow:  # faktyczna klasa QMainWindow tworzona dynamicznie
    def __new__(cls):  # tworzy instancje realnej klasy QMainWindow z PySide6
        # qtcore nie jest potrzebne na tym etapie
        qtw = importlib.import_module("PySide6.QtWidgets")
        qtgui = importlib.import_module("PySide6.QtGui")

        QMainWindow = qtw.QMainWindow
        QPixmap = qtgui.QPixmap

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
                self.setCentralWidget(ui.central_widget)

                self.worker = create_processing_worker()
                self.worker.frameReady.connect(self.on_frame)
                self.worker.status.connect(self.on_status)
                self.worker.metrics.connect(self.on_metrics)
                self.worker.gesture.connect(self.on_gesture)
                self.worker.startedOK.connect(self.on_started)
                self.worker.stoppedOK.connect(self.on_stopped)

                self.start_btn.clicked.connect(self.on_start)
                self.stop_btn.clicked.connect(self.on_stop)
                self.refresh_cams_btn.clicked.connect(self.populate_cameras)
                self.exec_actions_chk.stateChanged.connect(self.on_actions_toggle)
                self.preview_chk.stateChanged.connect(self.on_preview_toggle)

                self.populate_cameras()

            def populate_cameras(self):
                self.camera_combo.clear()
                cams = discover_cameras(max_index=10)
                for idx in cams:
                    self.camera_combo.addItem(f"Kamera {idx}", idx)
                if self.camera_combo.count() == 0:
                    self.camera_combo.addItem("Brak kamer", -1)
                self.status_label.setText(
                    f"Status: Wykryto kamery: {', '.join(map(str, cams)) if cams else 'brak'}"
                )

            def on_start(self):
                if self.worker.isRunning():  # zabezpieczenie przed wielokrotnym startem
                    return
                cam_data = self.camera_combo.currentData()
                if cam_data is None or cam_data < 0:
                    self.status_label.setText("Status: Wybierz prawidlowa kamere")
                    return
                actions_enabled = self.exec_actions_chk.isChecked()
                preview_enabled = self.preview_chk.isChecked()
                self.worker.configure(cam_data, actions_enabled, preview_enabled)
                self.worker.start()
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.camera_combo.setEnabled(False)
                self.refresh_cams_btn.setEnabled(False)

            def on_stop(self):
                if self.worker.isRunning():
                    self.worker.stop()
                    self.worker.wait(2000)

            def on_actions_toggle(self, state):
                qtcore = importlib.import_module("PySide6.QtCore")
                self.worker.set_actions_enabled(state == qtcore.Qt.Checked)

            def on_preview_toggle(self, state):
                qtcore = importlib.import_module("PySide6.QtCore")
                enabled = state == qtcore.Qt.Checked
                self.worker.set_preview_enabled(enabled)
                self.video_label.setVisible(enabled)
                if not enabled:
                    self.video_label.setText("Podglad ukryty")

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
                        f"Gesture: {result.name} ({int(result.confidence * 100)}%)"
                    )
                else:
                    self.gesture_label.setText("Gesture: None")

            def on_started(self):
                self.status_label.setText("Status: Przetwarzanie uruchomione")

            def on_stopped(self):
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.camera_combo.setEnabled(True)
                self.refresh_cams_btn.setEnabled(True)

            def closeEvent(self, event):  # noqa: N802
                try:
                    if self.worker.isRunning():
                        self.worker.stop()
                        self.worker.wait(2000)
                except Exception as e:
                    logger.debug("MainWindow.closeEvent: stop/wait error: %s", e)
                super().closeEvent(event)

        return _MainWindow()
