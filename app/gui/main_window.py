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
                self.setWindowTitle("SysTouch")

                # Ustaw ikonę aplikacji
                from pathlib import Path

                QIcon = qtgui.QIcon
                icon_path = (
                    Path(__file__).resolve().parent.parent.parent / "SysTouchIco.jpg"
                )
                if icon_path.exists():
                    self.setWindowIcon(QIcon(str(icon_path)))

                self.setMinimumSize(1100, 750)
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
                self.mode_switch = ui.mode_switch
                self.mode_text_label = ui.mode_text_label
                self.status_label = ui.status_label
                self.fps_label = ui.fps_label
                self.gesture_label = ui.gesture_label
                self.left_hand_label = ui.left_hand_label
                self.right_hand_label = ui.right_hand_label
                # przyciski obslugi nagrywania alfabetu i treningu modelu
                self.record_btn = ui.record_btn
                self.samples_btn = ui.samples_btn
                self.train_btn = ui.train_btn
                # panel PJM
                self.pjm_group = ui.pjm_group
                self.pjm_letter_label = ui.pjm_letter_label
                self.pjm_conf_label = ui.pjm_conf_label
                self.pjm_time_label = ui.pjm_time_label
                self.pjm_total_label = ui.pjm_total_label
                self.pjm_rate_label = ui.pjm_rate_label
                self.pjm_unique_label = ui.pjm_unique_label
                self.pjm_top_label = ui.pjm_top_label
                self.pjm_clear_btn = ui.pjm_clear_btn
                self.pjm_export_btn = ui.pjm_export_btn
                self.pjm_reload_btn = ui.pjm_reload_btn
                self.pjm_history_edit = ui.pjm_history_edit
                self.pjm_copy_history_btn = ui.pjm_copy_history_btn
                self.setCentralWidget(ui.central_widget)

                # przechowuje referencje do workera; tworzy worker przy starcie (QThread nie jest restartowalny)
                self.worker: ProcessingWorkerProtocol | None = None
                self.mode = "gestures"
                self._translator_available = False
                self._translator_error: str | None = None
                self._normalizer = None
                self._translator = None

                # podpina sygnaly podstawowych widzetow sterujacych
                self.start_btn.clicked.connect(self.on_start)
                self.stop_btn.clicked.connect(self.on_stop)
                self.refresh_cams_btn.clicked.connect(self.populate_cameras)
                self.exec_actions_chk.stateChanged.connect(self.on_actions_toggle)
                self.preview_chk.stateChanged.connect(self.on_preview_toggle)
                self.mode_switch.toggled.connect(self.on_mode_changed)
                self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
                # podpina sygnaly przyciskow translatora/nagrywania
                self.record_btn.clicked.connect(self.on_record_sign_language)
                self.samples_btn.clicked.connect(self.on_show_samples)
                self.train_btn.clicked.connect(self.on_train_sign_language)
                # podpina sygnaly panelu PJM
                self.pjm_clear_btn.clicked.connect(self.on_pjm_clear_stats)
                self.pjm_export_btn.clicked.connect(self.on_pjm_export_stats)
                self.pjm_reload_btn.clicked.connect(self.on_pjm_reload_model)
                self.pjm_copy_history_btn.clicked.connect(self.on_pjm_copy_history)

                self.populate_cameras()
                # wykonuje auto-start jesli wykryto kamere
                self._auto_start_if_possible()

                # uruchamia timer odswiezania listy kamer (hot-plug)
                self._cams_timer = QTimer(self)
                self._cams_timer.setInterval(int(CAMERA_SCAN_INTERVAL_MS))
                self._cams_timer.timeout.connect(self._on_cams_tick)
                self._cams_timer.start()

                self._init_translator_dependencies()

            def _init_translator_dependencies(self) -> None:
                # inicjalizuje translator PJM i (opcjonalnie) normalizer
                self._translator_available = False
                self._translator_error = None

                # init translator (obowiazkowy)
                try:
                    sign_language = importlib.import_module(
                        "app.sign_language.translator"
                    )
                    SignTranslator = getattr(sign_language, "SignTranslator")
                    self._translator = SignTranslator(
                        buffer_size=7,
                        min_hold_ms=600,
                        confidence_entry=0.85,
                        confidence_exit=0.65,
                        enable_dynamic_gestures=True,
                    )
                    self._translator_available = True
                    logger.info(
                        "[PJM] Translator zainicjalizowany: %d klas, buffer=%d, min_hold=%dms, conf_entry=%.2f",
                        len(self._translator.classes),
                        self._translator.buffer_size,
                        self._translator.min_hold_ms,
                        self._translator.confidence_entry,
                    )
                except Exception as exc:  # pragma: no cover
                    self._translator_available = False
                    self._translator_error = str(exc)
                    self._translator = None
                    self.mode_switch.setChecked(False)
                    logger.warning("[PJM] Translator niedostepny: %s", exc)
                    try:
                        self.status_label.setText(
                            f"Status: Translator PJM niedostepny ({exc.__class__.__name__})"
                        )
                    except Exception:
                        pass

                # init normalizer (opcjonalny)
                try:
                    normalizer_mod = importlib.import_module(
                        "app.sign_language.normalizer"
                    )
                    MediaPipeNormalizerCls = getattr(
                        normalizer_mod, "MediaPipeNormalizer"
                    )
                    self._normalizer = MediaPipeNormalizerCls()
                    logger.info(
                        "[PJM] Normalizer zainicjalizowany (MediaPipeNormalizer)"
                    )
                except Exception as exc:  # pragma: no cover
                    self._normalizer = None
                    logger.warning("[PJM] Normalizer niedostepny: %s", exc)

            def on_record_sign_language(self):
                # uruchamia proces nagrywania datasetu liter w osobnym procesie pythona
                # zabezpieczone haslem z config.py
                import os
                import subprocess
                import sys

                from PySide6.QtWidgets import QInputDialog, QLineEdit, QMessageBox

                from app.gesture_engine.config import (
                    PJM_LABELS,
                    RECORDING_CLIP_SECONDS,
                    RECORDING_COUNTDOWN,
                    RECORDING_PASSWORD,
                    RECORDING_REPEATS,
                )

                # sprawdz czy kamera jest zajeta przez worker
                if self.worker is not None and self.worker.isRunning():
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Zatrzymac aplikacje?")
                    msg_box.setText(
                        "Aplikacja glowna uzywa kamery. Zatrzymac ja przed nagrywaniem?"
                    )
                    msg_box.setIcon(QMessageBox.Question)

                    # polskie przyciski
                    tak_btn = msg_box.addButton("Tak", QMessageBox.YesRole)
                    msg_box.addButton("Nie", QMessageBox.NoRole)
                    msg_box.setDefaultButton(tak_btn)

                    msg_box.exec()

                    if msg_box.clickedButton() != tak_btn:
                        self.status_label.setText("Status: Nagrywanie anulowane")
                        return

                    # zatrzymaj worker
                    try:
                        self.worker.stop()
                        self.worker.wait(2000)
                        self._destroy_worker()
                        self.start_btn.setEnabled(True)
                        self.stop_btn.setEnabled(False)
                        self.camera_combo.setEnabled(True)
                        self.refresh_cams_btn.setEnabled(True)
                        logger.info("[PJM] Worker zatrzymany przed nagrywaniem")
                    except Exception as e:
                        logger.error("[PJM] Blad zatrzymywania workera: %s", e)
                        self.status_label.setText(
                            "Status: Blad zatrzymywania aplikacji"
                        )
                        return

                # dialog hasla
                password, ok = QInputDialog.getText(
                    self,
                    "Haslo wymagane",
                    "Wprowadz haslo do nagrywania alfabetu:",
                    QLineEdit.Password,
                )

                if not ok:
                    self.status_label.setText("Status: Nagrywanie anulowane")
                    return

                if password != RECORDING_PASSWORD:
                    # Wyświetl wyraźny komunikat o błędzie
                    QMessageBox.critical(
                        self,
                        "Nieprawidłowe hasło",
                        "Wprowadzone hasło jest nieprawidłowe.\nDostęp do nagrywania alfabetu został odmówiony.",
                    )
                    self.status_label.setText("Status: Nieprawidlowe haslo")
                    logger.warning("[PJM] Nieprawidlowe haslo do nagrywania")
                    return

                try:
                    base_dir = os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )

                    # przygotuj liste etykiet jako string
                    labels_str = ",".join(PJM_LABELS)

                    # Pobierz aktualnie wybraną kamerę z GUI
                    current_camera = self.camera_combo.currentData()
                    if not isinstance(current_camera, int) or current_camera < 0:
                        current_camera = 0  # fallback na domyślną
                        logger.warning("[PJM] Nieprawidłowa kamera, używam 0")

                    logger.info(
                        "[PJM] Uruchamianie nagrywania z kamerą: %d", current_camera
                    )

                    cmd = [
                        sys.executable,
                        "-m",
                        "tools.collect_dataset",
                        "--camera",
                        str(current_camera),  # Przekaż wybraną kamerę
                        "--labels",
                        labels_str,
                        "--clip-seconds",
                        str(RECORDING_CLIP_SECONDS),
                        "--countdown",
                        str(RECORDING_COUNTDOWN),
                        "--repeats",
                        str(RECORDING_REPEATS),
                        "--interactive",
                        "--show-landmarks",  # pokazuj landmarki dloni
                    ]
                    subprocess.Popen(cmd, cwd=base_dir, shell=False)  # nosec B603
                    self.status_label.setText(
                        f"Status: Uruchomiono nagrywanie {len(PJM_LABELS)} liter (kamera {current_camera})"
                    )
                    logger.info(
                        "[PJM] Uruchomiono nagrywanie: %d liter, %d powtorzen, %.1fs/probka, kamera=%d",
                        len(PJM_LABELS),
                        RECORDING_REPEATS,
                        RECORDING_CLIP_SECONDS,
                        current_camera,
                    )
                except Exception as exc:
                    self.status_label.setText(f"Status: Blad nagrywania: {exc}")
                    logger.error("on_record_sign_language error: %s", exc)

            def on_show_samples(self):
                # otwiera folder z nagranymi probkami w Eksploratorze
                import os

                base_dir = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                samples_dir = os.path.join(base_dir, "data", "collected")

                if os.path.exists(samples_dir):
                    try:
                        os.startfile(samples_dir)  # Windows
                        self.status_label.setText("Status: Otwarto folder z probkami")
                    except Exception as exc:
                        self.status_label.setText(
                            f"Status: Blad otwarcia folderu: {exc}"
                        )
                        logger.error("on_show_samples error: %s", exc)
                else:
                    self.status_label.setText("Status: Brak folderu z probkami")
                    logger.warning("Folder %s nie istnieje", samples_dir)

            def on_train_sign_language(self):
                # konsoliduje nowe probki i uruchamia trening modelu z progress barem
                from PySide6.QtCore import Qt
                from PySide6.QtWidgets import QProgressDialog

                from app.gui.training_thread import TrainingThread

                # Progress dialog
                self.progress_dialog = QProgressDialog(
                    "Przygotowanie...", "Anuluj", 0, 100, self
                )
                self.progress_dialog.setWindowTitle("Trening modelu PJM")
                self.progress_dialog.setWindowModality(Qt.WindowModal)
                self.progress_dialog.setMinimumDuration(0)

                # Training thread
                self.training_thread = TrainingThread(self)
                self.training_thread.progress_updated.connect(
                    self._on_training_progress
                )
                self.training_thread.training_finished.connect(
                    self._on_training_finished
                )
                self.training_thread.training_error.connect(self._on_training_error)

                # Anulowanie
                self.progress_dialog.canceled.connect(self.training_thread.cancel)

                # Start
                self.training_thread.start()
                self.progress_dialog.show()

                self.status_label.setText("Status: Trening w toku...")
                logger.info("[PJM] Uruchomiono trening z progress barem")

            def _on_training_progress(self, current: int, total: int, message: str):
                """Callback postępu treningu."""
                self.progress_dialog.setValue(current)
                self.progress_dialog.setLabelText(message)

            def _on_training_finished(self, results: dict):
                """Callback zakończenia treningu - pokazuje wyniki."""
                from PySide6.QtWidgets import QMessageBox

                from app.gui.training_results_dialog import TrainingResultsDialog

                self.progress_dialog.close()
                self.status_label.setText("Status: Trening zakończony!")
                logger.info("[PJM] Trening zakończony pomyślnie")

                # Dialog z wynikami
                dialog = TrainingResultsDialog(results, self)
                dialog.exec()

                # Zaproponuj przeładowanie modelu
                reply = QMessageBox.question(
                    self,
                    "Przeładować model?",
                    "Trening zakończony. Czy chcesz przeładować wytrenowany model?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.on_pjm_reload_model()

            def _on_training_error(self, error_msg: str):
                """Callback błędu treningu."""
                from PySide6.QtWidgets import QMessageBox

                self.progress_dialog.close()
                self.status_label.setText("Status: Błąd treningu")
                logger.error("[PJM] Błąd treningu: %s", error_msg)

                QMessageBox.critical(
                    self,
                    "Błąd treningu",
                    f"Wystąpił błąd podczas treningu:\n\n{error_msg}",
                )

            def on_pjm_copy_history(self):
                # kopiuje historie liter do schowka
                if not self._translator:
                    return

                try:
                    history = self._translator.get_history(format_groups=False)
                    if history:
                        from PySide6.QtWidgets import QApplication

                        QApplication.clipboard().setText(history)
                        self.status_label.setText(
                            f"Status: Skopiowano {len(history)} liter do schowka"
                        )
                        logger.info("[PJM] Historia skopiowana: %d liter", len(history))
                    else:
                        self.status_label.setText("Status: Historia jest pusta")
                except Exception as exc:
                    self.status_label.setText(f"Status: Blad kopiowania: {exc}")
                    logger.error("[PJM] Blad kopiowania historii: %s", exc)

            def on_pjm_clear_stats(self):
                # czysci statystyki translatora
                if self._translator:
                    self._translator.reset(keep_stats=False)
                    self._update_pjm_stats_display()
                    self.status_label.setText("Status: Statystyki PJM wyczyszczone")
                    logger.info("[PJM] Statystyki wyczyszczone")

            def on_pjm_export_stats(self):
                # eksportuje pełne dane sesji (historia + statystyki) do pliku CSV
                if not self._translator:
                    return

                import csv
                import datetime
                from pathlib import Path

                from PySide6.QtWidgets import QFileDialog

                try:
                    stats = self._translator.get_statistics()
                    history = self._translator.get_history(format_groups=False)

                    # generuj nazwe pliku z timestamp (bez domyslnej sciezki)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_filename = f"pjm_raport_{timestamp}.csv"

                    # otworz dialog wyboru lokalizacji zapisu (bez domyslnej sciezki)
                    filepath, _ = QFileDialog.getSaveFileName(
                        self,
                        "Zapisz raport PJM",
                        default_filename,
                        "Pliki CSV (*.csv);;Wszystkie pliki (*.*)",
                    )

                    # jesli uzytkownik anulował
                    if not filepath:
                        self.status_label.setText("Status: Eksport anulowany")
                        return

                    # zapisz pełny raport
                    with open(filepath, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)

                        # sekcja 1: HISTORIA SESJI
                        writer.writerow(["=== HISTORIA SESJI ==="])
                        writer.writerow([])
                        writer.writerow(
                            [
                                "Data",
                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            ]
                        )
                        writer.writerow(
                            ["Czas sesji (s)", f"{stats['session_duration_s']:.1f}"]
                        )
                        writer.writerow([])
                        writer.writerow(["Historia liter (sekwencja):"])
                        # podziel historie na linie po 60 znakow dla czytelnosci
                        if history:
                            for i in range(0, len(history), 60):
                                writer.writerow([history[i : i + 60]])
                        else:
                            writer.writerow(["(pusta)"])
                        writer.writerow([])

                        # sekcja 2: STATYSTYKI LITER
                        writer.writerow(["=== STATYSTYKI LITER ==="])
                        writer.writerow([])
                        writer.writerow(["Litera", "Liczba_wykryc", "Procent"])

                        total = stats["total_detections"]
                        for letter, count in sorted(
                            stats["letter_counts"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        ):
                            percent = (count / total * 100) if total > 0 else 0
                            writer.writerow([letter, count, f"{percent:.1f}%"])

                        writer.writerow([])

                        # sekcja 3: PODSUMOWANIE
                        writer.writerow(["=== PODSUMOWANIE ==="])
                        writer.writerow([])
                        writer.writerow(
                            ["Calkowite wykrycia", stats["total_detections"]]
                        )
                        writer.writerow(["Unikalne litery", stats["unique_letters"]])
                        writer.writerow(
                            ["Czas sesji (s)", f"{stats['session_duration_s']:.1f}"]
                        )
                        writer.writerow(
                            ["Wykryc/min", f"{stats['detections_per_minute']:.1f}"]
                        )
                        writer.writerow([])
                        writer.writerow(["Top 10 najczesciej uzywanych liter:"])
                        for i, (letter, count) in enumerate(stats["most_common"], 1):
                            writer.writerow([f"{i}.", letter, count])

                    # wyswietl komunikat sukcesu z nazwa pliku
                    saved_filename = Path(filepath).name
                    self.status_label.setText(
                        f"Status: Raport wyeksportowany: {saved_filename}"
                    )
                    logger.info("[PJM] Raport wyeksportowany: %s", filepath)
                except Exception as exc:
                    self.status_label.setText(f"Status: Blad eksportu: {exc}")
                    logger.error("[PJM] Blad eksportu: %s", exc)

            def on_pjm_reload_model(self):
                # przeladowuje model PJM z dysku (hot-reload)
                logger.info("[PJM] Rozpoczynam przeladowanie modelu...")
                self.status_label.setText("Status: Przeladowanie modelu PJM...")

                try:
                    # wymus odswiezenie importow (usuwa cache modulu)
                    import sys

                    if "app.sign_language.translator" in sys.modules:
                        del sys.modules["app.sign_language.translator"]
                    if "app.sign_language.normalizer" in sys.modules:
                        del sys.modules["app.sign_language.normalizer"]

                    # przeladuj translator i normalizer (normalizer opcjonalny)
                    sign_language = importlib.import_module(
                        "app.sign_language.translator"
                    )
                    SignTranslator = getattr(sign_language, "SignTranslator")
                    try:
                        normalizer_mod = importlib.import_module(
                            "app.sign_language.normalizer"
                        )
                        MediaPipeNormalizerCls = getattr(
                            normalizer_mod, "MediaPipeNormalizer"
                        )
                    except Exception:
                        MediaPipeNormalizerCls = None

                    # zapisz stare referencje (na wypadek bledu)
                    old_translator = self._translator
                    old_normalizer = self._normalizer

                    # stworz nowe instancje
                    self._normalizer = (
                        MediaPipeNormalizerCls() if MediaPipeNormalizerCls else None
                    )
                    self._translator = SignTranslator()

                    logger.info(
                        "[PJM] Model przeladowany: %d klas, input_size=%d, buffer=%d",
                        len(self._translator.classes),
                        self._translator.model_input_size,
                        self._translator.buffer_size,
                    )

                    # jesli worker istnieje i pracuje, zaktualizuj jego referencje
                    if self.worker and self.mode == "translator":
                        self.worker.set_mode(
                            mode="translator",
                            translator=self._translator,
                            normalizer=self._normalizer,
                        )
                        logger.info("[PJM] Worker zaktualizowany z nowym modelem")

                    self.status_label.setText(
                        f"Status: Model przeladowany (input={self._translator.model_input_size}D, klasy={len(self._translator.classes)})"
                    )

                    # wyswietl informacje o usuniętych cechach
                    if len(self._translator.zero_var_indices) > 0:
                        logger.info(
                            "[PJM] Model usunal %d cech z zerowa wariancja: %s",
                            len(self._translator.zero_var_indices),
                            list(self._translator.zero_var_indices),
                        )

                    # natychmiast odswiez wyswietlanie statystyk w GUI
                    self._update_pjm_stats_display()

                except Exception as exc:
                    logger.error("[PJM] Blad przeladowania modelu: %s", exc)
                    self.status_label.setText(
                        f"Status: Blad przeladowania modelu: {exc}"
                    )
                    # przywroc stare referencje jesli sa
                    if "old_translator" in locals():
                        self._translator = old_translator
                        self._normalizer = old_normalizer
                    logger.error("[PJM] Blad eksportu statystyk: %s", exc)

            def _update_pjm_stats_display(self):
                # aktualizuje wyswietlanie statystyk w panelu PJM
                if not self._translator:
                    return

                state = self._translator.get_state()

                # aktualna litera
                if state["current_letter"]:
                    self.pjm_letter_label.setText(state["current_letter"])
                    self.pjm_conf_label.setText(
                        f"Pewnosc: {state['confidence']*100:.0f}%"
                    )
                    self.pjm_time_label.setText(f"Czas: {state['time_held_ms']:.0f}ms")
                else:
                    self.pjm_letter_label.setText("--")
                    self.pjm_conf_label.setText("Pewnosc: --%")
                    self.pjm_time_label.setText("Czas: 0ms")

                # historia liter
                history = self._translator.get_history(format_groups=True)
                self.pjm_history_edit.setText(history)
                # przewin do konca
                self.pjm_history_edit.setCursorPosition(len(history))

                # statystyki sesji
                self.pjm_total_label.setText(
                    f"Wykryto liter: {state['total_detections']}"
                )
                self.pjm_rate_label.setText(
                    f"Wykryc/min: {state['detections_per_minute']:.1f}"
                )
                self.pjm_unique_label.setText(f"Unikalne: {state['unique_letters']}")

                # top 5 liter
                if self._translator.letter_stats:
                    top5 = self._translator.letter_stats.most_common(5)
                    top_str = ", ".join(
                        [f"{letter}({count})" for letter, count in top5]
                    )
                    self.pjm_top_label.setText(f"Top 5: {top_str}")
                else:
                    self.pjm_top_label.setText("Top 5: --")

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
                mode = "translator" if self.mode_switch.isChecked() else "gestures"
                try:
                    self._destroy_worker()
                except Exception as e:
                    logger.debug("_restart_with_camera: destroy error: %s", e)
                self.worker = self._create_worker()
                self.worker.configure(
                    cam_data,
                    actions_enabled,
                    preview_enabled,
                    mode,
                )
                # natychmiast ustawia stan akcji/podgladu przed startem
                try:
                    self.worker.set_actions_enabled(actions_enabled)
                    self.worker.set_preview_enabled(preview_enabled)
                    self.worker.set_mode(
                        mode if isinstance(mode, str) else "gestures",
                        self._translator if self._translator_available else None,
                        self._normalizer if self._translator_available else None,
                    )
                except Exception as e:
                    logger.debug("_restart_with_camera: sync pre-start error: %s", e)
                self.worker.start()
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                # Pozwól na zmianę kamery podczas działania
                self.camera_combo.setEnabled(True)
                self.refresh_cams_btn.setEnabled(True)
                logger.info(
                    "[Camera] _restart_with_camera: Camera controls remain ENABLED during operation"
                )

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

                logger.info(
                    "[Camera] populate_cameras: START - current_data=%s, current_text=%s",
                    current_data,
                    current_text,
                )

                self.camera_combo.blockSignals(True)
                self.camera_combo.clear()
                sources = discover_camera_sources(max_index=int(CAMERA_MAX_INDEX_SCAN))

                logger.info(
                    "[Camera] populate_cameras: Discovered %d cameras", len(sources)
                )
                for i, (source, name) in enumerate(sources):
                    logger.info("[Camera]   [%d] source=%s, name=%s", i, source, name)
                    self.camera_combo.addItem(name, source)

                if self.camera_combo.count() == 0:
                    self.camera_combo.addItem("Brak kamer", -1)
                    logger.warning("[Camera] populate_cameras: No cameras found!")
                else:
                    # proba przywrocenia poprzedniego wyboru (po data, a nastepnie po tekscie)
                    if current_data is not None:
                        idx = self.camera_combo.findData(current_data)
                        logger.info(
                            "[Camera] populate_cameras: Trying to restore by data=%s -> idx=%d",
                            current_data,
                            idx,
                        )
                        if idx >= 0:
                            self.camera_combo.setCurrentIndex(idx)
                            logger.info(
                                "[Camera] populate_cameras: Restored camera by data at index %d",
                                idx,
                            )
                        else:
                            # fallback po tekscie
                            if current_text:
                                idx2 = self.camera_combo.findText(current_text)
                                logger.info(
                                    "[Camera] populate_cameras: Fallback to text=%s -> idx=%d",
                                    current_text,
                                    idx2,
                                )
                                if idx2 >= 0:
                                    self.camera_combo.setCurrentIndex(idx2)
                                    logger.info(
                                        "[Camera] populate_cameras: Restored camera by text at index %d",
                                        idx2,
                                    )
                    else:
                        logger.info(
                            "[Camera] populate_cameras: No previous selection to restore"
                        )

                final_index = self.camera_combo.currentIndex()
                final_data = self.camera_combo.currentData()
                final_text = self.camera_combo.currentText()
                logger.info(
                    "[Camera] populate_cameras: FINAL - index=%d, data=%s, text=%s",
                    final_index,
                    final_data,
                    final_text,
                )

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
                mode = "translator" if self.mode_switch.isChecked() else "gestures"
                self.worker = self._create_worker()
                self.worker.configure(
                    cam_data,
                    actions_enabled,
                    preview_enabled,
                    mode,
                )
                try:
                    self.worker.set_actions_enabled(actions_enabled)
                    self.worker.set_preview_enabled(preview_enabled)
                    self.worker.set_mode(
                        mode if isinstance(mode, str) else "gestures",
                        self._translator if self._translator_available else None,
                        self._normalizer if self._translator_available else None,
                    )
                except Exception as e:
                    logger.debug("on_start: sync pre-start error: %s", e)
                self.worker.start()
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                # Pozwól na zmianę kamery podczas działania
                self.camera_combo.setEnabled(True)
                self.refresh_cams_btn.setEnabled(True)
                logger.info(
                    "[Camera] on_start: Camera controls remain ENABLED during operation"
                )

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
                logger.info("[Camera] on_camera_changed: TRIGGERED - index=%d", _index)

                if self.worker is None or not self.worker.isRunning():
                    logger.info(
                        "[Camera] on_camera_changed: Worker not running, ignoring change"
                    )
                    return

                cd_any: Any = self.camera_combo.currentData()
                cam_text = self.camera_combo.currentText()
                logger.info(
                    "[Camera] on_camera_changed: New selection - data=%s, text=%s",
                    cd_any,
                    cam_text,
                )

                if not isinstance(cd_any, (int, str)):
                    logger.warning(
                        "[Camera] on_camera_changed: Invalid camera data type: %s",
                        type(cd_any),
                    )
                    return

                cam_data = cast(Union[int, str], cd_any)
                logger.info(
                    "[Camera] on_camera_changed: Starting camera switch to: %s (%s)",
                    cam_data,
                    cam_text,
                )

                self.status_label.setText(
                    "Status: Zmiana kamery - restart przetwarzania..."
                )
                try:
                    logger.info("[Camera] on_camera_changed: Stopping worker...")
                    self.worker.stop()
                    self.worker.wait(2000)
                    logger.info("[Camera] on_camera_changed: Worker stopped")
                except Exception as e:
                    logger.debug("on_camera_changed: stop/wait error: %s", e)
                try:
                    self._disconnect_worker_signals(self.worker)
                except Exception as e:
                    logger.debug("on_camera_changed: disconnect error: %s", e)

                actions_enabled = self.exec_actions_chk.isChecked()
                preview_enabled = self.preview_chk.isChecked()
                mode = "translator" if self.mode_switch.isChecked() else "gestures"

                logger.info(
                    "[Camera] on_camera_changed: Creating new worker with cam=%s, mode=%s",
                    cam_data,
                    mode,
                )
                self.worker = self._create_worker()
                self.worker.configure(
                    cam_data,
                    actions_enabled,
                    preview_enabled,
                    mode,
                )
                try:
                    self.worker.set_actions_enabled(actions_enabled)
                    self.worker.set_preview_enabled(preview_enabled)
                    self.worker.set_mode(
                        mode,
                        self._translator if self._translator_available else None,
                        self._normalizer if self._translator_available else None,
                    )
                except Exception as e:
                    logger.debug("on_camera_changed: sync pre-start error: %s", e)

                logger.info(
                    "[Camera] on_camera_changed: Starting worker with new camera..."
                )
                self.worker.start()
                self.status_label.setText(
                    "Status: Przetwarzanie uruchomione (nowa kamera)"
                )
                logger.info(
                    "[Camera] on_camera_changed: COMPLETE - switched to camera %s",
                    cam_text,
                )

            def on_actions_toggle(self, _state: int):
                enabled = self.exec_actions_chk.isChecked()
                worker = self.worker
                worker_running = worker is not None and worker.isRunning()
                if worker_running and worker is not None:  # mypy guard
                    try:
                        worker.set_actions_enabled(enabled)
                    except Exception as e:
                        logger.debug("on_actions_toggle: worker sync error: %s", e)
                status = (
                    "Status: Akcje włączone" if enabled else "Status: Akcje wyłączone"
                )
                if not worker_running:
                    status += " (zmiana zadziała przy starcie)"
                self.status_label.setText(status)

            def on_preview_toggle(self, _state: int):
                enabled = self.preview_chk.isChecked()
                worker = self.worker
                worker_running = worker is not None and worker.isRunning()
                if worker_running and worker is not None:  # mypy guard
                    try:
                        worker.set_preview_enabled(enabled)
                    except Exception as e:
                        logger.debug("on_preview_toggle: worker sync error: %s", e)
                status = (
                    "Status: Podgląd włączony"
                    if enabled
                    else "Status: Podgląd wyłączony"
                )
                if not worker_running:
                    status += " (zmiana zadziała przy starcie)"
                self.status_label.setText(status)

            def on_mode_changed(self, checked: bool):
                # checked = True -> Tlumacz, False -> Gesty
                mode = "translator" if checked else "gestures"

                # aktualizuj tekst i kolor labela
                if checked:
                    self.mode_text_label.setText("Tlumacz")
                    self.mode_text_label.setStyleSheet(
                        "font-weight: bold; color: #4CAF50;"
                    )
                else:
                    self.mode_text_label.setText("Gesty")
                    self.mode_text_label.setStyleSheet(
                        "font-weight: bold; color: #4CAF50;"
                    )

                # pozwalamy pozostac w trybie translator nawet jesli niedostepny, tylko informujemy
                if mode == "translator" and not self._translator_available:
                    err = self._translator_error or "Model pjm niedostepny"
                    try:
                        self.status_label.setText(f"Status: {err}")
                    except Exception:
                        pass
                try:
                    self.mode = str(mode)
                except Exception:
                    self.mode = "gestures"

                # pokaz/ukryj panel PJM w zaleznosci od trybu
                is_translator_mode = self.mode == "translator"
                self.pjm_group.setVisible(is_translator_mode)

                # ukryj Left/Right w trybie translator (pokazywaly gesty, nie litery)
                self.left_hand_label.setVisible(not is_translator_mode)
                self.right_hand_label.setVisible(not is_translator_mode)

                # ukryj checkbox "Wykonuj akcje" w trybie translator
                self.exec_actions_chk.setVisible(not is_translator_mode)

                # pokaz przyciski treningu, nagrywania i próbek tylko w trybie translator
                self.train_btn.setVisible(is_translator_mode)
                self.record_btn.setVisible(is_translator_mode)
                self.samples_btn.setVisible(is_translator_mode)

                if is_translator_mode and self._translator:
                    # reset stanu ale zachowaj statystyki
                    self._translator.reset(keep_stats=True)
                    self._update_pjm_stats_display()

                if self.worker is not None:
                    try:
                        self.worker.set_mode(
                            self.mode,
                            self._translator if self._translator_available else None,
                            self._normalizer if self._translator_available else None,
                        )
                    except Exception as e:
                        logger.debug("on_mode_changed: worker.set_mode error: %s", e)
                    else:
                        # komunikat trybu zawsze, ale gdy brak translatora, poprzedni status bledu zostaje
                        if not (
                            mode == "translator" and not self._translator_available
                        ):
                            self.status_label.setText(
                                f"Status: Tryb {'tłumacz' if self.mode == 'translator' else 'gesty'}"
                            )

            def on_frame(self, qimg):
                # aktualizuje podglad wideo w etykiecie
                try:
                    self.video_label.setPixmap(QPixmap.fromImage(qimg))
                except Exception as e:
                    logger.debug("on_frame: setPixmap error: %s", e)

            def on_status(self, text: str):
                # aktualizuje status w ui
                try:
                    self.status_label.setText(f"Status: {text}")
                except Exception as e:
                    logger.debug("on_status: setText error: %s", e)

            def on_metrics(self, fps: int, frametime_ms: int):
                # aktualizuje metryki fps i frametime
                try:
                    self.fps_label.setText(f"FPS: {fps} ({frametime_ms} ms)")
                except Exception as e:
                    logger.debug("on_metrics: setText error: %s", e)

            def on_gesture(self, result):
                if self.mode == "translator":
                    # w trybie tłumacza zawsze aktualizuj panel PJM (nawet gdy brak wykrycia)
                    self._update_pjm_stats_display()
                    if result.name:
                        self.gesture_label.setText(f"Tłumacz: {result.name}")
                    else:
                        self.gesture_label.setText("Tłumacz: Brak")
                    return
                if result.name:
                    self.gesture_label.setText(
                        f"Gest (najlepszy): {result.name} ({int(result.confidence * 100)}%)"
                    )
                else:
                    self.gesture_label.setText("Gest (najlepszy): Brak")

            def on_hands(self, per_hand: List[object]):
                # per_hand to lista SingleHandResult z polami: handedness ("Left"/"Right"), name, confidence
                left_txt = "Lewa: Brak"
                right_txt = "Prawa: Brak"
                try:
                    for h in per_hand:
                        handed = getattr(h, "handedness", None)
                        name = getattr(h, "name", None)
                        conf = getattr(h, "confidence", 0.0)
                        if handed and str(handed).lower().startswith("left"):
                            left_txt = (
                                f"Lewa: {name} ({int(conf * 100)}%)"
                                if name
                                else "Lewa: Brak"
                            )
                        elif handed and str(handed).lower().startswith("right"):
                            right_txt = (
                                f"Prawa: {name} ({int(conf * 100)}%)"
                                if name
                                else "Prawa: Brak"
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
                # zatrzymuje timer skanowania kamer
                try:
                    if hasattr(self, "_cams_timer") and self._cams_timer is not None:
                        self._cams_timer.stop()
                        logger.debug("MainWindow.closeEvent: cams_timer zatrzymany")
                except Exception as e:
                    logger.debug("MainWindow.closeEvent: timer stop error: %s", e)
                # niszczy worker i zwalnia zasoby
                try:
                    self._destroy_worker()
                except Exception as e:
                    logger.debug("MainWindow.closeEvent: destroy error: %s", e)
                super().closeEvent(event)

        return _MainWindow()
