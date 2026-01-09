from __future__ import annotations

import importlib
from dataclasses import dataclass


class SwitchToggle:
    """Android-style switch toggle widget"""

    def __new__(cls, width: int = 60, height: int = 28):
        qtcore = importlib.import_module("PySide6.QtCore")
        qtgui = importlib.import_module("PySide6.QtGui")
        qtw = importlib.import_module("PySide6.QtWidgets")

        QWidget = qtw.QWidget
        QPropertyAnimation = qtcore.QPropertyAnimation
        QEasingCurve = qtcore.QEasingCurve
        Qt = qtcore.Qt
        pyqtSignal = qtcore.Signal
        pyqtProperty = qtcore.Property
        QPainter = qtgui.QPainter
        QColor = qtgui.QColor
        QBrush = qtgui.QBrush
        QPen = qtgui.QPen

        class _SwitchToggle(QWidget):  # type: ignore[valid-type, misc]
            toggled = pyqtSignal(bool)

            def __init__(self, parent=None):
                super().__init__(parent)
                self.setFixedSize(width, height)
                self._checked = False
                self._circle_position = 3
                self._animation = QPropertyAnimation(self, b"circle_position", self)
                self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
                self._animation.setDuration(120)

                # kolory
                self._bg_color_off = QColor("#555555")
                self._bg_color_on = QColor("#4CAF50")
                self._circle_color = QColor("#FFFFFF")
                self._bg_color_on_translator = QColor("#2196F3")

                self.setCursor(Qt.CursorShape.PointingHandCursor)

            def paintEvent(self, event):
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                # tlo (zaokraglony prostokat)
                if self._checked:
                    bg_color = self._bg_color_on_translator
                else:
                    bg_color = self._bg_color_off

                painter.setBrush(QBrush(bg_color))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawRoundedRect(0, 0, width, height, height / 2, height / 2)

                # kolko (thumb)
                circle_diameter = height - 6
                painter.setBrush(QBrush(self._circle_color))
                painter.drawEllipse(
                    int(self._circle_position),
                    3,
                    circle_diameter,
                    circle_diameter,
                )

            def mousePressEvent(self, event):
                if event.button() == Qt.MouseButton.LeftButton:
                    self.toggle()

            def isChecked(self):
                return self._checked

            def setChecked(self, checked: bool):
                if self._checked == checked:
                    return
                self._checked = checked
                self._move_circle()
                self.toggled.emit(self._checked)

            def toggle(self):
                self.setChecked(not self._checked)

            def _move_circle(self):
                if self._checked:
                    end_pos = width - height + 3
                else:
                    end_pos = 3
                self._animation.setStartValue(self._circle_position)
                self._animation.setEndValue(end_pos)
                self._animation.start()

            def get_circle_position(self):
                return self._circle_position

            def set_circle_position(self, pos):
                self._circle_position = pos
                self.update()

            circle_position = pyqtProperty(
                int, fget=get_circle_position, fset=set_circle_position
            )

        return _SwitchToggle()


@dataclass(slots=True)
class UIRefs:
    # przechowuje referencje do kluczowych widzetow wykorzystywanych przez mainWindow
    title_label: object
    video_label: object
    camera_combo: object
    refresh_cams_btn: object
    start_btn: object
    stop_btn: object
    exec_actions_chk: object
    preview_chk: object
    mode_switch: object
    mode_text_label: object
    status_label: object
    fps_label: object
    gesture_label: object
    left_hand_label: object
    right_hand_label: object
    central_widget: object
    # przyciski dla nagrywania alfabetu i treningu modelu jezyka migowego
    record_btn: object
    train_btn: object
    # panel PJM - wyswietlanie liter i statystyk
    pjm_group: object
    pjm_letter_label: object
    pjm_conf_label: object
    pjm_time_label: object
    pjm_total_label: object
    pjm_rate_label: object
    pjm_unique_label: object
    pjm_top_label: object
    pjm_clear_btn: object
    pjm_export_btn: object
    pjm_reload_btn: object
    pjm_history_edit: object
    pjm_copy_history_btn: object


def create_separator():
    # tworzy poziomy separator
    qtw = importlib.import_module("PySide6.QtWidgets")
    QFrame = qtw.QFrame
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet("color: #444;")
    return line


def build_ui(display_width: int, display_height: int) -> UIRefs:
    # buduje layout i zwraca referencje do elementow ui
    # importuje PySide6 dynamicznie aby uniknac twardej zaleznosci przy imporcie modulu
    qtcore = importlib.import_module("PySide6.QtCore")
    qtw = importlib.import_module("PySide6.QtWidgets")

    Qt = qtcore.Qt
    QLabel = qtw.QLabel
    QHBoxLayout = qtw.QHBoxLayout
    QVBoxLayout = qtw.QVBoxLayout
    QComboBox = qtw.QComboBox
    QCheckBox = qtw.QCheckBox
    QPushButton = qtw.QPushButton
    QGroupBox = qtw.QGroupBox
    QWidget = qtw.QWidget

    title_label = QLabel("SysTouch - Sterowanie gestami")
    title_label.setObjectName("Title")

    video_label = QLabel("Brak podgladu")
    video_label.setObjectName("Video")
    video_label.setAlignment(Qt.AlignCenter)
    video_label.setFixedSize(display_width, display_height)

    camera_combo = QComboBox()
    refresh_cams_btn = QPushButton("Odswiez kamery")
    start_btn = QPushButton("Start")
    stop_btn = QPushButton("Stop")
    stop_btn.setEnabled(False)
    exec_actions_chk = QCheckBox("Wykonuj akcje")
    exec_actions_chk.setChecked(True)
    preview_chk = QCheckBox("Pokaz podglad")
    preview_chk.setChecked(True)

    # Android-style switch toggle
    mode_label = QLabel("Tryb:")
    mode_switch = SwitchToggle(width=60, height=28)
    mode_switch.setChecked(False)  # type: ignore[attr-defined]  # False = Gesty, True = Tlumacz
    mode_text_label = QLabel("Gesty")
    mode_text_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
    mode_text_label.setMinimumWidth(70)

    record_btn = QPushButton("Nagraj alfabet")
    train_btn = QPushButton("Wytrenuj model")
    record_btn.setToolTip("Rozpoczyna nagrywanie datasetu liter (osobny proces)")
    train_btn.setToolTip("Trenuje model na nagranym datasiecie (osobny proces)")
    # ukryj domyslnie - pokaza sie tylko w trybie translator
    record_btn.setVisible(False)
    train_btn.setVisible(False)

    status_label = QLabel("Status: gotowe")
    fps_label = QLabel("FPS: 0")
    gesture_label = QLabel("Gesture (best): None")
    left_hand_label = QLabel("Left: None")
    right_hand_label = QLabel("Right: None")

    top_bar = QHBoxLayout()
    top_bar.addWidget(QLabel("Kamera:"))
    top_bar.addWidget(camera_combo)
    top_bar.addWidget(refresh_cams_btn)
    top_bar.addStretch(1)
    top_bar.addWidget(exec_actions_chk)
    top_bar.addWidget(preview_chk)
    top_bar.addWidget(mode_label)
    top_bar.addWidget(mode_switch)
    top_bar.addWidget(mode_text_label)

    ctrl_bar = QHBoxLayout()
    ctrl_bar.addWidget(start_btn)
    ctrl_bar.addWidget(stop_btn)
    ctrl_bar.addWidget(record_btn)
    ctrl_bar.addWidget(train_btn)
    ctrl_bar.addStretch(1)
    ctrl_bar.addWidget(fps_label)

    # Panel PJM - wyswietlanie liter i statystyk
    pjm_group = QGroupBox("Alfabet PJM")
    pjm_layout = QVBoxLayout()

    # Aktualna litera (duzy font)
    pjm_letter_label = QLabel("--")
    pjm_letter_label.setStyleSheet(
        "font-size: 72px; font-weight: bold; color: #4CAF50;"
    )
    pjm_letter_label.setAlignment(Qt.AlignCenter)
    pjm_layout.addWidget(pjm_letter_label)

    # Confidence i czas
    pjm_conf_label = QLabel("Pewnosc: --%")
    pjm_conf_label.setStyleSheet("font-size: 16px;")
    pjm_conf_label.setAlignment(Qt.AlignCenter)
    pjm_layout.addWidget(pjm_conf_label)

    pjm_time_label = QLabel("Czas: 0ms")
    pjm_time_label.setStyleSheet("font-size: 14px; color: #888;")
    pjm_time_label.setAlignment(Qt.AlignCenter)
    pjm_layout.addWidget(pjm_time_label)

    # Separator
    pjm_layout.addWidget(create_separator())

    # Historia liter
    pjm_history_label = QLabel("Historia liter:")
    pjm_history_label.setStyleSheet("font-size: 14px; font-weight: bold;")
    pjm_layout.addWidget(pjm_history_label)

    QLineEdit = qtw.QLineEdit
    pjm_history_edit = QLineEdit()
    pjm_history_edit.setReadOnly(True)
    pjm_history_edit.setStyleSheet(
        "font-family: 'Courier New', monospace; font-size: 12px; background: #2b2b2b; color: #e0e0e0; padding: 5px;"
    )
    pjm_history_edit.setPlaceholderText("Wykryte litery pojawia sie tutaj...")
    pjm_layout.addWidget(pjm_history_edit)

    pjm_copy_history_btn = QPushButton("Kopiuj historie")
    pjm_copy_history_btn.setStyleSheet("font-size: 11px;")
    pjm_layout.addWidget(pjm_copy_history_btn)

    # Separator
    pjm_layout.addWidget(create_separator())

    # Statystyki sesji
    pjm_stats_label = QLabel("Statystyki sesji:")
    pjm_stats_label.setStyleSheet("font-size: 14px; font-weight: bold;")
    pjm_layout.addWidget(pjm_stats_label)

    pjm_total_label = QLabel("Wykryto liter: 0")
    pjm_total_label.setStyleSheet("font-size: 12px;")
    pjm_layout.addWidget(pjm_total_label)

    pjm_rate_label = QLabel("Wykryc/min: 0.0")
    pjm_rate_label.setStyleSheet("font-size: 12px;")
    pjm_layout.addWidget(pjm_rate_label)

    pjm_unique_label = QLabel("Unikalne: 0")
    pjm_unique_label.setStyleSheet("font-size: 12px;")
    pjm_layout.addWidget(pjm_unique_label)

    # Top litery
    pjm_top_label = QLabel("Top 5: --")
    pjm_top_label.setStyleSheet("font-size: 11px; color: #666;")
    pjm_top_label.setWordWrap(True)
    pjm_layout.addWidget(pjm_top_label)

    # Przyciski akcji
    pjm_buttons_layout = QHBoxLayout()
    pjm_clear_btn = QPushButton("Wyczysc")
    pjm_export_btn = QPushButton("Eksportuj")
    pjm_reload_btn = QPushButton("Przeladuj Model")
    pjm_reload_btn.setToolTip("Przeladowuje model PJM z dysku (po przetrenowaniu)")
    pjm_buttons_layout.addWidget(pjm_clear_btn)
    pjm_buttons_layout.addWidget(pjm_export_btn)
    pjm_buttons_layout.addWidget(pjm_reload_btn)
    pjm_layout.addLayout(pjm_buttons_layout)

    pjm_group.setLayout(pjm_layout)
    pjm_group.setVisible(False)  # ukryj domyslnie, pokaze sie w trybie translator
    pjm_group.setMaximumWidth(
        350
    )  # ogranicza szerokosc panelu aby nie zaslanialo wideo

    info_group = QGroupBox("Informacje")
    info_layout = QVBoxLayout()
    info_layout.addWidget(gesture_label)
    info_layout.addWidget(left_hand_label)
    info_layout.addWidget(right_hand_label)
    info_layout.addWidget(status_label)
    info_group.setLayout(info_layout)

    # tworzy horizontal layout dla wideo i panelu PJM (obok siebie)
    video_and_stats_layout = QHBoxLayout()
    video_and_stats_layout.addWidget(video_label, alignment=Qt.AlignCenter)
    video_and_stats_layout.addWidget(pjm_group)
    video_and_stats_layout.addStretch(0)  # panel PJM przylega do prawej krawedzi wideo

    central = QWidget()
    main_layout = QVBoxLayout()
    main_layout.addWidget(title_label)
    main_layout.addLayout(top_bar)
    main_layout.addLayout(video_and_stats_layout)  # wideo + panel PJM obok siebie
    main_layout.addLayout(ctrl_bar)
    main_layout.addWidget(info_group)
    central.setLayout(main_layout)

    return UIRefs(
        title_label=title_label,
        video_label=video_label,
        camera_combo=camera_combo,
        refresh_cams_btn=refresh_cams_btn,
        start_btn=start_btn,
        stop_btn=stop_btn,
        exec_actions_chk=exec_actions_chk,
        preview_chk=preview_chk,
        mode_switch=mode_switch,
        mode_text_label=mode_text_label,
        status_label=status_label,
        fps_label=fps_label,
        gesture_label=gesture_label,
        left_hand_label=left_hand_label,
        right_hand_label=right_hand_label,
        central_widget=central,
        record_btn=record_btn,
        train_btn=train_btn,
        pjm_group=pjm_group,
        pjm_letter_label=pjm_letter_label,
        pjm_conf_label=pjm_conf_label,
        pjm_time_label=pjm_time_label,
        pjm_total_label=pjm_total_label,
        pjm_rate_label=pjm_rate_label,
        pjm_unique_label=pjm_unique_label,
        pjm_top_label=pjm_top_label,
        pjm_clear_btn=pjm_clear_btn,
        pjm_export_btn=pjm_export_btn,
        pjm_reload_btn=pjm_reload_btn,
        pjm_history_edit=pjm_history_edit,
        pjm_copy_history_btn=pjm_copy_history_btn,
    )
