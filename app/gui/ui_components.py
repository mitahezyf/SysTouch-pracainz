from __future__ import annotations

import importlib
from dataclasses import dataclass


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
    mode_combo: object
    status_label: object
    fps_label: object
    gesture_label: object
    left_hand_label: object
    right_hand_label: object
    central_widget: object
    # przyciski dla nagrywania alfabetu i treningu modelu jezyka migowego
    record_btn: object
    train_btn: object


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
    mode_label = QLabel("Tryb:")
    mode_combo = QComboBox()
    mode_combo.addItem("Gesty", "gestures")
    mode_combo.addItem("Tlumacz", "translator")
    mode_combo.setCurrentIndex(0)

    record_btn = QPushButton("Nagraj alfabet")
    train_btn = QPushButton("Wytrenuj model")
    record_btn.setToolTip("Rozpoczyna nagrywanie datasetu liter (osobny proces)")
    train_btn.setToolTip("Trenuje model na nagranym datasiecie (osobny proces)")

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
    top_bar.addWidget(mode_combo)

    ctrl_bar = QHBoxLayout()
    ctrl_bar.addWidget(start_btn)
    ctrl_bar.addWidget(stop_btn)
    ctrl_bar.addWidget(record_btn)
    ctrl_bar.addWidget(train_btn)
    ctrl_bar.addStretch(1)
    ctrl_bar.addWidget(fps_label)

    info_group = QGroupBox("Informacje")
    info_layout = QVBoxLayout()
    info_layout.addWidget(gesture_label)
    info_layout.addWidget(left_hand_label)
    info_layout.addWidget(right_hand_label)
    info_layout.addWidget(status_label)
    info_group.setLayout(info_layout)

    central = QWidget()
    main_layout = QVBoxLayout()
    main_layout.addWidget(title_label)
    main_layout.addLayout(top_bar)
    main_layout.addWidget(video_label, alignment=Qt.AlignCenter)
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
        mode_combo=mode_combo,
        status_label=status_label,
        fps_label=fps_label,
        gesture_label=gesture_label,
        left_hand_label=left_hand_label,
        right_hand_label=right_hand_label,
        central_widget=central,
        record_btn=record_btn,
        train_btn=train_btn,
    )
