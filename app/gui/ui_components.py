from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(slots=True)
class UIRefs:
    # trzyma referencje do kluczowych widzetow uzywanych przez MainWindow
    title_label: object
    video_label: object
    camera_combo: object
    refresh_cams_btn: object
    start_btn: object
    stop_btn: object
    exec_actions_chk: object
    preview_chk: object
    status_label: object
    fps_label: object
    gesture_label: object
    central_widget: object


def build_ui(display_width: int, display_height: int) -> UIRefs:
    """Buduje layout i zwraca referencje do elementow UI.

    importuje PySide6 dynamicznie, aby uniknac zaleznosci przy imporcie modulow
    """
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
    exec_actions_chk.setChecked(False)
    preview_chk = QCheckBox("Pokaz podglad")
    preview_chk.setChecked(True)

    status_label = QLabel("Status: gotowe")
    fps_label = QLabel("FPS: 0")
    gesture_label = QLabel("Gesture: None")

    top_bar = QHBoxLayout()
    top_bar.addWidget(QLabel("Kamera:"))
    top_bar.addWidget(camera_combo)
    top_bar.addWidget(refresh_cams_btn)
    top_bar.addStretch(1)
    top_bar.addWidget(exec_actions_chk)
    top_bar.addWidget(preview_chk)

    ctrl_bar = QHBoxLayout()
    ctrl_bar.addWidget(start_btn)
    ctrl_bar.addWidget(stop_btn)
    ctrl_bar.addStretch(1)
    ctrl_bar.addWidget(fps_label)

    info_group = QGroupBox("Informacje")
    info_layout = QVBoxLayout()
    info_layout.addWidget(gesture_label)
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
        status_label=status_label,
        fps_label=fps_label,
        gesture_label=gesture_label,
        central_widget=central,
    )
