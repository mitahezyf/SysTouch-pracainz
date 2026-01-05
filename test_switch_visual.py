"""Test wizualny Android-style switch toggle"""

import sys

from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from app.gui.ui_components import SwitchToggle


def main():
    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("Test Switch Toggle")
    layout = QVBoxLayout()

    label = QLabel("Tryb: Gesty")
    label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;")

    switch = SwitchToggle(width=60, height=28)

    def on_toggle(checked):
        if checked:
            label.setText("Tryb: Tlumacz")
            label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        else:
            label.setText("Tryb: Gesty")
            label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;")

    switch.toggled.connect(on_toggle)

    layout.addWidget(label)
    layout.addWidget(switch)

    window.setLayout(layout)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
