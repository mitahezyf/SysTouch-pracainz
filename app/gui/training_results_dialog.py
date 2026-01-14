"""
Dialog wyświetlający wyniki treningu modelu PJM w języku polskim.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class TrainingResultsDialog(QDialog):
    """Dialog wyświetlający wyniki treningu modelu."""

    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.results = results
        self._init_ui()

    def _init_ui(self):
        """Inicjalizuje interfejs dialogu."""
        self.setWindowTitle("Wyniki treningu modelu PJM")
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout()

        # Podsumowanie
        summary_group = QGroupBox("Podsumowanie")
        summary_layout = QVBoxLayout()

        test_acc = self.results["test_accuracy"] * 100
        val_acc = self.results["val_accuracy"] * 100
        num_classes = self.results["num_classes"]
        train_samples = self.results["train_samples"]
        val_samples = self.results["val_samples"]
        test_samples = self.results["test_samples"]

        summary_layout.addWidget(QLabel(f"<b>Dokładność testowa:</b> {test_acc:.2f}%"))
        summary_layout.addWidget(
            QLabel(f"<b>Dokładność walidacyjna:</b> {val_acc:.2f}%")
        )
        summary_layout.addWidget(QLabel(f"<b>Liczba klas (liter):</b> {num_classes}"))

        stats_row = QHBoxLayout()
        stats_row.addWidget(QLabel(f"Próbki treningowe: {train_samples}"))
        stats_row.addWidget(QLabel(f"Próbki walidacyjne: {val_samples}"))
        stats_row.addWidget(QLabel(f"Próbki testowe: {test_samples}"))
        summary_layout.addLayout(stats_row)

        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # Tabela per-class stats
        table_group = QGroupBox("Statystyki dla każdej litery")
        table_layout = QVBoxLayout()

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["Litera", "Próbki treningowe", "Precyzja", "Czułość", "Wynik F1"]
        )

        per_class = self.results["per_class_stats"]
        table.setRowCount(len(per_class))

        for i, (cls, stats) in enumerate(per_class.items()):
            table.setItem(i, 0, QTableWidgetItem(cls))
            table.setItem(i, 1, QTableWidgetItem(str(stats["train_samples"])))
            table.setItem(i, 2, QTableWidgetItem(f"{stats['precision']:.2f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{stats['recall']:.2f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{stats['f1-score']:.2f}"))

            # Center alignment
            for col in range(5):
                if table.item(i, col):
                    table.item(i, col).setTextAlignment(Qt.AlignCenter)

        table.resizeColumnsToContents()
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)

        table_layout.addWidget(table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # Przyciski
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.setLayout(layout)
