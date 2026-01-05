"""Testy dla app.gesture_engine.actions.volume_helpers."""

from app.gesture_engine.actions.volume_helpers import quantize_pct


class TestQuantizePct:
    """Testy dla funkcji quantize_pct."""

    def test_quantize_standard_values(self) -> None:
        """sprawdza kwantyzacje standardowych wartosci."""
        assert quantize_pct(0.0, step=5) == 0
        assert quantize_pct(5.0, step=5) == 5
        assert quantize_pct(50.0, step=5) == 50
        assert quantize_pct(100.0, step=5) == 100

    def test_quantize_rounding(self) -> None:
        """sprawdza zaokraglanie do najblizszego kroku."""
        # funkcja najpierw konwertuje do int(), wiec 7.5 -> 7 -> 5
        assert quantize_pct(7.0, step=5) == 5
        assert quantize_pct(7.5, step=5) == 5  # int(7.5) = 7, round(7/5=1.4) = 1 -> 5
        assert quantize_pct(8.0, step=5) == 10
        # inne wartosci
        assert quantize_pct(33.0, step=5) == 35
        assert (
            quantize_pct(32.4, step=5) == 30
        )  # int(32.4) = 32, round(32/5=6.4) = 6 -> 30
        assert quantize_pct(67.0, step=5) == 65
        assert (
            quantize_pct(67.6, step=5) == 65
        )  # int(67.6) = 67, round(67/5=13.4) = 13 -> 65

    def test_quantize_clamping(self) -> None:
        """sprawdza obcinanie wartosci poza zakresem 0-100."""
        # ponizej 0
        assert quantize_pct(-10.0, step=5) == 0
        assert quantize_pct(-0.1, step=5) == 0
        # powyzej 100
        assert quantize_pct(105.0, step=5) == 100
        assert quantize_pct(200.0, step=5) == 100

    def test_quantize_custom_step(self) -> None:
        """sprawdza kwantyzacje z niestandardowym krokiem."""
        assert quantize_pct(50.0, step=10) == 50
        assert quantize_pct(45.0, step=10) == 40  # int(45) = 45, round(45/10)*10 = 40
        assert quantize_pct(46.0, step=10) == 50  # int(46) = 46, round(46/10)*10 = 50
        assert quantize_pct(44.0, step=10) == 40
        assert quantize_pct(25.0, step=25) == 25
        assert quantize_pct(30.0, step=25) == 25  # int(30) = 30, round(30/25)*25 = 25
        assert quantize_pct(38.0, step=25) == 50  # int(38) = 38, round(38/25)*25 = 50

    def test_quantize_edge_cases(self) -> None:
        """sprawdza przypadki brzegowe."""
        # step=1 -> kazda wartosc calkowita (int() obcina czesc dziesietna)
        assert quantize_pct(42.7, step=1) == 42  # int(42.7) = 42
        assert quantize_pct(42.3, step=1) == 42
        assert quantize_pct(43.0, step=1) == 43
        # step=100 -> tylko 0 lub 100
        assert quantize_pct(49.9, step=100) == 0
        assert quantize_pct(50.0, step=100) == 0  # round(0.5) = 0 (banker's rounding)
        assert quantize_pct(51.0, step=100) == 100  # round(0.51) = 1 -> 100

    def test_quantize_robustness(self) -> None:
        """sprawdza odpornosc na nietypowe wejscia."""
        # wartosci float - int() obcina czesc dziesietna przed round()
        assert (
            quantize_pct(12.34, step=5) == 10
        )  # int(12.34) = 12, round(2.4) = 2 -> 10
        assert (
            quantize_pct(12.99, step=5) == 10
        )  # int(12.99) = 12, round(2.4) = 2 -> 10
        assert quantize_pct(13.0, step=5) == 15  # int(13) = 13, round(2.6) = 3 -> 15
        assert (
            quantize_pct(87.65, step=5) == 85
        )  # int(87.65) = 87, round(17.4) = 17 -> 85
        assert quantize_pct(88.0, step=5) == 90
        # bardzo male wartosci
        assert quantize_pct(0.001, step=5) == 0
        assert (
            quantize_pct(99.999, step=5) == 100
        )  # int(99.999) = 99, round(19.8) = 20 -> 100
