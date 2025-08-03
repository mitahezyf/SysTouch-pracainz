import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
from app.utils.performance import PerformanceTracker

# test początkowego stanu - zanim cokolwiek zostanie zaktualizowane
def test_initial_state():
    tracker = PerformanceTracker()
    assert tracker.fps == 0
    assert tracker.delay_ms == 0
    assert tracker.frametime_ms == 0  # alias dla delay_ms

# po 50ms update() powinien ustawic delay w okolicach 50ms i fps ~20
def test_update_increases_delay_and_sets_fps():
    tracker = PerformanceTracker()

    time.sleep(0.05)  # 50ms = ~20 FPS
    tracker.update()

    assert 45 <= tracker.delay_ms <= 65  # sprawdzamy tolerancję
    assert 15 <= tracker.fps <= 25       # sprawdzamy czy fps tez się ustawiło
    assert tracker.frametime_ms == tracker.delay_ms  # property działa poprawnie

# drugi update po krotszym czasie daje mniejsze delay
def test_multiple_updates_reduce_delta():
    tracker = PerformanceTracker()
    time.sleep(0.1)  # pierwszy sleep - duzy delay
    tracker.update()
    delay_1 = tracker.delay_ms

    time.sleep(0.01)  # drugi sleep - mniejszy delay
    tracker.update()
    delay_2 = tracker.delay_ms

    assert delay_2 < delay_1  # sprawdzamy, czy delay się zmniejszył
    assert tracker.fps > 0    # fps powinno byc sensowne
