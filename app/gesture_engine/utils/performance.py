import time


class PerformanceTracker:
    def __init__(self):
        # przechowuje czas ostatniej klatki
        self.prev_time = time.time()
        # przechowuje opoznienie miedzy klatkami [ms]
        self.delay_ms = 0
        self.fps = 0

    def update(self):
        # oblicza roznice czasu miedzy klatkami
        current_time = time.time()
        delta = current_time - self.prev_time

        # przelicza na milisekundy
        self.delay_ms = int(delta * 1000)

        # wylicza fps
        self.fps = int(1 / (delta + 1e-9))
        self.prev_time = current_time

    @property
    def frametime_ms(self):
        return self.delay_ms
