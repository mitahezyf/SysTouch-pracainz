import time


class PerformanceTracker:
    def __init__(self):
        # czas ostatniej klatki
        self.prev_time = time.time()
        # opoznienie miedzy klatkami
        self.delay_ms = 0
        self.fps = 0

    def update(self):
        # roznica czasu miedzy klatkami
        current_time = time.time()
        delta = current_time - self.prev_time

        # przeliczenie na ms
        self.delay_ms = int(delta * 1000)

        # liczenie fps
        self.fps = int(1 / (delta + 1e-9))
        self.prev_time = current_time

    @property
    def frametime_ms(self):
        return self.delay_ms
