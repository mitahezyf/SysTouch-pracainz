# leniwe ladowanie symboli, aby unikac ciezkich importow (np. cv2) przy imporcie pakietu
__all__ = [
    "PerformanceTracker",
    "ThreadedCapture",
]


def __getattr__(name):  # PEP 562
    if name == "PerformanceTracker":
        from .performance import PerformanceTracker

        return PerformanceTracker
    if name == "ThreadedCapture":
        from .video_capture import ThreadedCapture

        return ThreadedCapture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
