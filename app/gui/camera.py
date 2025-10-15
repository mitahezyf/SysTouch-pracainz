from __future__ import annotations

from app.gesture_engine.logger import logger


def discover_cameras(max_index: int = 10) -> list[int]:
    """Wykrywa dostepne kamery do podanego indeksu.

    - zwraca liste indeksow kamer, ktore dalo sie otworzyc
    - uzywa cv2 jesli jest dostepne; w przeciwnym razie zwraca pusta liste
    """
    try:  # pragma: no cover
        import cv2
    except Exception:
        return []

    available: list[int] = []
    for idx in range(max_index):
        cap = None
        try:
            if hasattr(cv2, "CAP_DSHOW"):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                available.append(idx)
        except Exception as e:
            logger.debug("discover_cameras: open error idx=%s: %s", idx, e)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    logger.debug("discover_cameras: release error idx=%s: %s", idx, e)
    return available
