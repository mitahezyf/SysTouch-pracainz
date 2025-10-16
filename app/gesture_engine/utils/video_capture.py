from threading import Thread
from typing import Any, Union, cast

from app.gesture_engine.config import (
    CAMERA_INDEX,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    TARGET_CAMERA_FPS,
)
from app.gesture_engine.logger import logger

cv2: Any  # typ ogolny dla cv2 (modul lub stub)

# Bezpieczny import cv2 - w CI lub srodowiskach bez OpenCV pozwalamy na import
# modulu poprzez stub, aby testy mogly patchowac cv2.VideoCapture.
try:  # pragma: no cover
    import cv2 as _cv2

    cv2 = cast(Any, _cv2)
except Exception:  # pragma: no cover

    class _CV2Stub:  # minimalny stub potrzebny w testach
        # wartosci zgodne z OpenCV
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5

        class VideoCapture:
            def __init__(self, *_, **__):
                raise ImportError(
                    "cv2 (OpenCV) nie jest zainstalowane - uzyto stubu. Zainstaluj opencv-python."
                )

        def __getattr__(self, name):  # dla ewentualnych innych atrybutow
            raise ImportError(
                f"cv2 atrybut '{name}' nie jest dostepny w trybie stub. Zainstaluj opencv-python."
            )

    cv2 = cast(Any, _CV2Stub())


# klasa do obslugi przechwytywania obrazu z kamery
class ThreadedCapture:
    def __init__(self, camera_index: Union[int, str, None] = None):
        """
        Inicjalizuje watkowe przechwytywanie obrazu.

        :param camera_index: indeks kamery lub nazwa DirectShow w formacie "video=<NAZWA>".
        Jesli None, uzywa wartosci z configu (CAMERA_INDEX).
        """
        # walidacja atrybutow cv2 po imporcie (wychwytuje niepelne/bledne instalacje)
        if not hasattr(cv2, "VideoCapture"):
            raise RuntimeError(
                "cv2.VideoCapture niedostepne. Sprawdz instalacje OpenCV (opencv-python), usun konflikty (np. kilka wariantow) i srodowisko PATH/DLL."
            )

        # ustal zrodlo kamery (int lub string)
        self._camera_source: Union[int, str]
        if camera_index is None:
            self._camera_source = int(CAMERA_INDEX)
        else:
            self._camera_source = camera_index

        cap = None

        # jesli podano nazwe urzadzenia DirectShow: probuje tylko z CAP_DSHOW
        if isinstance(self._camera_source, str):
            try:
                backend = getattr(cv2, "CAP_DSHOW", 0)
                cap = (
                    cv2.VideoCapture(self._camera_source, backend)
                    if backend != 0
                    else cv2.VideoCapture(self._camera_source)
                )
                if cap is not None and cap.isOpened():
                    logger.info(
                        f"Kamera otwarta backend={backend}, source={self._camera_source}"
                    )
                else:
                    if cap is not None:
                        cap.release()
                        cap = None
            except Exception as e:
                logger.debug(
                    f"VideoCapture init fail for source '{self._camera_source}': {e}"
                )
                cap = None
        else:
            # proba z backendami Windows (DirectShow/MSMF), potem domyslny
            backends = []
            if hasattr(cv2, "CAP_DSHOW"):
                backends.append(cv2.CAP_DSHOW)
            if hasattr(cv2, "CAP_MSMF"):
                backends.append(cv2.CAP_MSMF)
            backends.append(0)  # domyslny

            for be in backends:
                try:
                    cap = (
                        cv2.VideoCapture(int(self._camera_source), be)
                        if be != 0
                        else cv2.VideoCapture(int(self._camera_source))
                    )
                    if cap is not None and cap.isOpened():
                        logger.info(
                            f"Kamera otwarta backend={be}, index={self._camera_source}"
                        )
                        break
                    else:
                        if cap is not None:
                            cap.release()
                        cap = None
                except Exception as e:
                    logger.debug(f"VideoCapture init fail backend={be}: {e}")
                    cap = None

        if cap is None:
            raise RuntimeError(
                "Nie udalo sie otworzyc kamery. Sprawdz zrodlo kamery (index lub 'video=<NAZWA>'), uprawnienia i instalacje sterownikow/DirectShow."
            )

        self.cap = cap
        # ustawienia wymiary/fps (ignorowane jesli backend nie wspiera)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_CAMERA_FPS)
        except Exception as e:
            logger.debug(f"Ustawienia kamery pominiete: {e}")

        self.ret, self.frame = self.cap.read()

        if not self.ret or self.frame is None:
            logger.warning("Nie udalo sie pobrac pierwszej klatki z kamery.")

        logger.info(
            f"Uruchomiono kamere (source={self._camera_source}, res={CAPTURE_WIDTH}x{CAPTURE_HEIGHT}, fps={TARGET_CAMERA_FPS})"
        )

        self.running = True

        # start nowego watku do pobierania klatek
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    # aktualizacja ramki - frame
    def update(self):
        while self.running:
            try:
                self.ret, self.frame = self.cap.read()
            except Exception as e:
                logger.debug(f"Blad odczytu klatki: {e}")
                self.ret, self.frame = False, None

    # zwraca ostatni frame i status
    def read(self):
        return self.ret, self.frame

    # zatrzymuje watek i zwalnia kamere
    def stop(self):
        self.running = False
        self.thread.join()
        try:
            self.cap.release()
        except Exception as e:
            logger.debug(f"Wyjatek przy zwalnianiu kamery: {e}")
        logger.info("Kamera zostala zwolniona i watek zakonczony.")
