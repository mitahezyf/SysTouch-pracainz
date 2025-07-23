import cv2
from threading import Thread

from config import CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT, TARGET_CAMERA_FPS

#klasa do obslugi przechwytywania obrazu z kamery
class ThreadedCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_CAMERA_FPS)

        self.ret, self.frame = self.cap.read()
        self.running = True

        #start nowego watku do pobierania klatek
        self.thread = Thread(target = self.update, daemon = True)
        self.thread.start()

    #aktualizacja ramki - frame
    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    #zwraca ostatni frame i status
    def read(self):
        return self.ret, self.frame

    #zatrzymuje watek i zwalnia kamere
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
