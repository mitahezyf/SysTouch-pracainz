# kamera i przechwytywanie
CAMERA_INDEX = 0

# rozdzielczosc przechwytywania
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080

# rozdzielczosc wyswietlania
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480


# opcjonalne ustawienia ramki (przyklad)
# przyklad: frame_width = 1280
# przyklad: frame_height = 720

TARGET_CAMERA_FPS = 60


# debug i diagnostyka
DEBUG_MODE = True
SHOW_FPS = True
SHOW_DELAY = True


# progi gestow
CLICK_THRESHOLD = 0.5
HOLD_THRESHOLD = 2.0
SCROLL_THRESHOLD = 5
SCROLL_SENSITIVITY = 30
MOUSE_MOVING_SMOOTHING = 0.7
FLEX_THRESHOLD = 0.001
VOLUME_THRESHOLD = 0.5

# procentowy prog akceptacji gestu
GESTURE_CONFIDENCE_THRESHOLD = 0.9


# ustawienia scrolla
MAX_SCROLL_SPEED = 10
SCROLL_BASE_INTERVAL = 0.3


# renderowanie landmarkow
DRAW_EVERY_N_FRAMES = 3

LANDMARK_CIRCLE_RADIUS = 2
LANDMARK_LINE_THICKNESS = 1
LANDMARK_COLOR = (0, 255, 0)  # zielony
CONNECTION_COLOR = (0, 0, 255)  # niebieski

LABEL_FONT_SCALE = 0.5
LABEL_THICKNESS = 1
LABEL_COLOR = (0, 255, 0)  # zielony


VOLUME_CONFIRMATION_DELAY = 2

# rozszerzone ustawienia dla gestu klikniecia (pod przyszla stabilizacje/fsm)
CONFIDENCE_MIN = 0.98  # minimalna akceptowalna pewnosc pojedynczej klatki
STABLE_FRAMES_PRESS = (
    3  # ile kolejnych klatek musi spelniac warunek, zeby wejsc w pressed
)
STABLE_FRAMES_RELEASE = 2  # ile kolejnych klatek potwierdza zwolnienie
RELEASE_COOLDOWN = 0.2  # czas blokady po zwolnieniu [s]
CLICK_DISTANCE_CLOSE = 0.015  # dystans kciuk-wskazujacy uznawany za "zamkniety"
CLICK_DISTANCE_OPEN = 0.030  # dystans uznawany za "otwarty" (histereza)
SMOOTHING_ALPHA = 0.0  # wspolczynnik wygladzania (0 = wyl.)
PRIMARY_HAND = "Left"  # sterujaca reka: "Left" | "Right" | "Auto"
LOG_LEVEL = "INFO"  # domyslny poziom logowania
