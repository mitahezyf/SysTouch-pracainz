from utils.geometry import distance
from config import CLICK_TRESHOLD

# klasa do reprezentacji gestu
class Gesture:
    def __init__(self, name, confidence, landmarks):
        self.name = name
        self.confidence = confidence
        self.landmarks = landmarks


# funkcja do rozpoznawania pozycji punktow do sterowania kursorem
def is_mouse_control_pose(landmarks):
    index = landmarks.landmark[8]
    middle = landmarks.landmark[12]
    ring = landmarks.landmark[16]
    pinky = landmarks.landmark[20]

    # palec wskazujacy i srodkowy – wyprostowane
    index_extended = index.y < landmarks.landmark[5].y
    middle_extended = middle.y < landmarks.landmark[9].y

    # palec serdeczny i maly – zgiete
    ring_bent = ring.y > landmarks.landmark[13].y
    pinky_bent = pinky.y > landmarks.landmark[17].y

    return index_extended and middle_extended and ring_bent and pinky_bent


#glowna funkcja do rozpoznawania gestu z punktow dloni
def detect_gesture(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]

    dist = distance(thumb_tip, index_tip)
    click_confidence = max(0.0, 1.0 - (dist / CLICK_TRESHOLD))

    # -------------------------------------------------------------------------
    # PRIORYTET GESTU 1: STEROWANIE MYSZKA
    # -------------------------------------------------------------------------
    if is_mouse_control_pose(landmarks):
        return Gesture("move_mouse", 1.0, landmarks)

    # -------------------------------------------------------------------------
    # PRIORYTET GESTU 2: CLICK
    # -------------------------------------------------------------------------
    if click_confidence > 0.9:
        return Gesture("click", click_confidence, landmarks)

    return None




