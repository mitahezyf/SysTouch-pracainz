import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 1, circle_radius = 1)

hands = mp_hands.Hands(max_num_hands = 4, min_detection_confidence = .9, min_tracking_confidence = .9)
face = mp_face.FaceMesh(max_num_faces = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

cam = cv2.VideoCapture(0)

while True:
    ret, image = cam.read()
    image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    face_results = face.process(image)
    hand_results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.namedWindow("wykrywacz", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wykrywacz", 800, 600)

    cv2.imshow("wykrywacz", image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()






