import csv
import os
import time

import cv2

from app.gesture_engine.detector.hand_tracker import HandTracker
from app.gesture_trainer.normalizer import HandNormalizer

DATA_FILE = "app/sign_language/data/dataset_pjm.csv"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _detect_gui_support() -> bool:
    if not hasattr(cv2, "imshow"):
        return False
    try:
        cv2.namedWindow("_test_gui_", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_test_gui_")
        return True
    except Exception:
        return False


def record_data(
    show_ui: bool | None = None,
    target_samples_per_letter: int = 100,
    camera_index: int = 0,
    letters: list[str] | None = None,
    max_empty_frames_per_letter: int = 300,
):
    # nagrywa gesty liter do CSV
    # parametry: show_ui wymusza tryb okienkowy; target_samples_per_letter liczba klatek na litere
    # camera_index indeks kamery; letters podzbior liter; max_empty_frames_per_letter limit pustych klatek
    tracker = HandTracker()
    normalizer = HandNormalizer()
    letters = letters or CLASSES

    # tworzy katalog wyjsciowy jesli nie istnieje
    out_dir = os.path.dirname(DATA_FILE)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # tworzy plik csv z naglowkiem jesli brak
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [
                f"p{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")
            ]
            writer.writerow(header)

    # wykrywa wsparcie gui
    gui_supported = _detect_gui_support()
    if show_ui is None:
        show_ui = gui_supported
    elif show_ui and not gui_supported:
        print("[recorder] wymuszono show_ui ale brak wsparcia highgui - headless")
        show_ui = False

    # otwiera kamere
    cap = cv2.VideoCapture(camera_index)
    if not cap or not cap.isOpened():
        print(f"[recorder] nie mozna otworzyc kamery (index={camera_index})")
        return

    print("=== START NAGRYWANIA ===")
    print(
        f"[recorder] tryb gui: {'ON' if show_ui else 'OFF (headless)'} | litery: {''.join(letters)}"
    )

    for letter in letters:
        print(f"\n>>> litera: {letter} (odliczanie 5s) <<<")
        for i in range(5, 0, -1):
            if show_ui:
                ret, frame = cap.read()
                if not ret:
                    print("[recorder] brak ramki w odliczaniu - przerwano")
                    cap.release()
                    return
                try:
                    cv2.putText(
                        frame,
                        f"Letter {letter} in {i}",
                        (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Recorder", frame)
                    if cv2.waitKey(1000) & 0xFF == 27:
                        print("[recorder] esc - wyjscie")
                        cap.release()
                        if show_ui:
                            cv2.destroyAllWindows()
                        return
                except Exception as e:
                    print(f"[recorder] highgui exception: {e} -> headless")
                    show_ui = False
                    break
            else:
                print(i)
                time.sleep(1)

        print(f"[recorder] nagrywanie litery {letter}...")
        samples_collected = 0
        empty_frames = 0
        while samples_collected < target_samples_per_letter:
            ret, frame = cap.read()
            if not ret:
                print("[recorder] brak ramki - koncze litere")
                break
            # konwertuje kolor bezpiecznie - w testach stub cv2 moze nie miec stalej COLOR_BGR2RGB
            if hasattr(cv2, "COLOR_BGR2RGB"):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame  # fallback bez konwersji
            results = tracker.process(rgb)

            # fallback: uzywa tracker.get_landmarks jezeli process nie zwraca wynikow
            if not results and hasattr(tracker, "get_landmarks"):
                lm_list = tracker.get_landmarks()
                if lm_list:

                    class _LM:
                        def __init__(self, x, y, z=0.0):
                            self.x = x
                            self.y = y
                            self.z = z

                    class _Hand:
                        def __init__(self, pts):
                            self.landmark = [
                                (
                                    _LM(*p)
                                    if isinstance(p, (list, tuple))
                                    else _LM(p.x, p.y, getattr(p, "z", 0.0))
                                )
                                for p in pts
                            ]

                    class _Res:
                        def __init__(self, hands):
                            self.multi_hand_landmarks = hands

                    results = _Res([_Hand(lm_list[0])])

            first_hand_landmarks = None
            if results and getattr(results, "multi_hand_landmarks", None):
                first_hand_landmarks = results.multi_hand_landmarks[0]
            if first_hand_landmarks:
                pts = [
                    (lm.x, lm.y, getattr(lm, "z", 0.0))
                    for lm in first_hand_landmarks.landmark
                ]
                norm = normalizer.normalize(pts)
                with open(DATA_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([letter] + list(norm))
                samples_collected += 1
                empty_frames = 0  # resetuje licznik pustych klatek
                if samples_collected % 10 == 0:
                    print(
                        f"[recorder] {letter}: {samples_collected}/{target_samples_per_letter}"
                    )
                if show_ui:
                    try:
                        cv2.putText(
                            frame,
                            f"{samples_collected}/{target_samples_per_letter}",
                            (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    except Exception:
                        pass
            else:
                empty_frames += 1
                # loguje okresowo brak detekcji
                if empty_frames % 50 == 0:
                    print(
                        f"[recorder] brak detekcji dloni dla '{letter}' przez {empty_frames} klatek"
                    )
                if empty_frames >= max_empty_frames_per_letter:
                    print(
                        f"[recorder] przekroczono limit pustych klatek ({max_empty_frames_per_letter}) dla '{letter}' - dalej"
                    )
                    break

            if show_ui:
                try:
                    cv2.imshow("Recorder", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        print("[recorder] esc - przerwano")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                except Exception as e:
                    print(f"[recorder] highgui w petli: {e} -> headless")
                    show_ui = False
            else:
                time.sleep(0.01)
        print(
            f"[recorder] litera {letter} zakonczona - zapisano {samples_collected} probek"
        )

    cap.release()
    if show_ui:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    print("[recorder] zakonczono nagrywanie")


if __name__ == "__main__":
    record_data()
