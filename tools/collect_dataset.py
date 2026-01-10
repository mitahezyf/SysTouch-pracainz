from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# zapewnia importy z katalogu glownego repo przy uruchomieniu jako skrypt z tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.gesture_engine.detector.hand_tracker import HandTracker
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.visualizer import Visualizer
from app.gesture_trainer.collection_summary import (
    format_session_summary,
    summarize_session_dir,
)
from app.gesture_trainer.dataset_collector import (
    ClipCSVWriter,
    CollectionConfig,
    FrameFeaturePipeline,
    build_row,
    build_session_id,
    ensure_dirs,
    safe_close,
    write_session_meta,
)


def _parse_labels(value: str) -> list[str]:
    labels = [x.strip() for x in value.split(",") if x.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("brak etykiet")
    return labels


def _countdown(seconds: float) -> None:
    end = time.time() + seconds
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            break
        logger.info("start za %.1fs", remaining)
        time.sleep(min(0.5, remaining))


@dataclass(frozen=True, slots=True)
class _OverlayState:
    label: str
    phase: str
    seconds_left: float
    handedness: str | None
    frame_idx: int
    ok: bool
    performer: str
    repetition: int
    repetitions_total: int
    handedness_required: str | None = None


def _draw_overlay(frame_bgr: np.ndarray, state: _OverlayState) -> np.ndarray:
    # rysuje prosty overlay na klatce (bez diakrytykow)
    out = frame_bgr
    h, w = out.shape[:2]

    bar_h = 80
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)

    color = (0, 200, 0) if state.ok else (0, 0, 255)

    line1 = (
        f"label: {state.label} ({state.repetition}/{state.repetitions_total})   who: {state.performer}"
        f"   phase: {state.phase}   left: {state.seconds_left:.1f}s"
    )
    line2 = f"hand: {state.handedness or '-'}   req: {state.handedness_required or '-'}   frame: {state.frame_idx}"

    cv2.putText(out, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(out, line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    hint = "keys: n/space/enter=next, r=repeat, q/esc=quit"
    cv2.putText(
        out,
        hint,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return out


def _key_is_next(key: int) -> bool:
    # spacja lub enter lub n
    return key in (ord("n"), 13, 32)


def _normalize_handedness(label: str | None) -> str | None:
    if label is None:
        return None
    val = label.strip().lower()
    if val.startswith("r"):
        return "Right"
    if val.startswith("l"):
        return "Left"
    return None


def _parse_bool(value: str) -> bool:
    val = value.strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"niepoprawna wartosc bool: {value!r}")


# pomocniczo odwraca etykiete handedness jesli obraz byl flipowany w poziomie


def _flip_handedness_label(label: str | None) -> str | None:
    norm = _normalize_handedness(label)
    if norm is None:
        return None
    if norm == "Left":
        return "Right"
    return "Left"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zbieranie klipow i cech z kamery (PJM) - zapis MP4 + CSV",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--capture-width",
        type=int,
        default=1920,
        help="docelowa szerokosc przechwytywania (kamera moze zignorowac)",
    )
    parser.add_argument(
        "--capture-height",
        type=int,
        default=1080,
        help="docelowa wysokosc przechwytywania (kamera moze zignorowac)",
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=30.0,
        help="docelowy fps przechwytywania (kamera moze zignorowac)",
    )
    parser.add_argument("--labels", type=_parse_labels, required=True, help="np. A,B,C")
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--countdown", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("data/collected"))
    parser.add_argument(
        "--mirror-left",
        type=_parse_bool,
        default=True,
        help="czy mirrorowac lewa dlon (true/false); ustaw false jesli zawsze pokazujesz prawa",
    )
    parser.add_argument("--no-landmarks", action="store_true")
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="pokazuje podglad i pozwala sterowac next/repeat klawiszami",
    )
    parser.add_argument(
        "--require-handedness",
        type=str,
        default="",
        help="opcjonalnie: Right lub Left (klatki z inna reka beda oznaczane jako has_hand=0)",
    )
    parser.add_argument(
        "--performers",
        type=str,
        default="Krzysiek,Werka",
        help="lista po przecinku, np. Krzysiek,Werka; overlay pokazuje kto ma wykonywac gest",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="ile razy powtorzyc kazda litere (na performerow rotacyjnie)",
    )
    parser.add_argument(
        "--show-landmarks",
        action="store_true",
        help="rysuje landmarki mediapipe na podgladzie (tylko --interactive)",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.6,
        help="skala okna podgladu, np. 1.0/1.5/2.0 (tylko --interactive)",
    )
    parser.add_argument(
        "--ui-max-width",
        type=int,
        default=1600,
        help="maks. szerokosc podgladu (skalowanie dla duzych rozdzielczosci)",
    )
    parser.add_argument(
        "--ui-max-height",
        type=int,
        default=900,
        help="maks. wysokosc podgladu (skalowanie dla duzych rozdzielczosci)",
    )
    parser.add_argument(
        "--handedness-mode",
        choices=["mediapipe", "flip"],
        default="mediapipe",
        help=(
            "jak interpretowac reke: mediapipe=ufa etykiecie mediapipe; "
            "flip=flipuje klatke przed mediapipe i odwraca etykiete Left/Right; "
            "przydatne gdy zawsze pokazujesz prawa reke, a mediapipe widzi ja jako Left"
        ),
    )

    args = parser.parse_args(argv)

    if args.capture_width <= 0 or args.capture_height <= 0:
        logger.error("--capture-width/--capture-height musza byc > 0")
        return 2
    if float(args.capture_fps) <= 0:
        logger.error("--capture-fps musi byc > 0")
        return 2
    if args.ui_max_width <= 0 or args.ui_max_height <= 0:
        logger.error("--ui-max-width/--ui-max-height musza byc > 0")
        return 2

    if args.no_landmarks and args.no_features:
        logger.error("musisz zapisac landmarks albo features")
        return 2

    required_hand = _normalize_handedness(args.require_handedness)
    if args.require_handedness and required_hand is None:
        logger.error("niepoprawne --require-handedness: %s", args.require_handedness)
        return 2

    # w trybie flip: domyslnie akceptuj obie rece, chyba ze user wymusil konkretna
    # if args.handedness_mode == "flip" and not args.require_handedness:
    #     required_hand = "Right"

    performers = [p.strip() for p in str(args.performers).split(",") if p.strip()]
    if not performers:
        logger.error("brak performerow")
        return 2
    repeats = int(args.repeats)
    if repeats <= 0:
        logger.error("--repeats musi byc > 0")
        return 2

    session_id = build_session_id()
    cfg = CollectionConfig(
        output_dir=args.output,
        session_id=session_id,
        mirror_left=bool(args.mirror_left),
        include_landmarks=not bool(args.no_landmarks),
        include_features=not bool(args.no_features),
    )

    session_dir, clips_dir, features_dir = ensure_dirs(cfg)
    write_session_meta(
        cfg,
        labels=list(args.labels),
        extra={
            "camera": int(args.camera),
            "clip_seconds": float(args.clip_seconds),
            "countdown": float(args.countdown),
            "fps": float(args.fps),
            "require_handedness": required_hand or "",
            "handedness_mode": str(args.handedness_mode),
            "interactive": bool(args.interactive),
            "performers": performers,
            "repeats": repeats,
        },
    )

    logger.info("session_id=%s", session_id)
    logger.info("output=%s", session_dir)

    tracker = HandTracker(max_num_hands=1)
    pipeline = FrameFeaturePipeline(mirror_left=cfg.mirror_left)

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        logger.error("nie mozna otworzyc kamery index=%s", args.camera)
        return 3

    # ustawia parametry przechwytywania (kamera moze je zignorowac)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.capture_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.capture_height))
    cap.set(cv2.CAP_PROP_FPS, float(args.capture_fps))

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps)

        if width != int(args.capture_width) or height != int(args.capture_height):
            logger.warning(
                "kamera nie ustawila zadanej rozdzielczosci %dx%d, uzywam %dx%d",
                int(args.capture_width),
                int(args.capture_height),
                width,
                height,
            )
        if fps < float(args.capture_fps) - 0.1:
            logger.warning(
                "kamera nie ustawila zadanego fps=%.1f, uzywam fps=%.1f",
                float(args.capture_fps),
                fps,
            )

        # nadpisuje fps dla video writer i sleep - trzymamy sie realnego fps kamery
        args.fps = fps

        # inicjalizuje visualizer pod rozmiar strumienia z kamery; display bedzie skalowany przez opencv
        visualizer = Visualizer(
            capture_size=(width, height), display_size=(width, height)
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]

        window_name = "dataset collector" if args.interactive else ""
        if args.interactive:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # skaluje podglad do rozmiaru ekranu, zeby overlay sie miescil
            scale_ui = min(
                float(args.ui_max_width) / float(width),
                float(args.ui_max_height) / float(height),
                1.0,
            )
            ui_w = max(640, int(width * scale_ui))
            ui_h = max(480, int(height * scale_ui))

            # uwzglednia dodatkowa skale z --window-scale
            scale_win = float(args.window_scale)
            if scale_win <= 0:
                logger.error("--window-scale musi byc > 0")
                return 2
            ui_w = int(ui_w * scale_win)
            ui_h = int(ui_h * scale_win)

            try:
                cv2.resizeWindow(window_name, ui_w, ui_h)
            except Exception:
                logger.debug("nie mozna ustawic rozmiaru okna")

            logger.info(
                "capture=%dx%d@%.1ffps ui=%dx%d", width, height, fps, ui_w, ui_h
            )

        label_idx = 0
        rep_global_idx = 0
        while label_idx < len(args.labels):
            label = args.labels[label_idx]

            rep_in_label = 1
            while rep_in_label <= repeats:
                performer = performers[rep_global_idx % len(performers)]

                if not args.interactive:
                    logger.info(
                        "przygotuj gest: %s (%d/%d), kto: %s",
                        label,
                        rep_in_label,
                        repeats,
                        performer,
                    )
                    _countdown(float(args.countdown))

                else:
                    logger.info(
                        "tryb interaktywny: gest %s (%d/%d), kto: %s - wcisnij next",
                        label,
                        rep_in_label,
                        repeats,
                        performer,
                    )
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            continue

                        state = _OverlayState(
                            label=label,
                            phase="ready",
                            seconds_left=0.0,
                            handedness=None,
                            frame_idx=0,
                            ok=True,
                            performer=performer,
                            repetition=rep_in_label,
                            repetitions_total=repeats,
                        )
                        vis = frame
                        if args.show_landmarks:
                            rgb_vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                            res_vis = tracker.process(rgb_vis)
                            if res_vis and res_vis.multi_hand_landmarks:
                                visualizer.draw_landmarks(
                                    vis, res_vis.multi_hand_landmarks[0]
                                )
                        vis = _draw_overlay(vis, state)
                        cv2.imshow(window_name, vis)
                        key = cv2.waitKey(10) & 0xFF
                        if key in (ord("q"), 27):
                            return 0
                        if key == ord("r"):
                            continue
                        if _key_is_next(key):
                            break

                    # countdown 2s z podgladem
                    cd_end = time.time() + float(args.countdown)
                    while True:
                        remaining = cd_end - time.time()
                        if remaining <= 0:
                            break
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        state = _OverlayState(
                            label=label,
                            phase="countdown",
                            seconds_left=max(0.0, remaining),
                            handedness=None,
                            frame_idx=0,
                            ok=True,
                            performer=performer,
                            repetition=rep_in_label,
                            repetitions_total=repeats,
                        )
                        vis = frame
                        if args.show_landmarks:
                            rgb_vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                            res_vis = tracker.process(rgb_vis)
                            if res_vis and res_vis.multi_hand_landmarks:
                                visualizer.draw_landmarks(
                                    vis, res_vis.multi_hand_landmarks[0]
                                )
                        vis = _draw_overlay(vis, state)
                        cv2.imshow(window_name, vis)
                        key = cv2.waitKey(10) & 0xFF
                        if key in (ord("q"), 27):
                            return 0

                clip_id = f"{label}_{performer}_{rep_in_label}_{int(time.time())}"
                video_path = clips_dir / f"{clip_id}.mp4"
                csv_path = features_dir / f"{clip_id}.csv"

                writer = None
                video = None
                frames_total = 0
                frames_hand_ok = 0
                frames_hand_bad = 0
                frames_hand_unknown = 0
                try:
                    writer = ClipCSVWriter(
                        csv_path,
                        include_landmarks=cfg.include_landmarks,
                        include_features=cfg.include_features,
                    )
                    video = cv2.VideoWriter(
                        str(video_path), fourcc, fps, (width, height)
                    )

                    start = time.time()
                    frame_idx = 0
                    while True:
                        now = time.time()
                        elapsed = now - start
                        if elapsed >= float(args.clip_seconds):
                            break

                        ret, frame = cap.read()
                        if not ret:
                            continue

                        video.write(frame)

                        # pomocniczo odwraca etykiete handedness jesli obraz byl flipowany w poziomie
                        if args.handedness_mode == "flip":
                            frame_for_mp = cv2.flip(frame, 1)
                        else:
                            frame_for_mp = frame

                        rgb = cv2.cvtColor(frame_for_mp, cv2.COLOR_BGR2RGB)
                        results = tracker.process(rgb)

                        handedness = None
                        landmarks21 = None
                        features63 = None
                        mirror_applied = False
                        has_hand = False
                        ok = True

                        if results and results.multi_hand_landmarks:
                            hand_lm = results.multi_hand_landmarks[0]

                            # pobierz surowe handedness z mediapipe
                            handedness_raw = None
                            if results.multi_handedness:
                                handedness_raw = (
                                    results.multi_handedness[0].classification[0].label
                                )

                            # w trybie flip: obraz byl juz flipowany, wiec MediaPipe zwraca prawidlowa etykiete
                            # w trybie mediapipe: MediaPipe widzi lustrzany obraz z kamery

                            # dla zapisu w CSV: w trybie mediapipe nie flipujemy (zapisujemy co widzi MP)
                            # w trybie flip: rowniez nie flipujemy (MediaPipe juz dostal poprawny obraz)
                            handedness = handedness_raw

                            # sprawdz required_hand
                            hand_matches_requirement = True
                            if required_hand is not None and handedness_raw is not None:
                                # w obu trybach: porownujemy bezposrednio
                                # w flip: MP juz dostal flipowany obraz, wiec jego etykieta jest prawidlowa
                                # w mediapipe: MP widzi lustro, wiec jego etykieta jest odwrocona
                                if args.handedness_mode == "flip":
                                    # flip: MP zwraca prawidlowa etykiete (Right dla prawej reki)
                                    hand_matches_requirement = (
                                        handedness_raw == required_hand
                                    )
                                else:
                                    # mediapipe: MP zwraca odwrocona etykiete (Left dla prawej reki)
                                    # wiec musimy porownac z odwroconym required_hand
                                    expected = _flip_handedness_label(required_hand)
                                    hand_matches_requirement = (
                                        handedness_raw == expected
                                    )

                            # jesli mediapipe nie zwraca handedness, nie blokuje klatki
                            if required_hand is not None and handedness_raw is None:
                                frames_hand_unknown += 1
                                has_hand = True
                                frames_hand_ok += 1
                            # jesli wymuszamy handedness, a mediapipe zwraca inna reke, to oznacza brak dloni
                            elif (
                                required_hand is not None
                                and not hand_matches_requirement
                            ):
                                ok = False
                                has_hand = False
                                frames_hand_bad += 1
                            else:
                                has_hand = True
                                frames_hand_ok += 1

                            # ekstrahuj landmarks i features ZAWSZE gdy mediapipe wykryje reke
                            if has_hand:
                                landmarks21 = np.array(
                                    [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                                    dtype=np.float32,
                                )

                                if cfg.include_features:
                                    features63, mirror_applied = pipeline.compute(
                                        landmarks21, handedness
                                    )
                        else:
                            ok = False

                        frames_total += 1

                        row = build_row(
                            session_id=cfg.session_id,
                            clip_id=clip_id,
                            label=label,
                            frame_idx=frame_idx,
                            timestamp_ms=int(elapsed * 1000),
                            handedness=handedness,
                            has_hand=has_hand,
                            mirror_applied=mirror_applied,
                            landmarks21=landmarks21,
                            features63=features63,
                            include_landmarks=cfg.include_landmarks,
                            include_features=cfg.include_features,
                        )
                        writer.write(row)

                        if args.interactive:
                            remaining = max(0.0, float(args.clip_seconds) - elapsed)
                            state = _OverlayState(
                                label=label,
                                phase="rec",
                                seconds_left=remaining,
                                handedness=handedness,
                                frame_idx=frame_idx,
                                ok=ok,
                                performer=performer,
                                repetition=rep_in_label,
                                repetitions_total=repeats,
                                handedness_required=required_hand,
                            )
                            vis = frame
                            if (
                                args.show_landmarks
                                and results
                                and results.multi_hand_landmarks
                            ):
                                # gdy uzywamy flip do mediapipe, landmarki sa w przestrzeni sfipowanej
                                # wiec rysujemy je na tej samej klatce co detekcja
                                if args.handedness_mode == "flip":
                                    vis = frame_for_mp
                                visualizer.draw_landmarks(
                                    vis, results.multi_hand_landmarks[0]
                                )
                            vis = _draw_overlay(vis, state)
                            cv2.imshow(window_name, vis)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (ord("q"), 27):
                                return 0
                            if key == ord("r"):
                                raise RuntimeError("repeat")

                        frame_idx += 1
                        time.sleep(max(0.0, (1.0 / fps) - (time.time() - now)))

                    ratio = (frames_hand_ok / frames_total) if frames_total else 0.0
                    if frames_hand_unknown:
                        logger.info(
                            "handedness unknown w %d/%d klatek (%.0f%%) dla %s",
                            frames_hand_unknown,
                            frames_total,
                            (
                                (frames_hand_unknown / frames_total) * 100.0
                                if frames_total
                                else 0.0
                            ),
                            clip_id,
                        )
                    if ratio < 0.7:
                        logger.warning(
                            "niska jakosc klipu %s: hand_ok=%.0f%% (ok=%d bad=%d total=%d)",
                            clip_id,
                            ratio * 100.0,
                            frames_hand_ok,
                            frames_hand_bad,
                            frames_total,
                        )

                    logger.info(
                        "zapisano klip %s -> %s / %s (hand_ok=%.0f%%)",
                        clip_id,
                        video_path.name,
                        csv_path.name,
                        ratio * 100.0,
                    )

                    rep_in_label += 1
                    rep_global_idx += 1
                except RuntimeError as e:
                    if str(e) == "repeat":
                        logger.info(
                            "powtarzam: %s (%d/%d), kto: %s",
                            label,
                            rep_in_label,
                            repeats,
                            performer,
                        )
                        # kasuje pliki biezacego klipu i powtarza te sama litere
                        safe_close(writer)
                        writer = None
                        if video is not None:
                            video.release()
                            video = None
                        try:
                            if video_path.exists():
                                video_path.unlink()
                        except Exception:
                            logger.exception("nie mozna usunac %s", video_path)
                        try:
                            if csv_path.exists():
                                csv_path.unlink()
                        except Exception:
                            logger.exception("nie mozna usunac %s", csv_path)
                        continue
                    raise
                finally:
                    safe_close(writer)
                    if video is not None:
                        video.release()

            label_idx += 1

    finally:
        cap.release()
        if args.interactive:
            try:
                cv2.destroyAllWindows()
            except Exception:
                logger.debug("nie mozna zamknac okien opencv")

    try:
        summary = summarize_session_dir(session_dir)
        logger.info("\n%s", format_session_summary(summary))
    except Exception:
        logger.exception("nie mozna wygenerowac podsumowania sesji")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
