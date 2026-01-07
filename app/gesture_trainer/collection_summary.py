from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.gesture_engine.logger import logger


@dataclass(frozen=True, slots=True)
class ClipSummary:
    clip_id: str
    label: str
    frames_total: int
    frames_hand_ok: int
    frames_hand_bad: int

    @property
    def hand_ok_ratio(self) -> float:
        if self.frames_total <= 0:
            return 0.0
        return self.frames_hand_ok / self.frames_total


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_dir: Path
    clips: list[ClipSummary]

    def by_label(self) -> dict[str, list[ClipSummary]]:
        out: dict[str, list[ClipSummary]] = {}
        for c in self.clips:
            out.setdefault(c.label, []).append(c)
        return out


def summarize_clip_csv(path: Path) -> ClipSummary:
    # czyta CSV i liczy jakosc (has_hand=1) oraz ile klatek ma zly handedness (brak danych)
    frames_total = 0
    frames_hand_ok = 0
    frames_hand_bad = 0
    clip_id = path.stem
    label = ""

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            frames_total += 1
            if not label:
                label = row.get("label", "")

            has_hand = (row.get("has_hand") or "0").strip()
            if has_hand == "1":
                frames_hand_ok += 1
            else:
                frames_hand_bad += 1

    if not label:
        label = "?"

    return ClipSummary(
        clip_id=clip_id,
        label=label,
        frames_total=frames_total,
        frames_hand_ok=frames_hand_ok,
        frames_hand_bad=frames_hand_bad,
    )


def summarize_session_dir(session_dir: Path) -> SessionSummary:
    features_dir = session_dir / "features"
    if not features_dir.exists():
        raise FileNotFoundError(f"brak katalogu: {features_dir}")

    clips: list[ClipSummary] = []
    for p in sorted(features_dir.glob("*.csv")):
        try:
            clips.append(summarize_clip_csv(p))
        except Exception:
            logger.exception("nie mozna podsumowac %s", p)

    return SessionSummary(session_dir=session_dir, clips=clips)


def format_session_summary(summary: SessionSummary) -> str:
    # buduje krotki, czytelny raport tekstowy
    lines: list[str] = []
    lines.append(f"session: {summary.session_dir}")
    lines.append(f"clips: {len(summary.clips)}")

    by_label = summary.by_label()
    if by_label:
        lines.append("per label:")

    for label in sorted(by_label.keys()):
        items = by_label[label]
        total = sum(c.frames_total for c in items)
        ok = sum(c.frames_hand_ok for c in items)
        ratio = (ok / total) if total else 0.0
        lines.append(
            f"- {label}: clips={len(items)} frames_ok={ok}/{total} ({ratio:.0%})"
        )

    # top najgorsze klipy
    worst = sorted(summary.clips, key=lambda c: c.hand_ok_ratio)[:5]
    if worst:
        lines.append("worst clips:")
        for c in worst:
            lines.append(
                f"- {c.clip_id}: {c.label} ok={c.frames_hand_ok}/{c.frames_total} ({c.hand_ok_ratio:.0%})"
            )

    return "\n".join(lines)
