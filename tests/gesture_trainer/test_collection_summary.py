from pathlib import Path

from app.gesture_trainer.collection_summary import (
    format_session_summary,
    summarize_clip_csv,
)


def test_summarize_clip_csv_counts(tmp_path: Path) -> None:
    p = tmp_path / "A_123.csv"
    p.write_text(
        "session_id,clip_id,label,frame_idx,timestamp_ms,handedness,has_hand,mirror_applied\n"
        "s,c,A,0,0,Right,1,0\n"
        "s,c,A,1,10,Right,0,0\n",
        encoding="utf-8",
    )

    clip = summarize_clip_csv(p)
    assert clip.clip_id == "A_123"
    assert clip.label == "A"
    assert clip.frames_total == 2
    assert clip.frames_hand_ok == 1
    assert clip.frames_hand_bad == 1


def test_format_session_summary_smoke(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess"
    (session_dir / "features").mkdir(parents=True)

    csv_path = session_dir / "features" / "B_1.csv"
    csv_path.write_text(
        "session_id,clip_id,label,frame_idx,timestamp_ms,handedness,has_hand,mirror_applied\n"
        "s,c,B,0,0,Right,1,0\n",
        encoding="utf-8",
    )

    # import lokalny zeby nie robic kolejnego helpera do budowy session
    from app.gesture_trainer.collection_summary import summarize_session_dir

    summary = summarize_session_dir(session_dir)
    report = format_session_summary(summary)
    assert "session:" in report
    assert "- B:" in report
