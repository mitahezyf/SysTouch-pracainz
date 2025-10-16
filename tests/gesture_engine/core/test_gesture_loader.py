import json
from pathlib import Path

from app.gesture_engine.core.gesture_loader import load_gestures


def write_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def minimal_static(name: str, priority: int = 0):
    return {
        "type": "static",
        "name": name,
        "namespace": "test",
        "priority": priority,
        # fingers/predicates/action/stabilization opcjonalne -> domysly
    }


def test_load_gestures_from_dir_and_file(tmp_path: Path):
    d = tmp_path / "gestures"
    d.mkdir()
    # plik nie-json powinien byc pominiety
    (d / "readme.txt").write_text("ignore", encoding="utf-8")

    f1 = d / "a.json"
    f2 = d / "b.json"
    write_json(f1, minimal_static("A", priority=10))
    write_json(f2, [minimal_static("B", priority=20)])

    # dodatkowy pojedynczy plik poza katalogiem
    f3 = tmp_path / "c.json"
    write_json(f3, minimal_static("C", priority=5))

    out = load_gestures([str(d), str(f3)])
    # powinno posortowac malejaco po priority: B(20), A(10), C(5)
    names = [g["name"] for g in out]
    assert names == ["B", "A", "C"]


def test_load_gestures_skips_invalid_file_and_item(tmp_path: Path):
    d = tmp_path / "g"
    d.mkdir()
    bad_json = d / "bad.json"
    bad_json.write_text("{not-json}", encoding="utf-8")

    mixed = d / "mixed.json"
    data = [
        minimal_static("OK1"),
        {"type": "unknown", "name": "bad", "namespace": "t"},  # invalid item
        minimal_static("OK2"),
    ]
    write_json(mixed, data)

    out = load_gestures([str(d)])
    names = [g["name"] for g in out]
    assert names == ["OK1", "OK2"]
