# ladowanie gestow z plikow json i normalizacja
# prosta walidacja i kontynuacja przy bledach

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .gesture_schema import normalize_gesture_def


def _is_json(path: str) -> bool:
    return path.lower().endswith(".json")


def load_gestures(paths: List[str]) -> List[Dict[str, Any]]:
    # wczytuje wszystkie pliki json z listy sciezek (pliki i katalogi rekurencyjnie)
    # zwraca liste gestow po normalizacji
    results: List[Dict[str, Any]] = []
    to_scan: List[str] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        if os.path.isfile(p) and _is_json(p):
            to_scan.append(p)
        elif os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in files:
                    if _is_json(f):
                        to_scan.append(os.path.join(root, f))
    for fp in to_scan:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                raw = [raw]
            if not isinstance(raw, list):
                continue
            for item in raw:
                try:
                    norm = normalize_gesture_def(item)
                    results.append(norm)
                except Exception:
                    # pomijamy bledny wpis
                    pass
        except Exception:
            # pomijamy bledny plik
            pass
    # sort po priorytecie malejaco
    results.sort(key=lambda g: int(g.get("priority", 0)), reverse=True)
    return results
