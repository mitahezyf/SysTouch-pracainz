from __future__ import annotations

import sys
from typing import List, Tuple, Union

from app.gesture_engine.logger import logger


def discover_cameras(max_index: int = 10) -> list[int]:
    # wykrywa dostepne kamery do podanego indeksu
    # zwraca liste indeksow kamer ktore udalo sie otworzyc
    # uzywa cv2 jesli dostepne inaczej zwraca pusta liste
    # na windows preferuje backend directShow aby uniknac problemow msmf
    try:  # pragma: no cover
        import cv2
    except Exception:
        return []

    available: list[int] = []
    for idx in range(max_index):
        cap = None
        try:
            # na Windows wymusza DirectShow przy skanowaniu indeksow
            if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                available.append(idx)
        except Exception as e:
            logger.debug("discover_cameras: open error idx=%s: %s", idx, e)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    logger.debug("discover_cameras: release error idx=%s: %s", idx, e)
    return available


def _win_list_camera_names() -> list[str]:
    # pobiera nazwy kamer z wmi (windows) best-effort moze zwrocic pusta liste
    # korzysta z pywin32 win32com.client bez dodatkowych zaleznosci
    try:
        if sys.platform != "win32":
            return []
        import win32com.client

        wmi = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        svc = wmi.ConnectServer(".", "root\\cimv2")
        queries = [
            "SELECT Name FROM Win32_PnPEntity WHERE PNPClass = 'Camera'",
            "SELECT Name FROM Win32_PnPEntity WHERE PNPClass = 'Image'",
            "SELECT Name FROM Win32_PnPEntity WHERE Name LIKE '%Camera%'",
            "SELECT Name FROM Win32_PnPEntity WHERE Name LIKE '%Webcam%'",
            "SELECT Name FROM Win32_PnPEntity WHERE Name LIKE '%OBS%'",
        ]
        names: list[str] = []
        seen: set[str] = set()
        for q in queries:
            try:
                items = svc.ExecQuery(q)
                for it in items:
                    nm = str(getattr(it, "Name", "")).strip()
                    if nm and nm not in seen:
                        names.append(nm)
                        seen.add(nm)
            except Exception as e:
                logger.debug("_win_list_camera_names: query error %s: %s", q, e)
                continue
        return names
    except Exception as e:
        logger.debug("_win_list_camera_names: WMI error: %s", e)
        return []


def discover_camera_names(max_index: int = 10) -> List[Tuple[int, str]]:
    # zwraca liste par (index nazwa) do wyswietlenia w gui
    # na windows probuje pobrac nazwy urzadzen przez wmi
    # mapuje nazwy do wykrytych indeksow gdy liczba sie zgadza inaczej stosuje etykiety "Kamera {idx}"
    indices = discover_cameras(max_index=max_index)
    names: list[str] = []
    if sys.platform == "win32":  # pragma: no cover
        names = _win_list_camera_names()

    pairs: List[Tuple[int, str]] = []
    if names and len(names) == len(indices):
        for idx, nm in zip(indices, names):
            pairs.append((idx, nm))
    else:
        for idx in indices:
            pairs.append((idx, f"Kamera {idx}"))
    return pairs


def discover_camera_sources(max_index: int = 10) -> List[Tuple[Union[int, str], str]]:
    # zwraca zrodla kamer do gui (source display_name)
    # na windows uzywa nazw z wmi jako etykiet ale zrodlem pozostaje indeks
    # zawsze zwraca indeksy jako source zgodne z opencv
    indices = discover_cameras(max_index=max_index)
    names: list[str] = []
    if sys.platform == "win32":  # pragma: no cover
        names = _win_list_camera_names()

    pairs: List[Tuple[Union[int, str], str]] = []
    for i, idx in enumerate(indices):
        label = None
        if names and i < len(names):
            label = names[i]
        pairs.append((idx, label or f"Kamera {idx}"))
    return pairs
