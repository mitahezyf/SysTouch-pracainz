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

        # Szersze zapytania, aby znaleźć również wirtualne kamery (OBS, itp.)
        queries = [
            "SELECT Name FROM Win32_PnPEntity WHERE PNPClass = 'Camera'",
            "SELECT Name FROM Win32_PnPEntity WHERE PNPClass = 'Image'",
            "SELECT Name FROM Win32_PnPEntity WHERE Name LIKE '%Camera%'",
            "SELECT Name FROM Win32_PnPEntity WHERE Name LIKE '%Webcam%'",
            "SELECT Name FROM Win32_PnPEntity WHERE Name LIKE '%OBS%'",
            "SELECT Name FROM Win32_PnPEntity WHERE Name LIKE '%Virtual%'",
            "SELECT Name FROM Win32_PnPEntity WHERE Name LIKE '%Video%'",
            # Szukaj również w kategorii Video
            "SELECT Name FROM Win32_PnPEntity WHERE PNPClass = 'Media'",
        ]
        names: list[str] = []
        seen: set[str] = set()

        # Lista słów kluczowych, które wykluczają urządzenie z listy kamer
        blacklist_keywords = [
            "MFP",
            "Printer",
            "Scanner",
            "Scan",
            "Print",
            "ACPI",
            "x64",
            "x86",
            "Composite",
            "Laser",
            "Inkjet",
            "Copier",
            "Audio",  # Wykluczamy urządzenia audio
        ]

        logger.debug(
            "_win_list_camera_names: Starting WMI scan with %d queries", len(queries)
        )

        for q in queries:
            try:
                items = svc.ExecQuery(q)
                for it in items:
                    nm = str(getattr(it, "Name", "")).strip()
                    if nm and nm not in seen:
                        # Sprawdź czy nazwa zawiera któreś z blacklisted słów
                        is_blacklisted = any(
                            keyword.lower() in nm.lower()
                            for keyword in blacklist_keywords
                        )
                        if not is_blacklisted:
                            names.append(nm)
                            seen.add(nm)
                            logger.debug(
                                "_win_list_camera_names: Accepted camera: %s", nm
                            )
                        else:
                            logger.debug(
                                "_win_list_camera_names: Filtered out (blacklisted): %s",
                                nm,
                            )
            except Exception as e:
                logger.debug("_win_list_camera_names: query error %s: %s", q, e)
                continue

        logger.debug(
            "_win_list_camera_names: WMI scan complete, found %d cameras", len(names)
        )
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
    logger.info("[Camera Discovery] Found %d camera indices: %s", len(indices), indices)

    names: list[str] = []
    if sys.platform == "win32":  # pragma: no cover
        names = _win_list_camera_names()
        logger.info(
            "[Camera Discovery] Retrieved %d camera names from WMI: %s",
            len(names),
            names,
        )

    pairs: List[Tuple[Union[int, str], str]] = []

    # Używaj dostępnych nazw z WMI dla pierwszych N kamer
    # Dla pozostałych spróbuj pobrać nazwę z OpenCV lub użyj generycznej etykiety
    for i, idx in enumerate(indices):
        label = None
        if names and i < len(names):
            label = names[i]

        # Jeśli WMI nie znalazło nazwy, spróbuj pobrać backend name z OpenCV
        if not label:
            try:
                import cv2

                cap = None
                if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(idx)

                if cap and cap.isOpened():
                    # Próba pobrania nazwy kamery (nie wszystkie backend'y to wspierają)
                    backend_name = (
                        cap.getBackendName() if hasattr(cap, "getBackendName") else None
                    )
                    if backend_name:
                        label = f"{backend_name} {idx}"
                        logger.debug(
                            "[Camera Discovery] Got backend name for idx %d: %s",
                            idx,
                            backend_name,
                        )

                if cap:
                    cap.release()
            except Exception as e:
                logger.debug(
                    "[Camera Discovery] Failed to get OpenCV name for idx %d: %s",
                    idx,
                    e,
                )

        final_label = label or f"Kamera {idx}"
        pairs.append((idx, final_label))
        logger.info(
            "[Camera Discovery] Camera %d: index=%d, label='%s' (from_wmi=%s)",
            i,
            idx,
            final_label,
            bool(names and i < len(names)),
        )

    logger.info("[Camera Discovery] Returning %d camera sources", len(pairs))
    return pairs
