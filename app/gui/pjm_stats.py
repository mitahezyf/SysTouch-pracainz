# czysty modul do budowania tekstow statystyk PJM bez zaleznosci od Qt
from typing import Any, Protocol


class TranslatorProtocol(Protocol):
    # minimalna definicja translatora dla statystyk
    def get_state(self) -> dict[str, Any]: ...

    def get_statistics(self) -> dict[str, Any]: ...

    def get_history(self, format_groups: bool = False) -> str: ...


def build_pjm_display(translator: TranslatorProtocol | None) -> dict[str, str]:
    # zwraca slownik z kluczami odpowiadajacymi labelkom UI
    # jesli translator == None, zwraca puste/domyslne wartosci
    if not translator:
        return {
            "letter": "--",
            "confidence": "Pewnosc: --%",
            "time": "Czas: 0ms",
            "total": "Wykryto liter: 0",
            "rate": "Wykryc/min: 0.0",
            "unique": "Unikalne: 0",
            "top": "Top 5: --",
            "history": "",
        }

    state = translator.get_state()
    stats = translator.get_statistics()

    # aktualna litera i pewnosc
    letter = state.get("current_letter", "unknown")
    if letter == "unknown":
        letter_text = "--"
        conf_text = "Pewnosc: --%"
    else:
        letter_text = letter
        conf = float(state.get("confidence", 0.0)) * 100
        conf_text = f"Pewnosc: {conf:.1f}%"

    # czas trzymania
    time_ms = int(state.get("time_held_ms", 0))
    time_text = f"Czas: {time_ms}ms"

    # statystyki sesji
    total = stats.get("total_detections", 0)
    rate = stats.get("detections_per_minute", 0.0)
    unique = stats.get("unique_letters", 0)

    total_text = f"Wykryto liter: {total}"
    rate_text = f"Wykryc/min: {rate:.1f}"
    unique_text = f"Unikalne: {unique}"

    # top 5
    most_common = stats.get("most_common", [])
    if most_common:
        top_items = [f"{k}:{v}" for k, v in most_common[:5]]
        top_text = "Top 5: " + ", ".join(top_items)
    else:
        top_text = "Top 5: --"

    # historia liter
    history = translator.get_history(format_groups=False)
    history_text = history or ""

    return {
        "letter": letter_text,
        "confidence": conf_text,
        "time": time_text,
        "total": total_text,
        "rate": rate_text,
        "unique": unique_text,
        "top": top_text,
        "history": history_text,
    }
