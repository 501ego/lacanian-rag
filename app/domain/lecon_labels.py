"""Lesson label utilities for Lacan seminars."""

from typing import Dict, Optional, Tuple
import re

LECON_LABELS: Dict[str, Dict[str, str]] = {"en": {}, "es": {}}

_MONTHS: Dict[str, Tuple[str, str]] = {
    "janvier": ("January", "enero"),
    "fevrier": ("February", "febrero"),
    "février": ("February", "febrero"),
    "mars": ("March", "marzo"),
    "avril": ("April", "abril"),
    "mai": ("May", "mayo"),
    "juin": ("June", "junio"),
    "juillet": ("July", "julio"),
    "aout": ("August", "agosto"),
    "août": ("August", "agosto"),
    "septembre": ("September", "septiembre"),
    "octobre": ("October", "octubre"),
    "novembre": ("November", "noviembre"),
    "decembre": ("December", "diciembre"),
    "décembre": ("December", "diciembre"),
}

_UNKNOWN_PATTERN = re.compile(r"^(Leçon|Séance)\s+UNKNOWN\s+(\d+)$")
_DATE_PATTERN = re.compile(
    r"^(Leçon|Séance)\s+(\d+)\s+(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})$"
)


def get_lecon_label(
    lecon_raw: Optional[str],
    language_code: str,
    unknown_label: str,
) -> str:
    if not lecon_raw:
        return unknown_label

    override = LECON_LABELS.get(language_code, {}).get(lecon_raw)
    if override:
        return override

    normalized = re.sub(r"_+", " ", lecon_raw).strip()
    normalized = re.sub(r"\s+", " ", normalized)

    match = _UNKNOWN_PATTERN.match(normalized)
    if match:
        label_base = _unknown_label_for_kind(match.group(1), language_code, unknown_label)
        return f"{label_base} {int(match.group(2))}"

    match = _DATE_PATTERN.match(normalized)
    if not match:
        return lecon_raw

    kind, number, day, month, year = match.groups()
    month_name = _translate_month(month, language_code)
    if not month_name:
        return lecon_raw

    prefix, date_label = _label_tokens(kind, language_code)
    return f"{prefix} {int(number)} - {date_label} {int(day)} {month_name} {year}"


def _label_tokens(kind: str, language_code: str) -> Tuple[str, str]:
    if kind == "Séance":
        return ("Session", "Date") if language_code == "en" else ("Sesión", "Fecha")
    return ("Lesson", "Date") if language_code == "en" else ("Lección", "Fecha")


def _unknown_label_for_kind(kind: str, language_code: str, default_label: str) -> str:
    if kind == "Séance":
        return "Unknown session" if language_code == "en" else "Sesión desconocida"
    return default_label


def _translate_month(month_token: str, language_code: str) -> Optional[str]:
    key = month_token.lower()
    month = _MONTHS.get(key)
    if not month:
        return None
    return month[0] if language_code == "en" else month[1]
