import re

from app.domain import lecon_labels


def test_get_lecon_label_returns_unknown_on_empty():
    assert lecon_labels.get_lecon_label(None, "en", "Unknown lesson") == "Unknown lesson"


def test_get_lecon_label_uses_override(monkeypatch):
    monkeypatch.setitem(lecon_labels.LECON_LABELS, "en", {"Custom": "Override"})
    assert lecon_labels.get_lecon_label("Custom", "en", "Unknown") == "Override"


def test_get_lecon_label_unknown_pattern():
    label = lecon_labels.get_lecon_label("Leçon UNKNOWN 2", "en", "Unknown lesson")
    assert label == "Unknown lesson 2"


def test_get_lecon_label_date_pattern_en():
    label = lecon_labels.get_lecon_label("Leçon 2 13 janvier 1964", "en", "Unknown")
    assert label == "Lesson 2 - Date 13 January 1964"


def test_get_lecon_label_returns_raw_when_pattern_fails():
    raw = "Leçon 99 foo 1964"
    assert lecon_labels.get_lecon_label(raw, "en", "Unknown") == raw


def test_translate_month_handles_missing():
    assert lecon_labels._translate_month("nonexistent", "en") is None


def test_unknown_pattern_regex_matches():
    assert re.match(lecon_labels._UNKNOWN_PATTERN, "Leçon UNKNOWN 1")
