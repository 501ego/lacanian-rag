from app.domain.lacan_canon import LACAN_CANON


def test_lacan_canon_contains_languages():
    assert "en" in LACAN_CANON
    assert "es" in LACAN_CANON


def test_lacan_canon_known_entry():
    assert "S1_Ecrits_techniques" in LACAN_CANON["en"]
    entry = LACAN_CANON["en"]["S1_Ecrits_techniques"]
    assert "titulo" in entry
