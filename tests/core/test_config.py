from app.core.config import AppConfig, JSON_KEYS, LANGUAGE_NAMES, UI_TEXT


def test_app_config_defaults():
    config = AppConfig()
    assert config.top_k == 5
    assert config.max_sources == 4
    assert config.chunk_size > 0
    assert str(config.vector_index_path).endswith("vector_index/lacan.index")


def test_constants_have_expected_keys():
    assert "en" in LANGUAGE_NAMES
    assert "es" in LANGUAGE_NAMES
    assert "ask_question" in UI_TEXT
    assert "sources" in JSON_KEYS
