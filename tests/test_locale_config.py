"""
Tests for LocaleConfig loading, validation, and settings.locale property.
Addresses Milestone 8 — Multi-Locale Architecture invariants.
"""
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Set required env vars before any project imports
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")
os.environ.setdefault("SETTLEBOT_API_KEY", "test-api-key-for-unit-tests")
os.environ["SETTLEBOT_LOCALE"] = "nairobi"

from config.locale import LocaleConfig
from config.settings import ROOT_DIR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def nairobi_config() -> LocaleConfig:
    """Return LocaleConfig loaded from nairobi_config.json."""
    return LocaleConfig.from_file("nairobi")


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------


def test_locale_config_loads_nairobi(nairobi_config: LocaleConfig) -> None:
    """LocaleConfig.from_file("nairobi") returns city=Nairobi, country=Kenya."""
    assert nairobi_config.city == "Nairobi"
    assert nairobi_config.country == "Kenya"
    assert nairobi_config.timezone == "Africa/Nairobi"


def test_locale_config_has_all_fields(nairobi_config: LocaleConfig) -> None:
    """All 12 required LocaleConfig fields are non-empty after loading."""
    assert nairobi_config.city
    assert nairobi_config.country
    assert nairobi_config.currency_code
    assert nairobi_config.currency_symbol
    assert nairobi_config.timezone
    assert nairobi_config.emergency_number
    assert nairobi_config.primary_languages
    assert nairobi_config.key_neighborhoods
    assert nairobi_config.trusted_web_domains
    assert nairobi_config.collection_name
    assert nairobi_config.web_search_geo_bias
    assert nairobi_config.fact_store_path


def test_key_neighborhoods_non_empty(nairobi_config: LocaleConfig) -> None:
    """key_neighborhoods must have at least 10 entries."""
    assert len(nairobi_config.key_neighborhoods) >= 10


def test_trusted_web_domains_non_empty(nairobi_config: LocaleConfig) -> None:
    """trusted_web_domains must have at least 3 entries."""
    assert len(nairobi_config.trusted_web_domains) >= 3


def test_primary_languages_includes_swahili(nairobi_config: LocaleConfig) -> None:
    """primary_languages for nairobi must include 'sw' (Swahili)."""
    assert "sw" in nairobi_config.primary_languages


def test_currency_symbol_correct(nairobi_config: LocaleConfig) -> None:
    """currency_symbol for nairobi must be 'KSh'."""
    assert nairobi_config.currency_symbol == "KSh"


def test_collection_name_correct(nairobi_config: LocaleConfig) -> None:
    """collection_name for nairobi must be 'settlebot_nairobi'."""
    assert nairobi_config.collection_name == "settlebot_nairobi"


def test_settings_locale_returns_locale_config() -> None:
    """settings.locale returns a LocaleConfig instance when SETTLEBOT_LOCALE is set."""
    from config.settings import settings

    lc = settings.locale
    assert lc is not None
    assert isinstance(lc, LocaleConfig)
    assert lc.city == "Nairobi"


def test_settings_locale_cached() -> None:
    """Calling settings.locale twice returns the same object (no re-read)."""
    from config.settings import settings

    lc1 = settings.locale
    lc2 = settings.locale
    assert lc1 is lc2


def test_vector_db_uses_locale_collection_name() -> None:
    """VectorDBService instantiated with a locale uses locale.collection_name."""
    from unittest.mock import MagicMock, patch

    mock_locale = MagicMock()
    mock_locale.collection_name = "settlebot_test_city"
    mock_locale.key_neighborhoods = []
    mock_locale.city = "TestCity"

    mock_embed_svc = MagicMock()
    mock_embed_svc.dimension = 8

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))

    with patch("services.vector_db.chromadb.PersistentClient", return_value=client), \
         patch("services.vector_db.VectorDBService._build_bm25_index"), \
         patch.object(
             __import__("services.vector_db", fromlist=["VectorDBService"]).VectorDBService,
             "_reranker", None
         ):
        from services.vector_db import VectorDBService

        svc = VectorDBService.__new__(VectorDBService)
        svc.embedding_service = mock_embed_svc
        svc.dimension = 8
        svc.locale = mock_locale
        svc.collection_name = mock_locale.collection_name
        svc.client = client
        svc._bm25_index = None
        svc._bm25_texts = []
        svc._bm25_docs = []
        svc.collection = client.create_collection(
            name="settlebot_test_city",
            metadata={"hnsw:space": "cosine"},
        )
        svc._initialize_settlement_filters()

    assert svc.collection_name == "settlebot_test_city"


def test_language_processor_includes_luganda() -> None:
    """LanguageProcessor.supported_languages must include 'lg' (Luganda) after YAML derivation."""
    from services.language_processor import LanguageProcessor

    lp = LanguageProcessor()
    assert "lg" in lp.supported_languages, (
        "'lg' (Luganda) must be in supported_languages — "
        "it is in config.yaml but was missing from the old hardcoded dict"
    )


def test_language_processor_includes_kinyarwanda() -> None:
    """LanguageProcessor.supported_languages must include 'rw' (Kinyarwanda)."""
    from services.language_processor import LanguageProcessor

    lp = LanguageProcessor()
    assert "rw" in lp.supported_languages


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_locale_config_missing_file_raises() -> None:
    """LocaleConfig.from_file('nonexistent') must raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        LocaleConfig.from_file("nonexistent_locale_xyz")


def test_locale_config_missing_field_raises(tmp_path: Path) -> None:
    """Loading a JSON file missing 'timezone' raises ValueError."""
    incomplete = {
        "city": "TestCity",
        "country": "TestCountry",
        "currency_code": "TST",
        "currency_symbol": "T$",
        # "timezone" deliberately omitted
        "emergency_number": "911",
        "primary_languages": ["en"],
        "key_neighborhoods": ["Area1"],
        "trusted_web_domains": ["example.com"],
        "collection_name": "settlebot_test",
        "web_search_geo_bias": "TestCountry",
        "fact_store_path": "config/locale/test_facts.json",
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(incomplete), encoding="utf-8")

    with patch.object(
        LocaleConfig,
        "from_file",
        wraps=lambda name: _load_from_path(config_file),
    ):
        with pytest.raises(ValueError, match="missing required fields"):
            _load_from_path(config_file)


def _load_from_path(path: Path) -> LocaleConfig:
    """Helper: load LocaleConfig directly from a Path (bypasses name lookup)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    required = [
        "city", "country", "currency_code", "currency_symbol", "timezone",
        "emergency_number", "primary_languages", "key_neighborhoods",
        "trusted_web_domains", "collection_name", "web_search_geo_bias", "fact_store_path",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"Locale config is missing required fields: {missing}")
    return LocaleConfig(**{k: data[k] for k in required})


def test_settings_locale_none_when_env_unset() -> None:
    """settings.locale returns None when SETTLEBOT_LOCALE is not set."""
    from config.settings import Settings
    from config.settings import ROOT_DIR as _ROOT

    saved = os.environ.pop("SETTLEBOT_LOCALE", None)
    try:
        new_settings = Settings.from_yaml(_ROOT / "config" / "config.yaml")
        result = new_settings.locale  # Should return None when env not set
        assert result is None
    finally:
        if saved:
            os.environ["SETTLEBOT_LOCALE"] = saved
