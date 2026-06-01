import json
from dataclasses import dataclass

from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

from config.settings import ROOT_DIR


class EmergencyContact(BaseModel):
    """
    Verified emergency contact entry for a locale deployment.
    :field number: str - The emergency contact phone number.
    :field verified_date: str - ISO date when this entry was last verified by an operator.
    :field official_source_url: str - URL of the official source; empty string when unavailable.
    """

    number: str
    verified_date: str
    official_source_url: str


class HospitalEntry(BaseModel):
    """
    Verified hospital entry for a locale deployment.
    :field name: str - Full hospital name.
    :field address: str - Physical address.
    :field phone: Optional[str] - Contact phone number; None when not available.
    :field email: Optional[str] - Contact email address; None when not available.
    :field website: Optional[str] - Official website URL; None when not available.
    :field notes: Optional[str] - Descriptive notes about the facility.
    :field verified_date: str - ISO date when this entry was last verified by an operator.
    :field official_source_url: str - URL of the official source; empty string when unavailable.
    """

    name: str
    address: str
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    notes: Optional[str] = None
    verified_date: str
    official_source_url: str


class UniversityEntry(BaseModel):
    """
    Verified university contact entry for a locale deployment.
    :field name: str - Full university name.
    :field address: str - Physical address.
    :field phone: Optional[str] - Contact phone number; None when not available.
    :field email: Optional[str] - Contact email address; None when not available.
    :field website: Optional[str] - Official website URL; None when not available.
    :field verified_date: str - ISO date when this entry was last verified by an operator.
    :field official_source_url: str - URL of the official source; empty string when unavailable.
    """

    name: str
    address: str
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    verified_date: str
    official_source_url: str


class GovernmentOfficeEntry(BaseModel):
    """
    Verified government office entry for a locale deployment.
    :field name: str - Full office or department name.
    :field address: str - Physical address.
    :field phone: Optional[str] - Contact phone number; None when not available.
    :field hours: Optional[str] - Operating hours; None when not available.
    :field services: Optional[str] - Description of services offered; None when not available.
    :field verified_date: str - ISO date when this entry was last verified by an operator.
    :field official_source_url: str - URL of the official source; empty string when unavailable.
    """

    name: str
    address: str
    phone: Optional[str] = None
    hours: Optional[str] = None
    services: Optional[str] = None
    verified_date: str
    official_source_url: str


class LocaleFactStore(BaseModel):
    """
    Validated collection of verified settlement facts for one locale deployment.
    Loaded at startup from a JSON file in config/locale/. Every phone number,
    address, and contact detail that may appear in a generated response must
    originate from this store.
    :field emergency_contacts: Dict[str, EmergencyContact] - Service-name-keyed emergency contacts.
    :field hospitals: List[HospitalEntry] - Verified hospital records.
    :field universities: List[UniversityEntry] - Verified university contact records.
    :field government_offices: List[GovernmentOfficeEntry] - Verified government office records.
    :field trusted_domains: List[str] - Domain suffixes whose Tavily results are
        trusted for LLM injection. Results from domains not on this list are
        discarded before context assembly.
    """

    emergency_contacts: Dict[str, EmergencyContact]
    hospitals: List[HospitalEntry]
    universities: List[UniversityEntry]
    government_offices: List[GovernmentOfficeEntry]
    trusted_domains: List[str]


@dataclass
class LocaleConfig:
    """
    Deployment configuration for one SettleBot locale.
    All locale-specific values (city, country, currency, timezone, neighbourhoods)
    flow through this object. No service may use hardcoded city, country, currency,
    or timezone literals; read from this object instead.
    :field city: str - Display-case city name (e.g. "Nairobi").
    :field country: str - Display-case country name (e.g. "Kenya").
    :field currency_code: str - ISO 4217 code (e.g. "KES").
    :field currency_symbol: str - Short currency symbol (e.g. "KSh").
    :field timezone: str - Pytz-compatible timezone string (e.g. "Africa/Nairobi").
    :field emergency_number: str - Primary emergency services number (e.g. "999").
    :field primary_languages: List[str] - ISO 639-1 language codes in priority order.
    :field key_neighborhoods: List[str] - Well-known neighbourhood names for this city.
    :field trusted_web_domains: List[str] - Domain suffixes trusted for Tavily results.
    :field collection_name: str - ChromaDB collection name for this locale.
    :field web_search_geo_bias: str - Country name appended to Tavily queries.
    :field fact_store_path: str - Relative path to the LocaleFactStore JSON file.
    """

    city: str
    country: str
    currency_code: str
    currency_symbol: str
    timezone: str
    emergency_number: str
    primary_languages: List[str]
    key_neighborhoods: List[str]
    trusted_web_domains: List[str]
    collection_name: str
    web_search_geo_bias: str
    fact_store_path: str

    @classmethod
    def from_file(cls, locale_name: str) -> "LocaleConfig":
        """
        Load and return a LocaleConfig from config/locale/{locale_name}_config.json.
        :param locale_name: str - Locale identifier matching SETTLEBOT_LOCALE (e.g. "kampala").
        :return: LocaleConfig - Validated locale configuration instance.
        :raises ValueError: When the JSON file is missing or required fields are absent.
        """
        config_file: Path = (
            ROOT_DIR / "config" / "locale" / f"{locale_name}_config.json"
        )
        if not config_file.exists():
            raise ValueError(
                f"Locale config file not found: {config_file}. "
                f"Create config/locale/{locale_name}_config.json to enable this locale."
            )
        try:
            with open(config_file, encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Locale config file is not valid JSON for locale '{locale_name}': {exc}"
            ) from exc

        required_fields = [
            "city",
            "country",
            "currency_code",
            "currency_symbol",
            "timezone",
            "emergency_number",
            "primary_languages",
            "key_neighborhoods",
            "trusted_web_domains",
            "collection_name",
            "web_search_geo_bias",
            "fact_store_path",
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(
                f"Locale config for '{locale_name}' is missing required fields: {missing}"
            )

        return cls(
            city=data["city"],
            country=data["country"],
            currency_code=data["currency_code"],
            currency_symbol=data["currency_symbol"],
            timezone=data["timezone"],
            emergency_number=data["emergency_number"],
            primary_languages=data["primary_languages"],
            key_neighborhoods=data["key_neighborhoods"],
            trusted_web_domains=data["trusted_web_domains"],
            collection_name=data["collection_name"],
            web_search_geo_bias=data["web_search_geo_bias"],
            fact_store_path=data["fact_store_path"],
        )


def load_fact_store(locale_name: str) -> LocaleFactStore:
    """
    Read, parse, and validate the fact store JSON for the given locale.
    :param locale_name: str - The locale identifier (e.g. "kampala"). Maps to
        config/locale/{locale_name}.json.
    :return: LocaleFactStore - Validated fact store instance.
    :raises ValueError: When the JSON file is missing or fails schema validation.
    """
    locale_file: Path = ROOT_DIR / "config" / "locale" / f"{locale_name}.json"
    if not locale_file.exists():
        raise ValueError(
            f"Locale fact store file not found: {locale_file}. "
            f"Create config/locale/{locale_name}.json to enable this locale."
        )
    try:
        with open(locale_file, encoding="utf-8") as fh:
            data = json.load(fh)
        return LocaleFactStore.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Locale fact store file is invalid for locale '{locale_name}': {exc}"
        ) from exc
