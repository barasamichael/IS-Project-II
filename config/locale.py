import json

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


def load_fact_store(locale_name: str) -> LocaleFactStore:
    """
    Read, parse, and validate the fact store JSON for the given locale.
    :param locale_name: str - The locale identifier (e.g. "nairobi"). Maps to
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
