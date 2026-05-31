import json
import re

from typing import TYPE_CHECKING

from config.constants import PHONE_RE

if TYPE_CHECKING:
    from config.locale import LocaleFactStore


def normalise_phone(phone: str) -> str:
    """
    Produce a canonical comparison form for a phone number string by stripping
    spaces, hyphens, parentheses, and leading zeros.
    Examples: "+254 20 2845000" and "+254-20-2845000" both normalise to
    "+25420 2845000" → "+254202845000".
    :param phone: str - Raw phone number string as it appears in text or the fact store.
    :return: str - Normalised phone string suitable for set membership comparison.
    """
    stripped = re.sub(r"[\s\-\(\)]", "", phone)
    if stripped.startswith("0") and not stripped.startswith("+"):
        stripped = stripped.lstrip("0")
    return stripped


def extract_phones_from_context(
    context_text: str,
    fact_store: "LocaleFactStore",
) -> set:
    """
    Build the set of verified (normalised) phone numbers from two sources:
    the retrieved context text and the fact store's emergency_contacts.
    Any number present in either source is considered verified for audit purposes.
    :param context_text: str - The full enhanced context string passed to the LLM.
    :param fact_store: LocaleFactStore - The loaded locale fact store.
    :return: set[str] - Set of normalised phone strings that are considered verified.
    """
    verified: set = set()

    for raw in PHONE_RE.findall(context_text):
        verified.add(normalise_phone(raw))

    emergency_json = json.dumps(
        {k: v.model_dump() for k, v in fact_store.emergency_contacts.items()}
    )
    for raw in PHONE_RE.findall(emergency_json):
        verified.add(normalise_phone(raw))

    return verified
