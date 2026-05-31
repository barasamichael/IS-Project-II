import re

INTENT_OFF_TOPIC_THRESHOLD = 0.40
INTENT_EMBEDDING_CACHE_SIZE = 512
TAVILY_MAX_RESULTS = 3
TAVILY_TIMEOUT_SECONDS = 4
LLM_RESPONSE_MAX_TOKENS = 2048
LLM_EMERGENCY_MAX_TOKENS = 4096
LLM_RESPONSE_TEMPERATURE = 0.2
QUERY_MAX_LENGTH = 2000
QUERY_MIN_LENGTH = 3
EMBEDDING_CONTEXT_PREFIX = "International student in {city} {country}: "

GROUNDING_RULE = """GROUNDING RULE (NON-NEGOTIABLE):
You may only state a phone number, email address, physical address,
fee amount, or operating hours if that value appears VERBATIM in the
RETRIEVED SETTLEMENT INFORMATION or ESSENTIAL SETTLEMENT INFORMATION
sections provided below.
If you cannot find a contact detail in the provided context, write:
"Contact details not available — verify at [official source URL if known]."
Never invent, approximate, or infer contact information."""

PHONE_RE = re.compile(
    r"(\+254[\s\-]?\d[\d\s\-]{7,}|\b0[17]\d{2}[\s\-]?\d{3}[\s\-]?\d{3}\b)"
)

ERROR_CODE_INVALID_FILENAME = "INVALID_FILENAME"
ERROR_CODE_INTENT_RECOGNITION_FAILED = "INTENT_RECOGNITION_FAILED"
ERROR_CODE_RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_WAIT_MIN = 1
LLM_RETRY_WAIT_MAX = 8
LLM_CALL_TIMEOUT_SECONDS = 15
LLM_TAVILY_TIMEOUT_SECONDS = 4

EMBEDDING_LRU_CACHE_SIZE = 512
TAVILY_CACHE_TTL = 3600
TAVILY_CACHE_MAXSIZE = 200
