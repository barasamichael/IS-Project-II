import re

INJECTION_PATTERNS = re.compile(
    r"(ignore|disregard|forget).{0,30}(previous|above|prior).{0,30}"
    r"(instruction|prompt|system)",
    re.IGNORECASE | re.DOTALL,
)

_CONTENT_HARD_CAP = 1500


def sanitise_web_content(text: str) -> str:
    """
    Sanitise web content before injection into the LLM context window.

    Detects prompt injection patterns and replaces the entire content with a
    safe redaction string. Hard-caps clean content at 1500 characters to limit
    context bloat.

    :param text: str - Raw content string retrieved from a Tavily search result.
    :return: str - Sanitised content safe for LLM injection, capped at 1500 chars.
    """
    if INJECTION_PATTERNS.search(text):
        return "[Web content redacted: contained unsafe patterns]"
    return text[:_CONTENT_HARD_CAP]
