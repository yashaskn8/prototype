import re

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(the\s+)?system\s+prompt",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?developer\s+message",
    r"you\s+are\s+chatgpt",
    r"jailbreak",
    r"follow\s+these\s+instructions\s+instead",
    r"exfiltrat(e|ion)",
    r"reveal\s+secrets?",
]

QUERY_ABUSE_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"show\s+.*other\s+(customer|tenant)",
    r"reveal\s+confidential",
    r"dump\s+(the\s+)?database",
]


def detect_document_prompt_injection(text):
    lowered = (text or "").lower()
    flags = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered, flags=re.I):
            flags.append(pattern)
    return flags


def query_is_control_abuse(text):
    lowered = (text or "").lower()
    return any(re.search(pattern, lowered, flags=re.I) for pattern in QUERY_ABUSE_PATTERNS)
