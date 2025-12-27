import re

BLOCKED_WORDS = ["password", "secret", "token", "apikey"]

def sanitize_query(query: str) -> bool:
    return not any(word in query.lower() for word in BLOCKED_WORDS)

def redact_sensitive_data(text: str) -> str:
    text = re.sub(r'AKIA[0-9A-Z]{16}', '[REDACTED]', text)
    text = re.sub(
        r'(?i)password\s*=\s*\S+',
        'password=[REDACTED]',
        text
    )
    return text
