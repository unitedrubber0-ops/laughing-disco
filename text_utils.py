import json
import logging
import re

logger = logging.getLogger(__name__)

def _coerce_to_text(value):
    """
    Ensure we return a string: if value is a dict with common keys, return the text field.
    Otherwise JSON-dump or str() it.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, dict):
        # If common field exists, prefer it
        for k in ('text', 'ocr_text', 'raw_text', 'content'):
            if k in value:
                try:
                    return str(value[k])
                except Exception:
                    pass
        # fallback to compact json
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    # fallback for other non-str types
    try:
        return str(value)
    except Exception:
        return ""

def clean_text_encoding(text):
    """Clean and normalize text encoding issues."""
    text = _coerce_to_text(text)
    if not text:
        return ""
    try:
        # Replace common problematic characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'[\r\n]+', '\n', text)       # Normalize newlines
        text = re.sub(r'\s+', ' ', text)            # Normalize spaces
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text encoding: {e}")
        return text