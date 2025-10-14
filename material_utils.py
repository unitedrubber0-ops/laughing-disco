"""
Utilities for material handling and text normalization.
"""
import re
import json
import logging
import traceback
import math
import inspect
from typing import Any

def safe_material_lookup_entry(standard_raw, grade_raw, lookup_fn, material_df=None):
    """
    Enhanced material lookup with flexible argument handling.
    Coerces inputs to strings and handles functions that expect different numbers of arguments.
    """
    try:
        # Log types to find leak
        if isinstance(standard_raw, dict) or isinstance(grade_raw, dict):
            logging.warning("safe_material_lookup_entry: dict-like input detected. standard repr=%r grade repr=%r",
                            standard_raw, grade_raw)

        # Coerce with normalize helpers (safe for dicts)
        std = normalize_standard(standard_raw)
        grd = normalize_grade(grade_raw)

        logging.debug("safe_material_lookup_entry: normalized standard=%r grade=%r", std, grd)

        # Check function signature and call appropriately
        sig = inspect.signature(lookup_fn)
        params = len(sig.parameters)
        logging.debug(f"safe_material_lookup_entry: lookup_fn expects {params} parameters")
        
        # Prefer to pass material_df only if the function expects it
        if params >= 3:
            return lookup_fn(std, grd, material_df)
        else:
            return lookup_fn(std, grd)

    except Exception as e:
        logging.error("Error in material lookup: %s\nstandard_raw repr=%r\ngrade_raw repr=%r\nTraceback:\n%s",
                      e, standard_raw, grade_raw, traceback.format_exc())
        # Fallback attempts
        try:
            return lookup_fn(std, grd)
        except TypeError:
            if material_df is not None:
                return lookup_fn(std, grd, material_df)
            return "Not Found"

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
        for k in ('text', 'ocr_text', 'raw_text', 'content', 'STANDARD', 'standard', 'value', 'name'):
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

def _coerce_to_str_maybe_dict(val, keys_to_try=('STANDARD','standard','value','name')):
    """
    If val is a dict, try common keys; otherwise return str(val) (or '' if None).
    """
    if val is None:
        return ''
    if isinstance(val, dict):
        for k in keys_to_try:
            if k in val and val[k] is not None:
                return str(val[k])
        # fallback: return repr so logs show the dict content
        return repr(val)
    if not isinstance(val, str):
        try:
            return str(val)
        except Exception:
            return repr(val)
    return val

def normalize_standard(std):
    try:
        std = _coerce_to_str_maybe_dict(std, keys_to_try=('STANDARD','standard','std'))
        std = std.upper().strip()
        # apply existing cleanup rules (example)
        std = re.sub(r'MPAPS\s*F\s*[-_]?\s*(\d+)', r'MPAPS F-\1', std)
        std = re.sub(r'\s+', ' ', std).strip()
        return std
    except Exception as e:
        logging.exception("normalize_standard failed for input: %r; returning empty string", std)
        return ''

def normalize_grade(grd):
    try:
        grd = _coerce_to_str_maybe_dict(grd, keys_to_try=('GRADE','grade','grd'))
        grd = grd.upper().strip()
        grd = re.sub(r'[^A-Z0-9\-]', '', grd)       # remove weird punctuation
        grd = re.sub(r'GRADE|TYPE|CLASS', '', grd)
        grd = grd.replace(' ', '').replace('_', '').replace('/', '')
        grd = re.sub(r'([A-Z0-9\-]+)[XZ]$', r'\1', grd)
        return grd
    except Exception as e:
        logging.exception("normalize_grade failed for input: %r; returning empty string", grd)
        return ''

def _process_extraction_result(val: Any) -> str:
    """Process an extraction result, converting to string or empty string if None."""
    if val is None:
        return ''
    return str(val)

# Diameter extraction helpers
_DIA_PATTERNS = [
    r'(?:DIA|DIAMETER|Ø|PHI|⌀)\s*[:=]?\s*([+-]?\d+\.?\d*)\s*(MM|IN|INCH|CM)?',  # e.g., DIA: 85 mm
    r'\bD\s*[:=]?\s*([+-]?\d+\.?\d*)\s*(MM|IN|CM)?\b'
]

def extract_diameter(text: str) -> tuple[float, str | None] | None:
    """Extract diameter value and unit from text, returning (value, unit) or None."""
    t = (text or "").upper()
    for p in _DIA_PATTERNS:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                unit = (m.group(2) or '').upper().strip() or None
                return val, unit
            except Exception:
                continue
    return None

def development_length_from_diameter(dia_value: float, unit: str | None = None) -> float:
    """Return circumference = pi * D. Unit left as-is."""
    return math.pi * dia_value

def are_rings_empty(rings_info: dict) -> bool:
    """Return True if both types and count are missing/empty."""
    if rings_info is None:
        return True
    return not rings_info.get('types') and not rings_info.get('count')

def safe_search(pattern: str, text: str | None, flags: int = 0) -> re.Match | None:
    """Safely perform regex search on text that might be None or non-string."""
    try:
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        return re.search(pattern, text, flags=flags)
    except Exception as e:
        logging.exception("safe_search failed for pattern %r on text %r: %s", pattern, text, e)
        return None