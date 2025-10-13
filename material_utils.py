"""
Utilities for material handling and text normalization.
"""
import re
import logging

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

def safe_search(pattern, text, flags=0):
    """
    Safely perform regex search on text that might be None or non-string.
    """
    try:
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        return re.search(pattern, text, flags=flags)
    except Exception as e:
        logging.exception("safe_search failed for pattern %r on text %r: %s", pattern, text, e)
        return None