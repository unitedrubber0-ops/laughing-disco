"""
Utilities for material handling and text normalization.
"""
import re
import logging
import traceback
import math
import inspect
from typing import Any

def safe_material_lookup_entry(standard_raw, grade_raw, material_df, lookup_fn):
    """
    Call lookup_fn with appropriate number of arguments based on its signature.
    Handles both 2-arg (standard, grade) and 3-arg (standard, grade, material_df) function styles.
    Also coerces dict inputs to strings and provides detailed error logging.
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
        try:
            sig = inspect.signature(lookup_fn)
            params = len(sig.parameters)
            logging.debug(f"safe_material_lookup_entry: lookup_fn expects {params} parameters")
            
            if params >= 3:
                result = lookup_fn(std, grd, material_df)
            else:
                result = lookup_fn(std, grd)
                
            return result if result else "Not Found"
            
        except Exception as e:
            logging.warning(f"safe_material_lookup_entry signature inspection failed: {e}, trying fallback calls")
            # Fallback attempts
            try:
                return lookup_fn(std, grd)
            except TypeError:
                return lookup_fn(std, grd, material_df)

    except Exception as e:
        logging.error("Error in material lookup: %s\nstandard_raw repr=%r\ngrade_raw repr=%r\nTraceback:\n%s",
                      e, standard_raw, grade_raw, traceback.format_exc())
        # return a safe fallback so rest of pipeline doesn't crash
        return "Not Found"

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

# Mapping derived from ASTM D2000 Table X1.1 (Table of "Polymers Most Often Used").
# Source: ASTM D2000 (uploaded PDFs). See file citations in the chat.
_D2000_MATERIAL_TO_POLYMER = {
    "AA": "Natural rubber / reclaimed rubber / SBR / butyl / EP (polybutadiene / polyisoprene)",
    "AK": "Polysulfides",
    "BA": "Ethylene propylene (EPDM) / high-temperature SBR / butyl",
    "BC": "Chloroprene polymers (Neoprene)",
    "BE": "Chloroprene polymers (Neoprene)",
    "BF": "Nitrile rubber (NBR) polymers",
    "BG": "NBR polymers and urethanes",
    "BK": "NBR",
    "CA": "Ethylene propylene (EPDM)",
    "CE": "Chlorosulfonated polyethylene (Hypalon)",
    "CH": "NBR and epichlorohydrin polymers",
    "DA": "Ethylene propylene polymers",
    "DE": "CM / CSM (chlorosulfonated / chlorinated polyethylenes)",
    "DF": "Polyacrylic (butyl-acrylate type)",
    "DH": "Polyacrylic polymers / HNBR",
    "EE": "AEM (ethylene acrylic elastomers)",
    "EH": "ACM (polyacrylate)",
    "EK": "FZ (fluoro-elastomer family; see standard)",
    "FC": "Silicones (high temperature/high strength)",
    "FE": "Silicones",
    "FK": "Fluorinated silicones",
    "GE": "Silicones",
    "HK": "Fluorinated elastomers (Viton / Fluorel / similar)",
    "KK": "Perfluoroelastomers",
}

# Build an ordered list of keys (longer keys first is safe but keys here are 2 letters).
_D2000_KEYS = sorted(_D2000_MATERIAL_TO_POLYMER.keys(), key=lambda k: -len(k))

def _find_designation_in_text(text):
    """
    Return the first material designation key found in `text` (e.g. 'BC', 'HK'), or None.
    The search is case-insensitive and tolerant of compact forms like 'M2BC507' or '4CA720'.
    """
    if not text:
        return None
    t = text.upper()
    for key in _D2000_KEYS:
        # ensure key isn't surrounded by letters (but digits or other chars are fine).
        # (?<![A-Z]) asserts previous char is NOT an ASCII letter.
        # (?![A-Z]) asserts next char is NOT an ASCII letter.
        pattern = rf"(?<![A-Z]){re.escape(key)}(?![A-Z])"
        if re.search(pattern, t):
            return key
    return None

def detect_d2000_polymer(callout_text):
    """
    Given a string containing an ASTM D2000 callout (or drawing text),
    return a tuple (designation, polymer_string). If nothing found, (None, None).
    Example input: "ASTM D2000 M2BC 507 A14 EO34" -> ('BC', 'Chloroprene polymers (Neoprene)')
    """
    try:
        if not callout_text or not isinstance(callout_text, str):
            return None, None
        # quick normalization
        s = callout_text.strip()
        key = _find_designation_in_text(s)
        if not key:
            # As a fallback, try to split tokens and match tokens like "CA" or "M2BC"
            tokens = re.split(r"[\s,;/]+", s.upper())
            for tok in tokens:
                # try to find any key inside token (use direct substring)
                for k in _D2000_KEYS:
                    if k in tok:
                        return k, _D2000_MATERIAL_TO_POLYMER.get(k)
            return None, None
        return key, _D2000_MATERIAL_TO_POLYMER.get(key)
    except Exception:
        logging.exception("Error in detect_d2000_polymer")
        return None, None

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