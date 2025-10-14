"""
Enhanced extraction utilities for rings and coordinates.
"""
import re
import math
import logging
from typing import List, Tuple, Optional

# Base components for ring pattern matching
_RING_COMPONENTS = {
    'quantity': r'(?:(\d+)\s*(?:X|TIMES|PCS)?)',
    'type': r'(?:RING(?:S)?|REINFORCEMENT)',
    'position': r'(?:@\s*\d+|INNER|OUTER|MIDDLE)',
    'material': r'(?:STAINLESS\s*(?:STEEL|WIRE)|[A-Z]+\s*STEEL|[A-Z]{2,4})'
}

# Dynamic patterns built from components
_RING_COUNT_PATTERNS = [
    fr"{_RING_COMPONENTS['quantity']}\s*{_RING_COMPONENTS['type']}",
    fr"{_RING_COMPONENTS['type']}\s*{_RING_COMPONENTS['quantity']}",
    fr"(?:NO\.?\s*OF\s*)?{_RING_COMPONENTS['type']}\s*[:=\-]?\s*{_RING_COMPONENTS['quantity']}"
]

_RING_TYPE_PATTERNS = [
    fr"{_RING_COMPONENTS['position']}\s*[:=\-]?\s*{_RING_COMPONENTS['material']}",
    fr"{_RING_COMPONENTS['type']}\s*[:=\-]?\s*{_RING_COMPONENTS['material']}",
    fr"{_RING_COMPONENTS['material']}\s*{_RING_COMPONENTS['type']}"
]

# Patterns for explicit coordinates and points
_COORD_PATTERNS = [
    r'\(?\s*X\s*[:=]\s*([+-]?\d+\.?\d*)\s*[,\s]\s*Y\s*[:=]\s*([+-]?\d+\.?\d*)\s*\)?',  # X:12 Y:34
    r'\(?\s*([+-]?\d+\.?\d*)\s*[,\s]\s*([+-]?\d+\.?\d*)\s*\)',                        # (12,34) or 12,34
    r'POINT\s*\d+\s*[:=]?\s*\(?([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)\)?',             # POINT 1: (12,34)
    r'COORD(?:S|INATES)?[^\d\n]*[:=\-]?\s*([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)',     # COORD: 12 34
]

# Patterns for direct development-length values
_DEV_LENGTH_PATTERNS = [
    r'DEVELOP(?:MENT)?\s*LENGTH\s*[:=]?\s*([+-]?\d+\.?\d*)\s*(MM|IN|INCH|CM)?',
    r'DEV(?:\.| |EL)?\s*LENGTH\s*[:=]?\s*([+-]?\d+\.?\d*)\s*(MM|IN|INCH|CM)?',
]

def extract_rings_info(text: str) -> dict:
    """
    Extract rings information from text.
    Returns a dict with keys: count, types, raw_matches.
    """
    result = {
        "count": None,
        "types": [],
        "raw_matches": []
    }
    try:
        if not text:
            return result
            
        t = str(text).upper()
        
        # Look for explicit rings mentions
        rings_patterns = [
            r'RINGS?\s*:\s*([^\n.,;]+(?:[.,;]\s*[^\n.,;]+)*)',
            r'RINGS?\s*-\s*([^\n.,;]+(?:[.,;]\s*[^\n.,;]+)*)',
            r'(\d+\s*X\s*RING[^\n.,;]*(?:[.,;]\s*[^\n.,;]+)*)',
            r'(RING\s*REINFORCEMENT[^\n.,;]*(?:[.,;]\s*[^\n.,;]+)*)',
        ]
        
        for pattern in rings_patterns:
            matches = re.finditer(pattern, t, re.IGNORECASE)
            for m in matches:
                # Safely extract matched text
                if m.lastindex and m.lastindex >= 1:
                    val = m.group(1).strip()
                else:
                    val = m.group(0).strip()
                
                if not val:
                    continue
                    
                # Clean the value
                val = re.sub(r'\s+', ' ', val)
                val = val.strip(' ,.-;')
                
                # Only add if meaningful
                if len(val) > 3:
                    result["raw_matches"].append({
                        "text": val,
                        "pattern": pattern
                    })
                    
                    # Try to extract count
                    count_match = re.search(r'(\d+)\s*X', val, re.IGNORECASE)
                    if count_match:
                        try:
                            result["count"] = int(count_match.group(1))
                        except (ValueError, TypeError):
                            pass
                    
                    # Extract type information
                    if 'REINFORCEMENT' in val:
                        result["types"].append('REINFORCEMENT')
                    if 'STAINLESS' in val:
                        result["types"].append('STAINLESS')
                    if 'WIRE' in val:
                        result["types"].append('WIRE')
        
        # Deduplicate types while preserving order
        result["types"] = list(dict.fromkeys(result["types"]))
        
        return result
        
    except Exception as e:
        logger.error(f"Error in extract_rings_info: {e}")
        return result

def extract_coordinates(text: str) -> List[Tuple[float, float]]:
    """
    Extract explicit coordinate pairs found in text. If none found, will try a
    safe fallback only when coordinate-related keywords exist nearby.
    """
    t = (text or "")
    coords = []
    for pat in _COORD_PATTERNS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            try:
                x = float(m.group(1))
                y = float(m.group(2))
                coords.append((x, y))
            except Exception:
                continue

    # Remove duplicates (rounded)
    uniq = []
    seen = set()
    for x, y in coords:
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            uniq.append((x, y))
    coords = uniq

    # Safe fallback: if no coords but coordinate-related keywords exist,
    # attempt to group nearby long numeric runs into pairs. ONLY do if keywords present.
    if not coords:
        if re.search(r'POINT|COORD|X\s*[:=]|Y\s*[:=]|DEVELOP|DEV LENGTH|POINT\s*\d+', t, flags=re.IGNORECASE):
            # find all numbers
            nums = re.findall(r'([+-]?\d+\.?\d*)', t)
            if len(nums) >= 4:  # at least two pairs
                # try to pair as (n0,n1),(n2,n3),...
                pairs = []
                try:
                    for i in range(0, len(nums)-1, 2):
                        x = float(nums[i])
                        y = float(nums[i+1])
                        pairs.append((x, y))
                    # Heuristic: accept only if at least 3 pairs to avoid false positives
                    if len(pairs) >= 3:
                        coords = pairs
                except Exception:
                    coords = []

    return coords

def polyline_length(coords: List[Tuple[float, float]]) -> Optional[float]:
    """Calculate the total length of a polyline from its coordinates."""
    if not coords or len(coords) < 2:
        return None
    total = 0.0
    for i in range(1, len(coords)):
        x0, y0 = coords[i-1]
        x1, y1 = coords[i]
        total += math.hypot(x1 - x0, y1 - y0)
    return total

# try to extract a printed diameter or phi symbol
_DIA_PATTERNS = [
    r'(?:DIA|DIAMETER|Ø|PHI|⌀)\s*[:=]?\s*([+-]?\d+\.?\d*)\s*(MM|IN|INCH|CM)?',  # e.g., DIA: 85 mm
    r'\bD\s*[:=]?\s*([+-]?\d+\.?\d*)\s*(MM|IN|CM)?\b'
]

def extract_diameter(text: str) -> Optional[Tuple[float, Optional[str]]]:
    """Extract diameter value and optional unit from text using various patterns."""
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

def development_length_from_diameter(dia_value: float, unit: Optional[str] = None) -> float:
    """Return circumference = pi * D. Unit left as-is."""
    return math.pi * dia_value

def extract_development_length(text: str) -> Optional[Tuple[float, Optional[str]]]:
    """
    Extract explicit development length value and unit from text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Optional tuple of (value, unit). Unit may be None if not specified.
    """
    t = (text or "").upper()
    for pat in _DEV_LENGTH_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                unit = (m.group(2) or '').upper().strip() or None
                return (val, unit)
            except Exception:
                continue
    # also try loose 'DEV LENGTH 123' without unit
    m = re.search(r'DEV(?:\.|EL)?\s*LENGTH\s*[:=]?\s*([+-]?\d+\.?\d*)', t, flags=re.IGNORECASE)
    if m:
        try:
            return (float(m.group(1)), None)
        except Exception:
            pass
    return None

def snippet_around(text: str, match_pat: str) -> Optional[str]:
    """Get text snippet around a matching pattern for context debugging."""
    m = re.search(match_pat, text, flags=re.IGNORECASE)
    if not m:
        return None
    start = max(0, m.start() - 120)
    end = min(len(text), m.end() + 120)
    return text[start:end]