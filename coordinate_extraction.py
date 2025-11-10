import re
import logging
from math import sqrt

_FLOAT_RE = re.compile(r'-?\d+(?:[.,]\d+)?')

def _to_float_token(tok):
    if tok is None:
        return None
    s = str(tok).strip()
    if ',' in s and '.' in s:
        s = s.replace(',', '')
    elif ',' in s:
        s = s.replace(',', '.')
    try:
        return float(s)
    except Exception:
        return None

def parse_coordinate_lines(text):
    """
    Robustly parse lines containing P0/P1... coords.
    Returns list of dicts: [{'point':'P0','x':..,'y':..,'z':..,'r':..,'raw_line':..},...]
    """
    coords = []
    for line in text.splitlines():
        if not line or line.strip() == '':
            continue
        # collect numeric tokens
        nums_raw = _FLOAT_RE.findall(line)
        nums = [_to_float_token(t) for t in nums_raw]
        nums = [n for n in nums if n is not None]
        # try to find a label at line start if present
        m_label = re.match(r'^(P\d+|POINT\s*\d+|P\s*\d+)\b', line, re.IGNORECASE)
        label = m_label.group(1).replace(' ', '') if m_label else None

        if len(nums) < 3:
            logging.debug("ignored malformed coord line (less than 3 numeric tokens): %r", line)
            continue

        x, y, z = nums[0], nums[1], nums[2]
        r = nums[3] if len(nums) >= 4 else None
        coords.append({'point': label, 'x': x, 'y': y, 'z': z, 'r': r, 'raw_line': line})
    return coords

def validate_coords_list(coords):
    """Ensure coords is list[dict] and each dict has numeric x,y. Return cleaned coords."""
    if not isinstance(coords, list):
        raise ValueError("coords must be a list")
    cleaned = []
    for i, c in enumerate(coords):
        if not isinstance(c, dict):
            logging.warning("Found non-dict coord at index %d: %r — skipping", i, c)
            continue
        x = c.get('x'); y = c.get('y')
        if x is None or y is None:
            logging.warning("Invalid coordinates: Missing x/y in point %s (raw: %r)", c.get('point'), c.get('raw_line'))
            continue
        try:
            c['x'] = float(x); c['y'] = float(y)
        except Exception:
            logging.warning("Invalid coordinates: Invalid x/y type in point %s, skipping. raw: %r", c.get('point'), c.get('raw_line'))
            continue
        cleaned.append(c)
    return cleaned

def polyline_length_from_coords(coords):
    if not coords or len(coords) < 2:
        return 0.0
    total = 0.0
    prev = coords[0]
    for c in coords[1:]:
        dx = c['x'] - prev['x']
        dy = c['y'] - prev['y']
        dz = 0.0
        if prev.get('z') is not None and c.get('z') is not None:
            try:
                dz = float(c['z']) - float(prev['z'])
            except Exception:
                dz = 0.0
        total += sqrt(dx*dx + dy*dy + dz*dz)
        prev = c
    return total

def safe_development_length(coords, centerline_length_mm=None, max_reasonable_mm=20000):
    coords = validate_coords_list(coords)
    computed = polyline_length_from_coords(coords)
    if computed <= 0 or computed is None:
        if centerline_length_mm:
            logging.warning("Computed development length non-positive (%.3f). Falling back to centerline_length=%.3f mm", computed, centerline_length_mm)
            return float(centerline_length_mm), True, "fallback_nonpositive"
        return None, True, "nonpositive_no_centerline"
    if computed > max_reasonable_mm:
        logging.warning("Implausible computed development length %.3f mm (> %d mm). Falling back to centerline_length=%s",
                        computed, max_reasonable_mm, centerline_length_mm)
        # log small sample for debug
        for i, c in enumerate(coords[:8]):
            logging.warning("coord[%d]: %s", i, c.get('raw_line', c))
        if centerline_length_mm:
            return float(centerline_length_mm), True, "fallback_too_large"
        return None, True, "too_large_no_centerline"
    return computed, False, "computed_ok"

def extract_coordinates_from_text(text):
    """
    Enhanced coordinate extraction for the specific PDF table format
    """
    if not text:
        return []

    # Try to find the coordinate section
    coord_pattern = r'COORDS\s+POINTS\s+(.*?)(?:\n\s*\n|\Z)'
    coord_section_match = re.search(coord_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if coord_section_match:
        coord_section = coord_section_match.group(1)
    else:
        # If no coords section found, use the whole text
        coord_section = text

    # Use the robust parser to handle coordinates
    coords = parse_coordinate_lines(coord_section)
    coords = validate_coords_list(coords)
    return coords