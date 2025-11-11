"""
Tolerance and burst pressure handling for MPAPS standards.
"""
import re
import logging
from typing import Optional, Dict, Any, Tuple, List
from mpaps_burst import normalize_grade, get_burst_pressure_from_table_iv

#######################
# MPAPS F-6032 Tables
#######################

# Burst pressure constant for MPAPS F-6032
MPAPS_F6032_BURST_PRESSURE_MPA = 2.0

# TABLE 1: DIMENSIONS TYPE I (Bulk Hose) - MPAPS F-6032
TABLE_1_DATA = [
    # Nominal ID, ID (mm), ID Tolerance, OD (mm), OD Tolerance
    ('5/32', 3.97, 0.4, 9.3, 0.6),
    ('3/16', 4.76, 0.4, 10.3, 0.6),
    ('7/32', 5.56, 0.4, 11.3, 0.6),
    ('1/4', 5.9, 0.4, 12.25, 0.6),
    ('5/16', 7.14, 0.4, 13.5, 0.6),
    ('3/8', 9.0, 0.4, 15.37, 0.6),
    ('1/2', 12.0, 0.58, 20.3, 0.8),
    ('5/8', 15.1, 0.79, 24.62, 0.8),
    ('3/4', 18.4, 0.79, 28.35, 0.8),
    ('1', 24.6, 0.79, 34.9, 1.0)
]

# Map from nominal ID to tolerances for F-6032 TABLE 1
F6032_TABLE1 = {
    row[1]: {  # Use ID mm as key
        'nominal_id_mm': row[1],
        'id_tolerance_mm': row[2],
        'nominal_od_mm': row[3],
        'od_tolerance_mm': row[4]
    }
    for row in TABLE_1_DATA
}

def apply_mpaps_f6032_dimensions(result: dict, id_value_mm: float):
    """
    Populate ID/OD nominal + tolerances for MPAPS F-6032 and ensure NO wall thickness tolerance.
    Mutates and returns result dict.
    """
    # find exact nominal match first (your existing logic may find within tolerance)
    closest = None
    for key, row in F6032_TABLE1.items():
        # if exact or within 0.5 mm
        if abs(float(key) - float(id_value_mm)) <= 0.5:
            closest = row
            break
    if closest is None:
        # fallback: find best approximate by difference
        # choose nearest nominal by abs diff
        candidate = min(F6032_TABLE1.items(), key=lambda kv: abs(kv[0] - float(id_value_mm)))
        closest = candidate[1]
    # populate result fields (use keys your pipeline expects)
    result['id_nominal_mm'] = float(closest['nominal_id_mm'])
    result['id_tolerance_mm'] = float(closest.get('id_tolerance_mm')) if closest.get('id_tolerance_mm') is not None else None
    result['od_nominal_mm'] = float(closest['nominal_od_mm'])
    result['od_tolerance_mm'] = float(closest.get('od_tolerance_mm')) if closest.get('od_tolerance_mm') is not None else None

    # CRITICAL: for F-6032, explicitly DO NOT set wall/thickness tolerance
    # There should be NO wall thickness tolerance for F-6032 per your requirement
    result['thickness_mm'] = result.get('thickness_mm') or None  # allow computed thickness if you want
    result['thickness_tolerance_mm'] = None  # explicit: no tolerance

    # Material will be looked up from material_data.csv through get_material_from_standard()
    # which correctly maps MPAPS F-6032 Type I to "INNER NBR OUTER:ECO"
    
    # also mark dimension source for traceability
    result['dimension_source'] = 'MPAPS F-6032 TABLE 1'
    return result

#######################
# MPAPS F-30/F-1 Tables 
#######################

# Maximum acceptable difference for nominal value matching (mm)
MAX_ACCEPT_DIFF_MM = 0.5  # Allow up to 0.5mm difference for practical measurements

def canonical_standard(standard_raw: str) -> str:
    """
    Return canonical standard identity strings:
      - Returns 'MPAPS F-6032', 'MPAPS F-30', or the cleaned input
      - Maps F-1 to F-30 (per Table 4 mapping)
    """
    if not standard_raw:
        return ''
    s = standard_raw.upper().replace('_', ' ').strip()
    # normalize common OCR variants
    s = re.sub(r'\s+', ' ', s)
    if 'F-6032' in s or 'F6032' in s:
        return 'MPAPS F-6032'
    if 'F-30' in s or 'F30' in s:
        return 'MPAPS F-30'
    if 'F-1' in s or 'F1' in s:
        # treat F-1 as F-30 (per your Table 4 mapping)
        return 'MPAPS F-30'
    return s

def process_mpaps_dimensions(result: dict) -> dict:
    """
    Populate MPAPS-driven nominal/tolerance fields and prefer F-30/F-1 table values
    for Grade 1 / 1BF hoses (ID tol = ±0.50 mm; wall tol = ±0.80 mm).
    Will only apply F-6032 rules if the canonical standard explicitly equals MPAPS F-6032.
    Mutates and returns result.
    """
    if not isinstance(result, dict):
        return result

    # normalize standard and grade
    std_raw = result.get('standard', '') or ''
    grade_raw = result.get('grade', '') or ''
    canonical = canonical_standard(std_raw)
    result['standard_canonical'] = canonical

    # try numeric ID
    id_val = result.get('id_nominal_mm')
    try:
        if id_val is not None:
            id_val = float(id_val)
            result['id_nominal_mm'] = id_val
    except Exception:
        id_val = None

    # 1) If standard is explicitly MPAPS F-6032 -> use TABLE 1 logic (no wall tol)
    if canonical == 'MPAPS F-6032' and id_val is not None:
        result = apply_mpaps_f6032_dimensions(result, id_val)
        return result

    # 2) If standard is MPAPS F-30 (or F-1 mapped to F-30), apply F-30/F-1 rules
    if canonical == 'MPAPS F-30' and id_val is not None:
        # normalize grade string e.g. "1BF", "1B", "1"
        g = (grade_raw or "").upper().replace(' ', '').replace('-', '')
        # handle "GRADE1BF" etc
        g = re.sub(r'GRADE', '', g)

        # If Grade 1 or 1BF -> use Grade1/BF table entry (TABLE 4 / TABLE 8 semantics)
        if g.startswith('1') or '1B' in g or '1BF' in g:
            # Use helper that already knows TABLE_8/ TABLE_4 semantics
            try:
                grade_entry = get_grade1bf_tolerances(id_val)
            except Exception:
                grade_entry = None

            if isinstance(grade_entry, dict):
                # ID tolerance -> ensure ±0.5 (use table value)
                id_tol = grade_entry.get('id_tolerance_mm')
                if id_tol is not None:
                    result['id_tolerance_mm'] = float(id_tol)

                # Wall thickness nominal -> set to table wall if missing
                wall_mm = grade_entry.get('wall_mm')
                if wall_mm is not None and (result.get('thickness_mm') in (None, '', 'Not Found')):
                    result['thickness_mm'] = float(wall_mm)

                # Wall thickness tolerance -> set to table wall_tol (±0.8)
                wall_tol = grade_entry.get('wall_tolerance_mm')
                if wall_tol is not None:
                    result['thickness_tolerance_mm'] = float(wall_tol)

                # If OD missing, compute OD = ID + 2*thickness
                if result.get('od_nominal_mm') in (None, '', 'Not Found'):
                    try:
                        if result.get('id_nominal_mm') is not None and result.get('thickness_mm') is not None:
                            result['od_nominal_mm'] = round(float(result['id_nominal_mm']) + 2.0 * float(result['thickness_mm']), 3)
                    except Exception:
                        pass

                # mark where values came from
                result.setdefault('dimension_sources', []).append('MPAPS F-30/F-1 TABLE 4/8 (Grade 1/BF)')

            return result

        # For other grades in F-30, you can implement TABLE III/IV as before.
        # (Keep existing F-30 burst-pressure logic, etc.)
        return result

    # 3) Fallback: not enough info or unknown standard -> return unchanged
    return result

def normalize_standard_and_grade(standard_raw: str, grade_raw: str):
    """
    Normalize strings from OCR/text so downstream lookups use canonical keys.
    Returns (standard_norm, grade_norm)
    """
    std = canonical_standard(standard_raw)
    if 'F-6032' in std or 'F6032' in std:
        std = 'MPAPS F-6032'
    elif 'F-30' in std or 'F30' in std:
        std = 'MPAPS F-30'
    elif 'F-1' in std or 'F1' in std:
        std = 'MPAPS F-1'
    else:
        std = std  # keep what we got
    # grade: remove extra spaces and OCR noise, convert "GRADE 1 B" -> "1B"
    g = (grade_raw or "").upper()
    g = re.sub(r'GRADE\s*', '', g)
    g = re.sub(r'\s+', '', g)  # 1 B -> 1B
    return std, g

# Burst pressure tables for MPAPS F-30/F-1
# Grade lookup guide for TABLE IV:
# - Grade 1BF: Use "Suffix F, Grade 1" column (column 6)
# - Grade 2B:  Use "Suffix B, Grade 2" column (column 3)
# - Grade 1B/3B: Use "Suffix B, Grade 1&3" column (column 2)
# - For Grade 1A/2A/3A: Use TABLE III instead (fixed values)
TABLE_IV_BURST_PRESSURE = [
    # Column indices and what they contain:
    # [0] Over (mm), [1] Thru (mm)
    # [2] Suffix B Grade 1&3: For grades 1B and 3B
    # [3] Suffix B Grade 2:   For grade 2B
    # [4] Suffix C Grade 1:   For grade 1C
    # [5] Suffix C Grade 2:   For grade 2C
    # [6] Suffix F Grade 1:   For grade 1BF
    # [7] Suffix F Grade 2:   For grade 2F
    (0, 24, 1.72, 1.72, 2.10, 1.72, 2.5, 2.25),   # e.g. for 15mm ID: 1BF->2.5 MPa, 2B->1.72 MPa
    (24, 32, 1.38, 1.21, 1.90, 1.38, 2.1, 2.25),  # e.g. for 25mm ID: 1B->1.38 MPa, 2B->1.21 MPa
    (32, 44, 1.21, 1.21, 1.72, 1.38, 2.1, 2.25),
    (44, 50.8, 1.00, 1.21, 1.38, 1.21, 1.55, 2.25),
    (51, 65, 0.90, 0.69, 1.38, 1.21, 1.38, 1.21),
    (65, 76, 0.62, 0.69, 1.38, 1.03, 1.38, 1.21),
    (76, 102, None, 0.55, 1.38, 0.86, 1.38, 1.03)
]

# TABLE 4: Grade 1 (EPDM/Premium) Formed Hose Dimensions - MPAPS F-30/F-1
TABLE_4_GRADE_1_DATA = [
    # (nominal_id_in, id_mm, id_tol_mm, wall_mm, wall_tol_mm)
    ('1/4', 5.9, 0.5, 4.95, 0.65),
    ('3/8', 9.0, 0.5, 4.95, 0.65),
    ('1/2', 12.0, 0.5, 4.95, 0.65),
    ('5/8', 15.1, 0.5, 4.95, 0.65),
    ('3/4', 18.4, 0.5, 4.95, 0.65),
    ('7/8', 21.3, 0.5, 4.95, 0.65),
    ('1', 24.6, 0.5, 4.95, 0.65),
    ('>1.0-2.0', None, 0.5, 4.95, 0.65),  # For ID range >25.4 - 50.8 mm
    ('>2.0-2.5', None, 0.5, 4.95, 0.65)   # For ID range >50.8 - 62.7 mm
]

    # Grade 1 (EPDM/Premium) ranges for IDs larger than 1 inch
TABLE_4_GRADE_1_RANGES = [
    (25.4, 50.8, 4.95, 0.65, 0.5),  # min_id_mm, max_id_mm, wall_mm, wall_tol_mm, id_tol_mm
    (50.8, 62.7, 4.95, 0.65, 0.5)
]

# TABLE 8: Grade 1 (EPDM/Premium) Suffix BF Hose Dimensions - MPAPS F-30/F-1
TABLE_8_GRADE_1BF_DATA = [
    # (nominal_id_in, id_mm, id_tol_mm, od_mm, wall_mm, wall_tol_mm)
    ('5/8', 15.1, 0.5, 28.3, 4.95, 0.195),
    ('3/4', 18.4, 0.5, 28.3, 4.95, 0.195),
    ('7/8', 21.3, 0.5, 29.7, 4.95, 0.195),
    ('1', 24.6, 0.5, 34.5, 4.95, 0.195),
    ('>1.0-2.0', None, 0.5, None, 5.35, 0.211),  # For ID range >25.4 - 50.8 mm
    ('>2.0-2.5', None, 0.5, None, 5.35, 0.211)   # For ID range >50.8 - 62.7 mm
]

# Grade 1BF ranges for IDs larger than 1 inch
TABLE_8_GRADE_1BF_RANGES = [
    (25.4, 50.8, 5.35, 0.211),  # min_id_mm, max_id_mm, wall_mm, wall_tol_mm
    (50.8, 62.7, 5.35, 0.211)
]

# ---------------------------
# Grade 1 BF tolerance helper
# ---------------------------
# Based on TABLE 8 (Grade 1BF Dimensions and tolerances)
GRADE_1_BF_TOLERANCE_ENTRIES = [
    # (nominal_in, actual_id_mm, id_tol_mm, wall_mm, wall_tol_mm)
    ('5/8',  15.1, 0.5, 4.95, 0.8),
    ('3/4',  18.4, 0.5, 4.95, 0.8),
    ('7/8',  21.3, 0.5, 4.95, 0.8),
    ('1',    24.6, 0.5, 4.95, 0.8),
    ('>1.0-2.0', 38.4, 0.5, 4.95, 0.8),  # For ID range >26 to <50.8 mm
    ('2.0-2.5', 56.6, 0.5, 5.35, 0.8)     # For ID range ≥50.8 to <63.5 mm
]

def get_grade1bf_tolerances(id_value_mm: float) -> dict:
    """
    Return a tolerance record for Grade 1 (EPDM/PREMIUM) SUFFIX BF hose
    using TABLE 8 logic with exact matches and ranges.
    Output keys:
      - nominal_in: Nominal size in inches or range
      - nominal_id_mm: Actual inside diameter representative
      - id_tolerance_mm: ID tolerance (±mm)
      - wall_mm: Wall thickness (mm)
      - wall_tolerance_mm: Wall thickness tolerance (±mm)
      - difference_mm: Absolute difference between id_value_mm and nominal_id_mm
    
    TABLE 8 Reference:
    | Actual ID (mm) | ID Tolerance (±mm) | Wall Thickness (mm) | Wall Thickness Tolerance (±mm) |
    |---------------|-------------------|-------------------|------------------------------|
    | 15.1 | 0.5 | 4.95 | 0.8 |
    | 18.4 | 0.5 | 4.95 | 0.8 |
    | 21.3 | 0.5 | 4.95 | 0.8 |
    | 24.6 | 0.5 | 4.95 | 0.8 |
    | >26 <50.8 | 0.5 | 4.95 | 0.8 |
    | ≥50.8 <63.5 | 0.5 | 5.35 | 0.8 |
    """
    try:
        id_val = float(id_value_mm)
    except Exception:
        logging.error(f"Invalid ID value: {id_value_mm}")
        return {
            'nominal_in': None,
            'nominal_id_mm': None,
            'id_tolerance_mm': None,
            'wall_mm': None,
            'wall_tolerance_mm': None,
            'difference_mm': None
        }

    # Handle special ranges first
    if 50.8 <= id_val < 63.5:
        return {
            'nominal_in': '2.0-2.5',
            'nominal_id_mm': id_val,  # Use actual value since it's in range
            'id_tolerance_mm': 0.5,
            'wall_mm': 5.35,
            'wall_tolerance_mm': 0.8,
            'difference_mm': 0.0
        }
    elif 26.0 <= id_val < 50.8:
        return {
            'nominal_in': '>1.0-2.0',
            'nominal_id_mm': id_val,  # Use actual value since it's in range
            'id_tolerance_mm': 0.5,
            'wall_mm': 4.95,
            'wall_tolerance_mm': 0.8,
            'difference_mm': 0.0
        }

    # find nearest entry by absolute difference for standard sizes
    best = None
    best_diff = float('inf')
    for entry in GRADE_1_BF_TOLERANCE_ENTRIES:
        nominal_in, nominal_id_mm, id_tol, wall_mm, wall_tol = entry
        if nominal_id_mm is None:
            continue
        diff = abs(nominal_id_mm - id_val)
        if diff < best_diff:
            best_diff = diff
            best = {
                'nominal_in': nominal_in,
                'nominal_id_mm': nominal_id_mm,
                'id_tolerance_mm': id_tol,
                'wall_mm': wall_mm,
                'wall_tolerance_mm': wall_tol,
                'difference_mm': diff
            }

    return best or {
        'nominal_in': None,
        'nominal_id_mm': None,
        'id_tolerance_mm': None,
        'wall_mm': None,
        'wall_tolerance_mm': None,
        'difference_mm': None
    }

# TABLE 8-A: Grade 2 (Silicone) Suffix BF Hose Dimensions - MPAPS F-30/F-1
TABLE_8A_GRADE_2BF_DATA = [
    # (nominal_id_in, id_mm, id_tol_mm, wall_mm, wall_tol_mm)
    ('1/2', 12.0, 0.7, 6.35, 0.95),
    ('5/8', 15.1, 0.7, 6.35, 0.95),
    ('3/4', 18.4, 0.8, 6.35, 0.95),
    ('7/8', 21.3, 0.5, 6.35, 0.95),
    ('1', 24.6, 0.8, 7.3, 1.10),
    ('>1.0-2.5', None, 0.8, 7.3, 1.10)  # For ID range >25.4 - 63.5 mm
]

# Grade 2BF range for IDs larger than 1 inch
TABLE_8A_GRADE_2BF_RANGES = [
    (25.4, 63.5, 7.3, 1.10)  # min_id_mm, max_id_mm, wall_mm, wall_tol_mm
]

# TABLE 4-A: Grade 2 (Silicone) Formed Hose Dimensions - MPAPS F-30/F-1
TABLE_4A_GRADE_2_DATA = [
    # (nominal_id_in, id_mm, id_tol_mm, wall_mm, wall_tol_mm)
    ('1/4', 5.9, 0.5, 4.95, 0.65),
    ('3/8', 9.0, 0.5, 4.95, 0.65),
    ('1/2', 12.0, 0.5, 4.95, 0.65),
    ('5/8', 15.1, 0.5, 4.95, 0.65),
    ('3/4', 18.4, 0.5, 4.95, 0.65),
    ('7/8', 21.3, 0.5, 4.95, 0.65),
    ('1', 24.6, 0.5, 4.95, 0.65),
    ('>1.0 - 2.0', None, 0.5, 4.95, 0.65),  # For ID range >25.4 - 50.8 mm
    ('>2.0', None, 0.5, 4.95, 0.65)         # For ID range >50.8 - 62.7 mm
]

# Grade 2 (Silicone) ranges for IDs larger than 1 inch
TABLE_4A_GRADE_2_RANGES = [
    (25.4, 50.8, 4.95, 0.65),  # min_id_mm, max_id_mm, wall_mm, wall_tol_mm
    (50.8, 62.7, 4.95, 0.65)
]

# TABLE 8: Grade 1 (EPDM/Premium) Suffix BF Hose Dimensions (MPAPS F-30/F-1)
TABLE_8A_GRADE_2BF_DATA = [
    # (nominal ID inches, ID mm, ID tol mm, wall mm, wall tol mm)
    ('1/2', 12.0, 0.7, 6.35, 0.95),
    ('5/8', 15.1, 0.7, 6.35, 0.95),
    ('3/4', 18.4, 0.8, 6.35, 0.95),
    ('7/8', 21.3, 0.5, 6.35, 0.95),
    ('1', 24.6, 0.8, 7.3, 1.10),
    ('>1.0-2.5', None, 0.8, 7.3, 1.10)     # For ID range >26-63.5 mm
]

# Grade 2BF ranges for IDs larger than 1 inch
# TABLE 8A_GRADE_2BF_RANGES = [
# (26.0, 63.5, 7.3, 1.10)   # min_id_mm, max_id_mm, wall_mm, wall_tol_mm
# ]

#######################
# Burst Pressure Functions 
#######################

def lookup_burst_pressure(standard_norm: str, grade_norm: str, id_mm: float) -> Optional[float]:
    """
    Unified lookup that routes to the correct table based on standard_norm.
    Returns burst_pressure_MPa (float) or None if not found.
    """
    # prefer MPAPS F-30 / F-1 rules if standard includes either
    if 'F-30' in standard_norm or 'F-1' in standard_norm:
        logging.info(f"Looking up burst pressure in TABLE IV for standard={standard_norm}, grade={grade_norm}, id={id_mm}")
        return get_burst_pressure_from_table_iv(id_mm, grade_norm)
    # fallback to F-6032 table lookup
    if 'F-6032' in standard_norm:
        logging.info(f"Using F-6032 default burst pressure for standard={standard_norm}")
        return MPAPS_F6032_BURST_PRESSURE_MPA
    # last resort: None
    logging.warning(f"No burst pressure table found for standard={standard_norm}")
    return None
        
def get_burst_pressure_from_table_iv(id_value_mm: float, grade_norm: str) -> Optional[float]:
    """
    Get burst pressure from TABLE IV based on ID and grade.
    Returns float MPa or None.
    
    Args:
        id_value_mm: ID in mm
        grade_norm: Normalized grade string (e.g., "1B", "2B", etc.)
        
    Returns:
        Burst pressure in MPa or None if not found
    """
    logging.info(f"Looking up burst pressure for ID {id_value_mm}mm with grade {grade_norm}")
    
    # Find the matching ID range in TABLE IV
    for min_id, max_id, b_1_3, b_2, c_1, c_2, f_1, f_2 in TABLE_IV_BURST_PRESSURE:
        # inclusive with tiny epsilon
        if (min_id - 1e-6) <= id_value_mm <= (max_id + 1e-6):
            logging.info(f"Found ID range {min_id}-{max_id}mm in TABLE IV")
            # Select correct column based on grade
            if any(g in grade_norm for g in ['1B', 'B1', '3B', 'B3']):
                return b_1_3  # Suffix B Grade 1&3 column
            elif any(g in grade_norm for g in ['2B', 'B2']):
                return b_2   # Suffix B Grade 2 column
            elif any(g in grade_norm for g in ['1BF', 'BF1']):
                return f_1   # Suffix F Grade 1 column
                
    logging.warning(f"No matching ID range found for {id_value_mm}mm in TABLE IV")
    return None
    logging.info(f"Looking up burst pressure for ID {id_value}mm with grade {grade_str}")
    for min_id, max_id, b_1_3, b_2, c_1, c_2, f_1, f_2 in TABLE_IV_BURST_PRESSURE:
        if min_id < id_value <= max_id:
            logging.info(f"Found ID range {min_id}-{max_id}mm in TABLE IV")
            # Select correct column based on grade
            if any(g in grade_str for g in ['1B', 'B1', '3B', 'B3']):
                return b_1_3  # Suffix B Grade 1&3 column
            elif any(g in grade_str for g in ['2B', 'B2']):
                return b_2   # Suffix B Grade 2 column
            elif any(g in grade_str for g in ['1BF', 'BF1']):
                return f_1   # Suffix F Grade 1 column
    
    logging.warning(f"No matching ID range found for {id_value}mm in TABLE IV")            
    return None

def get_burst_pressure() -> float:
    """Get the standard burst pressure for MPAPS F-6032 materials."""
    return MPAPS_F6032_BURST_PRESSURE_MPA

def get_mpaps_f6032_dimensions_from_table(id_value: float) -> Optional[Dict[str, Any]]:
    """
    Get MPAPS F-6032 dimensions and tolerances from TABLE 1.
    Uses only the mm values in square brackets.
    
    Args:
        id_value: Inside diameter in mm
        
    Returns:
        Dict with dimension info or None if not found
    """
    try:
        id_val = float(id_value)
        logging.info(f"MPAPS F-6032 dimension lookup for ID: {id_val}mm")
        
        # Find the closest nominal ID
        closest_match = None
        min_diff = float('inf')
        
        for row in TABLE_1_DATA:
            nominal_id, nominal_id_mm, id_tol, nominal_od_mm, od_tol = row
            diff = abs(nominal_id_mm - id_val)
            
            if diff < min_diff:
                min_diff = diff
                closest_match = {
                    'nominal_id_inches': nominal_id,
                    'nominal_id_mm': nominal_id_mm,
                    'id_tolerance_mm': id_tol,
                    'nominal_od_mm': nominal_od_mm,
                    'od_tolerance_mm': od_tol,
                    'id_formatted': f"{nominal_id_mm:.2f} ± {id_tol:.2f} mm",
                    'od_formatted': f"{nominal_od_mm:.2f} ± {od_tol:.2f} mm",
                    'difference_mm': diff
                }
        
        if closest_match and min_diff <= MAX_ACCEPT_DIFF_MM:
            logging.info(f"Found MPAPS F-6032 dimension match: {closest_match}")
            logging.info(f"Exact match found within {MAX_ACCEPT_DIFF_MM}mm tolerance (diff={min_diff:.3f}mm)")
            return closest_match
        elif closest_match:
            logging.warning(f"Nearest MPAPS F-6032 nominal for ID {id_val}mm is {min_diff:.3f}mm away (tolerance={MAX_ACCEPT_DIFF_MM}mm)")
            logging.info(f"Using nearest match: {closest_match}")
            logging.debug(f"Match details - Nominal: {closest_match['nominal_id_mm']}mm, ID Tol: ±{closest_match['id_tolerance_mm']}mm")
            return closest_match
        else:
            logging.warning(f"No MPAPS F-6032 dimension match found for ID {id_val}mm")
            return None
            
    except Exception as e:
        logging.error(f"Error in MPAPS F-6032 dimension lookup: {e}")
        return None

def apply_mpaps_f6032_rules(results: Dict[str, Any]) -> None:
    """
    Apply ONLY MPAPS F-6032 rules to analysis results.
    Uses TABLE 1 for dimensions and tolerances, and fixed 2.0 MPa burst pressure.
    """
    # Check multiple fields for MPAPS F-6032 indication
    material = results.get('material')
    standard = results.get('standard')
    specification = results.get('specification')
    
    # Only apply if it's MPAPS F-6032 - use strict checking
    is_f6032 = False
    
    if standard:
        std_upper = str(standard).upper()
        if 'MPAPS F-6032' in std_upper or 'MPAPSF6032' in std_upper:
            is_f6032 = True
    
    if material and not is_f6032:  # Only check material if standard didn't match
        mat_upper = str(material).upper()
        if 'MPAPS F-6032' in mat_upper or 'MPAPSF6032' in mat_upper:
            is_f6032 = True
    
    if not is_f6032:
        return  # Exit early if not F-6032
        
    logging.info("Applying MPAPS F-6032 rules to results")
    
    # FORCEFULLY set dimension source to ensure F-6032 rules take precedence
    results['dimension_source'] = "MPAPS F-6032 TABLE 1"
    logging.info("Set dimension_source to MPAPS F-6032 TABLE 1")
    
    logging.info("DEBUG MPAPS F-6032: standard=%r, material=%r, dimension_source=%r",
                 results.get('standard'),
                 results.get('material'),
                 results.get('dimension_source'))
    logging.info("DEBUG DIMENSIONS: %r", results.get('dimensions', {}))
    
    # Get dimensions from all possible locations
    dimensions = results.get('dimensions', {})
    
    # Check for ID in multiple locations with type conversion
    id_val = None
    for id_key in ['id1', 'ID1', 'ID', 'id']:
        val = dimensions.get(id_key) or results.get(id_key)
        if val and str(val).strip().lower() != 'not found':
            try:
                if isinstance(val, str):
                    val_clean = re.sub(r'[^\d.-]', '', val)
                    id_val = float(val_clean)
                else:
                    id_val = float(val)
                logging.info(f"Found valid ID value {id_val} from key {id_key}")
                break
            except (ValueError, TypeError) as e:
                logging.warning(f"Failed to parse ID value '{val}': {e}")
                continue
    
    # fallback: try to parse from raw_text if id_val is None
    if id_val is None:
        raw_text = results.get('raw_text') or results.get('ocr_text') or results.get('text')
        if raw_text:
            m = re.search(r'HOSE\s+ID\s*[=:]?\s*([\d.]+)', raw_text, re.IGNORECASE)
            if m:
                try:
                    id_val = float(m.group(1))
                    logging.info(f"Fallback parsed ID from raw text: {id_val}mm")
                except Exception as e:
                    logging.warning(f"Fallback parse failed for ID value {m.group(1)}: {e}")
    
    # For MPAPS F-6032, use TABLE 1 for dimensions and tolerances
    if id_val is not None:
        logging.info(f"Processing MPAPS F-6032 dimensions for ID: {id_val}mm")
        
        # Get dimensions from TABLE 1
        table_data = get_mpaps_f6032_dimensions_from_table(id_val)
        
        if table_data:
            # Set ID tolerance from TABLE 1
            results['id_tolerance'] = table_data['id_formatted']
            logging.info(f"Set ID tolerance from TABLE 1: {table_data['id_formatted']}")
            
            # Set OD tolerance from TABLE 1
            results['od_tolerance'] = table_data['od_formatted']
            logging.info(f"Set OD tolerance from TABLE 1: {table_data['od_formatted']}")
            
            # Update dimensions with nominal values from TABLE 1
            dimensions['id1'] = table_data['nominal_id_mm']
            dimensions['id2'] = table_data['nominal_id_mm']
            dimensions['od1'] = table_data['nominal_od_mm'] 
            dimensions['od2'] = table_data['nominal_od_mm']
            
            logging.info(f"Updated dimensions with TABLE 1 nominals: ID={table_data['nominal_id_mm']}mm, OD={table_data['nominal_od_mm']}mm")
            
            # Add nominal reference
            results['nominal_id_inches'] = table_data['nominal_id_inches']
            results['nominal_id_mm'] = table_data['nominal_id_mm']
            results['dimension_source'] = "MPAPS F-6032 TABLE 1"
        else:
            results['id_tolerance'] = "N/A"
            results['od_tolerance'] = "N/A"
            logging.warning(f"Could not find MPAPS F-6032 dimensions for ID: {id_val}mm")
    else:
        results['id_tolerance'] = "N/A"
        results['od_tolerance'] = "N/A"
        logging.warning("No valid ID value found for MPAPS F-6032 dimension calculation")
            
    # After dimension match, ensure tolerances are set
    if table_data:
        result = {}
        result['id_nominal_mm'] = table_data['nominal_id_mm']
        result['id_tolerance_mm'] = table_data.get('id_tolerance_mm')
        
        # Handle OD tolerance
        if 'nominal_od_mm' in table_data and table_data['nominal_od_mm'] != 'Not Found':
            result['od_nominal_mm'] = table_data['nominal_od_mm']
            result['od_tolerance_mm'] = table_data.get('od_tolerance_mm', 1.0)  # fallback 1.0 mm
            
        # Handle thickness
        if table_data.get('thickness') and table_data['thickness'] != 'Not Found':
            result['thickness_mm'] = table_data['thickness']
            result['thickness_tolerance_mm'] = table_data.get('thickness_tolerance_mm', 0.25)
        else:
            # Try compute from OD & ID
            if result.get('od_nominal_mm') and result.get('id_nominal_mm'):
                try:
                    t = (float(result['od_nominal_mm']) - float(result['id_nominal_mm'])) / 2.0
                    result['thickness_mm'] = round(t, 3)
                    result['thickness_tolerance_mm'] = 0.25
                except Exception:
                    logging.warning("Failed to compute thickness from OD-ID")
                    result['thickness_mm'] = None
                    result['thickness_tolerance_mm'] = None
                    
        # Update results with computed values
        results.update(result)
        logging.info("Set tolerances from TABLE 1 or computed values")
    else:
        logging.warning("No dimension match found - using fallback tolerances")
        # Set fallback tolerances rather than N/A
        if id_val is not None:
            results['id_tolerance_mm'] = 0.79  # Default from TABLE 1
            results['od_tolerance_mm'] = 1.0
            results['thickness_tolerance_mm'] = 0.25
        
    # Now look up burst pressure based on normalized standard and grade
    standard_norm, grade_norm = normalize_standard_and_grade(results.get('standard', ''), results.get('grade', ''))
    logging.info(f"Normalized standard={standard_norm}, grade={grade_norm}")
    
    if id_val is not None:
        burst = lookup_burst_pressure(standard_norm, grade_norm, id_val)
        if burst is not None:
            results['burst_pressure_mpa'] = burst
            results['burst_pressure_psi'] = round(burst * 145.038, 2)
            results['burst_pressure'] = burst * 10  # Convert to bar
            results['burst_pressure_source'] = f"{standard_norm} TABLE IV ({burst} MPa)"
            logging.info(f"Set burst pressure from {standard_norm} lookup: {burst} MPa")
        else:
            logging.warning(f"No burst pressure found for standard={standard_norm}, grade={grade_norm}, id={id_val}mm")
            results['burst_pressure_mpa'] = None
            results['burst_pressure_psi'] = None
            results['burst_pressure'] = None
            results['burst_pressure_source'] = "No matching burst pressure found"

#######################
# MPAPS F-30/F-1 Functions
#######################

def apply_mpaps_f30_f1_rules(results: Dict[str, Any]) -> None:
    """
    Apply ONLY MPAPS F-30/F-1 rules to analysis results.
    Uses TABLE III and IV for burst pressure and existing tolerance rules.
    """
    standard = results.get('standard', '')
    grade = results.get('grade', '')
    
    # Only apply if it's MPAPS F-30 or F-1 - use strict checking
    is_f30_f1 = False
    
    if standard:
        std_upper = str(standard).upper()
        if any(pat in std_upper for pat in ['MPAPS F-30', 'MPAPS F-1', 'MPAPSF30']):
            is_f30_f1 = True
    
    if not is_f30_f1:
        return  # Exit early if not F-30/F-1
        
    logging.info("Applying MPAPS F-30/F-1 rules (TABLE III/IV burst pressure)")
    
    # Clear MPAPS F-6032 tolerances only when appropriate
    if 'id_tolerance' in results and results.get('dimension_source') == "MPAPS F-6032 TABLE 1":
        # clear only if current standard is definitely F-30/F-1 and NOT F-6032
        std = str(results.get('standard','')).upper()
        if ('MPAPS F-30' in std or 'MPAPS F-1' in std) and 'F-6032' not in std:
            results['id_tolerance'] = "N/A"
            results['od_tolerance'] = "N/A"
            logging.info("Cleared MPAPS F-6032 tolerances because standard is F-30/F-1")
        else:
            logging.info("Retaining MPAPS F-6032 tolerances (dimension_source indicates TABLE 1)")
            
    # Set F-30/F-1 burst pressure from TABLE IV if available
    dimensions = results.get('dimensions', {})
    id_val = None
    
    # Get ID value for burst pressure lookup
    for id_key in ['id1', 'ID1', 'ID', 'id']:
        val = dimensions.get(id_key) or results.get(id_key)
        if val and str(val).strip().lower() != 'not found':
            try:
                if isinstance(val, str):
                    val_clean = re.sub(r'[^\d.-]', '', val)
                    id_val = float(val_clean)
                else:
                    id_val = float(val)
                logging.info(f"Found valid ID value {id_val} from key {id_key}")
                break
            except (ValueError, TypeError) as e:
                logging.warning(f"Failed to parse ID value '{val}': {e}")
                continue
    
    # Get ID value for burst pressure lookup
    dimensions = results.get('dimensions', {})
    id_val = None
    
    for id_key in ['id1', 'ID1', 'ID', 'id']:
        val = dimensions.get(id_key) or results.get(id_key)
        if val and str(val).strip().lower() != 'not found':
            try:
                if isinstance(val, str):
                    val_clean = re.sub(r'[^\d.-]', '', val)
                    id_val = float(val_clean)
                else:
                    id_val = float(val)
                logging.info(f"Found valid ID value {id_val} from key {id_key}")
                break
            except (ValueError, TypeError) as e:
                logging.warning(f"Failed to parse ID value '{val}': {e}")
                continue
    
    if id_val is None:
        logging.warning("No valid ID value found for burst pressure calculation")
        return
        
    # Apply burst pressure based on grade and ID using TABLE III/IV
    burst_pressure = get_burst_pressure_from_tables(grade, id_val)
    if burst_pressure is not None:
        results['burst_pressure_mpa'] = burst_pressure
        results['burst_pressure_psi'] = round(burst_pressure * 145.038, 2)
        results['burst_pressure'] = burst_pressure * 10  # Convert to bar
        results['burst_pressure_source'] = f"MPAPS F-30/F-1 TABLE {'III' if any(g in str(grade).upper() for g in ['1A', '2A', '3A']) else 'IV'}"
        logging.info(f"Set burst pressure from {results['burst_pressure_source']}: {burst_pressure} MPa")
    else:
        logging.warning(f"Could not determine burst pressure for Grade {grade} with ID {id_val}mm")

def get_burst_pressure_from_tables(grade: str, id_mm: float) -> Optional[float]:
    """
    Get burst pressure from MPAPS F-30/F-1 TABLE III or IV based on grade and ID.
    TABLE III has fixed values for suffix A grades.
    TABLE IV is used for suffix B, C, and F grades with values based on ID ranges.
    
    Args:
        grade: Material grade (e.g., '1A', '2B', '1BF', etc.)
        id_mm: Inside diameter in millimeters
        
    Returns:
        Burst pressure in MPa or None if not found
    """
    try:
        if not grade or not id_mm:
            return None
            
        grade_upper = str(grade).upper()
        
        # TABLE III: Fixed values for suffix A grades
        if any(g in grade_upper for g in ['1A', '2A', '3A']):
            # TABLE III values in MPa
            if '1A' in grade_upper:
                return 2.75  # Grade 1A burst pressure
            elif '2A' in grade_upper:
                return 2.06  # Grade 2A burst pressure
            elif '3A' in grade_upper:
                return 2.75  # Grade 3A burst pressure
            return None
        
        # TABLE IV: Value ranges for suffix B, C, and F grades
        for row in TABLE_IV_BURST_PRESSURE:
            over, thru = row[0], row[1]
            
            if over < id_mm <= thru:
                if any(g in grade_upper for g in ['1BF', 'BF']):
                    return row[6]  # Suffix F Grade 1 (1BF)
                elif '2F' in grade_upper:
                    return row[7]  # Suffix F Grade 2
                elif any(g in grade_upper for g in ['1B', '3B', 'GRADE IB', 'IB', 'GRADE 1B']):
                    return row[2]  # Suffix B Grade 1&3
                elif any(g in grade_upper for g in ['2B', 'GRADE 2B']):
                    return row[3]  # Suffix B Grade 2
                elif '1C' in grade_upper:
                    return row[4]  # Suffix C Grade 1
                elif '2C' in grade_upper:
                    return row[5]  # Suffix C Grade 2
                
        logging.warning(f"No matching ID range found for {id_mm}mm in TABLE IV")
        return None
            
    except Exception as e:
        logging.error(f"Error in burst pressure lookup: {e}")
        return None

def is_grade_1bf(grade: str) -> bool:
    """
    Check if a grade string indicates Grade 1BF.
    
    Args:
        grade: Grade string to check
        
    Returns:
        True if grade indicates 1BF, False otherwise
    """
    if not grade:
        return False
        
    grade_upper = str(grade).upper()
    return '1BF' in grade_upper.replace(' ', '') or 'GRADE1BF' in grade_upper.replace(' ', '') or 'GRADE1-BF' in grade_upper.replace(' ', '')
    return any(g in grade_upper for g in ['1BF', 'BF'])

def apply_grade_1bf_rules(results: Dict[str, Any]) -> None:
    """
    Apply Grade 1BF specific rules to the results using TABLE 8 dimensions and tolerances.
    
    TABLE 8 Reference:
    | Actual ID (mm) | ID Tolerance (±mm) | Wall Thickness (mm) | Wall Thickness Tolerance (±mm) |
    |---------------|-------------------|-------------------|------------------------------|
    | 15.1 | 0.5 | 4.95 | 0.8 |
    | 18.4 | 0.5 | 4.95 | 0.8 |
    | 21.3 | 0.5 | 4.95 | 0.8 |
    | 24.6 | 0.5 | 4.95 | 0.8 |
    | >26 <50.8 | 0.5 | 4.95 | 0.8 |
    | ≥50.8 <63.5 | 0.5 | 5.35 | 0.8 |
    """
    grade = results.get('grade', '')
    if not is_grade_1bf(grade):
        return  # Exit early if not Grade 1BF
        
    logging.info("Applying Grade 1BF rules to results")
    
    # Get dimensions
    dimensions = results.get('dimensions', {})
    id_val = None
    
    for id_key in ['id1', 'ID1', 'ID', 'id']:
        val = dimensions.get(id_key) or results.get(id_key)
        if val and str(val).strip().lower() != 'not found':
            try:
                if isinstance(val, str):
                    val_clean = re.sub(r'[^\d.-]', '', val)
                    id_val = float(val_clean)
                else:
                    id_val = float(val)
                logging.info(f"Found valid ID value {id_val} from key {id_key}")
                break
            except (ValueError, TypeError) as e:
                logging.warning(f"Failed to parse ID value '{val}': {e}")
                continue
    
    if id_val is not None:
        logging.info(f"Processing Grade 1BF dimensions for ID: {id_val}mm")
        
        # Get matching tolerances from TABLE 8
        tol_rec = get_grade1bf_tolerances(id_val)
        logging.info(f"Grade 1BF tolerance lookup result: {tol_rec}")
        if tol_rec and tol_rec['id_tolerance_mm'] is not None:
            # Set ID tolerance with exact value for ranges or nominal value for standard sizes
            id_display = id_val if tol_rec['nominal_in'] in ['>1.0-2.0', '2.0-2.5'] else tol_rec['nominal_id_mm']
            results['id_tolerance'] = f"± {tol_rec['id_tolerance_mm']:.2f} mm"  # TABLE 8 format
            
            # Set wall thickness and tolerance
            results['wall_thickness'] = tol_rec['wall_mm']
            results['wall_thickness_tolerance'] = f"± {tol_rec['wall_tolerance_mm']:.2f} mm"
            
            # Calculate OD based on ID and wall thickness
            od_mm = id_display + (2 * tol_rec['wall_mm'])
            results['od_reference'] = od_mm
            results['od_tolerance'] = f"± {tol_rec['wall_tolerance_mm']:.2f} mm"
            
            # Log the matched details
            logging.info(f"TABLE 8 match: {tol_rec['nominal_in']} " +
                        f"(ID={id_display:.2f}±{tol_rec['id_tolerance_mm']:.2f}mm, " +
                        f"Wall={tol_rec['wall_mm']:.2f}±{tol_rec['wall_tolerance_mm']:.2f}mm)")
        else:
            logging.warning(f"No Grade 1BF dimension match found for ID {id_val}mm")
    else:
        logging.warning("No valid ID value found for Grade 1BF dimension calculation")

def apply_grade_1b_rules(results: Dict[str, Any]) -> None:
    """
    Apply Grade 1B specific rules to the results.
    Uses TABLE 4 for dimensions and tolerances.
    """
    grade = results.get('grade', '')
    if not grade:
        return False
        
    # Normalize grade string
    grade_str = re.sub(r'\s+', '', str(grade).upper())  # Remove spaces
    grade_str = re.sub(r'GRADE', '', grade_str)  # Remove "GRADE"
    
    if not (any(g in grade_str for g in ['1B', 'B1']) or
            '1B' in grade_str):
        logging.info(f"Not a Grade 1B specification: {grade}")
        return  # Exit early if not Grade 1B
        
    logging.info("Applying Grade 1B rules to results")
    logging.info(f"Processing Grade 1B with grade string: {grade}")

    # Get dimensions and burst pressure from TABLE IV
    dimensions = results.get('dimensions', {})
    id_val = None
    
    for id_key in ['id1', 'ID1', 'ID', 'id']:
        val = dimensions.get(id_key) or results.get(id_key)
        if val and str(val).strip().lower() != 'not found':
            try:
                if isinstance(val, str):
                    val_clean = re.sub(r'[^\d.-]', '', val)
                    id_val = float(val_clean)
                else:
                    id_val = float(val)
                logging.info(f"Found valid ID value {id_val} from key {id_key}")
                break
            except (ValueError, TypeError) as e:
                logging.warning(f"Failed to parse ID value '{val}': {e}")
                continue
    
    if id_val is not None:
        logging.info(f"Processing Grade 1B dimensions for ID: {id_val}mm")
        
        # Look up burst pressure in TABLE IV
        burst_pressure = get_burst_pressure_from_table_iv(id_val, grade)
        if burst_pressure is not None:
            results['burst_pressure_mpa'] = burst_pressure
            results['burst_pressure_psi'] = round(burst_pressure * 145.038, 2)
            results['burst_pressure'] = burst_pressure * 10  # Convert to bar
            results['burst_pressure_source'] = f"MPAPS F-30/F-1 TABLE IV ({burst_pressure} MPa)"
            logging.info(f"Set burst pressure from TABLE IV: {burst_pressure} MPa")
        
        # First check ranges since they take precedence over exact matches
        match_found = False
        logging.info(f"Checking TABLE 4 ranges first for ID {id_val}mm")
        
        for min_id, max_id, wall_mm, wall_tol, id_tol in TABLE_4_GRADE_1_RANGES:
            logging.info(f"Checking range {min_id}-{max_id}mm")
            if min_id < id_val <= max_id:  # Use < for min to match ">X" ranges
                results['wall_thickness'] = wall_mm
                results['wall_thickness_tolerance'] = f"± {wall_tol:.2f} mm"
                results['id_tolerance'] = f"± {id_tol:.2f} mm"
                
                # Calculate OD based on actual ID and wall thickness for range entries
                od_mm = id_val + (2 * wall_mm)
                results['od_reference'] = od_mm
                results['od_tolerance'] = f"± {wall_tol:.2f} mm"  # OD tolerance matches wall tolerance
                
                logging.info(f"Found matching range in TABLE 4: {min_id}-{max_id}mm")
                logging.info(f"Setting tolerances: ID={id_val:.2f}±{id_tol:.2f}mm, Wall={wall_mm:.2f}±{wall_tol:.2f}mm")
                match_found = True
                break
                
        # Only check exact matches if no range match found
        if not match_found:
            logging.info(f"No range match, checking TABLE 4 exact matches for ID {id_val}mm")
            
            for row in TABLE_4_GRADE_1_DATA:
                nom_in, nom_mm, id_tol, wall_mm, wall_tol = row
                
                # Skip range entries (which have None for nom_mm)
                if nom_mm is None:
                    continue
                    
                diff = abs(nom_mm - id_val)
                logging.info(f"Comparing with nominal {nom_in}\" ({nom_mm}mm) - difference: {diff:.3f}mm (max allowed: {MAX_ACCEPT_DIFF_MM}mm)")
                
                if diff <= MAX_ACCEPT_DIFF_MM:
                    results['id_tolerance'] = f"{nom_mm:.2f} ± {id_tol:.2f} mm"
                    results['wall_thickness'] = wall_mm
                    results['wall_thickness_tolerance'] = f"± {wall_tol:.2f} mm"
                    
                    # Calculate OD from ID and wall thickness
                    od_mm = nom_mm + (2 * wall_mm)
                    results['od_reference'] = od_mm
                    results['od_tolerance'] = f"± {wall_tol:.2f} mm"  # OD tolerance matches wall tolerance
                    
                    logging.info(f"Found matching ID in TABLE 4: {nom_in}\" ({nom_mm}mm)")
                    logging.info(f"Setting tolerances: ID={nom_mm:.2f}±{id_tol:.2f}mm, Wall={wall_mm:.2f}±{wall_tol:.2f}mm")
                    match_found = True
                    break
        if not match_found:
            logging.info(f"No exact match found, checking TABLE 4 ranges for ID {id_val}mm")
            for min_id, max_id, wall_mm, wall_tol in TABLE_4_GRADE_1_RANGES:
                logging.info(f"Checking range {min_id}-{max_id}mm")
                if min_id <= id_val <= max_id:
                    results['wall_thickness'] = wall_mm
                    results['wall_thickness_tolerance'] = wall_tol
                    results['id_tolerance'] = "± 0.5 mm"  # Standard tolerance for range entries
                    
                    # Calculate OD from ID and wall thickness for range entries
                    od_mm = id_val + (2 * wall_mm)
                    results['od_reference'] = od_mm
                    results['od_tolerance'] = f"{od_mm:.2f} mm"
                    
                    logging.info(f"ID {id_val}mm falls in range {min_id}-{max_id}mm")
                    match_found = True
                    break
                    
            if not match_found:
                logging.warning(f"No Grade 1B dimension match found for ID {id_val}mm")
    else:
        logging.warning("No valid ID value found for Grade 1B dimension calculation")

def apply_grade_2b_rules(results: Dict[str, Any]) -> None:
    """
    Apply Grade 2B specific rules to the results.
    Uses TABLE 4-A for dimensions and tolerances.
    """
    grade = results.get('grade', '')
    # Normalize grade string to handle variations like "GRADE 1 B", "1B", etc.
    grade_str = re.sub(r'\s+', '', str(grade).upper())  # Remove spaces
    grade_str = re.sub(r'GRADE', '', grade_str)  # Remove "GRADE"
    
    if not any(g in grade_str for g in ['2B', 'B2']):
        return  # Exit early if not Grade 2B
        
    logging.info("Applying Grade 2B rules to results")
    
    # Get dimensions
    dimensions = results.get('dimensions', {})
    id_val = None
    
    for id_key in ['id1', 'ID1', 'ID', 'id']:
        val = dimensions.get(id_key) or results.get(id_key)
        if val and str(val).strip().lower() != 'not found':
            try:
                if isinstance(val, str):
                    val_clean = re.sub(r'[^\d.-]', '', val)
                    id_val = float(val_clean)
                else:
                    id_val = float(val)
                logging.info(f"Found valid ID value {id_val} from key {id_key}")
                break
            except (ValueError, TypeError) as e:
                logging.warning(f"Failed to parse ID value '{val}': {e}")
                continue
    
    if id_val is not None:
        logging.info(f"Processing Grade 2B dimensions for ID: {id_val}mm")
        
        # First check exact matches in TABLE 4-A
        match_found = False
        for row in TABLE_4A_GRADE_2_DATA:
            nom_in, nom_mm, id_tol, wall_mm, wall_tol = row
            
            # Skip range entries (which have None for nom_mm)
            if nom_mm is None:
                continue
                
            if abs(nom_mm - id_val) <= MAX_ACCEPT_DIFF_MM:
                results['id_tolerance'] = f"{nom_mm:.2f} ± {id_tol:.2f} mm"
                results['wall_thickness'] = wall_mm
                results['wall_thickness_tolerance'] = wall_tol
                
                # Calculate OD based on ID and wall thickness for Grade 2
                od_mm = nom_mm + (2 * wall_mm)
                results['od_reference'] = od_mm
                results['od_tolerance'] = f"{od_mm:.2f} mm"
                
                logging.info(f"Found matching ID in TABLE 4-A: {nom_in}\" ({nom_mm}mm)")
                match_found = True
                break
        
        # If no exact match found, check ranges
        if not match_found:
            for min_id, max_id, wall_mm, wall_tol in TABLE_4A_GRADE_2_RANGES:
                if min_id <= id_val <= max_id:
                    results['wall_thickness'] = wall_mm
                    results['wall_thickness_tolerance'] = wall_tol
                    results['id_tolerance'] = "± 0.5 mm"  # Standard tolerance for range entries
                    
                    # Calculate OD based on ID and wall thickness for range entries
                    od_mm = id_val + (2 * wall_mm)
                    results['od_reference'] = od_mm
                    results['od_tolerance'] = f"{od_mm:.2f} mm"
                    
                    logging.info(f"ID {id_val}mm falls in range {min_id}-{max_id}mm")
                    match_found = True
                    break
                    
            if not match_found:
                logging.warning(f"No Grade 2B dimension match found for ID {id_val}mm")
    else:
        logging.warning("No valid ID value found for Grade 2B dimension calculation")

def apply_grade_2bf_rules(results: Dict[str, Any]) -> None:
    """
    Apply Grade 2BF specific rules to the results.
    Uses TABLE 8-A for dimensions and tolerances.
    """
    grade = results.get('grade', '')
    if not any(g in str(grade).upper() for g in ['2BF', 'BF2']):
        return  # Exit early if not Grade 2BF
        
    logging.info("Applying Grade 2BF rules to results")
    
    # Get dimensions
    dimensions = results.get('dimensions', {})
    id_val = None
    
    for id_key in ['id1', 'ID1', 'ID', 'id']:
        val = dimensions.get(id_key) or results.get(id_key)
        if val and str(val).strip().lower() != 'not found':
            try:
                if isinstance(val, str):
                    val_clean = re.sub(r'[^\d.-]', '', val)
                    id_val = float(val_clean)
                else:
                    id_val = float(val)
                logging.info(f"Found valid ID value {id_val} from key {id_key}")
                break
            except (ValueError, TypeError) as e:
                logging.warning(f"Failed to parse ID value '{val}': {e}")
                continue
    
    if id_val is not None:
        logging.info(f"Processing Grade 2BF dimensions for ID: {id_val}mm")
        
        # First check exact matches in TABLE 8-A
        match_found = False
        for row in TABLE_8A_GRADE_2BF_DATA:
            nom_in, nom_mm, id_tol, wall_mm, wall_tol = row
            
            # Skip range entries (which have None for nom_mm)
            if nom_mm is None:
                continue
                
            if abs(nom_mm - id_val) <= MAX_ACCEPT_DIFF_MM:
                results['id_tolerance'] = f"{nom_mm:.2f} ± {id_tol:.2f} mm"
                results['wall_thickness'] = wall_mm
                results['wall_thickness_tolerance'] = wall_tol
                
                # Calculate OD based on ID and wall thickness for Grade 2BF
                od_mm = nom_mm + (2 * wall_mm)
                results['od_reference'] = od_mm
                results['od_tolerance'] = f"{od_mm:.2f} mm"
                
                logging.info(f"Found matching ID in TABLE 8-A: {nom_in}\" ({nom_mm}mm)")
                match_found = True
                break
        
        # If no exact match found, check ranges
        if not match_found:
            for min_id, max_id, wall_mm, wall_tol in TABLE_8A_GRADE_2BF_RANGES:
                if min_id <= id_val <= max_id:
                    results['wall_thickness'] = wall_mm
                    results['wall_thickness_tolerance'] = wall_tol
                    results['id_tolerance'] = "± 0.8 mm"  # Standard tolerance for range entries
                    
                    # Calculate OD based on ID and wall thickness for range entries
                    od_mm = id_val + (2 * wall_mm)
                    results['od_reference'] = od_mm
                    results['od_tolerance'] = f"{od_mm:.2f} mm"
                    
                    logging.info(f"ID {id_val}mm falls in range {min_id}-{max_id}mm")
                    match_found = True
                    break
                    
            if not match_found:
                logging.warning(f"No Grade 2BF dimension match found for ID {id_val}mm")
    else:
        logging.warning("No valid ID value found for Grade 2BF dimension calculation")

#######################
# Backwards-compatible aliases (fix ImportError seen in debug_utils)
#######################
# Point the debug-compatible alias to the TABLE_8 6-field entries (nom_in, nom_mm, id_tol, od_mm, wall_mm, wall_tol)
_F30_BF_TABLE = TABLE_8_GRADE_1BF_DATA
_F30_BF_RANGES = TABLE_8_GRADE_1BF_RANGES

def get_burst_pressure() -> float:
    """Get the standard burst pressure for MPAPS F-6032 materials."""
    return MPAPS_F6032_BURST_PRESSURE_MPA