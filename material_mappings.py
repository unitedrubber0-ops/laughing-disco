"""
Module for mapping between different material standards and specifications.
"""
import logging
import re

logger = logging.getLogger(__name__)

def map_tms_to_mpaps_standard(tms_standard):
    """Map TMS standards to equivalent MPAPS standards"""
    mapping = {
        "TMS-6034": "MPAPS F-6034",
        "TMS-6028": "MPAPS F-6028", 
        "TMS-6032": "MPAPS F-6032",
        "TMS-6034 H-ANRX": "MPAPS F-6034",
    }
    
    for tms, mpaps in mapping.items():
        if tms in str(tms_standard):
            logger.info(f"Mapped {tms_standard} to {mpaps}")
            return mpaps
    
    return tms_standard

def debug_material_lookup(standard, grade):
    """Debug function to identify material lookup issues"""
    logger.info(f"Material lookup debug - Standard type: {type(standard)}, value: {standard}")
    logger.info(f"Material lookup debug - Grade type: {type(grade)}, value: {grade}")
    
    # If standard is a dict, extract the actual value
    if isinstance(standard, dict):
        logger.warning("Standard is a dictionary, attempting to extract value")
        # Try common keys
        for key in ['standard', 'specification', 'value', 'text']:
            if key in standard:
                standard = standard[key]
                break
        logger.info(f"Extracted standard: {standard}")
    
    # If grade is a dict, extract the actual value  
    if isinstance(grade, dict):
        logger.warning("Grade is a dictionary, attempting to extract value")
        for key in ['grade', 'type', 'value', 'text']:
            if key in grade:
                grade = grade[key]
                break
        logger.info(f"Extracted grade: {grade}")
    
    return standard, grade


# ============================================================================
# Authoritative Standard + Grade -> Material Mapping (from user-supplied table)
# ============================================================================

def _canon_standard(s):
    """Normalize standard string to canonical form used in lookup table"""
    if not s:
        return ''
    s = str(s).upper().strip()
    s = re.sub(r'\s+', ' ', s)
    # Map variants to canonical keys used in the table
    if 'F-30' in s or 'F30' in s or 'F-1' in s or 'F1' in s:
        return 'MPAPS F-30/F-1'
    if 'F-6032' in s or 'F6032' in s:
        return 'MPAPS F-6032'
    if 'F-6028' in s or 'F6028' in s:
        return 'MPAPS F-6028'
    if 'F-6034' in s or 'F6034' in s:
        return 'MPAPS F-6034'
    return s

def _canon_grade(g):
    """Normalize grade string to canonical form used in lookup table"""
    if not g:
        return ''
    s = str(g).upper().strip()
    # Remove 'GRADE' prefix: "GRADE IB" -> "IB", "GRADE 1B" -> "1B"
    s = re.sub(r'GRADE\s*', '', s)
    # Remove spaces except within specific contexts
    s = s.replace(' ', '')
    # Convert roman I -> 1 when it appears like "IB" or "I" followed by letter/digit
    # e.g., "IB" -> "1B", "IBF" -> "1BF"
    s = re.sub(r'^I(?=[A-Z0-9])', '1', s)
    # Normalize standalone "IB" -> "1B"
    s = re.sub(r'^IB$', '1B', s)
    # Normalize various BF/BFD/BD patterns with I -> 1
    s = re.sub(r'^I([BDF]+)$', lambda m: '1' + m.group(1), s)
    return s

# Authoritative mapping derived from the user's table image.
# Expand as needed to cover all standards/grades in use.
MATERIAL_LOOKUP_TABLE = {
    'MPAPS F-30/F-1': {
        '1B': ('P-EPDM', 'KEVLAR'),
        '1BF': ('P-EPDM', 'KEVLAR'),
        '1BFD': ('P-EPDM WITH SPRING INSERT', 'KEVLAR'),
        '2B': ('SILICONE', 'NOMEX 4 PLY'),
        '2C': ('SILICONE', 'NOMEX 4 PLY'),
        'J20CLASSA': ('SILICONE', 'NOMEX 4 PLY'),
        'J20CLASSB': ('P-NBR', 'KEVLAR'),
        'J20CLASSC': ('CR', 'KEVLAR'),
        'J20CLASSR': ('EPDM', 'KEVLAR'),
    },
    'MPAPS F-6032': {
        'TYPEI': ('INNER NBR OUTER:ECO', 'KEVLAR'),
    },
    'MPAPS F-6028': {
        '--': ('INNER:NBR OUTER:CR', 'KEVLAR'),
    },
    'MPAPS F-6034': {
        'H-AN': ('HIGH TEMP. SILICONE', 'NOMEX 4 PLY'),
        'H-ANR': ('INNER:FKM OUTER:HIGH TEMP. SILICONE', 'NOMEX 4 PLY'),
        'C-AN': ('HIGH TEAR SILICONE', 'NOMEX 4 PLY'),
        'GRADEC-BNR': ('CSM', 'KEVLAR'),
    },
}

def get_material_by_standard_grade(standard_raw: str, grade_raw: str):
    """
    Return (material_string, reinforcement_string) if mapping exists, else (None, None).
    This lookup is authoritative and should be consulted BEFORE fuzzy DB lookups.
    
    Args:
        standard_raw: Raw standard string (e.g., "MPAPS F-30", "F-30/F-1", etc.)
        grade_raw: Raw grade string (e.g., "GRADE IB", "1BF", "Grade 1B", etc.)
    
    Returns:
        Tuple: (material, reinforcement) if found in table, else (None, None)
    """
    std = _canon_standard(standard_raw)
    grd = _canon_grade(grade_raw)

    logger.debug(f"Authoritative lookup: standard={standard_raw} -> {std}, grade={grade_raw} -> {grd}")

    std_map = MATERIAL_LOOKUP_TABLE.get(std)
    if not std_map:
        # Fallback: try to find a key that contains the standard string
        for k in MATERIAL_LOOKUP_TABLE.keys():
            if k in std or std in k:
                std_map = MATERIAL_LOOKUP_TABLE[k]
                logger.debug(f"Found standard via substring match: {k}")
                break
    if not std_map:
        logger.debug(f"No standard entry found for {std}")
        return None, None

    # Direct exact match on grade
    if grd in std_map:
        result = std_map[grd]
        logger.info(f"Authoritative material mapping matched: {std} + {grd} -> {result}")
        return result

    # Relaxed matches: look for prefix matches
    # e.g., grade '1' matches '1B' or '1BF', grade '1BF' matches '1BF'
    for key in std_map.keys():
        if grd and (grd == key or grd.startswith(key) or key.startswith(grd)):
            result = std_map[key]
            logger.info(f"Authoritative material mapping matched (relaxed): {std} + {grd} -> {key} -> {result}")
            return result

    logger.debug(f"No grade match in {std} for {grd}")
    return None, None