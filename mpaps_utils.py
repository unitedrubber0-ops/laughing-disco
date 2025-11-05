"""
MPAPS F-6032 and F-30 tolerance and burst pressure handling.
"""
import re
import logging
from typing import Optional, Dict, Any, Tuple, List

# Constants for burst pressure
MPAPS_F6032_BURST_PRESSURE_MPA = 2.0

# Maximum acceptable difference for nominal value matching (mm)
MAX_ACCEPT_DIFF_MM = 0.5  # Allow up to 0.5mm difference for practical measurements

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

# TABLE VII-B: SUFFIX BF HOSE ID AND WALL DIMENSIONS AND TOLERANCES (MPAPS F-30/F-1 only)
_F30_BF_TABLE = [
    # (nominal ID inches, actual ID mm, ID tolerance mm, OD mm, wall thickness mm, wall tolerance mm)
    ('5/8', 15.1, 0.8, 25.0, 4.95, 0.8),
    ('3/4', 18.4, 0.8, 28.3, 4.95, 0.8),
    ('7/8', 21.3, 0.8, 29.9, 4.95, 0.8),
    ('1', 24.6, 0.8, 34.5, 4.95, 0.8),
    ('>1.0 < 2.0', None, 0.8, None, 4.95, 0.8),   # For ID range 26 - 50.8 mm
    ('>2.0', None, 0.8, None, 5.35, 0.8)          # For ID range >50.8 - 63.5 mm
]

# Range data for MPAPS F-30/F-1 Suffix BF dimensions
_F30_BF_RANGES = [
    (26.0, 50.8, 4.95, 0.8),   # min_id, max_id, wall_thickness, wall_tolerance
    (50.8, 63.5, 5.35, 0.8)    # min_id, max_id, wall_thickness, wall_tolerance
]

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
            return closest_match
        elif closest_match:
            logging.warning(f"Nearest MPAPS F-6032 nominal for ID {id_val}mm is {min_diff:.2f}mm away")
            logging.info(f"Using nearest match: {closest_match}")
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
    
    if material:
        mat_upper = str(material).upper()
        if 'MPAPS F-6032' in mat_upper or 'MPAPSF6032' in mat_upper:
            is_f6032 = True
    
    if not is_f6032:
        return  # Exit early if not F-6032
        
    logging.info("Applying MPAPS F-6032 rules to results")
    
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
            
    # Set burst pressure for MPAPS F-6032 (default 2.0 MPa = 20 bar)
    results['burst_pressure_mpa'] = 2.0
    results['burst_pressure_psi'] = round(2.0 * 145.038, 2)
    results['burst_pressure'] = 20.0
    results['burst_pressure_source'] = "MPAPS F-6032 default (2.0 MPa)"
    logging.info("Set MPAPS F-6032 default burst pressure: 2.0 MPa")

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
        
    logging.info("Applying MPAPS F-30/F-1 rules to results")
    
    # Clear any MPAPS F-6032 tolerances that might have been set
    if 'id_tolerance' in results and 'TABLE 1' in str(results.get('dimension_source', '')):
        results['id_tolerance'] = "N/A"
        results['od_tolerance'] = "N/A" 
        logging.info("Cleared MPAPS F-6032 tolerances for F-30/F-1 part")
    
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
                elif any(g in grade_upper for g in ['1B', '3B']):
                    return row[2]  # Suffix B Grade 1&3
                elif '2B' in grade_upper:
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

def get_burst_pressure() -> float:
    """Get the standard burst pressure for MPAPS F-6032 materials."""
    return MPAPS_F6032_BURST_PRESSURE_MPA