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

def get_burst_pressure_from_tables(grade: str, id_value: float) -> Optional[float]:
    """
    Get burst pressure from tables based on grade and ID value for MPAPS F-30/F-1.
    Uses TABLE III for Suffix A and TABLE IV for Suffix B, C and F.
    
    Args:
        grade: Grade specification (e.g., '1BF', '2B', '1A', etc.)
        id_value: Inside diameter in mm
        
    Returns:
        Burst pressure in MPa or None if not found
    """
    try:
        grade_str = str(grade).upper().strip()
        id_val = float(id_value)
        
        logging.info(f"MPAPS F-30/F-1 burst pressure lookup: Grade='{grade_str}', ID={id_val}mm")
        
        # TABLE III: BURST STRENGTH FOR SUFFIX A (fixed values regardless of ID)
        if grade_str in ['1A', '2A', '3A']:
            if grade_str == '1A':
                logging.info("Using TABLE III: Grade 1A = 4.1 MPa")
                return 4.1  # MPa (600 psi)
            else:  # 2A or 3A
                logging.info("Using TABLE III: Grade 2A/3A = 2.1 MPa")
                return 2.1  # MPa (300 psi)
        
        # TABLE IV: BURST STRENGTH FOR SUFFIX B, C and F
        elif 'BF' in grade_str or 'B' in grade_str or 'C' in grade_str or 'F' in grade_str:
            # Determine which column to use based on grade
            if '1BF' in grade_str:
                # Grade 1BF -> Look in Suffix F, Grade 1 column
                column_name = "Suffix F, Grade 1"
                column_index = 6  # 7th column (0-based index)
            elif '2B' in grade_str:
                # Grade 2B -> Look in Suffix B, Grade 2 column
                column_name = "Suffix B, Grade 2"
                column_index = 3  # 4th column (0-based index)
            elif '1B' in grade_str or '3B' in grade_str:
                # Grade 1B or 3B -> Look in Suffix B, Grade 1&3 column
                column_name = "Suffix B, Grade 1&3"
                column_index = 2  # 3rd column (0-based index)
            elif '1C' in grade_str:
                # Grade 1C -> Look in Suffix C, Grade 1 column
                column_name = "Suffix C, Grade 1"
                column_index = 4
            elif '2C' in grade_str:
                # Grade 2C -> Look in Suffix C, Grade 2 column
                column_name = "Suffix C, Grade 2"
                column_index = 5
            elif '2F' in grade_str:
                # Grade 2F -> Look in Suffix F, Grade 2 column
                column_name = "Suffix F, Grade 2"
                column_index = 7
            else:
                logging.warning(f"Unrecognized grade pattern: {grade_str}")
                return None
            
            if column_index is not None:
                # Find the appropriate row based on ID
                for row in TABLE_IV_BURST_PRESSURE:
                    over, thru = row[0], row[1]
                    # Handle the special case for 51mm (inclusive of 51)
                    if over == 51 and id_val >= 51 and id_val <= thru:
                        burst_pressure = row[column_index]
                        if burst_pressure is not None:
                            logging.info(f"Found burst pressure {burst_pressure} MPa for ID {id_val}mm in range {over}-{thru}mm")
                            return burst_pressure
                    # Normal range lookup (over < id_val <= thru)
                    elif over < id_val <= thru:
                        burst_pressure = row[column_index]
                        if burst_pressure is not None:
                            logging.info(f"Found burst pressure {burst_pressure} MPa for ID {id_val}mm in range {over}-{thru}mm")
                            return burst_pressure
                
                logging.warning(f"No burst pressure found for Grade {grade_str} with ID {id_val}mm")
        
        logging.warning(f"Grade {grade_str} not found in burst pressure tables")
        return None
        
    except Exception as e:
        logging.error(f"Error in burst pressure lookup: {e}")
        return None

def get_burst_pressure() -> float:
    """Get the standard burst pressure for MPAPS F-6032 materials."""
    return MPAPS_F6032_BURST_PRESSURE_MPA

def apply_burst_pressure_rules(results: Dict[str, Any]) -> None:
    """
    Apply burst pressure rules based on grade and ID using tables.
    This should only be called for MPAPS F-30/F-1 standards.
    For MPAPS F-6032, burst pressure remains 2.0 MPa by default.
    """
    try:
        standard = results.get('standard', '')
        grade = results.get('grade', '')
        dimensions = results.get('dimensions', {})
        
        # Only apply table-based burst pressure for MPAPS F-30/F-1
        if not (is_mpaps_f30(str(standard).upper()) or 'F-1' in str(standard).upper()):
            logging.info(f"Skipping table-based burst pressure for standard: {standard}")
            return
            
        if not grade:
            logging.warning("No grade specified for burst pressure lookup")
            return
            
        # Get ID value for lookup
        id_val = None
        for id_key in ['id1', 'ID1', 'ID']:
            val = dimensions.get(id_key) or results.get(id_key)
            if val and str(val).strip().lower() != 'not found':
                try:
                    if isinstance(val, str):
                        val_clean = re.sub(r'[^\d.-]', '', val)
                        id_val = float(val_clean)
                    else:
                        id_val = float(val)
                    break
                except (ValueError, TypeError):
                    continue
        
        if id_val is None:
            logging.warning("No valid ID value found for burst pressure lookup")
            return
            
        # Get burst pressure from tables
        burst_pressure_mpa = get_burst_pressure_from_tables(grade, id_val)
        
        if burst_pressure_mpa is not None:
            # Update results with table-based burst pressure
            results['burst_pressure_mpa'] = burst_pressure_mpa
            results['burst_pressure_psi'] = round(burst_pressure_mpa * 145.038, 2)
            results['burst_pressure'] = burst_pressure_mpa * 10.0  # Convert MPa to bar
            
            # Add note about source
            results['burst_pressure_source'] = f"Table lookup for MPAPS F-30/F-1 Grade {grade}, ID {id_val}mm"
            
            logging.info(f"Applied burst pressure from tables: {burst_pressure_mpa} MPa "
                        f"({results['burst_pressure_psi']} PSI, {results['burst_pressure']} bar)")
        else:
            logging.info(f"No table-based burst pressure found for Grade {grade}, using existing values")
            
    except Exception as e:
        logging.error(f"Error applying burst pressure rules: {e}")

def is_grade_1bf(grade: str) -> bool:
    """Check if grade specification matches Grade 1B or 1BF."""
    if not grade:
        return False
        
    grade_str = str(grade).upper().strip()
    return any(pattern in grade_str for pattern in ['1B', '1BF', 'GRADE 1B', 'GRADE 1BF'])

def get_f30_bf_dimensions(id_value: float) -> Optional[Dict[str, Any]]:
    """
    Get Suffix BF dimensions and tolerances from TABLE VII-B for MPAPS F-30/F-1 standards.
    Uses exact matches for standard sizes (5/8" to 1") and range-based lookup for larger sizes.
    
    Args:
        id_value: Inside diameter in mm
        
    Returns:
        Dict with dimension info (tolerances, wall thickness, etc.) or None if not found
    """
    try:
        id_val = _parse_dimension(id_value)
        if id_val is None:
            return None
            
        logging.info(f"MPAPS F-30/F-1 Suffix BF dimension lookup for ID: {id_val}mm")
        
        # First check exact matches in TABLE VII-B
        for row in _F30_BF_TABLE:
            nominal_in, actual_id_mm, id_tol_mm, od_mm, wall_mm, wall_tol_mm = row
            if actual_id_mm is not None and abs(actual_id_mm - id_val) <= MAX_ACCEPT_DIFF_MM:
                logging.info(f"Found exact match in TABLE VII-B: {nominal_in} inch nominal")
                return {
                    'id_tolerance': f"{id_val:.1f} ± {id_tol_mm} mm",
                    'wall_thickness': wall_mm,
                    'wall_thickness_value': wall_mm,
                    'wall_thickness_tolerance': f"{wall_mm:.2f} ± {wall_tol_mm} mm",
                    'nominal_inches': nominal_in,
                    'od_reference': od_mm,
                    'nearest_id': actual_id_mm
                }
        
        # If no exact match, check ranges
        for min_id, max_id, wall_mm, wall_tol_mm in _F30_BF_RANGES:
            if min_id <= id_val <= max_id:
                logging.info(f"Found range match in TABLE VII-B: {min_id}-{max_id}mm")
                return {
                    'id_tolerance': f"{id_val:.1f} ± 0.8 mm",  # Fixed tolerance per table
                    'wall_thickness': wall_mm,
                    'wall_thickness_value': wall_mm,
                    'wall_thickness_tolerance': f"{wall_mm:.2f} ± {wall_tol_mm} mm",
                    'nominal_inches': f">{min_id/25.4:.1f}",  # Convert to inches for display
                    'od_reference': None,  # No fixed OD for range-based sizes
                    'nearest_id': id_val,  # Use actual ID as no nominal exists
                    'wall_tolerance_value': wall_tol_mm  # Keep numeric value
                }
        
    except Exception as e:
        logging.error(f"Error in Grade 1BF tolerance lookup: {e}")
        return None

def _parse_dimension(value: Any) -> Optional[float]:
    """Extract numeric value from dimension string or number."""
    if value is None:
        return None
        
    try:
        if isinstance(value, (int, float)):
            return float(value)
            
        # Handle string input
        value_str = str(value).strip()
        
        # Try direct float conversion first
        try:
            return float(value_str)
        except ValueError:
            pass
            
        # Look for first number in string
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value_str)
        if match:
            return float(match.group(0))
            
    except Exception as e:
        logging.warning(f"Failed to parse dimension value '{value}': {e}")
        return None

def _convert_to_mm(value: float) -> Tuple[float, bool]:
    """Convert value to mm if it appears to be in inches (< 3.0)."""
    if value < 3.0:  # Likely inches
        return value * 25.4, True
    return value, False

def get_mpaps_f6032_tolerance(value: Any, dimension_type: str) -> Optional[Dict[str, Any]]:
    """
    Get MPAPS F-6032 tolerance for a dimension value from TABLE 1.
    
    Args:
        value: Numeric value or string containing a number
        dimension_type: 'ID' or 'OD'
    
    Returns:
        Dict with tolerance info or None if value invalid
    """
    try:
        # Parse and validate input
        dim_num = _parse_dimension(value)
        if dim_num is None:
            return None

        is_id = dimension_type.upper().strip() == 'ID'
        lookup_val = dim_num
        
        # For OD lookup, find by matching ID first
        if not is_id:
            # Find best matching row by OD
            min_diff = float('inf')
            for row in TABLE_1_DATA:
                _, nominal_id_mm, _, nominal_od_mm, _ = row
                diff = abs(nominal_od_mm - dim_num)
                if diff < min_diff:
                    min_diff = diff
                    lookup_val = nominal_id_mm
        
        # Get data from TABLE 1
        table_data = get_mpaps_f6032_dimensions_from_table(lookup_val)
        
        if table_data:
            if is_id:
                return {
                    'nominal': table_data['nominal_id_mm'],
                    'tolerance': table_data['id_tolerance_mm'],
                    'formatted': table_data['id_formatted'],
                    'nominal_inches': table_data['nominal_id_inches']
                }
            else:
                return {
                    'nominal': table_data['nominal_od_mm'],
                    'tolerance': table_data['od_tolerance_mm'],
                    'formatted': table_data['od_formatted']
                }
                
        return None
                
    except Exception as e:
        logging.error(f"Error in MPAPS F-6032 tolerance lookup: {e}")
        return None
            
        # Convert to mm if needed
        value_mm, was_inches = _convert_to_mm(dim_num)
        
        # Select correct nominal/tolerance lists
        if dimension_type.upper() == 'ID':
            nominals = _MPAPS_ID_NOMINALS_MM
            tolerances = _MPAPS_ID_TOLS_MM
        elif dimension_type.upper() == 'OD':
            nominals = _MPAPS_OD_NOMINALS_MM
            tolerances = _MPAPS_OD_TOLS_MM
        else:
            raise ValueError(f"Invalid dimension type '{dimension_type}', must be 'ID' or 'OD'")
            
        # Find nearest nominal value with tolerance
        if not nominals:  # Handle empty lists
            return None
            
        # Find the closest nominal value
        closest_idx = None
        min_diff = float('inf')
        
        for i, nominal in enumerate(nominals):
            diff = abs(nominal - value_mm)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        # Accept nearest nominal and warn when the match is relatively far
        if closest_idx is None:
            logging.warning(f"No MPAPS F-6032 nominals available for lookup")
            return None
        if min_diff > MAX_ACCEPT_DIFF_MM:
            logging.warning(f"No close MPAPS F-6032 match found for {dimension_type} value {value_mm}mm "
                          f"(nearest {nominals[closest_idx]} mm, diff {min_diff:.3f}mm). "
                          "Returning nearest nominal — please verify.")
            
        nearest = nominals[closest_idx]
        tolerance = tolerances[closest_idx]
        
        # Build result
        try:
            result = {
                'original': dim_num,
                'value_mm': value_mm,
                'nearest_mm': nearest,
                'tol_mm': tolerance,
                'was_inches': was_inches,
                'formatted': f"{value_mm:.1f} ± {tolerance:.1f} mm"
            }
            
            logging.debug(f"MPAPS F-6032 {dimension_type} tolerance lookup: {result}")
            return result
        except Exception as e:
            logging.error(f"Error building result dict: {e}")
            return None

def is_mpaps_f30(standard: Optional[str]) -> bool:
    """Check if standard specification matches MPAPS F-30."""
    if not standard:
        return False
        
    try:
        std = str(standard).upper().strip()
        std = re.sub(r'[\s_-]+', '', std)
        return 'MPAPSF30' in std
    except (AttributeError, TypeError):
        return False

def is_mpaps_f6032(material: str) -> bool:
    """Check if material specification matches MPAPS F-6032."""
    if not material:
        return False
        
    mat = str(material).upper().strip()
    mat = re.sub(r'[\s_-]+', '', mat)  # Remove spaces, underscores, dashes
    
    # Check for various MPAPS F-6032 patterns
    patterns = [
        'MPAPSF6032',
        'MPAPSF-6032', 
        'MPAPS F6032',
        'MPAPS F-6032'
    ]
    
    return any(pattern.replace('-', '') in mat for pattern in patterns)

# Expose key functions at module level
__all__ = ['get_burst_pressure', 'is_grade_1bf', 'get_f30_bf_dimensions', 
           'get_mpaps_f6032_tolerance', 'is_mpaps_f6032', 'apply_mpaps_f6032_rules',
           'apply_grade_1bf_rules', 'get_burst_pressure_from_tables', 'apply_burst_pressure_rules',
           'get_mpaps_f6032_dimensions_from_table', 'apply_mpaps_f30_f1_rules']

def apply_mpaps_f30_f1_rules(results: Dict[str, Any]) -> None:
    """
    Apply MPAPS F-30/F-1 rules to analysis results.
    Uses TABLE III and IV for burst pressure and existing tolerance rules.
    """
    standard = results.get('standard', '')
    grade = results.get('grade', '')
    dimensions = results.get('dimensions', {})
    
    # Only apply if it's MPAPS F-30 or F-1
    is_f30_f1 = is_mpaps_f30(standard) or 'F-1' in str(standard).upper()
    
    if not is_f30_f1:
        return
        
    logging.info("Applying MPAPS F-30/F-1 rules to results")
    
    # Get ID for tolerance lookup
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
        logging.warning("No valid ID value found for F-30/F-1 rules")
        return
    
    # Check for MPAPS F-30 GRADE 1B
    if is_mpaps_f30(standard) and '1B' in str(grade).upper():
        logging.info("Applying MPAPS F-30 GRADE 1B rules to results")
        apply_mpaps_f30_grade_1b_rules(results)
    
    # Check for Grade 1B/1BF in any standard
    elif is_grade_1bf(grade):
        logging.info("Applying Grade 1B/1BF rules to results")
        apply_grade_1bf_rules(results)
    
    # Apply table-based burst pressure for all F-30/F-1 standards
    apply_burst_pressure_rules(results)
    
    # Check for OD in multiple locations with type conversion
    od_val = None
    for od_key in ['od1', 'OD1', 'OD']:
        val = str(dimensions.get(od_key, '') or results.get(od_key, '')).strip()
        if val and val.lower() != 'not found':
            try:
                od_val = float(re.sub(r'[^\d.-]', '', val))
                logging.info(f"Found valid OD value {od_val} from key {od_key}")
                break
            except (ValueError, TypeError):
                continue
    
    # Use TABLE 1 to get dimensions and tolerances
    if id_val is not None:
        logging.info(f"Looking up MPAPS F-6032 dimensions for ID: {id_val}mm")
        
        # Get dimensions from TABLE 1
        table_dimensions = get_mpaps_f6032_dimensions_from_table(id_val)
        
        if table_dimensions:
            # Set ID tolerance
            results['id_tolerance'] = table_dimensions['id_formatted']
            logging.info(f"Set ID tolerance from TABLE 1: {table_dimensions['id_formatted']}")
            
            # Set OD tolerance and nominal value
            results['od_tolerance'] = table_dimensions['od_formatted']
            results['od_nearest_nominal'] = table_dimensions['nominal_od_mm']
            
            # Update dimensions dictionary to use TABLE 1 values
            dimensions = results.get('dimensions', {})
            dimensions['id1'] = table_dimensions['nominal_id_mm']
            dimensions['od1'] = table_dimensions['nominal_od_mm']
            dimensions['od2'] = table_dimensions['nominal_od_mm']  # Same as od1 for consistency
            results['dimensions'] = dimensions
            
            logging.info(f"Set OD tolerance and nominal from TABLE 1: {table_dimensions['od_formatted']}")
            logging.info(f"Updated dimensions to use TABLE 1 values: ID={table_dimensions['nominal_id_mm']}mm, OD={table_dimensions['nominal_od_mm']}mm")
        else:
            # Fall back to existing tolerance calculation if TABLE 1 lookup fails
            logging.warning(f"No matching dimensions found in TABLE 1 for ID {id_val}mm")
            results['id_tolerance'] = "N/A"
            results['od_tolerance'] = "N/A"
    else:
        results['id_tolerance'] = "N/A"
        results['od_tolerance'] = "N/A"
        logging.warning("No valid ID value found for dimension lookup")
            
    # Set burst pressure with units (always 2.0 MPa for F-6032)
    results['burst_pressure_mpa'] = MPAPS_F6032_BURST_PRESSURE_MPA
    results['burst_pressure_psi'] = round(MPAPS_F6032_BURST_PRESSURE_MPA * 145.038, 2)  # Convert to PSI
    results['burst_pressure'] = MPAPS_F6032_BURST_PRESSURE_MPA * 10.0  # 2.0 MPa = 20 bar
    results['burst_pressure_source'] = "MPAPS F-6032 default (2.0 MPa)"
    logging.info("Set burst pressure values: %.1f MPa, %.2f PSI, %.1f bar", 
                 results['burst_pressure_mpa'], results['burst_pressure_psi'], results['burst_pressure'])

def apply_grade_1bf_rules(results: Dict[str, Any]) -> None:
    """
    Apply Grade 1B/1BF rules to analysis results.
    """
    dimensions = results.get('dimensions', {})
    
    # Get ID for tolerance lookup
    id_val = None
    for id_key in ['id1', 'ID1', 'ID']:
        val = str(dimensions.get(id_key, '') or results.get(id_key, '')).strip()
        if val and val.lower() != 'not found':
            try:
                id_val = float(re.sub(r'[^\d.-]', '', val))
                logging.info(f"Found valid ID value {id_val} from key {id_key}")
                break
            except (ValueError, TypeError):
                continue
    
    if id_val is not None:
        grade_1bf_info = get_f30_bf_dimensions(id_val)
        if grade_1bf_info:
            # Set tolerances and dimensions
            results['id_tolerance'] = grade_1bf_info['id_tolerance']
            results['wall_thickness'] = grade_1bf_info['wall_thickness']
            results['wall_thickness_tolerance'] = grade_1bf_info['wall_thickness_tolerance']
            
            # Set OD reference if available and OD not found
            if grade_1bf_info['od_reference']:
                if not dimensions.get('od1') or dimensions['od1'] == 'Not Found':
                    dimensions['od1'] = grade_1bf_info['od_reference']
                    dimensions['od2'] = grade_1bf_info['od_reference']
                    logging.info(f"Set OD reference: {grade_1bf_info['od_reference']} mm")
            
            # Set material and reinforcement for Grade 1B/1BF
            results['material'] = "EPDM"
            results['reinforcement'] = "STEEL WIRE"  # Based on suffix BF
            
            logging.info(f"Grade 1B/1BF rules applied: ID={id_val}, Wall={grade_1bf_info['wall_thickness']}mm")
        else:
            results['id_tolerance'] = "N/A"
            results['wall_thickness_tolerance'] = "N/A"
            logging.warning("Could not calculate Grade 1B/1BF tolerances")
    else:
        logging.warning("No valid ID value found for Grade 1B/1BF tolerance calculation")

def apply_mpaps_f30_grade_1b_rules(results: Dict[str, Any]) -> None:
    """
    Apply MPAPS F-30 GRADE 1B rules to analysis results.
    Uses the Grade 1B table for dimensions and tolerances.
    """
    standard = results.get('standard', '')
    grade = results.get('grade', '')
    
    if not (is_mpaps_f30(standard) and '1B' in str(grade).upper()):
        return
        
    logging.info("Applying MPAPS F-30 GRADE 1B rules to results")
    
    # Get dimensions from either top level or dimensions dict
    dimensions = results.get('dimensions', {})
    
    # Get ID for tolerance lookup
    id_val = results.get('id1') or dimensions.get('id1') or results.get('ID')
    if id_val is not None and id_val != "Not Found":
        grade_1b_info = get_f30_bf_dimensions(id_val)
        if grade_1b_info:
            results['id_tolerance'] = grade_1b_info['id_tolerance']
            results['wall_thickness'] = grade_1b_info['wall_thickness']
            results['wall_thickness_tolerance'] = grade_1b_info['wall_thickness_tolerance']
            
            # Set OD reference if available and OD not found
            if grade_1b_info['od_reference']:
                results['od1'] = grade_1b_info['od_reference']
                if 'dimensions' in results:
                    results['dimensions']['od1'] = grade_1b_info['od_reference']
                    results['dimensions']['od2'] = grade_1b_info['od_reference']
            
            logging.info(f"MPAPS F-30 GRADE 1B rules applied: ID={id_val}, Wall={grade_1b_info['wall_thickness']}mm")
    
    # Set burst pressure for MPAPS F-30 (typically 2.0 MPa = 20 bar)
    results['burst_pressure_mpa'] = 2.0
    results['burst_pressure'] = 20.0

def apply_mpaps_f6032_rules(results: Dict[str, Any]) -> None:
    """
    Apply MPAPS F-6032 rules to analysis results.
    Uses TABLE 1 for dimensions and tolerances, and fixed 2.0 MPa burst pressure.
    """
    # Check multiple fields for MPAPS F-6032 indication
    material = results.get('material')
    standard = results.get('standard')
    specification = results.get('specification')
    
    # Only apply if it's MPAPS F-6032
    is_f6032 = (material and is_mpaps_f6032(material)) or \
               (standard and is_mpaps_f6032(standard)) or \
               (specification and is_mpaps_f6032(specification))
    
    if not is_f6032:
        return
        
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
    
    # Check for OD in multiple locations with type conversion
    od_val = None
    for od_key in ['od1', 'OD1', 'OD']:
        val = str(dimensions.get(od_key, '') or results.get(od_key, '')).strip()
        if val and val.lower() != 'not found':
            try:
                od_val = float(re.sub(r'[^\d.-]', '', val))
                logging.info(f"Found valid OD value {od_val} from key {od_key}")
                break
            except (ValueError, TypeError):
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
            
            # Update dimensions with nominal values from TABLE 1 if not already set
            if not dimensions.get('id1') or dimensions['id1'] == 'Not Found':
                dimensions['id1'] = table_data['nominal_id_mm']
                dimensions['id2'] = table_data['nominal_id_mm']
                logging.info(f"Updated ID dimensions with nominal: {table_data['nominal_id_mm']}mm")
            
            if (not dimensions.get('od1') or dimensions['od1'] == 'Not Found') and od_val is None:
                dimensions['od1'] = table_data['nominal_od_mm']
                dimensions['od2'] = table_data['nominal_od_mm']
                logging.info(f"Updated OD dimensions with nominal: {table_data['nominal_od_mm']}mm")
            
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
    results['burst_pressure_mpa'] = MPAPS_F6032_BURST_PRESSURE_MPA
    results['burst_pressure_psi'] = round(MPAPS_F6032_BURST_PRESSURE_MPA * 145.038, 2)
    results['burst_pressure'] = MPAPS_F6032_BURST_PRESSURE_MPA * 10.0  # Convert to bar
    results['burst_pressure_source'] = "MPAPS F-6032 default (2.0 MPa)"
    logging.info("Set MPAPS F-6032 default burst pressure: 2.0 MPa")