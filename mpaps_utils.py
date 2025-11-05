"""
MPAPS F-6032 tolerance and burst pressure handling.
"""
import re
import logging
from typing import Optional, Dict, Any, Tuple

# Constants for burst pressure
MPAPS_F6032_BURST_PRESSURE_MPA = 2.0

# MPAPS F-6032 ID nominal values (mm) and tolerances (mm) from spec table
_MPAPS_ID_NOMINALS_MM = [3.97, 4.76, 5.56, 5.95, 7.14, 9.00, 12.00, 15.10, 18.40, 24.60]
_MPAPS_ID_TOLS_MM     = [0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.58, 0.79, 0.79, 0.79]

_MPAPS_OD_NOMINALS_MM = [9.3,  10.3, 11.3, 12.25, 13.5, 15.37, 20.3, 24.62, 28.35, 34.9]
_MPAPS_OD_TOLS_MM     = [0.6,  0.6,  0.6,  0.6,   0.6,  0.6,   0.8,  0.8,   0.8,   1.0]

# Grade 1B/1BF data tables from specification
_GRADE_1BF_ID_NOMINALS_MM = [15.1, 18.4, 21.3, 24.6, 62.7]
_GRADE_1BF_ID_TOLS_MM = [0.5, 0.5, 0.5, 0.5, 0.5]
_GRADE_1BF_OD_NOMINALS_MM = [25.0, 28.3, 29.9, 34.5, 73.4]
_GRADE_1BF_WALL_THICKNESS_MM = [4.95, 4.95, 4.95, 4.95, 5.35]
_GRADE_1BF_WALL_TOLS_MM = [0.8, 0.8, 0.8, 0.8, 0.8]

# Range data for Grade 1BF
_GRADE_1BF_RANGES = [
    (26.0, 50.8, 4.95, 0.8),   # min_id, max_id, wall_thickness, wall_tolerance
    (50.8, 63.5, 5.35, 0.8)    # min_id, max_id, wall_thickness, wall_tolerance
]

def get_burst_pressure() -> float:
    """Get the standard burst pressure for MPAPS F-6032 materials."""
    return MPAPS_F6032_BURST_PRESSURE_MPA

def is_grade_1bf(grade: str) -> bool:
    """Check if grade specification matches Grade 1B or 1BF."""
    if not grade:
        return False
        
    grade_str = str(grade).upper().strip()
    return any(pattern in grade_str for pattern in ['1B', '1BF', 'GRADE 1B', 'GRADE 1BF'])

def get_grade_1bf_tolerance(id_value: float) -> Optional[Dict[str, Any]]:
    """
    Get Grade 1B/1BF tolerance and wall thickness based on ID.
    """
    try:
        id_val = _parse_dimension(id_value)
        if id_val is None:
            return None
            
        # Find nearest nominal ID in Grade 1BF table
        closest_idx = None
        min_diff = float('inf')
        
        for i, nominal_id in enumerate(_GRADE_1BF_ID_NOMINALS_MM):
            diff = abs(nominal_id - id_val)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        # Check if we have a reasonable match (within 1mm)
        if closest_idx is None or min_diff > 1.0:
            logging.warning(f"No close Grade 1BF match found for ID {id_val}mm")
            return None
        
        # Use the matched values
        wall_thickness = _GRADE_1BF_WALL_THICKNESS_MM[closest_idx]
        wall_tolerance = _GRADE_1BF_WALL_TOLS_MM[closest_idx]
        od_reference = _GRADE_1BF_OD_NOMINALS_MM[closest_idx]
        id_tolerance = _GRADE_1BF_ID_TOLS_MM[closest_idx]
        nearest_id = _GRADE_1BF_ID_NOMINALS_MM[closest_idx]
        
        id_tolerance = _GRADE_1BF_ID_TOLS_MM[closest_idx]
        
        return {
            'id_tolerance': f"{id_val:.1f} ± {id_tolerance:.1f} mm",
            'wall_thickness': wall_thickness,
            'wall_thickness_value': wall_thickness,  # Keep numeric value for calculations
            'wall_thickness_tolerance': f"{wall_thickness:.2f} ± {wall_tolerance:.1f} mm",  # Format: "4.95 ± 0.8 mm"
            'wall_tolerance_value': wall_tolerance,  # Keep numeric value
            'od_reference': od_reference,
            'nearest_id': nearest_id
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
    Get MPAPS F-6032 tolerance for a dimension value.
    
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
        
        if closest_idx is None or min_diff > 2.0:  # Allow 2mm tolerance for matching
            logging.warning(f"No close nominal match found for {dimension_type} value {value_mm}mm")
            return None
            
        nearest = nominals[closest_idx]
        tolerance = tolerances[closest_idx]
        
        # Build result
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
        logging.error(f"Error in MPAPS F-6032 tolerance lookup: {e}")
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
__all__ = ['get_burst_pressure', 'is_grade_1bf', 'get_grade_1bf_tolerance', 
           'get_mpaps_f6032_tolerance', 'is_mpaps_f6032', 'apply_mpaps_f6032_rules',
           'apply_grade_1bf_rules']

def _apply_mpaps_specific_rules(results: Dict[str, Any]) -> None:
    """
    Apply specific MPAPS F-6032 rules.
    """
    # Get dimensions from all possible locations
    dimensions = results.get('dimensions', {})
    
    # Check for ID in multiple locations with type conversion
    id_val = None
    for id_key in ['id1', 'ID1', 'ID', 'id']:
        val = dimensions.get(id_key) or results.get(id_key)
        if val and str(val).strip().lower() != 'not found':
            try:
                # Extract numeric value from string if needed
                if isinstance(val, str):
                    # Remove non-numeric characters except decimal point and minus
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
    
    # Process ID tolerance if value found
    if id_val is not None:
        # Try MPAPS F-6032 tolerance first
        id_tol = get_mpaps_f6032_tolerance(id_val, 'ID')
        if not id_tol:
            # Fall back to Grade 1BF tolerance
            id_tol_info = get_grade_1bf_tolerance(id_val)
            if id_tol_info:
                id_tol = {'formatted': id_tol_info['id_tolerance']}
        
        if id_tol:
            results['id_tolerance'] = id_tol['formatted']
            logging.info(f"Set ID tolerance: {id_tol['formatted']}")
        else:
            results['id_tolerance'] = "N/A"
            logging.warning(f"Could not calculate ID tolerance for value: {id_val}")
    else:
        results['id_tolerance'] = "N/A"
        logging.warning("No valid ID value found for tolerance calculation")
            
    # Process OD tolerance if value found
    if od_val is not None:
        od_tol = get_mpaps_f6032_tolerance(od_val, 'OD')
        if od_tol:
            results['od_tolerance'] = od_tol['formatted']
            results['od_nearest_nominal'] = od_tol['nearest_mm']
            logging.info(f"Set OD tolerance: {od_tol['formatted']}")
        else:
            results['od_tolerance'] = "N/A"
            logging.warning(f"Could not calculate OD tolerance for value: {od_val}")
    else:
        results['od_tolerance'] = "N/A"
        logging.warning("No valid OD value found for tolerance calculation")
            
    # Set burst pressure with units
    results['burst_pressure_mpa'] = 2.0  # Standard MPAPS F-6032 value
    results['burst_pressure_psi'] = round(2.0 * 145.038, 2)  # Convert to PSI
    results['burst_pressure'] = 20.0  # 2.0 MPa = 20 bar
    logging.info("Set burst pressure values: 2.0 MPa, %.2f PSI, 20.0 bar", results['burst_pressure_psi'])

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
        grade_1bf_info = get_grade_1bf_tolerance(id_val)
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
        grade_1b_info = get_grade_1bf_tolerance(id_val)
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
    Modifies results dict in place to add tolerances and burst pressure.
    """
    # Check multiple fields for MPAPS F-6032 indication
    material = results.get('material')
    standard = results.get('standard')
    specification = results.get('specification')
    grade = results.get('grade', '')
    
    # Check for MPAPS F-6032
    if (material and is_mpaps_f6032(material)) or \
       (standard and is_mpaps_f6032(standard)) or \
       (specification and is_mpaps_f6032(specification)):
        logging.info("Applying MPAPS F-6032 rules to results")
        _apply_mpaps_specific_rules(results)
    
    # Check for MPAPS F-30 GRADE 1B
    elif is_mpaps_f30(standard) and '1B' in str(grade).upper():
        logging.info("Applying MPAPS F-30 GRADE 1B rules to results")
        apply_mpaps_f30_grade_1b_rules(results)
    
    # Check for Grade 1B/1BF in any standard
    elif is_grade_1bf(grade):
        logging.info("Applying Grade 1B/1BF rules to results")
        apply_grade_1bf_rules(results)
    else:
        return