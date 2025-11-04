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

def get_burst_pressure() -> float:
    """Get the standard burst pressure for MPAPS F-6032 materials."""
    return MPAPS_F6032_BURST_PRESSURE_MPA

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
            
        # Find nearest nominal value
        if not nominals:  # Handle empty lists
            return None
            
        diffs = [abs(n - value_mm) for n in nominals]
        idx = min(range(len(diffs)), key=lambda i: diffs[i])
        nearest = nominals[idx]
        tolerance = tolerances[idx]
        
        # Build result
        result = {
            'original': dim_num,
            'value_mm': value_mm,
            'nearest_mm': nearest,
            'tol_mm': tolerance,
            'was_inches': was_inches,
            'formatted': f"{dim_num:.2f} ± {tolerance:.2f} mm"
        }
        
        logging.debug(f"MPAPS F-6032 {dimension_type} tolerance lookup: {result}")
        return result
        
    except Exception as e:
        logging.error(f"Error in MPAPS F-6032 tolerance lookup: {e}")
        return None

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

def apply_mpaps_f6032_rules(results: Dict[str, Any]) -> None:
    """
    Apply MPAPS F-6032 rules to analysis results.
    Modifies results dict in place to add tolerances and burst pressure.
    """
    # Check multiple fields for MPAPS F-6032 indication
    material = results.get('material')
    standard = results.get('standard')
    specification = results.get('specification')
    
    is_mpaps = (material and is_mpaps_f6032(material)) or \
               (standard and is_mpaps_f6032(standard)) or \
               (specification and is_mpaps_f6032(specification))
    
    if not is_mpaps:
        return
        
    logging.info("Applying MPAPS F-6032 rules to results")
    
    # Get dimensions from all possible locations
    dimensions = results.get('dimensions', {})
    
    # Check for ID in multiple locations with type conversion
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
        id_tol = get_mpaps_f6032_tolerance(id_val, 'ID')
        if id_tol:
            results['id_tolerance'] = id_tol['formatted']
            results['id_nearest_nominal'] = id_tol['nearest_mm']
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