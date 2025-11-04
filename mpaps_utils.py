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
    return 'MPAPSF6032' in mat

def apply_mpaps_f6032_rules(results: Dict[str, Any]) -> None:
    """
    Apply MPAPS F-6032 rules to analysis results.
    Modifies results dict in place to add tolerances and burst pressure.
    """
    material = results.get('material')
    if not material or not is_mpaps_f6032(material):
        return
        
    logging.info("Applying MPAPS F-6032 rules to results")
    
    # Get dimensions from either top level or dimensions dict
    dimensions = results.get('dimensions', {})
    
    # Get ID tolerance if available
    id_val = results.get('id1') or dimensions.get('id1') or results.get('ID')
    if id_val is not None and id_val != "Not Found":
        id_tol = get_mpaps_f6032_tolerance(id_val, 'ID')
        if id_tol:
            results['id_tolerance'] = id_tol['formatted']
            results['id_nearest_nominal'] = id_tol['nearest_mm']
        else:
            results['id_tolerance'] = "Not Available"
            
    # Get OD tolerance if available  
    od_val = results.get('od1') or dimensions.get('od1') or results.get('OD')
    if od_val is not None and od_val != "Not Found":
        od_tol = get_mpaps_f6032_tolerance(od_val, 'OD')
        if od_tol:
            results['od_tolerance'] = od_tol['formatted']
            results['od_nearest_nominal'] = od_tol['nearest_mm']
        else:
            results['od_tolerance'] = "Not Available"
            
    # Set burst pressure
    results['burst_pressure_mpa'] = 2.0
    results['burst_pressure_psi'] = 290