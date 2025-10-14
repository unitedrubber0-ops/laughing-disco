import logging
import math
from typing import List, Dict, Union, Optional

logger = logging.getLogger(__name__)

def validate_coordinate_point(point: Dict) -> Optional[Dict]:
    """
    Validate and normalize a single coordinate point.
    Returns None if point is invalid.
    """
    if not isinstance(point, dict):
        return None
    
    try:
        x = float(point.get('x', 0))
        y = float(point.get('y', 0))
        z = float(point.get('z', 0))
        
        validated = {
            'point': str(point.get('point', '')),
            'x': x,
            'y': y,
            'z': z
        }
        
        # Optional radius
        if 'r' in point and point['r'] not in (None, ''):
            try:
                validated['r'] = float(point['r'])
            except (ValueError, TypeError):
                pass
                
        return validated
    except (ValueError, TypeError) as e:
        logger.debug(f"Could not validate point {point}: {e}")
        return None

def validate_coordinates(coords: List[Dict]) -> List[Dict]:
    """
    Validate a list of coordinate points.
    Returns only valid points with numeric x,y,z values.
    """
    if not isinstance(coords, list):
        logger.warning(f"Expected list of coordinates, got {type(coords)}")
        return []
        
    validated = []
    for p in coords:
        valid_point = validate_coordinate_point(p)
        if valid_point:
            validated.append(valid_point)
        
    if len(validated) < 2:
        logger.warning(f"Not enough valid coordinate points (found {len(validated)})")
    else:
        logger.info(f"Validated {len(validated)} coordinate points")
        
    return validated

def calculate_development_length(coords: List[Dict]) -> float:
    """
    Calculate development length from a list of validated coordinate points.
    """
    validated = validate_coordinates(coords)
    if len(validated) < 2:
        logger.warning("Not enough valid coordinates to compute development length")
        return 0.0
        
    try:
        total_length = 0.0
        for i in range(len(validated) - 1):
            p1 = validated[i]
            p2 = validated[i + 1]
            
            dx = p2['x'] - p1['x']
            dy = p2['y'] - p1['y']
            dz = p2['z'] - p1['z']
            
            segment_length = math.sqrt(dx*dx + dy*dy + dz*dz)
            total_length += segment_length
            
        logger.info(f"Calculated development length: {total_length:.2f}")
        return total_length
        
    except Exception as e:
        logger.error(f"Error calculating development length: {e}")
        return 0.0