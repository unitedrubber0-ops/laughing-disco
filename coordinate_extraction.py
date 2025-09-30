# temporary file for coordinate extraction function
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text_encoding(text):
    """
    Clean and normalize text.
    """
    if not text:
        return ""
    try:
        # Remove non-printable characters and normalize whitespace
        text = ''.join(char if char.isprintable() or char in '\n\t' else ' ' for char in text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text encoding: {e}")
        return text if text else ""

def extract_coordinates_from_text(text):
    """
    Enhanced coordinate extraction for the specific PDF table format
    """
    coordinates = []
    
    try:
        # Clean and normalize text
        text = clean_text_encoding(text)
        
        logger.info("Starting coordinate extraction...")
        
        # Look for the coordinate table section
        coord_section_match = re.search(r'COORDS\s+POINTS\s+(.*?)(?:\n\s*\n|\Z)', text, re.DOTALL | re.IGNORECASE)
        if coord_section_match:
            coord_section = coord_section_match.group(1)
            logger.info(f"Found coordinate section: {coord_section[:200]}...")
            
            # Pattern for coordinate lines: P0, P1, etc. with X, Y, Z, R (handles integers and floats)
            coord_pattern = r'P(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(?:\s+(-?\d+(?:\.\d+)?))?'
            
            matches = re.finditer(coord_pattern, coord_section)
            
            for match in matches:
                try:
                    point_num = int(match.group(1))
                    x = float(match.group(2))
                    y = float(match.group(3)) 
                    z = float(match.group(4))
                    r = match.group(5)  # Optional radius
                    
                    point_data = {
                        'point': f'P{point_num}',
                        'x': x,
                        'y': y,
                        'z': z
                    }
                    
                    if r:
                        point_data['r'] = float(r)
                    
                    coordinates.append(point_data)
                    logger.info(f"Extracted point {point_data['point']}: ({x}, {y}, {z}) R={r if r else 'None'}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid coordinate data at point P{point_num}: {e}")
                    continue
        
        # Alternative pattern for the entire table
        if not coordinates:
            logger.info("Trying alternative coordinate pattern...")
            # Pattern that matches the full table structure (handles integers and floats)
            alt_pattern = r'P(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(?:\s+(-?\d+(?:\.\d+)?))?\s*\n'
            matches = re.finditer(alt_pattern, text)
            
            for match in matches:
                try:
                    point_num = int(match.group(1))
                    x = float(match.group(2))
                    y = float(match.group(3))
                    z = float(match.group(4))
                    r = match.group(5)
                    
                    point_data = {
                        'point': f'P{point_num}',
                        'x': x,
                        'y': y, 
                        'z': z
                    }
                    
                    if r:
                        point_data['r'] = float(r)
                    
                    coordinates.append(point_data)
                    logger.info(f"Alt pattern match - Point {point_data['point']}: ({x}, {y}, {z}) R={r if r else 'None'}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid coordinate data at point P{point_num}: {e}")
                    continue
        
        # Sort by point number
        coordinates.sort(key=lambda p: int(p['point'][1:]))
        
        # Validate sequence
        if coordinates:
            point_numbers = {int(p['point'][1:]) for p in coordinates}
            expected_numbers = set(range(min(point_numbers), max(point_numbers) + 1))
            missing = expected_numbers - point_numbers
            
            if missing:
                logger.warning(f"Missing points in sequence: P{', P'.join(map(str, missing))}")
        
        logger.info(f"Successfully extracted {len(coordinates)} coordinate points")
        
        return coordinates
        
    except Exception as e:
        logger.error(f"Error extracting coordinates: {e}")
        return []