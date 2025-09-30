"""
Test script to verify coordinate extraction with both integer and decimal coordinates
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordinate_extraction import extract_coordinates_from_text
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_coordinate_extraction():
    """Test coordinate extraction with various formats"""
    
    # Test case 1: Mixed integer and decimal coordinates
    test_text = """
    COORDS POINTS
    P0 100 200 300 10
    P1 150.5 250.5 350.5 15.5
    P2 -200 -300.5 400 20
    P3 300.5 400 -500.5
    """
    
    logger.info("Testing coordinate extraction with mixed format coordinates...")
    coordinates = extract_coordinates_from_text(test_text)
    
    if not coordinates:
        logger.error("Failed to extract any coordinates")
        return False
        
    logger.info(f"Successfully extracted {len(coordinates)} points:")
    for point in coordinates:
        logger.info(f"Point {point['point']}: x={point['x']}, y={point['y']}, z={point['z']}" + 
                   (f", r={point['r']}" if 'r' in point else ""))
                   
    # Validate specific test cases
    expected_points = 4
    actual_points = len(coordinates)
    if actual_points != expected_points:
        logger.error(f"Expected {expected_points} points, but got {actual_points}")
        return False
        
    # Check P0 (all integers)
    p0 = next((p for p in coordinates if p['point'] == 'P0'), None)
    if not p0 or p0['x'] != 100 or p0['y'] != 200 or p0['z'] != 300 or p0['r'] != 10:
        logger.error("Failed to correctly parse integer coordinates for P0")
        return False
        
    # Check P1 (all decimals)
    p1 = next((p for p in coordinates if p['point'] == 'P1'), None)
    if not p1 or p1['x'] != 150.5 or p1['y'] != 250.5 or p1['z'] != 350.5 or p1['r'] != 15.5:
        logger.error("Failed to correctly parse decimal coordinates for P1")
        return False
        
    # Check P2 (mixed format with negative numbers)
    p2 = next((p for p in coordinates if p['point'] == 'P2'), None)
    if not p2 or p2['x'] != -200 or p2['y'] != -300.5 or p2['z'] != 400 or p2['r'] != 20:
        logger.error("Failed to correctly parse mixed format coordinates for P2")
        return False
        
    # Check P3 (mixed format without radius)
    p3 = next((p for p in coordinates if p['point'] == 'P3'), None)
    if not p3 or p3['x'] != 300.5 or p3['y'] != 400 or p3['z'] != -500.5 or 'r' in p3:
        logger.error("Failed to correctly parse coordinates with optional radius for P3")
        return False
    
    logger.info("All coordinate extraction tests passed!")
    return True

if __name__ == "__main__":
    success = test_coordinate_extraction()
    sys.exit(0 if success else 1)