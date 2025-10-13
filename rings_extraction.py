"""
Module for rings extraction functionality.
"""
import re
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

def extract_rings_info(text: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract rings information from the text using RingsExtractor.
    
    Args:
        text: The text to analyze, can be None
        
    Returns:
        Dict containing rings count and types
    """
    if not text:
        return {"count": None, "types": []}
        
    extractor = RingsExtractor()
    rings_text = extractor.clean_rings_text(text)
    count = None
    types = []
    
    # Extract ring count
    count_match = re.search(r'(?:RING|RINGS|COUNT)\s*[=:]\s*(\d+)', rings_text, re.IGNORECASE)
    if count_match:
        count = int(count_match.group(1))
    
    # Extract ring types
    type_patterns = [
        r'(?:INNER|INSIDE)[:=]\s*([A-Z]+)',
        r'(?:OUTER|OUTSIDE)[:=]\s*([A-Z]+)'
    ]
    
    for pattern in type_patterns:
        match = re.search(pattern, rings_text, re.IGNORECASE)
        if match:
            types.append(match.group(0).upper().replace(' ', ''))
    
    return {"count": count, "types": types}

def extract_coordinates(text: Optional[str] = None) -> List[Tuple[float, float]]:
    """
    Extract coordinate pairs from text.
    
    Args:
        text: Text containing coordinate information, can be None
        
    Returns:
        List of coordinate tuples (x, y)
    """
    if not text:
        return []
        
    coordinates = []
    lines = scan_text_by_lines(text)
    
    for line in lines:
        # Look for coordinate patterns like (x,y) or x,y
        matches = re.finditer(r'\(?(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\)?', line)
        for match in matches:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                coordinates.append((x, y))
            except ValueError:
                continue
    
    return coordinates

from typing import Sequence

def polyline_length(coordinates: Optional[Sequence[Tuple[float, float]]] = None) -> float:
    """
    Calculate the total length of a polyline defined by coordinates.
    
    Args:
        coordinates: Sequence of (x,y) coordinate tuples, can be None or empty
        
    Returns:
        Total length of the polyline (0.0 if coordinates are invalid/empty)
    """
    if not coordinates or len(coordinates) < 2:
        return 0.0
        
    try:
        total_length = 0.0
        for i in range(len(coordinates) - 1):
            # Convert coordinates to float if they're integers
            x1, y1 = float(coordinates[i][0]), float(coordinates[i][1])
            x2, y2 = float(coordinates[i+1][0]), float(coordinates[i+1][1])
            segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_length += segment_length
        return total_length
    except (TypeError, ValueError):
        return 0.0

def scan_text_by_lines(text: Optional[str] = None) -> Dict[str, Any]:
    """
    Clean and split text into lines, processing for both rings info and coordinates.
    
    Args:
        text: Text to process, can be None
        
    Returns:
        Dictionary containing processed rings_info and coordinates
    """
    if not text:
        return {
            'rings_info': {'count': None, 'types': []},
            'coordinates': []
        }
        
    # Process rings info
    rings_info = extract_rings_info(text)
    
    # Process coordinates
    coordinates = extract_coordinates(text)
    
    return {
        'rings_info': rings_info,
        'coordinates': coordinates
    }

class RingsExtractor:
    """
    Class to handle different methods of rings extraction from text.
    """
    
    @staticmethod
    def clean_rings_text(text):
        """Clean and normalize rings text."""
        if not isinstance(text, str):
            return "Not Found"
            
        # Remove extra whitespace and normalize spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common noise words
        noise_words = ['approx', 'approximately', 'about']
        for word in noise_words:
            text = re.sub(fr'\b{word}\b', '', text, flags=re.IGNORECASE)
            
        return text.strip()

    @staticmethod
    def extract_by_pattern(text):
        """
        Extract rings information using pattern matching.
        """
        if not isinstance(text, str):
            return "Not Found"
            
        try:
            # Clean the text first
            text = RingsExtractor.clean_rings_text(text)
            
            # Look for ring count patterns
            rings_patterns = [
                r'(?:with|having|includes?)\s+(\d+)\s*(?:rings?|reinforcements?)',
                r'(?:HOSE|HEATER).*?(\d+)\s*(?:rings?|reinforcements?)(?:\s+|$)',
                r'(\d+)\s*(?:rings?|reinforcements?)\s+(?:required|needed|specified|TYPE)',
                r'(?:rings?|reinforcements?)\s*(?:count|number|qty|quantity)?\s*[:-]?\s*(\d+)',
                r'reinforced\s+with\s+(\d+)\s*rings?'
            ]
            
            for pattern in rings_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
                    
            return "Not Found"
            
        except Exception as e:
            logger.error(f"Error extracting rings by pattern: {str(e)}")
            return "Not Found"

    @staticmethod
    def extract_by_regex(text):
        """
        Extract rings count using regular expressions.
        """
        if not isinstance(text, str):
            return "Not Found"
            
        try:
            text = RingsExtractor.clean_rings_text(text)
            
            # Look for direct ring count mentions
            rings_regex = r'(\d+)\s*(?:ring|rings|reinforcement|reinforcements)'
            match = re.search(rings_regex, text, re.IGNORECASE)
            if match:
                return match.group(1)
                
            return "Not Found"
            
        except Exception as e:
            logger.error(f"Error extracting rings by regex: {str(e)}")
            return "Not Found"

    @staticmethod
    def extract_by_context(text):
        """
        Extract rings information using context-aware analysis.
        """
        if not isinstance(text, str):
            return "Not Found"
            
        try:
            text = RingsExtractor.clean_rings_text(text)
            
            # Look for rings in product description sections
            desc_regex = r'description:?\s*(.*?rings.*?)(?:\n|$)'
            match = re.search(desc_regex, text, re.IGNORECASE)
            if match:
                # Extract number from description
                num_match = re.search(r'(\d+)\s*rings?', match.group(1), re.IGNORECASE)
                if num_match:
                    return num_match.group(1)
                    
            return "Not Found"
            
        except Exception as e:
            logger.error(f"Error extracting rings by context: {str(e)}")
            return "Not Found"
            
    @staticmethod
    def extract_rings(text):
        """
        Main function to extract rings information, trying multiple methods.
        """
        if not isinstance(text, str):
            return "Not Found"
            
        # Try different extraction methods in order of reliability
        extractors = [
            RingsExtractor.extract_by_pattern,
            RingsExtractor.extract_by_regex,
            RingsExtractor.extract_by_context
        ]
        
        for extractor in extractors:
            result = extractor(text)
            if result != "Not Found":
                return result
                
        return "Not Found"