"""
Module for cleaning and normalizing rings text extraction.
"""
import re
import logging

logger = logging.getLogger(__name__)

def clean_rings_text(rings_text):
    """Clean and normalize rings text with enhanced extraction"""
    if not rings_text or rings_text == "Not Found":
        return "Not Found"
    
    # Handle different input types
    if isinstance(rings_text, list):
        if rings_text:
            # Take first element if it's a list
            rings_text = rings_text[0]
        else:
            return "Not Found"
    
    if isinstance(rings_text, dict):
        # Extract from dictionary
        for key in ['text', 'description', 'type', 'value']:
            if key in rings_text and rings_text[key]:
                rings_text = rings_text[key]
                break
        else:
            return "Not Found"
    
    # Convert to string and clean
    rings_text = str(rings_text).strip()
    
    # Remove extra whitespace
    rings_text = re.sub(r'\s+', ' ', rings_text).strip()
    
    # Extract meaningful rings information
    rings_patterns = [
        r'(\d+X?\s*RING\s*REINFORCEMENT[^,.]*)',
        r'(RING[^,.]*REINFORCEMENT[^,.]*)',
        r'(\d+\s*X\s*@\d+\s*[^,.]*)',
    ]
    
    for pattern in rings_patterns:
        match = re.search(pattern, rings_text, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
            # Basic validation
            if len(cleaned) > 5:  # Meaningful content
                return cleaned
    
    # Final cleanup if no pattern matched
    rings_text = re.sub(r'\s*[.,;]\s*$', '', rings_text)
    
    return rings_text if len(rings_text.strip()) >= 3 else "Not Found"