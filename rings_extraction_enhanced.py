import re
import logging
import re
import logging
from text_utils import _coerce_to_text, clean_text_encoding

logger = logging.getLogger(__name__)

def extract_rings_from_text_specific(text):
    """Enhanced rings extraction for specific PDF format"""
    try:
        # Clean the text
        text = clean_text_encoding(text)
        
        # Use flexible pattern matching based on ring components
        components = {
            'quantity': r'(\d+)X?',
            'ring_type': r'RING(?:\s*REINFORCEMENT)?',
            'position': r'@\s*(\d+)',
            'material': r'(?:STAINLESS\s*WIRE|[A-Z]+\s*STEEL)'
        }
        
        # Look for combinations of components
        combined_pattern = fr"(?:{components['quantity']}\s*{components['ring_type']}.*?{components['position']}|{components['ring_type']}.*?{components['position']})"
        
        match = re.search(combined_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            # Extract the surrounding context to get full ring information
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            rings_text = text[start:end].strip()
            rings_text = re.sub(r'\s+', ' ', rings_text)  # Normalize spaces
            
            # Clean up the extracted text
            rings_text = rings_text.split('.')[0]  # Take only the first sentence
            rings_text = re.sub(r'[^\w\s@\-]', '', rings_text)  # Remove special chars except @-
            logger.info(f"Found rings information: {rings_text}")
            return rings_text
        
        logger.warning("No rings information found in text")
        return "Not Found"
        
    except Exception as e:
        logger.error(f"Error extracting rings: {e}")
        return "Not Found"