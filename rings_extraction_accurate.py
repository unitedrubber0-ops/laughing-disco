import re
import logging
from text_utils import _coerce_to_text

logger = logging.getLogger(__name__)

def clean_text_encoding_for_rings(text):
    """Enhanced text cleaning that preserves rings-related formatting"""
    if not text:
        return ""
        
    try:
        # Preserve rings-related patterns
        # Use component-based pattern building for maximum flexibility
        base_components = {
            'quantity': r'\d+X?',
            'type': r'RING(?:S|\s*REINFORCEMENT)?',
            'position': r'@\s*\d+',
            'material': r'(?:STAINLESS\s*(?:STEEL|WIRE)|[A-Z]+\s*STEEL|[A-Z]{2,4})'
        }
        
        rings_preserve_patterns = [
            fr'({base_components["quantity"]}\s*{base_components["type"]})',
            fr'({base_components["type"]})',
            fr'({base_components["material"]})',
            fr'({base_components["position"]}[^\.]*)'
        ]
        
        # First, extract and protect rings patterns
        protected_patterns = []
        for pattern in rings_preserve_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                protected_patterns.append(match.group(0))
        
        # Do basic cleaning
        text = ''.join(char if char.isprintable() or char in '\n\t' else ' ' for char in text)
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        # Restore protected patterns (though they shouldn't be affected by basic cleaning)
        # This is more for future-proofing
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error cleaning text for rings: {str(e)}")
        return text if text else ""

def extract_rings_from_text_accurate(text):
    """Extract ONLY rings information that is explicitly mentioned in the text"""
    try:
        # First coerce any input to text and clean it
        text = _coerce_to_text(text)
        text = clean_text_encoding_for_rings(text)
        
        # Look for explicit rings mentions - only return if explicitly found
        # Build comprehensive patterns from components
        base_components = {
            'quantity': r'(?:\d+X?|\d+\s*(?:PCS|TIMES))',
            'type': r'(?:RING(?:S|\s*REINFORCEMENT)?)',
            'position': r'(?:@\s*\d+|INNER|OUTER|MIDDLE)',
            'material': r'(?:STAINLESS\s*(?:STEEL|WIRE)|[A-Z]+\s*STEEL|[A-Z]{2,4})'
        }
        
        # Build flexible patterns that can match various combinations
        rings_patterns = [
            fr"{base_components['quantity']}\s*{base_components['type']}\s*{base_components['position']}?[^\n.,;]*",
            fr"{base_components['type']}\s*[:-]\s*([^\n.,;]+(?:[.,;]\s*[^\n.,;]+)*)",
            fr"{base_components['type']}\s*{base_components['material']}[^\n.,;]*",
            fr"(?:{base_components['position']}\s*)?{base_components['type']}\s*(?:{base_components['material']})?[^\n.,;]*"
        ]
        
        for pattern in rings_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                rings_text = match.group(1).strip()
                # Clean up the text
                rings_text = re.sub(r'\s+', ' ', rings_text)  # Normalize spaces
                rings_text = rings_text.strip(' ,.-;')
                
                # Only return if we have meaningful content (not just "RING" alone)
                if len(rings_text) > 5 and 'RING' in rings_text.upper():
                    logger.info(f"Found explicit rings information: {rings_text}")
                    return rings_text
        
        # If we get here, no rings were explicitly mentioned
        logger.info("No explicit rings information found in PDF")
        return "No Rings"
        
    except Exception as e:
        logger.error(f"Error extracting rings: {e}")
        return "No Rings"