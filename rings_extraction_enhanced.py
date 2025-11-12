import re
import logging
import re
import logging
from text_utils import _coerce_to_text, clean_text_encoding

logger = logging.getLogger(__name__)

def extract_rings_from_text_specific(text):
    """Enhanced rings extraction for specific PDF format - returns only explicitly found rings"""
    try:
        # Clean and normalize input text
        text = clean_text_encoding(_coerce_to_text(text))
        if not text:
            logger.info("No text provided for rings extraction")
            return "No Rings"
            
        # Look for explicit rings mentions - no fallbacks
        rings_patterns = [
            r'RINGS?\s*:\s*([^\n.,;]+(?:[.,;]\s*[^\n.,;]+)*)',  # RINGS: followed by text
            r'RINGS?\s*-\s*([^\n.,;]+(?:[.,;]\s*[^\n.,;]+)*)',  # RINGS - followed by text
            r'(\d+\s*X\s*RING[^\n.,;]*(?:[.,;]\s*[^\n.,;]+)*)',  # 2X RING etc.
            r'(RING\s*REINFORCEMENT[^\n.,;]*(?:[.,;]\s*[^\n.,;]+)*)',  # RING REINFORCEMENT
        ]
        
        for pattern in rings_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Safely extract the text
                if match.lastindex and match.lastindex >= 1:
                    rings_text = match.group(1).strip()
                else:
                    rings_text = match.group(0).strip()
                
                # Clean up the text
                rings_text = re.sub(r'\s+', ' ', rings_text)
                rings_text = rings_text.strip(' ,.-;')
                
                # Only return if we have meaningful content with "RING" in it
                if len(rings_text) > 5 and 'RING' in rings_text.upper():
                    logger.info(f"Found explicit rings information: {rings_text}")
                    return rings_text
        
        # If we get here, no rings were explicitly mentioned
        logger.info("No explicit rings information found in PDF - returning 'No Rings'")
        return "No Rings"
        
    except Exception as e:
        logger.error(f"Error extracting rings: {e}")
        return "No Rings"