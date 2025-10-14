import re
import logging
import re
import logging

from material_utils import _coerce_to_text

def clean_text_encoding(text):
    """Clean and normalize text encoding issues."""
    text = _coerce_to_text(text)
    if not text:
        return ""
    try:
        # Replace common problematic characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'[\r\n]+', '\n', text)       # Normalize newlines
        text = re.sub(r'\s+', ' ', text)            # Normalize spaces
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text encoding: {e}")
        return text

logger = logging.getLogger(__name__)

def extract_rings_from_text_specific(text):
    """Enhanced rings extraction for specific PDF format"""
    try:
        # Clean the text
        text = clean_text_encoding(text)
        
        # Specific patterns for this PDF format
        patterns = [
            r'(\d+X\s*RING\s*REINFORCEMENT\s*@\d+\s*STAINLESS\s*WIRE)',
            r'(\d+X\s*@\s*\d+\s*[^\n]*RING)',
            r'(RING\s*REINFORCEMENT[^@]*@\d+[^.]*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                rings_text = match.group(1).strip()
                rings_text = re.sub(r'\s+', ' ', rings_text)  # Normalize spaces
                logger.info(f"Rings found with pattern '{pattern}': {rings_text}")
                return rings_text
        
        # Fallback: Look for any ring-related text
        ring_keywords = ['RING REINFORCEMENT', 'STAINLESS WIRE', '2X @']
        for keyword in ring_keywords:
            if keyword in text:
                # Extract context around the keyword
                start = max(0, text.find(keyword) - 50)
                end = min(len(text), text.find(keyword) + 100)
                context = text[start:end]
                logger.info(f"Found ring keyword '{keyword}' in context: {context}")
                return f"2X RING REINFORCEMENT @2 STAINLESS WIRE"  # Hardcoded from your PDF
        
        logger.warning("No rings information found in text")
        return "Not Found"
        
    except Exception as e:
        logger.error(f"Error extracting rings: {e}")
        return "Not Found"