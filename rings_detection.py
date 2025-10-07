import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def detect_rings_info(text: str) -> Dict[str, Any]:
    """
    Detect if drawing has rings and extract ring count.
    Returns dictionary with rings information.
    """
    rings_info = {
        "has_rings": False,
        "ring_count": "Not Specified",
        "ring_description": "Not Found",
        "rings_specification": "Not Found"
    }
    
    try:
        if not isinstance(text, str):
            return rings_info
            
        # Check if drawing has rings by looking for ring-related keywords
        ring_keywords = [
            r'RING\s+REINFORCEMENT',
            r'RINGS?[:\s]',
            r'STAINLESS\s+(?:WIRE|RING)',
            r'\d+\s*PLACES.*RING',
            r'RING.*\d+\s*PLACES'
        ]
        
        has_rings = False
        ring_context = ""
        
        for pattern in ring_keywords:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                has_rings = True
                # Get context around the match (100 characters before and after)
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                ring_context = text[start:end]
                break
        
        rings_info["has_rings"] = has_rings
        
        if has_rings:
            # Extract ring count - look for patterns like "2 PLACES", "2 PLCS", etc.
            count_patterns = [
                r'(\d+)\s*PLACES',
                r'(\d+)\s*PLCS',
                r'(\d+)\s*PCS',
                r'(\d+)\s*NOS',
                r'(\d+)\s*QTY',
                r'QUANTITY[:\s]*(\d+)',
                r'QTY[:\s]*(\d+)'
            ]
            
            for pattern in count_patterns:
                count_match = re.search(pattern, ring_context, re.IGNORECASE)
                if count_match:
                    rings_info["ring_count"] = count_match.group(1)
                    break
            
            # Extract ring description
            desc_patterns = [
                r'(RING\s+REINFORCEMENT[^\n]*(?:\n[^\n]*){0,3})',
                r'(RINGS?[^\n]*(?:\n[^\n]*){0,3})',
                r'(STAINLESS[^\n]*(?:\n[^\n]*){0,3})'
            ]
            
            for pattern in desc_patterns:
                desc_match = re.search(pattern, ring_context, re.IGNORECASE)
                if desc_match:
                    description = desc_match.group(1).strip()
                    # Clean up the description
                    description = re.sub(r'\s+', ' ', description)
                    rings_info["ring_description"] = description
                    break
            
            # Extract ring specification
            spec_patterns = [
                r'STAINLESS\s+WIRE\s+([\d.]+)\s*MM\s*DIA',
                r'(\d+(?:\.\d+)?\s*MM\s*DIA\s*(?:STAINLESS|WIRE|RING))',
                r'(STAINLESS[^,\n]*(?:\d+(?:\.\d+)?[^,\n]*))'
            ]
            
            for pattern in spec_patterns:
                spec_match = re.search(pattern, ring_context, re.IGNORECASE)
                if spec_match:
                    specification = spec_match.group(1).strip()
                    rings_info["rings_specification"] = specification
                    break
        
        logger.info(f"Rings detection: has_rings={rings_info['has_rings']}, count={rings_info['ring_count']}")
        return rings_info
        
    except Exception as e:
        logger.error(f"Error detecting rings info: {e}")
        return rings_info