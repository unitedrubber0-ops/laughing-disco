"""
Debug utilities for tolerance lookup and dimension extraction.
"""

import logging
from typing import Dict, Any

def debug_tolerance_lookup(results: Dict[str, Any]) -> None:
    """Debug function to trace tolerance lookup issues"""
    logging.info("=== DEBUG TOLERANCE LOOKUP ===")
    
    dimensions = results.get('dimensions', {})
    id_val = dimensions.get('id1')
    standard = results.get('standard')
    grade = results.get('grade')
    
    logging.info(f"ID value: {id_val}")
    logging.info(f"Standard: {standard}")
    logging.info(f"Grade: {grade}")
    
    if id_val and id_val != "Not Found":
        try:
            id_float = float(str(id_val).replace(',', '.'))
            logging.info(f"ID as float: {id_float}")
            
            # Check Grade 1BF table directly
            from mpaps_utils import _GRADE_1BF_ID_NOMINALS_MM, _GRADE_1BF_ID_TOLS_MM
            logging.info(f"Grade 1BF ID nominals: {_GRADE_1BF_ID_NOMINALS_MM}")
            logging.info(f"Grade 1BF ID tolerances: {_GRADE_1BF_ID_TOLS_MM}")
            
            # Check if ID matches any nominal
            for i, nominal in enumerate(_GRADE_1BF_ID_NOMINALS_MM):
                diff = abs(nominal - id_float)
                logging.info(f"ID {id_float} vs nominal {nominal}: diff={diff}")
                if diff < 0.1:  # Close match
                    logging.info(f"Close match found at index {i}: {nominal} ± {_GRADE_1BF_ID_TOLS_MM[i]}mm")
                    
        except Exception as e:
            logging.error(f"Error in tolerance debug: {e}")
    
    logging.info("=== END DEBUG ===")