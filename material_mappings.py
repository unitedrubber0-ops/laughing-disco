"""
Module for mapping between different material standards and specifications.
"""
import logging

logger = logging.getLogger(__name__)

def map_tms_to_mpaps_standard(tms_standard):
    """Map TMS standards to equivalent MPAPS standards"""
    mapping = {
        "TMS-6034": "MPAPS F-6034",
        "TMS-6028": "MPAPS F-6028", 
        "TMS-6032": "MPAPS F-6032",
        "TMS-6034 H-ANRX": "MPAPS F-6034",
    }
    
    for tms, mpaps in mapping.items():
        if tms in str(tms_standard):
            logger.info(f"Mapped {tms_standard} to {mpaps}")
            return mpaps
    
    return tms_standard

def debug_material_lookup(standard, grade):
    """Debug function to identify material lookup issues"""
    logger.info(f"Material lookup debug - Standard type: {type(standard)}, value: {standard}")
    logger.info(f"Material lookup debug - Grade type: {type(grade)}, value: {grade}")
    
    # If standard is a dict, extract the actual value
    if isinstance(standard, dict):
        logger.warning("Standard is a dictionary, attempting to extract value")
        # Try common keys
        for key in ['standard', 'specification', 'value', 'text']:
            if key in standard:
                standard = standard[key]
                break
        logger.info(f"Extracted standard: {standard}")
    
    # If grade is a dict, extract the actual value  
    if isinstance(grade, dict):
        logger.warning("Grade is a dictionary, attempting to extract value")
        for key in ['grade', 'type', 'value', 'text']:
            if key in grade:
                grade = grade[key]
                break
        logger.info(f"Extracted grade: {grade}")
    
    return standard, grade