import logging

logger = logging.getLogger(__name__)

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
        else:
            # If no specific key, take the first string value
            for value in standard.values():
                if isinstance(value, str):
                    standard = value
                    break
        logger.info(f"Extracted standard: {standard}")
    
    # If grade is a dict, extract the actual value  
    if isinstance(grade, dict):
        logger.warning("Grade is a dictionary, attempting to extract value")
        for key in ['grade', 'type', 'value', 'text']:
            if key in grade:
                grade = grade[key]
                break
        else:
            for value in grade.values():
                if isinstance(value, str):
                    grade = value
                    break
        logger.info(f"Extracted grade: {grade}")
    
    return str(standard), str(grade)

def map_tms_to_mpaps_standard(tms_standard):
    """Map TMS standards to equivalent MPAPS standards"""
    mapping = {
        "TMS-8034": "MPAPS F-6034",  # Your PDF shows TMS-8034
        "TMS-6034": "MPAPS F-6034",
        "TMS-6028": "MPAPS F-6028", 
        "TMS-6032": "MPAPS F-6032",
    }
    
    tms_standard = str(tms_standard).strip()
    for tms, mpaps in mapping.items():
        if tms in tms_standard:
            logger.info(f"Mapped {tms_standard} to {mpaps}")
            return mpaps
    
    return tms_standard