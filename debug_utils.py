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
            
            # Check if MPAPS F-6032
            if standard and 'MPAPS F-6032' in str(standard).upper():
                from mpaps_utils import TABLE_1_DATA
                logging.info("Checking MPAPS F-6032 TABLE 1 dimensions:")
                for row in TABLE_1_DATA:
                    nominal_id, nominal_id_mm, id_tol, nominal_od_mm, od_tol = row
                    diff = abs(nominal_id_mm - id_float)
                    logging.info(f"ID {id_float} vs nominal {nominal_id_mm} ({nominal_id}\"): "
                               f"diff={diff:.3f}mm, ID tol={id_tol}mm, OD={nominal_od_mm}±{od_tol}mm")
            
            # Check if MPAPS F-30/F-1 with Grade 1BF
            elif grade and ('1BF' in grade or 'BF' in grade):
                from mpaps_utils import _F30_BF_TABLE, _F30_BF_RANGES
                logging.info("Checking MPAPS F-30/F-1 TABLE VII-B dimensions:")
                
                # Check exact matches
                for row in _F30_BF_TABLE:
                    nom_in, nom_mm, id_tol, od_mm, wall_mm, wall_tol = row
                    if nom_mm:  # Skip range-based entries
                        diff = abs(id_float - nom_mm) if nom_mm else float('inf')
                        logging.info(f"ID {id_float} vs nominal {nom_mm}mm ({nom_in}\"): "
                                   f"diff={diff:.3f}mm, ID tol={id_tol}mm, "
                                   f"Wall={wall_mm}±{wall_tol}mm")
                
                # Check ranges
                for min_id, max_id, wall_mm, wall_tol in _F30_BF_RANGES:
                    if min_id <= id_float <= max_id:
                        logging.info(f"ID {id_float}mm in range {min_id}-{max_id}mm: "
                                   f"Wall={wall_mm}±{wall_tol}mm")
                
            # Check burst pressure tables
            if id_float:
                from mpaps_utils import TABLE_IV_BURST_PRESSURE
                logging.info("\nChecking burst pressure tables:")
                if grade:
                    logging.info(f"Grade {grade} lookup in TABLE IV:")
                    for row in TABLE_IV_BURST_PRESSURE:
                        over, thru = row[0], row[1]
                        if over < id_float <= thru:
                            grade_1_3 = row[2]  # B(1&3)
                            grade_2b = row[3]   # B(2)
                            grade_f1 = row[6]   # F(1)
                            logging.info(f"ID {id_float}mm in range {over}-{thru}mm:")
                            logging.info(f"  Suffix B Grade 1&3: {grade_1_3} MPa")
                            logging.info(f"  Suffix B Grade 2: {grade_2b} MPa")
                            logging.info(f"  Suffix F Grade 1: {grade_f1} MPa")
            
        except Exception as e:
            logging.error(f"Error in tolerance debug: {e}", exc_info=True)
    
    logging.info("=== END DEBUG ===")