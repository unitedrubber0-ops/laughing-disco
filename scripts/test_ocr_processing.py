"""
Test script to verify OCR text processing functionality, including rings extraction.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import process_ocr_text
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rings_extraction():
    """Test rings extraction with various text formats"""
    
    test_cases = [
        (
            """HOSE, HEATER, MPAPS F-30 GRADE 1B with 2 rings required""",
            "2"
        ),
        (
            """4717736X1 HEATER HOSE, 3 reinforcements TYPE 1B""",
            "3"
        ),
        (
            """STANDARD: MPAPS F-30\nGRADE: 1B\nRings: 4""",
            "4"
        ),
        (
            """Basic hose without explicit ring count""",
            "Not Found"
        ),
        (
            """HOSE reinforced with 5 rings TYPE 2B""",
            "5"
        )
    ]
    
    logger.info("Testing rings extraction with various formats...")
    
    success = True
    for test_text, expected_rings in test_cases:
        result = process_ocr_text(test_text)
        extracted_rings = result.get("rings", "Not Found") if result else "Not Found"
        
        if extracted_rings != expected_rings:
            logger.error(f"Ring extraction failed for text: {test_text}")
            logger.error(f"Expected: {expected_rings}, Got: {extracted_rings}")
            success = False
        else:
            logger.info(f"Successfully extracted rings '{extracted_rings}' from test case")
            
    return success

def run_all_tests():
    """Run all OCR processing tests"""
    all_passed = True
    
    # Run rings extraction tests
    logger.info("\nRunning rings extraction tests...")
    if not test_rings_extraction():
        logger.error("Rings extraction tests failed")
        all_passed = False
    else:
        logger.info("All rings extraction tests passed")
        
    return all_passed

if __name__ == "__main__":
    if run_all_tests():
        logger.info("\nAll OCR processing tests passed!")
        sys.exit(0)
    else:
        logger.error("\nSome OCR processing tests failed!")
        sys.exit(1)