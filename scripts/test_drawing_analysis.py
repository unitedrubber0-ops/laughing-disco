import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app import extract_text_from_pdf, process_ocr_text, extract_dimensions_from_text
from rings_detection import detect_rings_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_drawing(pdf_path):
    """Test analysis on a specific PDF drawing"""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # 1. Extract text
        text = extract_text_from_pdf(pdf_bytes)
        print("\n=== EXTRACTED TEXT (first 1000 chars) ===")
        print(text[:1000])
        
        # 2. Test rings detection
        rings_info = detect_rings_info(text)
        print("\n=== RINGS INFORMATION ===")
        for key, value in rings_info.items():
            print(f"{key}: {value}")
        
        # 3. Test dimension extraction
        dimensions = extract_dimensions_from_text(text)
        print("\n=== DIMENSIONS ===")
        for key, value in dimensions.items():
            print(f"{key}: {value}")
        
        # 4. Test full analysis
        analysis_results = process_ocr_text(text)
        if analysis_results:
            print("\n=== FULL ANALYSIS RESULTS ===")
            for key, value in analysis_results.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue}")
                else:
                    print(f"{key}: {value}")
        
        return {
            "rings_info": rings_info,
            "dimensions": dimensions,
            "analysis_results": analysis_results
        }
                
    except Exception as e:
        logger.error(f"Error testing drawing: {e}")
        raise

def validate_test_results(results):
    """Validate test results against expected values"""
    expected = {
        "rings_spec": "STAINLESS WIRE 2MM DIA",
        "id1": "65",
        "od1": "88.9",
        "id2": "82",
        "centerline_length": "152.0"
    }
    
    print("\n=== VALIDATION RESULTS ===")
    
    # Check rings
    rings_info = results.get("rings_info", {})
    if rings_info.get("rings_specification") == expected["rings_spec"]:
        print("✓ Rings specification matches expected value")
    else:
        print(f"✗ Rings specification mismatch:")
        print(f"  Expected: {expected['rings_spec']}")
        print(f"  Got: {rings_info.get('rings_specification')}")
    
    # Check dimensions
    dimensions = results.get("dimensions", {})
    for dim_key in ["id1", "od1", "id2", "centerline_length"]:
        if dimensions.get(dim_key) == expected[dim_key]:
            print(f"✓ {dim_key.upper()} matches expected value")
        else:
            print(f"✗ {dim_key.upper()} mismatch:")
            print(f"  Expected: {expected[dim_key]}")
            print(f"  Got: {dimensions.get(dim_key)}")

def main():
    """Main test function"""
    # Test specific drawing
    test_pdf = "3541592c1_s001-_r-d.pdf"  # Update this to your PDF name
    pdf_path = project_root / test_pdf
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    print(f"\nTesting drawing: {test_pdf}")
    print("=" * 80)
    
    results = test_specific_drawing(str(pdf_path))
    validate_test_results(results)

if __name__ == "__main__":
    main()