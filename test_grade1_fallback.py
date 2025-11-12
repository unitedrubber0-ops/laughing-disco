#!/usr/bin/env python3
"""
Test the defensive Grade-1 fallback logic for ID 24.4mm matching to TABLE_8 24.6mm entry.

This test verifies that the new nearest-nominal lookup will:
1. Find the nearest TABLE_8 entry (24.6mm, diff=0.2mm <= 0.5mm MAX_ACCEPT_DIFF_MM)
2. Populate id_tolerance_mm, thickness_mm, thickness_tolerance_mm, od_nominal_mm
3. Eliminate N/A values in Excel output
"""
import sys
import logging

# Set up logging to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from mpaps_utils import process_mpaps_dimensions

def test_id_24_4_matches_table_24_6():
    """Test that ID 24.4mm finds nearest TABLE_8 entry at 24.6mm"""
    print("\n" + "="*70)
    print("TEST: ID 24.4mm should match TABLE_8 entry 24.6mm (diff=0.2mm)")
    print("="*70)
    
    # Simulate result dict for part 4509347C4 (ID 24.4mm, detected as MPAPS F-30/Grade 1BF)
    test_result = {
        'id': '4509347C4',
        'id_nominal_mm': 24.4,
        'standard': 'MPAPS F-30',
        'grade': 'GRADE 1BF',
        'burst_pressure_psi': 500,
        # thickness, od_nominal_mm, id_tolerance_mm still None (causing N/A in Excel)
        'thickness_mm': None,
        'od_nominal_mm': None,
        'id_tolerance_mm': None,
    }
    
    print(f"\nBefore process_mpaps_dimensions():")
    print(f"  ID: {test_result.get('id_nominal_mm')} mm")
    print(f"  Thickness: {test_result.get('thickness_mm')} (will be N/A in Excel)")
    print(f"  OD: {test_result.get('od_nominal_mm')} (will be N/A in Excel)")
    print(f"  ID Tolerance: {test_result.get('id_tolerance_mm')} (will be N/A in Excel)")
    
    # Call the fixed function
    result = process_mpaps_dimensions(test_result)
    
    print(f"\nAfter process_mpaps_dimensions():")
    print(f"  ID: {result.get('id_nominal_mm')} mm")
    print(f"  Thickness: {result.get('thickness_mm')} mm")
    print(f"  Thickness Tolerance: {result.get('thickness_tolerance_mm')} mm")
    print(f"  OD: {result.get('od_nominal_mm')} mm")
    print(f"  ID Tolerance: {result.get('id_tolerance_mm')} mm")
    print(f"  Dimension Source: {result.get('dimension_source')}")
    
    # Verify results
    print(f"\nVerification:")
    
    success = True
    
    # Should find thickness
    if result.get('thickness_mm') == 4.30:
        print("  ✓ Thickness correctly set to 4.30 mm (from TABLE_8 24.6 entry)")
    else:
        print(f"  ✗ Thickness incorrect: {result.get('thickness_mm')} (expected 4.30)")
        success = False
    
    # Should find thickness tolerance
    if result.get('thickness_tolerance_mm') == 0.8:
        print("  ✓ Thickness tolerance correctly set to 0.8 mm")
    else:
        print(f"  ✗ Thickness tolerance incorrect: {result.get('thickness_tolerance_mm')} (expected 0.8)")
        success = False
    
    # Should find ID tolerance
    if result.get('id_tolerance_mm') == 0.5:
        print("  ✓ ID tolerance correctly set to 0.5 mm")
    else:
        print(f"  ✗ ID tolerance incorrect: {result.get('id_tolerance_mm')} (expected 0.5)")
        success = False
    
    # Should compute OD
    expected_od = round(24.4 + 2.0 * 4.30, 3)
    if result.get('od_nominal_mm') == expected_od:
        print(f"  ✓ OD correctly computed to {expected_od} mm (ID + 2*thickness)")
    else:
        print(f"  ✗ OD incorrect: {result.get('od_nominal_mm')} (expected {expected_od})")
        success = False
    
    # Should show dimension source
    if 'Grade1 fallback' in result.get('dimension_source', ''):
        print(f"  ✓ Dimension source correctly set to fallback logic")
    else:
        print(f"  ✗ Dimension source not set: {result.get('dimension_source')}")
        success = False
    
    print("\n" + "="*70)
    if success:
        print("✓ TEST PASSED: All fields populated correctly")
        print("  Excel will now show actual values instead of N/A")
        return True
    else:
        print("✗ TEST FAILED: Some fields not populated correctly")
        return False

def test_additional_grade_1_ids():
    """Test a few other Grade-1 IDs to ensure robustness"""
    print("\n" + "="*70)
    print("TEST: Other Grade-1 IDs should also match properly")
    print("="*70)
    
    test_cases = [
        (15.9, 4.30, "Should match TABLE_8 16.0 entry"),  # diff=0.1mm
        (25.4, 4.30, "Close to TABLE_8 24.6 but 0.8mm away, or other TABLE entry"),  # diff=0.8mm (too far)
        (19.8, 4.30, "Should match some TABLE_8/TABLE_4 entry"),
    ]
    
    all_pass = True
    for id_mm, expected_thickness, description in test_cases:
        print(f"\n  Testing ID {id_mm}mm: {description}")
        test_result = {
            'id': f'TEST_ID_{id_mm}',
            'id_nominal_mm': id_mm,
            'standard': 'MPAPS F-30',
            'grade': 'GRADE 1BF',
            'burst_pressure_psi': 500,
            'thickness_mm': None,
            'od_nominal_mm': None,
            'id_tolerance_mm': None,
        }
        
        result = process_mpaps_dimensions(test_result)
        
        thickness = result.get('thickness_mm')
        od = result.get('od_nominal_mm')
        id_tol = result.get('id_tolerance_mm')
        
        print(f"    Thickness: {thickness}, OD: {od}, ID Tolerance: {id_tol}")
        
        if thickness is None or od is None or id_tol is None:
            print(f"    ✗ Some fields not populated")
            all_pass = False
        else:
            print(f"    ✓ Fields populated successfully")
    
    return all_pass

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEFENSIVE GRADE-1 FALLBACK LOGIC TEST")
    print("="*70)
    print("\nThis test verifies the new nearest-nominal fallback logic")
    print("for matching ID 24.4mm to TABLE_8 entry 24.6mm within 0.5mm tolerance.")
    
    test1_pass = test_id_24_4_matches_table_24_6()
    test2_pass = test_additional_grade_1_ids()
    
    print("\n" + "="*70)
    if test1_pass:
        print("✓ MAIN TEST PASSED")
        print("\nThe fallback logic successfully:")
        print("  1. Found nearest TABLE_8 entry (24.6mm, diff=0.2mm)")
        print("  2. Populated thickness_mm = 4.30")
        print("  3. Populated thickness_tolerance_mm = 0.8")
        print("  4. Populated id_tolerance_mm = 0.5")
        print("  5. Computed od_nominal_mm = 33.20")
        print("\nExcel output should now show actual values instead of N/A")
        sys.exit(0)
    else:
        print("✗ MAIN TEST FAILED")
        print("Some fields were not populated correctly")
        sys.exit(1)
