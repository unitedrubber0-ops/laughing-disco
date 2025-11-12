#!/usr/bin/env python3
"""
Test script to verify authoritative Grade-1 thickness fix.

This test ensures:
1. Authoritative override sets thickness=4.30, tol=±0.80 for MPAPS F-30/Grade 1
2. ensure_result_fields() doesn't overwrite it with computed value from OD/ID
3. thickness_source is tracked to prevent overwrites
"""
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from mpaps_utils import process_mpaps_dimensions
from excel_output import ensure_result_fields

def test_authoritative_grade1_override():
    """Test that Grade-1 uses authoritative TABLE_4 thickness"""
    print("\n" + "="*70)
    print("TEST: Authoritative Grade-1 Thickness Override")
    print("="*70)
    
    # Simulate part 4509347C4 input
    test_result = {
        'id': '4509347C4',
        'standard': 'MPAPS F-30',
        'grade': 'GRADE 1BF',
        'id_nominal_mm': 24.4,
        'od_nominal_mm': 31.4,  # This might have been set by some earlier code
        'burst_pressure_psi': 500,
    }
    
    print(f"\nInput:")
    print(f"  Standard: {test_result.get('standard')}")
    print(f"  Grade: {test_result.get('grade')}")
    print(f"  ID: {test_result.get('id_nominal_mm')} mm")
    print(f"  OD (pre-set): {test_result.get('od_nominal_mm')} mm")
    print(f"  Thickness (before): {test_result.get('thickness_mm')}")
    
    # Step 1: Call process_mpaps_dimensions (should set authoritative thickness=4.30)
    print(f"\nStep 1: process_mpaps_dimensions()")
    result = process_mpaps_dimensions(test_result)
    
    print(f"  Thickness: {result.get('thickness_mm')} mm")
    print(f"  Thickness Tolerance: {result.get('thickness_tolerance_mm')} mm")
    print(f"  Thickness Source: {result.get('thickness_source')}")
    print(f"  OD: {result.get('od_nominal_mm')} mm")
    print(f"  ID Tolerance: {result.get('id_tolerance_mm')} mm")
    
    # Step 2: Call ensure_result_fields (should NOT overwrite thickness)
    print(f"\nStep 2: ensure_result_fields()")
    result2 = ensure_result_fields(result)
    
    print(f"  Thickness: {result2.get('thickness_mm')} mm")
    print(f"  Thickness Tolerance: {result2.get('thickness_tolerance_mm')} mm")
    print(f"  Thickness Source: {result2.get('thickness_source')}")
    print(f"  Thickness Formatted: {result2.get('thickness_formatted')}")
    print(f"  OD: {result2.get('od_nominal_mm')} mm")
    print(f"  OD Formatted: {result2.get('od_formatted')}")
    print(f"  ID: {result2.get('id_nominal_mm')} mm (±{result2.get('id_tolerance_mm')} mm)")
    
    print(f"\nVerification:")
    
    success = True
    
    # Check 1: Thickness should be 4.30 (not 3.50 computed from OD/ID)
    if result2.get('thickness_mm') == 4.30:
        print(f"  [OK] Thickness = 4.30 mm (AUTHORITATIVE from TABLE_4, not computed)")
    else:
        print(f"  [FAIL] Thickness = {result2.get('thickness_mm')} mm (expected 4.30)")
        success = False
    
    # Check 2: Tolerance should be 0.80 (not 0.25 computed)
    if result2.get('thickness_tolerance_mm') == 0.80:
        print(f"  [OK] Thickness Tolerance = 0.80 mm (AUTHORITATIVE from TABLE_4)")
    else:
        print(f"  [FAIL] Thickness Tolerance = {result2.get('thickness_tolerance_mm')} mm (expected 0.80)")
        success = False
    
    # Check 3: thickness_source should show TABLE_4
    thickness_source = result2.get('thickness_source')
    if 'TABLE_4' in str(thickness_source) or 'AUTHORITATIVE' in str(thickness_source):
        print(f"  [OK] Thickness Source = {thickness_source} (protected from overwrite)")
    else:
        print(f"  [FAIL] Thickness Source = {thickness_source} (expected TABLE_4_*)")
        success = False
    
    # Check 4: OD should be computed from ID + 2*thickness if not already set
    if result2.get('od_nominal_mm') is not None:
        expected_od = round(24.4 + 2*4.30, 3)
        if abs(float(result2.get('od_nominal_mm')) - expected_od) < 0.01:
            print(f"  [OK] OD = {result2.get('od_nominal_mm')} mm (computed from ID + 2*thickness)")
        else:
            print(f"  [WARN] OD = {result2.get('od_nominal_mm')} mm (expected ~{expected_od}, but value exists)")
    else:
        print(f"  [WARN] OD = None")
    
    # Check 5: ID tolerance should be 0.5
    if result2.get('id_tolerance_mm') == 0.5:
        print(f"  [OK] ID Tolerance = 0.5 mm (from TABLE_4/TABLE_8)")
    else:
        print(f"  [WARN] ID Tolerance = {result2.get('id_tolerance_mm')} mm (expected 0.5)")
    
    # Check 6: Excel formatted strings should show correct values
    thickness_fmt = result2.get('thickness_formatted')
    if thickness_fmt and '4.30' in str(thickness_fmt) and '0.80' in str(thickness_fmt):
        print(f"  [OK] Thickness Formatted = '{thickness_fmt}'")
    else:
        print(f"  [WARN] Thickness Formatted = '{thickness_fmt}' (expected '4.30 +/- 0.80 mm')")
    
    print("\n" + "="*70)
    if success:
        print("[PASS] TEST PASSED: Grade-1 uses authoritative TABLE_4 thickness")
        print("  Thickness is NOT overwritten by computed OD/ID formula")
        return True
    else:
        print("[FAIL] TEST FAILED: Some checks did not pass")
        return False

def test_multiple_id_values():
    """Test a few different ID values to ensure robustness"""
    print("\n" + "="*70)
    print("TEST: Multiple ID Values with Grade-1 Authorization")
    print("="*70)
    
    test_cases = [
        (24.4, "Close to nominal 24.6"),
        (15.9, "Close to nominal 16.0"),
        (25.4, "Exact nominal 25.4"),
    ]
    
    all_pass = True
    for id_mm, description in test_cases:
        print(f"\n  Testing ID {id_mm}mm: {description}")
        test_result = {
            'id': f'TEST_ID_{id_mm}',
            'standard': 'MPAPS F-30',
            'grade': '1B',
            'id_nominal_mm': id_mm,
        }
        
        result = process_mpaps_dimensions(test_result)
        result2 = ensure_result_fields(result)
        
        thickness = result2.get('thickness_mm')
        thickness_source = result2.get('thickness_source')
        
        if thickness == 4.30 and 'TABLE_4' in str(thickness_source):
            print(f"    [OK] Thickness = {thickness}, Source = {thickness_source}")
        else:
            print(f"    [FAIL] Thickness = {thickness}, Source = {thickness_source} (expected 4.30 from TABLE_4)")
            all_pass = False
    
    return all_pass

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AUTHORITATIVE GRADE-1 THICKNESS OVERRIDE TEST")
    print("="*70)
    print("\nVerifies that:")
    print("  1. Grade-1 parts use authoritative TABLE_4 thickness (4.30 mm, ±0.80)")
    print("  2. ensure_result_fields() doesn't overwrite with (OD-ID)/2 formula")
    print("  3. thickness_source tracking prevents accidental overwrites")
    
    test1_pass = test_authoritative_grade1_override()
    test2_pass = test_multiple_id_values()
    
    print("\n" + "="*70)
    if test1_pass and test2_pass:
        print("[PASS] ALL TESTS PASSED")
        print("\nExpected Excel output for part 4509347C4:")
        print("  ID1: 24.40 +/- 0.50 mm")
        print("  Thickness: 4.30 +/- 0.80 mm")
        print("  OD: 33.0 mm (or similar computed value)")
        print("\nNOT:")
        print("  Thickness: 3.50 +/- 0.25 mm (computed from 31.4-24.4)/2)")
        sys.exit(0)
    else:
        print("[FAIL] SOME TESTS FAILED")
        sys.exit(1)
