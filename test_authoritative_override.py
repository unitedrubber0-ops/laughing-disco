#!/usr/bin/env python3
"""
Test suite for authoritative TABLE-4/8 override for MPAPS F-30/Grade 1.
Verifies that thickness and OD are set from table and not overwritten by computed values.
"""

import logging
from mpaps_utils import process_mpaps_dimensions

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

def test_grade1_authoritative_override():
    """
    Test: MPAPS F-30, Grade 1, ID 24.4mm → TABLE-4 thickness 4.30 ± 0.80 mm, OD 33.20 mm
    Expected provenance: thickness_source='TABLE-4/8 Grade-1 authoritative'
    Expected OD: computed from ID + 2*thickness if not in TABLE-8, or from TABLE-8 if present
    """
    input_data = {
        'standard': 'MPAPS F-30',
        'grade': '1B',
        'id_nominal_mm': 24.4,
        'part_number': '4509347C4'
    }
    
    print("\n" + "="*80)
    print("TEST: MPAPS F-30, Grade 1, ID 24.4mm (Part 4509347C4)")
    print("="*80)
    print(f"INPUT: {input_data}")
    
    result = process_mpaps_dimensions(input_data)
    
    print(f"\nOUTPUT:")
    print(f"  thickness_mm: {result.get('thickness_mm')} mm")
    print(f"  thickness_tolerance_mm: {result.get('thickness_tolerance_mm')} mm")
    print(f"  thickness_source: {result.get('thickness_source')}")
    print(f"  od_nominal_mm: {result.get('od_nominal_mm')} mm")
    print(f"  od_source: {result.get('od_source')}")
    print(f"  id_nominal_mm: {result.get('id_nominal_mm')} mm")
    print(f"  id_tolerance_mm: {result.get('id_tolerance_mm')} mm")
    print(f"  dimension_source: {result.get('dimension_source')}")
    
    # Verify authoritative values
    expected_thickness = 4.30
    expected_thickness_tol = 0.80
    # For ID 24.4: OD = 24.4 + 2*4.3 = 33.0 mm (NOT 33.2 which is for ID 24.6)
    expected_od = 24.4 + 2 * 4.30
    expected_thickness_source = 'TABLE-4/8 Grade-1 authoritative'
    
    assertions = [
        (result.get('thickness_mm') == expected_thickness, 
         f"thickness_mm should be {expected_thickness}, got {result.get('thickness_mm')}"),
        (result.get('thickness_tolerance_mm') == expected_thickness_tol,
         f"thickness_tolerance_mm should be {expected_thickness_tol}, got {result.get('thickness_tolerance_mm')}"),
        (result.get('od_nominal_mm') == expected_od,
         f"od_nominal_mm should be {expected_od}, got {result.get('od_nominal_mm')}"),
        (result.get('thickness_source') == expected_thickness_source,
         f"thickness_source should be '{expected_thickness_source}', got '{result.get('thickness_source')}'"),
        (result.get('id_tolerance_mm') == 0.5,
         f"id_tolerance_mm should be 0.5, got {result.get('id_tolerance_mm')}"),
    ]
    
    all_passed = True
    for condition, message in assertions:
        if condition:
            print(f"  ✓ {message.split(' should ')[1].split(',')[0]}")
        else:
            print(f"  ✗ {message}")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
    
    return all_passed

def test_grade1bf_authoritative_override():
    """
    Test: MPAPS F-30, Grade 1BF, ID 24.6mm → TABLE-8 thickness 4.30 ± 0.80 mm, OD 33.20 mm
    Expected provenance: thickness_source='TABLE-4/8 Grade-1 authoritative'
    """
    input_data = {
        'standard': 'MPAPS F-30',
        'grade': '1BF',
        'id_nominal_mm': 24.6,
        'part_number': 'TEST-1BF'
    }
    
    print("\n" + "="*80)
    print("TEST: MPAPS F-30, Grade 1BF, ID 24.6mm (Exact TABLE-8 match)")
    print("="*80)
    print(f"INPUT: {input_data}")
    
    result = process_mpaps_dimensions(input_data)
    
    print(f"\nOUTPUT:")
    print(f"  thickness_mm: {result.get('thickness_mm')} mm")
    print(f"  thickness_tolerance_mm: {result.get('thickness_tolerance_mm')} mm")
    print(f"  thickness_source: {result.get('thickness_source')}")
    print(f"  od_nominal_mm: {result.get('od_nominal_mm')} mm")
    print(f"  od_source: {result.get('od_source')}")
    
    # Verify authoritative values from TABLE 8
    expected_thickness = 4.30
    expected_thickness_tol = 0.80
    expected_od = 33.20
    
    assertions = [
        (result.get('thickness_mm') == expected_thickness, 
         f"thickness_mm should be {expected_thickness}, got {result.get('thickness_mm')}"),
        (result.get('thickness_tolerance_mm') == expected_thickness_tol,
         f"thickness_tolerance_mm should be {expected_thickness_tol}, got {result.get('thickness_tolerance_mm')}"),
        (result.get('od_nominal_mm') == expected_od,
         f"od_nominal_mm should be {expected_od}, got {result.get('od_nominal_mm')}"),
    ]
    
    all_passed = True
    for condition, message in assertions:
        if condition:
            print(f"  ✓ {message.split(' should ')[1].split(',')[0]}")
        else:
            print(f"  ✗ {message}")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
    
    return all_passed

if __name__ == '__main__':
    test1 = test_grade1_authoritative_override()
    test2 = test_grade1bf_authoritative_override()
    
    print("\n" + "="*80)
    if test1 and test2:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        exit(0)
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        exit(1)
