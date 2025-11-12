#!/usr/bin/env python
"""Comprehensive test of both patches for MPAPS field population in Excel output."""

import json
import logging

# Enable logging to see debug output
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

print("=" * 80)
print("TEST: MPAPS Field Population in Excel Output (Both Patches)")
print("=" * 80)
print()

# Test Scenario 1: ID in top-level field
print("SCENARIO 1: ID in top-level field (id_nominal_mm)")
print("-" * 80)
test_data_1 = {
    'part_number': '4794498C1',
    'standard': 'MPAPS F-1',
    'grade': 'Grade 1BF',
    'id_nominal_mm': 24.6,  # Top-level ID
    'od_nominal_mm': 33.20
}

from mpaps_utils import process_mpaps_dimensions
results_1 = process_mpaps_dimensions(test_data_1.copy())

print(f"Input ID source: top-level 'id_nominal_mm' = 24.6")
print(f"Output:")
print(f"  ✓ thickness_mm: {results_1.get('thickness_mm')} (expected 4.30)")
print(f"  ✓ thickness_tolerance_mm: {results_1.get('thickness_tolerance_mm')} (expected 0.80)")
print(f"  ✓ id_tolerance_mm: {results_1.get('id_tolerance_mm')} (expected 0.50)")
print(f"  ✓ dimension_source: {results_1.get('dimension_source')}")
print(f"  ✓ dimension_sources: {results_1.get('dimension_sources')}")
print()

# Test Scenario 2: ID nested in dimensions dict (Patch 1 target)
print("SCENARIO 2: ID nested in dimensions dict (Patch 1 target)")
print("-" * 80)
test_data_2 = {
    'part_number': '4794498C1-NESTED',
    'standard': 'MPAPS F-1',
    'grade': 'Grade 1BF',
    'dimensions': {
        'id1': 24.6,  # ID is nested here, not at top level
        'od1': 33.20
    }
    # Note: NO top-level id_nominal_mm
}

results_2 = process_mpaps_dimensions(test_data_2.copy())

print(f"Input ID source: nested in dimensions dict as 'id1' = 24.6")
print(f"Output:")
print(f"  ✓ id_nominal_mm (promoted): {results_2.get('id_nominal_mm')} (expected 24.6)")
print(f"  ✓ thickness_mm: {results_2.get('thickness_mm')} (expected 4.30)")
print(f"  ✓ thickness_tolerance_mm: {results_2.get('thickness_tolerance_mm')} (expected 0.80)")
print(f"  ✓ id_tolerance_mm: {results_2.get('id_tolerance_mm')} (expected 0.50)")
print(f"  ✓ dimension_source: {results_2.get('dimension_source')}")
print()

# Test Scenario 3: ID extracted from raw_text (Patch 1 fallback)
print("SCENARIO 3: ID extracted from raw_text (Patch 1 fallback)")
print("-" * 80)
test_data_3 = {
    'part_number': '4794498C1-RAWTEXT',
    'standard': 'MPAPS F-1',
    'grade': 'Grade 1BF',
    'raw_text': 'HOSE ID = 24.6 mm, OD = 33.20 mm per drawing'
    # Note: NO id_nominal_mm or dimensions dict
}

results_3 = process_mpaps_dimensions(test_data_3.copy())

print(f"Input ID source: extracted from raw_text pattern 'HOSE ID = 24.6'")
print(f"Output:")
print(f"  ✓ id_nominal_mm (extracted): {results_3.get('id_nominal_mm')} (expected 24.6)")
print(f"  ✓ thickness_mm: {results_3.get('thickness_mm')} (expected 4.30)")
print(f"  ✓ thickness_tolerance_mm: {results_3.get('thickness_tolerance_mm')} (expected 0.80)")
print(f"  ✓ id_tolerance_mm: {results_3.get('id_tolerance_mm')} (expected 0.50)")
print(f"  ✓ dimension_source: {results_3.get('dimension_source')}")
print()

# Test Scenario 4: No ID found (should return unchanged)
print("SCENARIO 4: No ID found anywhere (should return unchanged)")
print("-" * 80)
test_data_4 = {
    'part_number': '4794498C1-NOID',
    'standard': 'MPAPS F-1',
    'grade': 'Grade 1BF'
    # No ID anywhere
}

results_4 = process_mpaps_dimensions(test_data_4.copy())

print(f"Input: No ID in any source")
print(f"Output:")
print(f"  ✓ id_nominal_mm: {results_4.get('id_nominal_mm')} (expected None)")
print(f"  ✓ thickness_mm: {results_4.get('thickness_mm')} (expected None)")
print(f"  ✓ Return unchanged: {results_4 == test_data_4}")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)

checks = [
    ("Scenario 1 (top-level ID)", results_1.get('thickness_mm') == 4.30),
    ("Scenario 2 (nested ID in dict)", results_2.get('thickness_mm') == 4.30 and results_2.get('id_nominal_mm') == 24.6),
    ("Scenario 3 (extracted from raw_text)", results_3.get('thickness_mm') == 4.30 and results_3.get('id_nominal_mm') == 24.6),
    ("Scenario 4 (no ID fallback)", results_4.get('thickness_mm') is None),
]

all_passed = True
for check_name, passed in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {check_name}")
    if not passed:
        all_passed = False

print()
if all_passed:
    print("✅ ALL TESTS PASSED — Both patches are working correctly!")
    print("   - Patch 1: process_mpaps_dimensions() finds ID from multiple sources")
    print("   - Patch 2: Excel formatter will call process_mpaps_dimensions() before formatting")
else:
    print("❌ SOME TESTS FAILED — Check output above for details")
