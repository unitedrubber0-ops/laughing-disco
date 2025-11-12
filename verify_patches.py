#!/usr/bin/env python
"""Verify all sanity checks are met."""

import logging
logging.basicConfig(level=logging.WARNING)

# Check 1: Verify TABLE_4_GRADE_1_DATA 1" row
from mpaps_utils import TABLE_4_GRADE_1_DATA

print("=" * 70)
print("SANITY CHECKLIST")
print("=" * 70)
print()

# Check 1
print("✓ Check 1: TABLE_4_GRADE_1_DATA 1\" row values:")
one_inch_row = [row for row in TABLE_4_GRADE_1_DATA if row[0] == '1'][0]
print(f"  Row: {one_inch_row}")
expected = ('1', 24.6, 0.5, 4.30, 0.80)
if one_inch_row == expected:
    print(f"  ✅ Correct! Row matches expected: {expected}")
else:
    print(f"  ❌ MISMATCH! Expected: {expected}")
print()

# Check 2: Verify all three code edits are in place
from mpaps_utils import process_mpaps_dimensions, apply_mpaps_f6032_rules, canonical_standard

print("✓ Check 2: Code patch detection:")

# Check for Patch 1 in app.py
with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
    app_content = f.read()
    if 'process_mpaps_dimensions' in app_content and 'Early F-30/F-1 table processing' in app_content:
        print("  ✅ Patch 1 found in app.py (early process_mpaps_dimensions call)")
    else:
        print("  ❌ Patch 1 NOT found in app.py")

# Check for Patch 2 in mpaps_utils.py
with open('mpaps_utils.py', 'r', encoding='utf-8', errors='ignore') as f:
    utils_content = f.read()
    if 'Canonical standard indicates MPAPS F-30/F-1' in utils_content:
        print("  ✅ Patch 2 found in mpaps_utils.py (canonical guard)")
    else:
        print("  ❌ Patch 2 NOT found in mpaps_utils.py")
    
    if 'Only compute thickness from OD/ID if thickness is not already set' in utils_content:
        print("  ✅ Patch 3 found in mpaps_utils.py (guarded thickness computation)")
    else:
        print("  ❌ Patch 3 NOT found in mpaps_utils.py")
print()

# Check 3: Run the critical test
print("✓ Check 3: Run test with part 4794498C1:")
test_data = {
    'part_number': '4794498C1',
    'standard': 'MPAPS F-1',
    'grade': 'Grade 1BF',
    'id_nominal_mm': 24.6,
    'od_nominal_mm': 33.20
}

results = process_mpaps_dimensions(test_data)
apply_mpaps_f6032_rules(results)

expected_values = {
    'id_nominal_mm': 24.6,
    'id_tolerance_mm': 0.5,
    'thickness_mm': 4.30,
    'thickness_tolerance_mm': 0.80,
    'od_nominal_mm': 33.20,
    'dimension_source': 'MPAPS F-30/F-1 TABLE 4/8 (Grade 1/BF)'
}

all_pass = True
for key, expected_val in expected_values.items():
    actual_val = results.get(key)
    if actual_val == expected_val:
        print(f"  ✅ {key}: {actual_val}")
    else:
        print(f"  ❌ {key}: got {actual_val}, expected {expected_val}")
        all_pass = False

print()
print("=" * 70)
if all_pass:
    print("🎉 ALL CHECKS PASSED! Patches are working correctly.")
else:
    print("⚠️  Some checks failed. Review output above.")
print("=" * 70)
