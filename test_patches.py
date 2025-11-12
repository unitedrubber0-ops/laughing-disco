#!/usr/bin/env python
"""Test the three critical patches to verify they work correctly."""

import json
import logging

# Enable logging to see debug output
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Test data for part 4794498C1
test_data = {
    'part_number': '4794498C1',
    'standard': 'MPAPS F-1',
    'grade': 'Grade 1BF',
    'id_nominal_mm': 24.6,
    'od_nominal_mm': 33.20
}

print('Testing part 4794498C1 with patches applied...')
print(f'Input: {json.dumps(test_data, indent=2)}')
print()

# Import and test
from mpaps_utils import process_mpaps_dimensions, apply_mpaps_f6032_rules

# Apply F-30/F-1 processing (should happen early per Patch 1)
results = process_mpaps_dimensions(test_data)
print(f'After process_mpaps_dimensions():')
print(f'  thickness_mm: {results.get("thickness_mm")}')
print(f'  thickness_tolerance_mm: {results.get("thickness_tolerance_mm")}')
print(f'  dimension_source: {results.get("dimension_source")}')
print()

# Now test that F-6032 doesn't clobber it
apply_mpaps_f6032_rules(results)
print(f'After apply_mpaps_f6032_rules():')
print(f'  thickness_mm: {results.get("thickness_mm")}')
print(f'  thickness_tolerance_mm: {results.get("thickness_tolerance_mm")}')
print(f'  dimension_source: {results.get("dimension_source")}')
print()

# Verify
if results.get('thickness_mm') == 4.30:
    print('✅ SUCCESS: thickness_mm is correct (4.30 mm, not 5.15 mm)')
else:
    print(f'❌ FAILED: thickness_mm is {results.get("thickness_mm")} (expected 4.30 mm)')
