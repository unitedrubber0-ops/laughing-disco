# MPAPS Excel Output Population — Complete Patch Summary

## Problem Statement

Excel output was showing many **N/A** values for MPAPS-derived fields (ID tolerance, wall thickness, thickness tolerance) because:

1. **`process_mpaps_dimensions()`** was returning early if `id_nominal_mm` wasn't at the top-level, without checking the `dimensions` dict or raw text
2. **`generate_corrected_excel_sheet()`** was calling `ensure_result_fields()` directly without first calling `process_mpaps_dimensions()`
3. Result: Fields like `thickness_mm`, `thickness_tolerance_mm`, `id_tolerance_mm` were missing when the formatter tried to use them

## Solution: Two Critical Patches

### Patch 1: Enhance `process_mpaps_dimensions()` in `mpaps_utils.py`

**Location:** Lines 104–217

**What Changed:**
- Now searches for ID value from **multiple sources** in this order:
  1. Top-level keys: `id_nominal_mm`, `id1`, `ID1`, `ID`, `id`, `nominal_id_mm`
  2. Nested `dimensions` dict: same key names
  3. **Fallback:** Parse raw_text for pattern `HOSE ID = <number>`
- **Only** returns unchanged if ID cannot be found in any source
- Guarantees `id_nominal_mm` is set before F-30/F-1 table logic runs

**Why It Helps:**
- Many analysis results store ID inside the `dimensions` nested dict instead of top-level
- The old code would give up immediately and not populate thickness/tolerances from TABLE 4/8
- Now it will find the ID and apply the correct Grade 1/1BF rules

**Example Flow:**
```
Input:  { dimensions: { id1: 24.6 }, standard: "MPAPS F-1", grade: "Grade 1BF" }
↓ (Patch 1 finds ID in dimensions.id1)
↓ (Sets id_nominal_mm = 24.6)
↓ (Calls get_grade1bf_tolerances(24.6))
Output: { thickness_mm: 4.30, thickness_tolerance_mm: 0.80, id_tolerance_mm: 0.50, ... }
```

### Patch 2: Call `process_mpaps_dimensions()` from `generate_corrected_excel_sheet()` in `excel_output.py`

**Location:** Lines 247–254 (before `ensure_result_fields()`)

**What Changed:**
```python
# Ensure MPAPS values have been applied (populate thickness, tolerances, etc.)
try:
    from mpaps_utils import process_mpaps_dimensions
    analysis_results = process_mpaps_dimensions(analysis_results or {})
    logging.debug("process_mpaps_dimensions called successfully before ensure_result_fields")
except Exception as e:
    logging.debug(f"process_mpaps_dimensions failed inside excel generator: {e}", exc_info=True)

analysis_results = ensure_result_fields(analysis_results)
```

**Why It Helps:**
- Even if upstream code doesn't call `process_mpaps_dimensions()`, the Excel formatter now guarantees it runs
- `ensure_result_fields()` will find populated `thickness_mm` and `thickness_tolerance_mm` fields
- Result: Excel shows actual values instead of "N/A" when MPAPS tables cover them

## Expected Behavior After Patches

For part **4794498C1** (MPAPS F-1 Grade 1BF):

### Before Patches:
```
ID1 AS PER 2D (MM)                    → N/A
ID TOLERANCE (MM)                      → N/A
THICKNESS AS PER 2D (MM)              → N/A
WALL THICKNESS TOLERANCE (MM)         → N/A
```

### After Patches:
```
ID1 AS PER 2D (MM)                    → 24.60 ± 0.50 mm
ID TOLERANCE (MM)                      → 0.50 mm
THICKNESS AS PER 2D (MM)              → 4.30 ± 0.80 mm
WALL THICKNESS TOLERANCE (MM)         → 0.80 mm
```

## Testing

### Test Scenarios Verified:

✅ **Scenario 1:** ID in top-level field → `thickness_mm = 4.30`

✅ **Scenario 2:** ID nested in `dimensions` dict (Patch 1 target) → ID promoted, thickness populated

✅ **Scenario 3:** ID extracted from raw_text pattern (Patch 1 fallback) → ID extracted, thickness populated

✅ **Scenario 4:** No ID found → Returns unchanged, no false values

All 4 scenarios passed. See `test_both_patches.py` for full test output.

## Implementation Details

### Patch 1 ID Resolution Priority:
1. Check top-level `id_nominal_mm`, `id1`, `ID1`, `ID`, `id`, `nominal_id_mm`
2. If not found, check nested `dimensions` dict for same keys
3. If still not found, search `raw_text`, `ocr_text`, or `text` for pattern `HOSE ID\s*[=:]?\s*([\d\.]+)`
4. If still not found, return result unchanged (early exit, show N/A as expected)

### Patch 2 Safety Measures:
- Wrapped in try/except to prevent Excel generation from failing if `process_mpaps_dimensions()` has issues
- Logs debug info for troubleshooting
- Gracefully falls back if import fails

## Commits

```
f620181 fix: Two critical patches to populate MPAPS fields in Excel output
3606a15 fix: Enhance Patch 2 to check dimension_sources list and set dimension_source field
472dbd9 fix: Apply three critical patches to prevent F-6032 from overriding F-30/F-1 Grade 1 rules
ca7968a fix: Update 1 inch nominal wall thickness to 4.30 mm per MPAPS F-1 TABLE 4/8
200bf6a fix: Call process_mpaps_dimensions() before Excel generation
```

## Next Steps

1. Run full analyzer for part **4794498C1** 
2. Verify Excel output shows:
   - `ID1 AS PER 2D (MM)` → `24.60 ± 0.50 mm`
   - `THICKNESS AS PER 2D (MM)` → `4.30 ± 0.80 mm`
   - `WALL THICKNESS TOLERANCE (MM)` → `0.80 mm`
3. Check logs for messages like:
   - `Processing Grade 1BF dimensions for ID: 24.6mm`
   - `process_mpaps_dimensions called successfully before ensure_result_fields`
4. Verify no more N/A values for MPAPS-covered fields

## Files Modified

- `mpaps_utils.py` - Enhanced `process_mpaps_dimensions()` with multi-source ID lookup
- `excel_output.py` - Added `process_mpaps_dimensions()` call before field formatting
