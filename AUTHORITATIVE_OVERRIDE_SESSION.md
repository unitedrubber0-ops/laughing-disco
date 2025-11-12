# MPAPS F-30 Grade-1 Authoritative Value Implementation — Session Summary

**Date:** November 12, 2025  
**Objective:** Make MPAPS F-30/Grade 1 thickness, OD, and tolerance values authoritative (from TABLE-4/8), prevent accidental overwrites by computed values, and add comprehensive provenance tracking.

---

## Problem Statement

Previously, the pipeline populated OD and thickness fields (fixing the N/A issue), but the values were not authoritative. For part **4509347C4** with ID 24.4 mm:

- **Actual Output:** OD = 31.40 mm, thickness = 3.50 ± 0.25 mm  
- **Expected (TABLE-4 Grade-1):** OD = 33.0 mm, thickness = 4.30 ± 0.80 mm  

**Root Cause:** The pipeline computed thickness as `(OD − ID) / 2` instead of using the authoritative TABLE-4 value directly. The OD source was not marked, allowing later computations to overwrite table-sourced values.

---

## Solution Implemented

### 1. **Provenance Marking in `process_mpaps_dimensions()` (mpaps_utils.py)**

When TABLE-4/8 Grade-1 values are applied, we now set explicit provenance flags:

```python
# Mark authoritative provenance for thickness
result['thickness_source'] = 'TABLE-4/8 Grade-1 authoritative'

# Mark OD source (TABLE-8 if available, else computed)
if od_from_entry is not None:
    result['od_source'] = 'TABLE-8 Grade-1 authoritative'
else:
    result['od_source'] = 'computed from ID+2*thickness (guarded)'

# Record dimension source
result['dimension_source'] = 'MPAPS F-30/F-1 TABLE 4/8 (Grade1 authoritative)'
```

**Benefits:**
- Downstream code can check `thickness_source` to detect authoritative values.
- Traceability for debugging and validation.
- Enables guarded computation (only compute if not already set from table).

---

### 2. **Guard Against Overwrites in Excel Output (excel_output.py)**

Two locations compute thickness from OD/ID. Both now check provenance before overwriting:

**Location 1: `guarantee_and_format_results()`**
```python
provenance = rr.get('thickness_source')
# Only compute thickness if not set by authoritative table
if (rr.get('thickness_mm') is None and odn is not None and idn is not None 
    and provenance != 'TABLE-4/8 Grade-1 authoritative'):
    # compute and set tolerance only if not from table
    if rr.get('thickness_tolerance_mm') is None:
        rr['thickness_tolerance_mm'] = 0.25
elif provenance == 'TABLE-4/8 Grade-1 authoritative':
    logging.debug(f"Thickness is authoritative from table...")
```

**Location 2: `ensure_result_fields()`**
```python
provenance_thickness = res.get('thickness_source')
if (thickness is None and od_nom is not None and id_nom is not None 
    and provenance_thickness != 'TABLE-4/8 Grade-1 authoritative'):
    # safe to compute
```

**Key Rule:** If `thickness_source == 'TABLE-4/8 Grade-1 authoritative'`, skip computation entirely.

---

### 3. **Debug Logging for Traceability**

Before Excel creation, log all key values and their sources:

```python
logging.info("DEBUG BEFORE EXCEL: part_number=%s, id_nominal_mm=%s, "
             "od_nominal_mm=%s, thickness_mm=%s, thickness_tolerance_mm=%s, "
             "dimension_source=%s, thickness_source=%s",
             analysis_results.get('part_number'),
             analysis_results.get('id_nominal_mm'),
             analysis_results.get('od_nominal_mm'),
             analysis_results.get('thickness_mm'),
             analysis_results.get('thickness_tolerance_mm'),
             analysis_results.get('dimension_source'),
             analysis_results.get('thickness_source'))
```

**Expected Log Output for 4509347C4:**
```
DEBUG BEFORE EXCEL: part_number=4509347C4, id_nominal_mm=24.4, id_tolerance_mm=0.5, 
od_nominal_mm=33.0, thickness_mm=4.3, thickness_tolerance_mm=0.8, 
dimension_source=MPAPS F-30/F-1 TABLE 4/8 (Grade1 authoritative), 
thickness_source=TABLE-4/8 Grade-1 authoritative
```

---

## Test Results

Created **`test_authoritative_override.py`** to verify the logic:

### Test Case 1: MPAPS F-30, Grade 1, ID 24.4mm (Part 4509347C4)

**Input:**
```python
{
    'standard': 'MPAPS F-30',
    'grade': '1B',
    'id_nominal_mm': 24.4,
    'part_number': '4509347C4'
}
```

**Output:**
```
✓ thickness_mm: 4.3 mm (matches TABLE-4)
✓ thickness_tolerance_mm: 0.8 mm (matches TABLE-4)
✓ thickness_source: 'TABLE-4/8 Grade-1 authoritative'
✓ od_nominal_mm: 33.0 mm (24.4 + 2*4.3, correctly computed)
✓ od_source: 'computed from ID+2*thickness (guarded)'
✓ id_tolerance_mm: 0.5 mm (from TABLE-4)
✓ dimension_source: 'MPAPS F-30/F-1 TABLE 4/8 (Grade1 authoritative)'
```

**Status:** ✓ PASSED

---

### Test Case 2: MPAPS F-30, Grade 1BF, ID 24.6mm (Exact TABLE-8 Match)

**Input:**
```python
{
    'standard': 'MPAPS F-30',
    'grade': '1BF',
    'id_nominal_mm': 24.6,
    'part_number': 'TEST-1BF'
}
```

**Output:**
```
✓ thickness_mm: 4.3 mm (from TABLE-8)
✓ thickness_tolerance_mm: 0.8 mm (from TABLE-8)
✓ thickness_source: 'TABLE-4/8 Grade-1 authoritative'
✓ od_nominal_mm: 33.2 mm (from TABLE-8 exact match)
```

**Status:** ✓ PASSED

---

## Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `mpaps_utils.py` | Added `thickness_source`, `od_source` provenance flags in Grade-1 fallback logic (lines 283–307). Updated `dimension_source` to indicate "authoritative". | Mark all table-sourced values so downstream code can skip computation. |
| `excel_output.py` | Added guards in `guarantee_and_format_results()` (lines 54–68) and `ensure_result_fields()` (lines 149–172) to skip thickness computation if `thickness_source == 'TABLE-4/8 Grade-1 authoritative'`. Added debug logging before Excel creation (lines 264–271). | Prevent accidental overwrites; trace value sources. |
| `test_authoritative_override.py` | New comprehensive test suite for authoritative override logic. | Verify that TABLE-4 Grade-1 values are set and not overwritten. |

---

## Expected Excel Output for 4509347C4

After running analysis and Excel generation, the part **4509347C4** should now show:

| Field | Expected Value | Source |
|-------|-----------------|--------|
| **ID1 AS PER 2D (MM)** | `24.40 ± 0.50 mm` | TABLE-4 Grade-1 exact match (nearest nominal 24.6 mm) |
| **OD1 AS PER 2D (MM)** | `33.00 mm` | Computed from ID + 2×thickness (guarded) |
| **THICKNESS AS PER 2D (MM)** | `4.30 ± 0.80 mm` | **TABLE-4 Grade-1 authoritative** ✓ |
| **WALL THICKNESS TOLERANCE (MM)** | `±0.80 mm` | TABLE-4 Grade-1 |

---

## Validation Steps (Run These to Confirm)

1. **Verify provenance is marked:**
   ```bash
   python test_authoritative_override.py
   ```
   Expected: ✓ ALL TESTS PASSED

2. **Run full analysis for 4509347C4 and inspect debug logs:**
   - Look for `DEBUG BEFORE EXCEL:` log line.
   - Confirm `thickness_source=TABLE-4/8 Grade-1 authoritative`.
   - Confirm `od_nominal_mm=33.00`.
   - Confirm `thickness_mm=4.30, thickness_tolerance_mm=0.80`.

3. **Generate Excel and verify columns:**
   - **THICKNESS AS PER 2D (MM)** should display `4.30 ± 0.80 mm`.
   - **OD1 AS PER 2D (MM)** should display `33.00 mm` (not `31.40 mm`).
   - No fallback computation should overwrite these values.

---

## Key Policy Summary

- **Policy:** For MPAPS F-30/F-1 + Grade 1/1B/1BF, TABLE-4/8 thickness and tolerance are **authoritative** and must not be overwritten.
- **Mechanism:** Provenance marking (`thickness_source`) + guarded computation (skip if marked as authoritative).
- **Fallback:** Tolerance is only set to 0.25 mm if not already provided by a table; table values always take priority.
- **OD Handling:** If TABLE-8 provides OD (as for 1BF), use it directly. Otherwise, compute from ID + 2×thickness (always safe since thickness is now authoritative).

---

## Git Commit

```
commit 8cc34a5
Author: GitHub Copilot
Date:   <timestamp>

Implement authoritative TABLE-4/8 Grade-1 thickness/OD with provenance marking and guards against overwrites

- Add thickness_source, od_source, dimension_source provenance flags in process_mpaps_dimensions()
- Guard thickness computation in Excel output to prevent overwrites of authoritative table values
- Add debug logging before Excel creation to trace value sources
- Create test_authoritative_override.py to verify Grade-1 authoritative logic
- Update dimension_source to indicate "authoritative" for Grade-1 values
```

---

## Next Steps (Optional Enhancements)

1. **Integration Test:** Run full end-to-end analysis for sample parts and verify Excel output matches TABLE-4 values.
2. **Regression Test:** Ensure F-6032 and other grades still work correctly (no unintended side effects).
3. **Documentation:** Add user guide explaining provenance fields and how to debug Excel output discrepancies.
4. **Performance:** If needed, cache table lookups to avoid repeated nearest-nominal searches.

---

## Questions Resolved

**Q: Why are OD values sometimes computed vs. from table?**  
A: TABLE-4 (formed hose) doesn't provide OD; only TABLE-8 (suffix BF) does. For TABLE-4 entries, OD is computed as ID + 2×thickness, which is always correct because thickness is now authoritative.

**Q: What if a user provides an OD value directly in the drawing?**  
A: Thickness computation checks if `thickness_mm` is None before computing. If a user-supplied OD exists and thickness is set from TABLE-4, we use TABLE-4 thickness (not computed), so the values are consistent.

**Q: How do we prevent future regressions?**  
A: The test suite (`test_authoritative_override.py`) ensures that for MPAPS F-30 Grade 1/1B/1BF with ID near 24 mm, thickness is always 4.30 ± 0.80 mm and marked as authoritative. Add this test to your CI/CD pipeline.

---

**Session Complete.** All changes committed and pushed to GitHub branch `update-mpaps-tables`.
