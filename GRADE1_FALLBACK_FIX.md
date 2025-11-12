# Grade-1 Defensive Fallback Lookup - Fix for N/A Excel Fields

## Problem Statement

Part `4509347C4` (ID 24.4mm) was detected as MPAPS F-30/Grade 1BF, and the analyzer set burst pressure correctly, but the Excel output showed:
- **OD**: N/A ❌
- **Thickness**: N/A ❌
- **ID Tolerance**: N/A ❌

Only the ID value (24.4mm) was populated.

### Root Cause

The original `process_mpaps_dimensions()` Grade-1 handling:
1. Called `get_grade1bf_tolerances(id_val)` looking for exact table match
2. For ID 24.4mm, no exact match exists (table has 16.0, 24.6, 25.4, etc.)
3. The function returned early without setting thickness/OD/id_tolerance
4. `ensure_result_fields()` in Excel generation left those as N/A

The issue: **Near-nominal IDs like 24.4mm were not matched to their nearest table entry (24.6mm) that exists within practical tolerance (0.5mm).**

---

## Solution: 4-Step Defensive Fallback Chain

When Grade-1 direct lookup fails, the code now attempts:

### Step 1: Try Exact Lookup
```python
grade_entry = get_grade1bf_tolerances(id_val)
```
- Calls existing helper function
- Returns immediately if found (typical case)

### Step 2: Nearest Nominal in TABLE_8 (Grade 1BF)
```python
for row in TABLE_8_GRADE_1BF_DATA:
    diff = abs(float(row[1]) - float(id_val))
    if diff < nearest_diff:
        nearest = row
        nearest_diff = diff

if nearest_diff <= MAX_ACCEPT_DIFF_MM (0.5mm):
    # Use this entry
    grade_entry = {...}
```

**Example**: For ID 24.4mm:
- Searches TABLE_8: finds entries 16.0, 24.6, 25.4, etc.
- 24.6mm has diff = 0.2mm ✓ (within 0.5mm tolerance)
- Selects 24.6 entry: thickness_mm=4.30, wall_tol=0.8, id_tol=0.5

### Step 3: Nearest Nominal in TABLE_4 (Grade 1)
```python
for row in TABLE_4_GRADE_1_DATA:
    diff = abs(float(row[1]) - float(id_val))
    if diff <= MAX_ACCEPT_DIFF_MM (0.5mm):
        grade_entry = {...}
        break
```

Similar search in Grade 1 table if TABLE_8 didn't match.

### Step 4: Range Tables (Fallback for Non-Standard Sizes)
```python
# Check TABLE_4_GRADE_1_RANGES and TABLE_8_GRADE_1BF_RANGES
for (min_mm, max_mm, wall_mm, wall_tol_mm, id_tol_mm) in TABLE_4_GRADE_1_RANGES:
    if min_mm <= id_val <= max_mm:
        # Use range-based values
        grade_entry = {...}
        break
```

For IDs that don't match nominal table, check if they fall within range definitions (e.g., ">2.0–3.0").

---

## Field Population Logic

Once a `grade_entry` dict is found (via any of the 4 steps):

### ID Tolerance
```python
id_tol = grade_entry.get('id_tolerance_mm')
if id_tol is None:
    id_tol = 0.5  # Safe default
result['id_tolerance_mm'] = float(id_tol)
```
- Typically 0.5mm for Grade 1/1BF per table
- Fallback to 0.5 if not specified

### Wall Thickness Nominal
```python
wall_mm = grade_entry.get('wall_mm')
if wall_mm is not None:
    result['thickness_mm'] = float(wall_mm)
```
- From table entry (e.g., 4.30 for 24.6mm entry, 4.95 for 16.0mm entry)
- Always set from table, never None if entry found

### Wall Thickness Tolerance
```python
wall_tol = grade_entry.get('wall_tolerance_mm')
if wall_tol is None:
    wall_tol = 0.8  # Grade 1 default per Table 4
result['thickness_tolerance_mm'] = float(wall_tol)
```
- From table entry (typically ±0.80 for Grade 1)
- Fallback to 0.8 if missing

### OD Nominal
```python
od_from_entry = grade_entry.get('od_mm')
if od_from_entry is not None:
    result['od_nominal_mm'] = float(od_from_entry)
else:
    # Compute: OD = ID + 2 × thickness
    result['od_nominal_mm'] = round(id_nominal + 2.0 * thickness_mm, 3)
```
- If table has OD (e.g., TABLE_8 columns), use it
- Otherwise compute from ID + 2×thickness
- For ID 24.4mm with thickness 4.30: OD = 24.4 + 2(4.30) = 33.0mm

### Dimension Source Tracking
```python
ds = 'MPAPS F-30/F-1 TABLE 4/8 (Grade1 fallback)'
result['dimension_source'] = ds
result['dimension_sources'].append(ds)
```
- Marks that dimensions came from fallback lookup
- Prevents F-6032 logic from overriding (checks dimension_source)

---

## Testing & Verification

### Test Case: ID 24.4mm → TABLE_8 24.6mm Match

```
Before fix:
  ID: 24.4mm ✓
  Thickness: None ❌ (Excel shows N/A)
  OD: None ❌ (Excel shows N/A)
  ID Tolerance: None ❌ (Excel shows N/A)

After fix:
  ID: 24.4mm ✓
  Thickness: 4.30 mm ✓
  Thickness Tolerance: 0.80 mm ✓
  OD: 33.0 mm ✓ (computed: 24.4 + 2×4.30)
  ID Tolerance: 0.5 mm ✓
  Dimension Source: MPAPS F-30/F-1 TABLE 4/8 (Grade1 fallback)
```

### Additional Test Cases

| ID (mm) | Nearest Table | Diff (mm) | Accept? | Thickness | OD |
|---------|---------------|-----------|---------|-----------|-----|
| 24.4 | TABLE_8 24.6 | 0.2 | ✓ | 4.30 | 33.0 |
| 15.9 | TABLE_8 16.0 | 0.1 | ✓ | 4.95 | 25.8 |
| 25.4 | TABLE_8 25.4 | 0.0 | ✓ | 4.30 | 34.0 |
| 19.8 | TABLE_8 20.0 | 0.2 | ✓ | 4.95 | 29.7 |

**All test cases pass**: Fields populated correctly, no N/A in Excel.

---

## Logging Output

When the fallback logic executes, watch for these log messages:

```
INFO - No direct Grade1/BF table match for ID=24.4; attempting nearest-nominal/range lookup
INFO - Nearest TABLE_8 match for ID 24.4: nominal=24.6mm diff=0.200mm
INFO - ID 24.4 matches dimension_source: MPAPS F-30/F-1 TABLE 4/8 (Grade1 fallback)
```

Or if TABLE_8 doesn't match:
```
INFO - Nearest TABLE_4 match for ID 24.4: nominal=24.6mm diff=0.200mm
```

Or for ranges:
```
INFO - ID 24.4 falls in TABLE_8 range 20.0-30.0 mm -> wall 4.30 mm
```

Or if nothing matches:
```
WARNING - Unable to determine Grade1 table entry for ID 24.4; leaving thickness/ID tol unset
```

---

## Excel Output Impact

### Before Fix
```
Part 4509347C4 (MPAPS F-30, Grade 1BF, ID 24.4mm):
  ID: 24.40 mm
  ID Tolerance: N/A
  Thickness: N/A
  Thickness Tolerance: N/A
  OD: N/A
```

### After Fix
```
Part 4509347C4 (MPAPS F-30, Grade 1BF, ID 24.4mm):
  ID: 24.40 ± 0.50 mm
  Thickness: 4.30 ± 0.80 mm
  OD: 33.0 mm
  (All values populated from TABLE_8 24.6 fallback match)
```

---

## Configuration

### MAX_ACCEPT_DIFF_MM
```python
MAX_ACCEPT_DIFF_MM = 0.5  # Allow up to 0.5mm difference for practical measurements
```

This is the tolerance for "near-nominal" matching:
- IDs within 0.5mm of table entry are considered matches
- Typical tolerance for manufacturing measurements
- Can be adjusted if needed, but 0.5mm is standard for hose specifications

### Table Data
The fallback uses:
- `TABLE_8_GRADE_1BF_DATA` - Grade 1BF nominal entries with all specs
- `TABLE_4_GRADE_1_DATA` - Grade 1 nominal entries with specs
- `TABLE_8_GRADE_1BF_RANGES` - Grade 1BF range definitions (e.g., >20-30mm)
- `TABLE_4_GRADE_1_RANGES` - Grade 1 range definitions

---

## Backward Compatibility

✅ **Fully backward compatible**:
- If exact match exists, uses it immediately (original behavior)
- Fallback chain only activates if exact lookup returns None
- Graceful error handling with try/except
- Safe defaults if any value missing (id_tol=0.5, wall_tol=0.8)
- Existing code unaffected (only Grade-1 path enhanced)

---

## Future Enhancements

1. **Caching**: Cache nearest matches for repeated IDs
2. **Alerting**: Warn if ID is outside all table ranges entirely
3. **User Override**: Allow specifying ID in comments for exact lookup
4. **CSV-based Tables**: Load table from CSV for easier maintenance

---

## Related Files Modified

- `mpaps_utils.py` - Enhanced `process_mpaps_dimensions()` Grade-1 handling
- `test_grade1_fallback.py` - New comprehensive test suite

---

## Verification Steps

1. **Run test suite**:
   ```bash
   python test_grade1_fallback.py
   ```
   Expected: All 4 test cases pass with correct thickness/OD values

2. **Check logs** for fallback activation:
   ```
   grep -i "nearest TABLE" app.log
   grep -i "Grade1 fallback" app.log
   ```

3. **Analyze test part** (4509347C4 or similar):
   ```bash
   python app.py --analyze 4509347C4.pdf
   ```
   Check Excel output:
   - OD should show value, not N/A
   - Thickness should show 4.30 ± 0.80
   - ID Tolerance should show 0.5

4. **Verify dimension source**:
   ```
   dimension_source: MPAPS F-30/F-1 TABLE 4/8 (Grade1 fallback)
   ```
   Should appear in logs for fallback-matched parts

---

## Sign-Off

✅ **Fix Applied**: Defensive fallback logic added to Grade-1 lookup
✅ **Tests Passed**: 4/4 test cases pass (4 different ID values)
✅ **Logging Added**: Detailed info about which source matched
✅ **Excel Impact**: N/A fields now show actual values
✅ **Backward Compatible**: No breaking changes

**Ready for production deployment.**
