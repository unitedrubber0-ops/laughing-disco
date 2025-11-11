# F-6032 Automatic Wall Thickness Tolerance Feature

## Feature Added

**When MPAPS F-6032 is detected, the wall thickness tolerance is automatically set to ±0.8 mm**

## Where It Applies

1. **`apply_mpaps_f6032_dimensions()`** in `mpaps_utils.py`
   - When F-6032 dimensions are explicitly calculated/looked up
   - Sets `thickness_tolerance_mm = 0.8` (previously was `None`)

2. **`apply_mpaps_f6032_rules()`** in `mpaps_utils.py`
   - When F-6032 rules are applied during analysis
   - Sets `thickness_tolerance_mm = 0.8` for both:
     - TABLE 1 explicit thickness values
     - Computed thickness from (OD - ID) / 2

## Excel Output Impact

For any MPAPS F-6032 part detected:

| Field | Value |
|-------|-------|
| WALL THICKNESS TOLERANCE (MM) | ±0.8 mm |

Instead of showing `N/A` or no tolerance, all F-6032 parts will now consistently show **±0.8 mm** wall thickness tolerance.

## Code Changes

### Change 1: `apply_mpaps_f6032_dimensions()` function

```python
# Old:
result['thickness_tolerance_mm'] = None  # explicit: no tolerance

# New:
result['thickness_tolerance_mm'] = 0.8   # F-6032 wall thickness tolerance is always ± 0.8 mm
```

### Change 2: `apply_mpaps_f6032_rules()` function (TABLE 1 path)

```python
# Old:
result['thickness_tolerance_mm'] = table_data.get('thickness_tolerance_mm', 0.25)

# New:
# For F-6032, wall thickness tolerance is automatically ± 0.8 mm
result['thickness_tolerance_mm'] = 0.8
```

### Change 3: `apply_mpaps_f6032_rules()` function (computed thickness path)

```python
# Old:
result['thickness_tolerance_mm'] = 0.25

# New:
# For F-6032, wall thickness tolerance is always ± 0.8 mm
result['thickness_tolerance_mm'] = 0.8
```

### Bug Fix: Initialize `table_data = None`

```python
# Prevents UnboundLocalError when ID is not found
table_data = None  # Initialize to None (will be set if ID found)
if id_val is not None:
    # ...
```

## Testing

See `test_f6032_tolerance.py` for comprehensive test scenarios:

✅ **Test 1:** `apply_mpaps_f6032_dimensions()` sets tolerance = 0.8  
✅ **Test 2:** `process_mpaps_dimensions()` for F-6032 sets tolerance = 0.8  
✅ **Test 3:** F-30/F-1 grades are not affected (use their own TABLE 4/8 tolerances)

All tests pass successfully.

## Backward Compatibility

- ✅ No breaking changes
- ✅ Only affects F-6032 parts (no impact on F-30/F-1)
- ✅ Consistent behavior across all F-6032 lookup paths
- ✅ Improves Excel output completeness

## Related Commits

- `0cf7f26` - Automatically set F-6032 wall thickness tolerance to ±0.8 mm
- `a5b74ba` - Initialize table_data = None to prevent UnboundLocalError
