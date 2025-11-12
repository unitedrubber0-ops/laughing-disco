#!/usr/bin/env python3
"""
Test script to verify authoritative material mapping.

This test ensures that the authoritative material lookup (from user's table)
takes precedence over fuzzy matching and CSV lookups.
"""
import sys
import logging

# Set up logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from material_mappings import (
    get_material_by_standard_grade,
    _canon_standard,
    _canon_grade
)

def test_canon_functions():
    """Test canonicalization functions"""
    print("\n=== Testing Canonicalization Functions ===")
    
    # Standard canonicalization
    test_cases_std = [
        ("F-30", "MPAPS F-30/F-1"),
        ("MPAPS F-30", "MPAPS F-30/F-1"),
        ("F-1", "MPAPS F-30/F-1"),
        ("MPAPS F-1", "MPAPS F-30/F-1"),
        ("F-30/F-1", "MPAPS F-30/F-1"),
        ("F-6032", "MPAPS F-6032"),
        ("MPAPS F-6032", "MPAPS F-6032"),
        ("F-6028", "MPAPS F-6028"),
        ("F-6034", "MPAPS F-6034"),
    ]
    
    for input_val, expected in test_cases_std:
        result = _canon_standard(input_val)
        status = "✓" if result == expected else "✗"
        print(f"{status} _canon_standard('{input_val}') = '{result}' (expected: '{expected}')")
        assert result == expected, f"Failed: {input_val} -> {result}, expected {expected}"
    
    # Grade canonicalization
    test_cases_grd = [
        ("GRADE IB", "1B"),
        ("IB", "1B"),
        ("1B", "1B"),
        ("GRADE 1B", "1B"),
        ("Grade IB", "1B"),
        ("1BF", "1BF"),
        ("GRADE IBF", "1BF"),
        ("IBF", "1BF"),
        ("1BFD", "1BFD"),
        ("GRADE IBFD", "1BFD"),
        ("IBFD", "1BFD"),
        ("2B", "2B"),
        ("GRADE 2B", "2B"),
        ("2C", "2C"),
    ]
    
    for input_val, expected in test_cases_grd:
        result = _canon_grade(input_val)
        status = "✓" if result == expected else "✗"
        print(f"{status} _canon_grade('{input_val}') = '{result}' (expected: '{expected}')")
        assert result == expected, f"Failed: {input_val} -> {result}, expected {expected}"
    
    print("✓ All canonicalization tests passed!\n")

def test_authoritative_lookup():
    """Test authoritative material mapping"""
    print("\n=== Testing Authoritative Material Lookup ===")
    
    test_cases = [
        # (standard_raw, grade_raw, expected_material, expected_reinforce, description)
        ("MPAPS F-30", "GRADE IB", "P-EPDM", "KEVLAR", "F-30 Grade 1B (IB variant)"),
        ("F-30", "1B", "P-EPDM", "KEVLAR", "F-30 Grade 1B (normalized)"),
        ("MPAPS F-1", "GRADE IBF", "P-EPDM", "KEVLAR", "F-1 Grade 1BF"),
        ("F-1", "1BF", "P-EPDM", "KEVLAR", "F-1 Grade 1BF (normalized)"),
        ("MPAPS F-30", "GRADE 1BFD", "P-EPDM WITH SPRING INSERT", "KEVLAR", "F-30 Grade 1BFD"),
        ("F-30", "2B", "SILICONE", "NOMEX 4 PLY", "F-30 Grade 2B"),
        ("F-30", "2C", "SILICONE", "NOMEX 4 PLY", "F-30 Grade 2C"),
        ("F-30", "J20CLASSA", "SILICONE", "NOMEX 4 PLY", "F-30 J20 Class A"),
        ("MPAPS F-6032", "TYPEI", "INNER NBR OUTER:ECO", "KEVLAR", "F-6032 Type I"),
        ("F-6032", "TYPE I", "INNER NBR OUTER:ECO", "KEVLAR", "F-6032 Type I (variant)"),
        ("F-6034", "H-AN", "HIGH TEMP. SILICONE", "NOMEX 4 PLY", "F-6034 H-AN"),
        # Non-existent mapping should return None
        ("MPAPS F-30", "NONEXISTENT", None, None, "Non-existent grade"),
        ("NONEXISTENT", "1B", None, None, "Non-existent standard"),
    ]
    
    for standard, grade, exp_mat, exp_reinf, description in test_cases:
        mat, reinf = get_material_by_standard_grade(standard, grade)
        
        if exp_mat is None:
            status = "✓" if mat is None and reinf is None else "✗"
            print(f"{status} {description}")
            print(f"   get_material_by_standard_grade('{standard}', '{grade}') = ({mat}, {reinf})")
            assert mat is None and reinf is None, f"Expected (None, None), got ({mat}, {reinf})"
        else:
            status = "✓" if (mat == exp_mat and reinf == exp_reinf) else "✗"
            print(f"{status} {description}")
            print(f"   get_material_by_standard_grade('{standard}', '{grade}')")
            print(f"   -> ({mat}, {reinf})")
            print(f"   Expected: ({exp_mat}, {exp_reinf})")
            assert mat == exp_mat and reinf == exp_reinf, \
                f"Mismatch: got ({mat}, {reinf}), expected ({exp_mat}, {exp_reinf})"
    
    print("\n✓ All authoritative lookup tests passed!\n")

def test_material_utils_integration():
    """Test integration into material_utils.safe_material_lookup_entry"""
    print("\n=== Testing Integration with material_utils ===")
    
    # Simple mock for testing without full DataFrame
    def mock_lookup_fn(standard, grade, df=None):
        """Mock fuzzy lookup that returns wrong answer"""
        # Simulate fuzzy matcher choosing F-6032 material for F-30 Grade 1B
        if "F-30" in str(standard) and "1B" in str(grade):
            return "INNER NBR OUTER:ECO"  # Wrong! Should be P-EPDM
        return "Not Found"
    
    from material_utils import safe_material_lookup_entry
    
    # Test: authoritative should win over fuzzy match
    result = safe_material_lookup_entry("MPAPS F-30", "GRADE IB", None, mock_lookup_fn)
    status = "✓" if result == "P-EPDM" else "✗"
    print(f"{status} Authoritative overrides fuzzy match for F-30 Grade 1B")
    print(f"   Got: {result} (expected: P-EPDM)")
    assert result == "P-EPDM", f"Expected P-EPDM, got {result}"
    
    # Test: fuzzy match used when no authoritative entry
    result = safe_material_lookup_entry("MPAPS F-30", "GRADE NONEXISTENT", None, mock_lookup_fn)
    status = "✓" if result == "Not Found" else "✗"
    print(f"{status} Fuzzy match fallback for non-existent grade")
    print(f"   Got: {result}")
    
    print("\n✓ Integration tests passed!\n")

if __name__ == "__main__":
    try:
        test_canon_functions()
        test_authoritative_lookup()
        test_material_utils_integration()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe authoritative material mapping is working correctly:")
        print("  • Canonicalization handles variants (F-30, F-1, Grade IB, 1BF, etc.)")
        print("  • Lookup returns correct material+reinforcement pairs")
        print("  • Integration ensures authoritative values override fuzzy matches")
        print("\nFor MPAPS F-30/Grade 1B:")
        print("  • get_material_by_standard_grade('MPAPS F-30', 'GRADE IB')")
        print("  • Returns: ('P-EPDM', 'KEVLAR')")
        print("  • Prevents F-6032 material fuzzy match from being selected")
        
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
