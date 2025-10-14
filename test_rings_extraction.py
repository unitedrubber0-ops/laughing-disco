"""
Test rings extraction functionality
"""
import unittest
from rings_extraction import extract_rings_info, extract_coordinates, polyline_length

class TestRingsExtraction(unittest.TestCase):
    """Test the rings extraction functions"""
    
    def test_extract_rings_basic(self):
        """Test basic rings extraction"""
        text = "Rings: 2\nInner: NBR\nOuter: CR"
        result = extract_rings_info(text)
        self.assertEqual(result['count'], 2)
        self.assertEqual(len(result['types']), 2)
        self.assertIn('INNER:NBR', result['types'])
        self.assertIn('OUTER:CR', result['types'])

    def test_extract_rings_complex(self):
        """Test more complex rings extraction"""
        text = "RING COUNT = 3\nRings: Steel, NBR\nPOINT 1: (0.0,0.0)\nPOINT 2: (10.0,0.0)\nPOINT 3: (10.0,5.0)"
        result = extract_rings_info(text)
        self.assertEqual(result['count'], 3)
        self.assertTrue(any('STEEL' in t for t in result['types']))
        self.assertTrue(any('NBR' in t for t in result['types']))

    def test_extract_rings_compact(self):
        """Test compact rings format"""
        text = "2R INNER:NBR OUTER:CR"
        result = extract_rings_info(text)
        self.assertEqual(result['count'], 2)
        self.assertIn('INNER:NBR', result['types'])
        self.assertIn('OUTER:CR', result['types'])

    def test_extract_coordinates(self):
        """Test coordinate extraction"""
        text = "COORD: 12.34,56.78; 23.45,67.89; 34.56,78.90"
        coords = extract_coordinates(text)
        self.assertEqual(len(coords), 3)
        self.assertEqual(coords[0], (12.34, 56.78))
        self.assertEqual(coords[1], (23.45, 67.89))
        self.assertEqual(coords[2], (34.56, 78.90))

    def test_polyline_length(self):
        """Test polyline length calculation"""
        coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)]
        length = polyline_length(coords)
        self.assertAlmostEqual(length, 15.0)  # 10 + 5 = 15 units

    def test_no_rings(self):
        """Test handling of text with no ring information"""
        text = "Some random text without any ring information"
        result = extract_rings_info(text)
        self.assertIsNone(result['count'])
        self.assertEqual(len(result['types']), 0)

    def test_no_coordinates(self):
        """Test handling of text with no coordinates"""
        text = "Some random text without any coordinates"
        coords = extract_coordinates(text)
        self.assertEqual(len(coords), 0)
        self.assertIsNone(polyline_length(coords))

if __name__ == '__main__':
    unittest.main()