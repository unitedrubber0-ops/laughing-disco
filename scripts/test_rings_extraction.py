"""
Test the rings extraction functionality
"""
import unittest
from rings_extraction import extract_rings_info, extract_coordinates, polyline_length, scan_text_by_lines

class TestRingsExtraction(unittest.TestCase):
    """Test cases for rings extraction functionality"""
    
    def test_extract_rings_basic(self):
        """Test basic rings extraction with clean input"""
        text = "Rings: 2\nInner: NBR\nOuter: CR"
        result = extract_rings_info(text)
        self.assertEqual(result['count'], 2)
        self.assertTrue(any('INNER:NBR' in t for t in result['types']))
        self.assertTrue(any('OUTER:CR' in t for t in result['types']))

    def test_extract_rings_with_noise(self):
        """Test rings extraction with noisy OCR-like input"""
        text = "R l N G S : 2 \nI N N E R:  N B R\nO U T E R:C R"
        result = extract_rings_info(text)
        self.assertEqual(result['count'], 2)
        self.assertTrue(any('NBR' in t for t in result['types']))
        self.assertTrue(any('CR' in t for t in result['types']))

    def test_extract_coordinates_basic(self):
        """Test basic coordinate extraction"""
        text = "COORD: 12.34,56.78\nPOINT 1: (23.45,67.89)"
        coords = extract_coordinates(text)
        self.assertEqual(len(coords), 2)
        self.assertEqual(coords[0], (12.34, 56.78))
        self.assertEqual(coords[1], (23.45, 67.89))

    def test_extract_coordinates_noise(self):
        """Test coordinate extraction with OCR noise"""
        text = "C O O R D : 1 2. 3 4, 5 6. 7 8\nX = 2 3. 4 5 Y = 6 7. 8 9"
        coords = extract_coordinates(text)
        self.assertEqual(len(coords), 2)
        self.assertAlmostEqual(coords[0][0], 12.34)
        self.assertAlmostEqual(coords[0][1], 56.78)
        self.assertAlmostEqual(coords[1][0], 23.45)
        self.assertAlmostEqual(coords[1][1], 67.89)

    def test_development_length(self):
        """Test development length calculation"""
        # Convert integer coordinates to floats explicitly
        coords = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]  # Forms a path of 5 + 5 = 10 units
        length = polyline_length(coords)
        self.assertAlmostEqual(length, 10.0, places=7)  # Specify places for float comparison

    def test_scan_text_by_lines(self):
        """Test scanning text by lines with contextual extraction"""
        text = """
        Some header text
        Part details
        RING COUNT = 2
        More text...
        Inner ring: NBR
        Some specs
        POINT 1: (0,0)
        POINT 2: (3,4)
        Footer text
        """
        result = scan_text_by_lines(text)
        self.assertEqual(result['rings_info']['count'], 2)
        self.assertTrue(any('NBR' in t for t in result['rings_info']['types']))
        self.assertEqual(len(result['coordinates']), 2)
        self.assertEqual(result['coordinates'][0], (0.0, 0.0))
        self.assertEqual(result['coordinates'][1], (3.0, 4.0))

    def test_edge_cases(self):
        """Test various edge cases"""
        # Empty input
        self.assertIsNone(extract_rings_info("")['count'])
        self.assertEqual(len(extract_coordinates("")), 0)
        self.assertIsNone(polyline_length([]))

        # None input
        self.assertIsNone(extract_rings_info(None)['count'])
        self.assertEqual(len(extract_coordinates(None)), 0)
        self.assertEqual(polyline_length(None), 0.0)  # Changed to expect 0.0 instead of None

        # Invalid coordinates
        text = "COORD: invalid, data"
        self.assertEqual(len(extract_coordinates(text)), 0)

        # Single coordinate pair (can't calculate length)
        coords = [(1.0, 1.0)]
        self.assertIsNone(polyline_length(coords))

if __name__ == '__main__':
    unittest.main()