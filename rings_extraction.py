"""
Module for rings extraction functionality.
"""
import re
import logging

logger = logging.getLogger(__name__)

class RingsExtractor:
    """
    Class to handle different methods of rings extraction from text.
    """
    
    @staticmethod
    def clean_rings_text(text):
        """Clean and normalize rings text."""
        if not isinstance(text, str):
            return "Not Found"
            
        # Remove extra whitespace and normalize spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common noise words
        noise_words = ['approx', 'approximately', 'about']
        for word in noise_words:
            text = re.sub(fr'\b{word}\b', '', text, flags=re.IGNORECASE)
            
        return text.strip()

    @staticmethod
    def extract_by_pattern(text):
        """
        Extract rings information using pattern matching.
        """
        if not isinstance(text, str):
            return "Not Found"
            
        try:
            # Clean the text first
            text = RingsExtractor.clean_rings_text(text)
            
            # Look for ring count patterns
            rings_patterns = [
                r'(?:with|having|includes?)\s+(\d+)\s*(?:rings?|reinforcements?)',
                r'(?:HOSE|HEATER).*?(\d+)\s*(?:rings?|reinforcements?)(?:\s+|$)',
                r'(\d+)\s*(?:rings?|reinforcements?)\s+(?:required|needed|specified|TYPE)',
                r'(?:rings?|reinforcements?)\s*(?:count|number|qty|quantity)?\s*[:-]?\s*(\d+)',
                r'reinforced\s+with\s+(\d+)\s*rings?'
            ]
            
            for pattern in rings_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
                    
            return "Not Found"
            
        except Exception as e:
            logger.error(f"Error extracting rings by pattern: {str(e)}")
            return "Not Found"

    @staticmethod
    def extract_by_regex(text):
        """
        Extract rings count using regular expressions.
        """
        if not isinstance(text, str):
            return "Not Found"
            
        try:
            text = RingsExtractor.clean_rings_text(text)
            
            # Look for direct ring count mentions
            rings_regex = r'(\d+)\s*(?:ring|rings|reinforcement|reinforcements)'
            match = re.search(rings_regex, text, re.IGNORECASE)
            if match:
                return match.group(1)
                
            return "Not Found"
            
        except Exception as e:
            logger.error(f"Error extracting rings by regex: {str(e)}")
            return "Not Found"

    @staticmethod
    def extract_by_context(text):
        """
        Extract rings information using context-aware analysis.
        """
        if not isinstance(text, str):
            return "Not Found"
            
        try:
            text = RingsExtractor.clean_rings_text(text)
            
            # Look for rings in product description sections
            desc_regex = r'description:?\s*(.*?rings.*?)(?:\n|$)'
            match = re.search(desc_regex, text, re.IGNORECASE)
            if match:
                # Extract number from description
                num_match = re.search(r'(\d+)\s*rings?', match.group(1), re.IGNORECASE)
                if num_match:
                    return num_match.group(1)
                    
            return "Not Found"
            
        except Exception as e:
            logger.error(f"Error extracting rings by context: {str(e)}")
            return "Not Found"
            
    @staticmethod
    def extract_rings(text):
        """
        Main function to extract rings information, trying multiple methods.
        """
        if not isinstance(text, str):
            return "Not Found"
            
        # Try different extraction methods in order of reliability
        extractors = [
            RingsExtractor.extract_by_pattern,
            RingsExtractor.extract_by_regex,
            RingsExtractor.extract_by_context
        ]
        
        for extractor in extractors:
            result = extractor(text)
            if result != "Not Found":
                return result
                
        return "Not Found"