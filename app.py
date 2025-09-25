import os
import re
import math
import base64
import pandas as pd
import io
import gc
import json
import logging
import tempfile
import numpy as np
import unicodedata
import openpyxl
from PIL import Image, ImageFilter

try:
    import cv2
    HAS_OPENCV = True
except Exception as e:
    cv2 = None
    HAS_OPENCV = False
    logging.warning(f"OpenCV (cv2) not available: {e}")
import pytesseract
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import fitz  # PyMuPDF
import pdf2image
from pdf2image import convert_from_path, convert_from_bytes
import google.generativeai as genai

# --- Basic Configuration ---
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Key Configuration ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    logging.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("GEMINI_API_KEY environment variable must be set to use Gemini AI features")

try:
    genai.configure(api_key=api_key)
    logging.info("Gemini API key configured successfully")
except Exception as e:
    logging.error(f"Failed to configure Gemini API key: {str(e)}")
    raise RuntimeError(f"Failed to initialize Gemini AI: {str(e)}")

# --- Load and Clean Material Database on Startup with Enhanced Debugging ---
def load_material_database():
    """Load and clean the material database from either Excel or CSV file."""
    try:
        # First try loading from Excel file
        material_df = pd.read_excel("MATERIAL WITH STANDARD.xlsx", sheet_name="Sheet1")
        source = "Excel"
    except FileNotFoundError:
        try:
            # Fall back to CSV if Excel file is not found
            material_df = pd.read_csv('material_data.csv')
            source = "CSV"
        except FileNotFoundError:
            logging.error("Neither MATERIAL WITH STANDARD.xlsx nor material_data.csv found. Material lookup will not work.")
            return None
    
    try:
        # Clean and standardize the data
        material_df.columns = material_df.columns.str.strip()
        material_df['STANDARD'] = material_df['STANDARD'].str.strip()
        material_df['GRADE'] = material_df['GRADE'].astype(str).str.strip()
        
        logging.info(f"Successfully loaded and cleaned material database from {source} with {len(material_df)} entries.")
        logging.info(f"Material database head (first 5 rows):\n{material_df.head().to_string()}")
        logging.info(f"Unique STANDARD values:\n{material_df['STANDARD'].unique().tolist()}")
        logging.info(f"Unique GRADE values:\n{material_df['GRADE'].unique().tolist()}")
        
        return material_df
    except FileNotFoundError as e:
        logging.error(f"Material database file not found: {str(e)}")
        return None
    except pd.errors.EmptyDataError:
        logging.error("Material database file is empty")
        return None
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing material database: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error processing material database: {str(e)}")
        return None

# Load the material database
material_df = load_material_database()

# --- String Normalization Helper ---
def normalize_for_comparison(text):
    """
    Converts text to a standardized format for reliable comparison.
    Preserves F-series standard formatting.
    """
    if not text:
        return ""
    
    # Convert to string and uppercase
    text = str(text).upper()
    
    # Preserve F-series standards (like F-1, F-30, etc.)
    text = re.sub(r'MPAPS\s*F\s*[-_]?\s*(\d+)', r'MPAPS F-\1', text)
    
    # Common OCR error corrections
    ocr_fixes = {
        'ВТРАР': 'STRAIGHT',
        'АК': 'AK',
        'ОЛГЮЛЕ': 'ANGLE',
        'ГРАД': 'GRADE',
        'ТУПЕ': 'TYPE',
        'О': 'O',
        'В': 'B',
    }
    for wrong, correct in ocr_fixes.items():
        text = text.replace(wrong, correct)
    
    # Remove grade/type prefix variations but preserve F-series
    if not re.search(r'F-\d+', text):
        prefixes = ['GRADE ', 'GRADE-', 'GRADE_', 'TYPE ', 'TYPE-', 'TYPE_', 'CLASS ', 'CLASS-', 'CLASS_']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
    
    # Remove all whitespace except around F-series
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert Roman numerals to numbers
    roman_to_num = {
        'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
        'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5',
        'І': '1', 'ІІ': '2', 'ІІІ': '3', 'ІV': '4', 'V': '5'
    }
    for roman, num in roman_to_num.items():
        text = text.replace(roman, num)
    
    return text
    
    # Common OCR error corrections
    ocr_fixes = {
        'ВТРАР': 'STRAIGHT',  # Russian OCR error
        'АК': 'AK',
        'ОЛГЮЛЕ': 'ANGLE',
        'ГРАД': 'GRADE',
        'ТУПЕ': 'TYPE',
        'О': 'O',  # Cyrillic O to Latin O
        'В': 'B',  # Cyrillic B to Latin B
    }
    for wrong, correct in ocr_fixes.items():
        text = text.replace(wrong, correct)
    
    # Remove grade/type prefix variations
    prefixes = ['GRADE ', 'GRADE-', 'GRADE_', 'TYPE ', 'TYPE-', 'TYPE_', 'CLASS ', 'CLASS-', 'CLASS_']
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
    
    # Normalize specification format
    text = re.sub(r'MPAPS\s*[-_]?\s*F\s*[-_]?\s*(\d+(?:\.\d+)?)', r'MPAPSF\1', text)
    text = re.sub(r'MPAPS\s+F\s*[-_]?\s*(\d+(?:\.\d+)?)', r'MPAPSF\1', text)
    text = re.sub(r'F-?(\d+(?:\.\d+)?)', r'F\1', text)
    
    # Remove all whitespace
    text = ''.join(text.split())
    
    # Convert Roman numerals to numbers (if needed) with better pattern matching
    roman_to_num = {
        'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
        'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5',  # Unicode Roman numerals
        'І': '1', 'ІІ': '2', 'ІІІ': '3', 'ІV': '4', 'V': '5'   # Similar-looking characters
    }
    for roman, num in roman_to_num.items():
        text = text.replace(roman, num)
    
    # Convert fraction-like text to decimals
    fraction_map = {
        '1/2': '.5', '1/4': '.25', '3/4': '.75',
        '1/3': '.33', '2/3': '.67',
        '1/8': '.125', '3/8': '.375', '5/8': '.625', '7/8': '.875'
    }
    for frac, dec in fraction_map.items():
        text = text.replace(frac, dec)
    
    # Strip special characters but preserve numbers and decimals
    text = re.sub(r'[^A-Z0-9\.]', '', text)
    
    return text

# --- Material Lookup Function ---
def get_material_from_standard(standard, grade):
    """
    Enhanced material lookup with improved F-series standard matching
    """
    if material_df is None:
        logging.error("Material database not loaded")
        return "Not Found"
    
    if standard == "Not Found" or grade == "Not Found":
        logging.warning("Standard or grade not provided")
        return "Not Found"
    
    try:
        # Clean inputs
        clean_standard = clean_text_encoding(str(standard))
        clean_grade = clean_text_encoding(str(grade))
        
        # Special handling for MPAPS F-1 -> MPAPS F-30/F-1 mapping
        if 'MPAPS F-1' in clean_standard.upper() or 'MPAPSF1' in clean_standard.upper():
            clean_standard = 'MPAPS F-30/F-1'
            logging.info(f"Mapping MPAPS F-1 to {clean_standard}")
        
        norm_standard = normalize_for_comparison(clean_standard)
        norm_grade = normalize_for_comparison(clean_grade)
        
        logging.info(f"Material lookup initiated:\n"
                    f"Original: Standard='{standard}', Grade='{grade}'\n"
                    f"Cleaned: Standard='{clean_standard}', Grade='{clean_grade}'\n"
                    f"Normalized: Standard='{norm_standard}', Grade='{norm_grade}'")
        
        # Stage 1: Exact match on cleaned values (less aggressive normalization)
        exact_matches = material_df[
            (material_df['STANDARD'].str.upper().str.strip() == clean_standard.upper()) &
            (material_df['GRADE'].astype(str).str.upper().str.strip() == clean_grade.upper())
        ]
        
        if not exact_matches.empty:
            material = exact_matches.iloc[0]['MATERIAL']
            logging.info(f"Exact match found: {material}")
            return material
        
        # Stage 2: Flexible matching for F-series standards
        best_match = None
        best_score = 0
        
        for idx, row in material_df.iterrows():
            db_standard = str(row['STANDARD']).upper().strip()
            db_grade = str(row['GRADE']).upper().strip()
            
            # Score for standard matching
            standard_score = 0
            if clean_standard.upper() in db_standard or db_standard in clean_standard.upper():
                standard_score = 1.0
            elif 'F-1' in clean_standard.upper() and 'F-30' in db_standard:
                standard_score = 0.9  # MPAPS F-1 should match MPAPS F-30/F-1
            elif any(term in db_standard for term in clean_standard.upper().split()):
                standard_score = 0.8
            
            # Score for grade matching
            grade_score = 0
            if clean_grade.upper() in db_grade or db_grade in clean_grade.upper():
                grade_score = 1.0
            elif norm_grade == normalize_for_comparison(db_grade):
                grade_score = 0.9
            
            # Combined score with priority on standard
            total_score = (standard_score * 0.7) + (grade_score * 0.3)
            
            if total_score > best_score:
                best_score = total_score
                best_match = row['MATERIAL']
                logging.debug(f"New best match: {best_match} (score: {best_score})")
        
        # Return match if score is sufficient
        if best_score >= 0.6:
            logging.info(f"Best match found: '{best_match}' (score: {best_score:.2f})")
            return best_match
        
        logging.warning(f"No suitable material match found for Standard: '{standard}', Grade: '{grade}'")
        return "Not Found"
    
    except Exception as e:
        logging.error(f"Error during material lookup: {str(e)}", exc_info=True)
        return "Not Found"
    
    try:
        # Clean and normalize inputs
        clean_standard = clean_text_encoding(str(standard))
        clean_grade = clean_text_encoding(str(grade))
        norm_standard = normalize_for_comparison(clean_standard)
        norm_grade = normalize_for_comparison(clean_grade)
        
        logging.info(f"Material lookup initiated:\n"
                    f"Original: Standard='{standard}', Grade='{grade}'\n"
                    f"Normalized: Standard='{norm_standard}', Grade='{norm_grade}'")
        
        # Stage 1: Exact match on normalized values
        exact_matches = material_df[
            (material_df['STANDARD'].apply(normalize_for_comparison) == norm_standard) &
            (material_df['GRADE'].apply(normalize_for_comparison) == norm_grade)
        ]
        
        if not exact_matches.empty:
            material = exact_matches.iloc[0]['MATERIAL']
            logging.info(f"Exact match found: {material}")
            return material
        
        logging.info("No exact match, proceeding with advanced matching...")
        
        # Stage 2: Advanced pattern matching with scoring
        best_match = None
        best_score = 0
        matches_found = []
        
        for idx, row in material_df.iterrows():
            db_standard = str(row['STANDARD'])
            db_grade = str(row['GRADE'])
            norm_db_standard = normalize_for_comparison(db_standard)
            norm_db_grade = normalize_for_comparison(db_grade)
            
            # Calculate match scores with detailed criteria
            standard_score = 0
            grade_score = 0
            
            # Standard matching criteria (hierarchical)
            if norm_standard == norm_db_standard:
                standard_score = 1.0
            elif norm_standard in norm_db_standard:
                standard_score = 0.9
            elif norm_db_standard in norm_standard:
                standard_score = 0.8
            else:
                # Check for F-series standards match
                std_pattern = re.search(r'F(\d+)', norm_standard)
                db_pattern = re.search(r'F(\d+)', norm_db_standard)
                if std_pattern and db_pattern:
                    if std_pattern.group(1) == db_pattern.group(1):
                        standard_score = 0.85
                
                # Check for common base standards
                base_patterns = ['MPAPS', 'SAE', 'ISO', 'DIN']
                for pattern in base_patterns:
                    if pattern in norm_standard and pattern in norm_db_standard:
                        standard_score = max(standard_score, 0.7)
                
                # Check for number matches within standard
                std_nums = re.findall(r'\d+', norm_standard)
                db_nums = re.findall(r'\d+', norm_db_standard)
                if set(std_nums) & set(db_nums):  # If any numbers match
                    standard_score = max(standard_score, 0.6)
            
            # Grade matching criteria (hierarchical)
            if norm_grade == norm_db_grade:
                grade_score = 1.0
            elif norm_grade.replace('1', 'I') == norm_db_grade or \
                 norm_grade.replace('I', '1') == norm_db_grade:
                grade_score = 0.9
            elif any(g in norm_db_grade for g in norm_grade.split()):
                grade_score = 0.8
            elif norm_grade in norm_db_grade or norm_db_grade in norm_grade:
                grade_score = 0.7
            
            # Special case scoring adjustments
            if 'F30' in norm_standard and 'F30' in norm_db_standard:
                standard_score = max(standard_score, 0.9)
            if 'F6032' in norm_standard and 'F6032' in norm_db_standard:
                standard_score = 1.0  # Direct F6032 match
            
            # Calculate weighted score with higher weight on standard
            total_score = (standard_score * 0.7) + (grade_score * 0.3)
            
            # Track all good matches for debugging
            if total_score >= 0.6:
                matches_found.append({
                    'material': row['MATERIAL'],
                    'standard': db_standard,
                    'grade': db_grade,
                    'score': total_score
                })
            
            if total_score > best_score:
                best_score = total_score
                best_match = row['MATERIAL']
        
        # Log all potential matches for debugging
        if matches_found:
            matches_str = "\n".join([
                f"Score {m['score']:.2f}: {m['material']} ({m['standard']} {m['grade']})"
                for m in sorted(matches_found, key=lambda x: x['score'], reverse=True)
            ])
            logging.info(f"Potential matches found:\n{matches_str}")
        
        # Return best match if score is sufficient
        if best_score >= 0.7:
            logging.info(f"Best match found: '{best_match}' (score: {best_score:.2f})")
            return best_match
        
        # Log detailed diagnostics for failed matches
        logging.warning(
            f"Material lookup failed:\n"
            f"Input Standard: '{standard}' (normalized: '{norm_standard}')\n"
            f"Input Grade: '{grade}' (normalized: '{norm_grade}')\n"
            f"Best match score: {best_score:.2f} (below threshold 0.7)\n"
            f"Available standards: {material_df['STANDARD'].unique().tolist()}\n"
            f"Available grades: {material_df['GRADE'].unique().tolist()}"
        )
        return "Not Found"
    
    except Exception as e:
        logging.error(f"Error during material lookup: {str(e)}", exc_info=True)
        logging.error(f"Failed lookup details - Standard: '{standard}', Grade: '{grade}'")
        return "Not Found"


    

from flask_cors import CORS
import base64
from io import BytesIO
import math

# --- Configuration ---
# 1. SET YOUR API KEY HERE
# It's best practice to set this as an environment variable, but you can paste it directly for testing.
# To set an environment variable:
# On Windows: set GEMINI_API_KEY=YOUR_API_KEY
# On macOS/Linux: export GEMINI_API_KEY=YOUR_API_KEY
# Gemini API key is already configured at the top of the file


# These settings are already configured at the top of the file

# Material database is already loaded at the top of the file from MATERIAL WITH STANDARD.xlsx



# --- Function to calculate development length using vector geometry ---
# --- Dimension and Coordinate Extraction Functions ---
def extract_dimensions_from_text(text):
    """
    Extract dimensions from the PDF text using regex patterns.
    Handles various formats and common OCR variations.
    """
    # Initialize results dictionary with default values
    dimensions = {
        "id1": "Not Found",
        "id2": "Not Found",
        "od1": "Not Found",
        "od2": "Not Found",
        "thickness": "Not Found",
        "centerline_length": "Not Found",
        "radius": "Not Found",
        "angle": "Not Found",
        "working_pressure": "Not Found",
        "burst_pressure": "Not Found",
        "coordinates": []  # List to store coordinate points
    }

    try:
        # Clean the text and normalize spacing
        text = clean_text_encoding(text)
        # Replace common OCR errors for diameter symbol
        text = text.replace('0/', 'Ø').replace('O/', 'Ø').replace('⌀', 'Ø').replace('O|', 'Ø').replace('0|', 'Ø')
        # Clean up spacing around diameter symbol
        text = re.sub(r'(?:INSIDE|INNER)\s*(?:DIAMETER|DIA\.?|DIAM\.?)?\s*(?:Ø|O|0)\s*', 'INSIDE Ø ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:I\.?D\.?|ID)\s*(?:Ø|O|0)\s*', 'ID Ø ', text, flags=re.IGNORECASE)
        # Normalize decimal separators and spacing
        text = re.sub(r'(\d)\s*[.,]\s*(\d)', r'\1.\2', text)
        # Remove any whitespace between numbers and units/symbols
        text = re.sub(r'(\d+(?:\.\d+)?)\s*(mm|MM|Ø|ID|O\.?D\.?)', r'\1 \2', text)
        # Clean up any double spaces
        text = re.sub(r'\s+', ' ', text)
        
        logger.debug(f"Cleaned text for dimension extraction: {text}")
        
        # Define regex patterns with variations
        patterns = {
            'id': [
                # Primary ID patterns with Ø symbol - these should match first
                r'INSIDE\s*Ø\s*(\d+(?:\.\d+)?)',  # Matches "INSIDE Ø 50.8"
                r'ID\s*Ø\s*(\d+(?:\.\d+)?)',      # Matches "ID Ø 50.8"
                # Hose ID patterns
                r'HOSE\s*ID\s*[=:]?\s*(\d+(?:\.\d+)?)',  # Matches "HOSE ID = 18.4"
                r'(?:HOSE\s+)?ID\s*[=:]?\s*(\d+(?:\.\d+)?)',  # More general pattern
                # Inside diameter variations
                r'(?:INSIDE|INNER)\s*(?:DIAMETER|DIA\.?|DIAM\.?)?\s*(?:Ø|⌀)?\s*(\d+(?:\.\d+)?)',
                # Standard ID formats
                r'(?:TUBING|HOSE)?\s*(?:I\.?D\.?|ID)\s*(?:AS\s+PER\s+2D|\(MM\))?\s*(?:=|:|IS|:=)?\s*(\d+(?:\.\d+)?)',
                r'(?:INSIDE|INNER)\s*(?:DIAMETER|DIA\.?|DIAM\.?)\s*(?:=|:|IS|:=)?\s*(\d+(?:\.\d+)?)',
                # Secondary variations
                r'BORE\s*(?:Ø|⌀)?\s*(?:=|:|IS|:=)?\s*(\d+(?:\.\d+)?)',
                r'(?:INT\.?|INTERNAL)\s*(?:DIA\.?|DIAM\.?)\s*(?:Ø|⌀)?\s*(?:=|:|IS|:=)?\s*(\d+(?:\.\d+)?)',
                # Fallback patterns
                r'(?:^|\s|:)(?:Ø|⌀)\s*(\d+(?:\.\d+)?)(?:\s*MM)?(?:\s|$)',  # Just the diameter symbol
                r'(?:^|\s|:)(\d+(?:\.\d+)?)\s*(?:MM\s+)?(?:ID|I\.D\.?)(?:\s|$)'  # Number followed by ID
            ],
            'od': [
                r'(?:TUBING|HOSE)?\s*OD\s*(?:AS\s+PER\s+2D|\(MM\))?\s*[=:]?\s*(\d+[.,]?\d*)',
                r'(?:OUTSIDE|OUTER)\s*(?:DIAMETER|DIA\.?)\s*[=:]?\s*(\d+[.,]?\d*)',
                r'O\.?D\.?\s*(?:AS\s+PER\s+2D|\(MM\))?\s*[=:]?\s*(\d+[.,]?\d*)'
            ],
            'thickness': [
                r'(?:WALL)?\s*THICKNESS\s*(?:AS\s+PER\s+2D|\(MM\))?\s*[=:]?\s*(\d+[.,]?\d*)',
                r'(?:WALL|TUBE)\s*(?:THK\.?|THICK\.?)\s*[=:]?\s*(\d+[.,]?\d*)',
                r'(?:THK|THICK)\.?\s*[=:]?\s*(\d+[.,]?\d*)'
            ],
            'centerline': [
                r'(?:APPROX\.?)?\s*CTRLINE\s*LENGTH\s*[=:]?\s*(\d+(?:\.\d+)?)',  # Matches "APPROX CTRLINE LENGTH = 489.67"
                r'(?:APPROX\.?)?\s*(?:CTRLINE|CENTERLINE|C/L)\s*(?:LENGTH)?\s*(?:AS\s+PER\s+2D|\(MM\))?\s*[=:]?\s*(\d+[.,]?\d*)',
                r'(?:LENGTH|LEN\.?)\s*(?:AS\s+PER\s+2D|\(MM\))?\s*[=:]?\s*(\d+[.,]?\d*)',
                r'(?:DEVELOPED|DEV\.?)\s*(?:LENGTH|LEN\.?)\s*[=:]?\s*(\d+[.,]?\d*)'
            ],
            'pressure': [
                r'MAX\s*(?:OPERATING|WORKING)?\s*PRESSURE[^.]*?(\d+(?:\.\d+)?)\s*kPag?',  # Matches "MAX OPERATING PRESSURE... 430 kPag"
                r'(?:OPERATING|WORKING)\s*PRESSURE[^.]*?(\d+(?:\.\d+)?)\s*kPag?',
                r'PRESSURE\s*RATING[^.]*?(\d+(?:\.\d+)?)\s*kPag?'
            ],
            'burst_pressure': [
                r'BURST\s*PRESSURE\s*(?:\(4×WP\))?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*kPag?',
                r'(?:MIN\.?|MINIMUM)?\s*BURST\s*(?:PRESSURE)?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*kPag?'
            ],
            'coordinates': [
                r'P(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)',  # Matches coordinate points P0, P1, etc.
                r'POINT\s*(\d+)\s*:\s*(?:X\s*=\s*)?(-?\d+\.\d+)\s*(?:Y\s*=\s*)?(-?\d+\.\d+)\s*(?:Z\s*=\s*)?(-?\d+\.\d+)'
            ],
            'radius': [
                r'(?:RADIUS|RAD\.?|R)\s*(?:=|:|IS|:=)?\s*(\d+[.,]?\d*)',
                r'R\s*(?:=|:|IS|:=)?\s*(\d+[.,]?\d*)'
            ],
            'angle': [
                r'(\d+[.,]?\d*)\s*(?:°|DEG\.?|DEGREES?)',
                r'(?:ANGLE|ANG\.?)\s*(?:=|:|IS|:=)?\s*(\d+[.,]?\d*)\s*(?:°|DEG\.?|DEGREES?)?'
            ]
        }
        
        # Extract dimensions using patterns
        for dim_type, pattern_list in patterns.items():
            logger.debug(f"Checking {dim_type} patterns...")
            for pattern in pattern_list:
                logger.debug(f"Trying pattern: {pattern}")
                matches = re.finditer(pattern, text, re.IGNORECASE)
                values = []
                
                for match in matches:
                    try:
                        value = match.group(1).replace(',', '.')
                        matched_text = match.group(0)  # Get the entire matched text
                        logger.debug(f"Found potential match: '{matched_text}' -> value: '{value}'")
                        
                        if value.replace('.', '', 1).replace('-', '', 1).isdigit():
                            # Convert to float to validate, but keep original decimal places
                            float_val = float(value)
                            # Preserve decimal places from original value
                            decimal_places = len(value.split('.')[-1]) if '.' in value else 0
                            values.append((float_val, decimal_places))
                            logger.debug(f"Successfully extracted value {float_val} with {decimal_places} decimal places")
                        else:
                            logger.debug(f"Skipped invalid numeric value: {value}")
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Error processing match: {e}")
                
                if values:
                    def format_value(val_tuple):
                        value, decimal_places = val_tuple
                        if decimal_places > 0:
                            return f"{value:.{decimal_places}f}"
                        return f"{value:.1f}" if value % 1 != 0 else f"{int(value)}"
                    
                    if dim_type == 'id':
                        dimensions['id1'] = format_value(values[0])
                        dimensions['id2'] = format_value(values[-1])  # Use last value if multiple found
                    elif dim_type == 'od':
                        dimensions['od1'] = format_value(values[0])
                        dimensions['od2'] = format_value(values[-1])
                    elif dim_type == 'thickness':
                        dimensions['thickness'] = format_value(values[0])
                    elif dim_type == 'centerline':
                        dimensions['centerline_length'] = format_value(values[0])
                    elif dim_type == 'radius':
                        dimensions['radius'] = format_value(values[0])
                    elif dim_type == 'angle':
                        dimensions['angle'] = format_value(values[0])
                    break  # Stop after first successful match for each dimension type
        
        # Cross-validate dimensions
        if dimensions['thickness'] == "Not Found":
            # Try to calculate thickness from ID/OD
            try:
                if dimensions['od1'] != "Not Found" and dimensions['id1'] != "Not Found":
                    od = float(dimensions['od1'])
                    id = float(dimensions['id1'])
                    if od > id:
                        thickness = (od - id) / 2
                        dimensions['thickness'] = str(round(thickness, 2))
            except (ValueError, TypeError):
                pass
        
        # Log the extraction results
        logger.info("Extracted dimensions:")
        for dim, value in dimensions.items():
            if value == "Not Found":
                logger.info(f"{dim}: {value} - No matching pattern found")
            else:
                logger.info(f"{dim}: {value}")
        
    except Exception as e:
        logger.error(f"Error extracting dimensions: {e}")
        logger.debug(f"Exception details:", exc_info=True)
    
    return dimensions

def extract_coordinates_from_text(text):
    """
    Extract coordinates from the PDF text using regex patterns.
    Handles various coordinate formats and includes error checking.
    """
    coordinates = []
    
    try:
        # Clean and normalize text
        text = clean_text_encoding(text)
        
        # Patterns for coordinate formats
        patterns = [
            # P1(x,y,z) format
            r'P(\d+)\s*[\(\[]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\)\]]\s*(?:R\s*=\s*(\d+\.?\d*))?',
            # P1 x y z format
            r'P(\d+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*(?:R\s*=\s*(\d+\.?\d*))?',
            # Point 1: x,y,z format
            r'(?:POINT|PT)\.?\s*(\d+)\s*[:=]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*(?:R\s*=\s*(\d+\.?\d*))?'
        ]
        
        # Track point numbers to avoid duplicates
        seen_points = set()
        
        # Extract coordinates using each pattern
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    point_num = int(match.group(1))
                    if point_num in seen_points:
                        continue
                        
                    point = {
                        'point': f'P{point_num}',
                        'x': float(match.group(2).replace(',', '.')),
                        'y': float(match.group(3).replace(',', '.')),
                        'z': float(match.group(4).replace(',', '.'))
                    }
                    
                    # Add radius if present (group 5)
                    if match.group(5):
                        point['r'] = float(match.group(5).replace(',', '.'))
                    
                    coordinates.append(point)
                    seen_points.add(point_num)
                    
                except (ValueError, TypeError) as e:
                    logging.warning(f"Invalid coordinate data at point P{point_num}: {e}")
                    continue
        
        # Sort coordinates by point number
        coordinates.sort(key=lambda p: int(p['point'][1:]))
        
        # Validate coordinate sequence
        if coordinates:
            expected_points = set(range(len(coordinates)))
            actual_points = {int(p['point'][1:]) for p in coordinates}
            missing_points = expected_points - actual_points
            
            if missing_points:
                logging.warning(f"Missing points in sequence: {missing_points}")
            
            # Log the extracted coordinates
            logging.info(f"Extracted {len(coordinates)} coordinate points:")
            for point in coordinates:
                logging.info(f"{point['point']}: ({point['x']}, {point['y']}, {point['z']})")
    
    except Exception as e:
        logging.error(f"Error extracting coordinates: {e}")
    
    return coordinates

def calculate_development_length(points):
    """
    Calculate the total development length considering both straight segments and bends.
    
    Args:
        points: List of point dictionaries containing x, y, z coordinates and optional radius
        
    Returns:
        float: Total development length in mm
    """
    try:
        if not points or len(points) < 2:
            logging.warning("Insufficient points for length calculation")
            return 0
            
        coordinates = []
        radii = []
        
        # Convert points to coordinate tuples and extract radii
        for point in points:
            try:
                x = float(point.get('x', 0))
                y = float(point.get('y', 0))
                z = float(point.get('z', 0))
                r = float(point.get('r', 0)) if point.get('r') is not None else 0
                coordinates.append((x, y, z))
                radii.append(r)
            except (ValueError, TypeError) as e:
                logging.warning(f"Invalid coordinate data: {e}")
                continue
        
        if len(coordinates) < 2:
            logging.warning("Insufficient valid coordinates for length calculation")
            return 0
        
        # Calculate total length using path length helper
        total_length = calculate_path_length(coordinates, radii)
        logging.info(f"Calculated development length: {total_length:.2f}mm")
        
        return round(total_length, 2)
            
    except Exception as e:
        logging.error(f"Error calculating development length: {e}")
        return 0
        if centerline != "Not Found" and str(centerline).replace('.', '', 1).replace('-', '', 1).isdigit():
            return round(float(centerline), 2)
        
        # Use default values if all else fails
        return round(2 * math.pi * 40 * (90 / 360), 2)  # 40mm radius, 90° angle default

def calculate_development_length(coordinates=None, radii=None, dimensions=None):
    """
    Calculate the total development length of a part considering bends and straight sections.
    
    Args:
        coordinates: List of (x,y,z) tuples representing path points
        radii: List of bend radii corresponding to each coordinate point
        dimensions: Dictionary containing part dimensions including centerline_length
    
    Returns:
        float: Total development length in mm
    """
    try:
        # If we have coordinates and radii, calculate path length
        if coordinates and radii and len(coordinates) >= 2:
            return round(calculate_path_length(coordinates, radii), 2)
            
        # Check for centerline length in dimensions
        if dimensions:
            centerline = dimensions.get("centerline_length", "Not Found")
            if centerline != "Not Found" and str(centerline).replace('.', '', 1).replace('-', '', 1).isdigit():
                return round(float(centerline), 2)
        
        # Use default values if all else fails
        return round(2 * math.pi * 40 * (90 / 360), 2)  # 40mm radius, 90° angle default
            
    except Exception as e:
        logging.error(f"Error calculating development length: {e}")
        return 0

# Removed duplicate function - using enhanced version below
    try:
        n = len(points)
        if n < 2:
            return 0
        
        # For just two points, return straight distance
        if n == 2:
            return math.dist(points[0], points[1])
        
        total_length = 0
        
        for i in range(n-1):
            # Calculate straight segment length
            straight_length = math.dist(points[i], points[i+1])
            total_length += straight_length
            
            # If this is a bend point (not first or last point)
            if i > 0 and i < n-1 and radii[i] and radii[i] > 0:
                try:
                    # Calculate vectors for incoming and outgoing segments
                    v1 = tuple(b-a for a, b in zip(points[i-1], points[i]))
                    v2 = tuple(b-a for a, b in zip(points[i], points[i+1]))
                    
                    # Calculate magnitudes
                    mag1 = math.sqrt(sum(x*x for x in v1))
                    mag2 = math.sqrt(sum(x*x for x in v2))
                    
                    if mag1 == 0 or mag2 == 0:
                        logging.warning(f"Zero magnitude vector at point {i}")
                        continue
                        
                    # Calculate dot product and angle
                    dot_product = sum(a*b for a,b in zip(v1, v2))
                    cos_theta = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
                    theta = math.acos(cos_theta)
                    
                    if theta == 0:
                        continue
                        
                    # Calculate bend adjustments
                    R = radii[i]
                    tangent_length = R * math.tan(theta / 2)
                    arc_length = R * theta
                    
                    # Subtract the overlap of tangent lines and add the arc length
                    total_length -= 2 * tangent_length
                    total_length += arc_length
                    logging.debug(f"Bend at point {i}: angle={math.degrees(theta):.1f}°, "
                                f"radius={R:.1f}mm, arc_length={arc_length:.1f}mm")
                    
                except Exception as e:
                    logging.warning(f"Error processing bend at point {i}: {e}")
                    continue
        
        return total_length
    
    except Exception as e:
        logging.error(f"Error calculating path length: {e}")
        return 0

def calculate_path_length(points, radii):
    """
    Calculate the total path length considering both straight segments and bends.
    
    Args:
        points: List of (x,y,z) tuples representing path points
        radii: List of bend radii (0 or None means no bend at that point)
    
    Returns:
        float: Total path length in mm
    """
    n = len(points)
    if n < 2:
        return 0
    
    # For just two points, return straight distance
    if n == 2:
        return math.dist(points[0], points[1])
    
    total_length = 0
    
    for i in range(n-1):
        # Calculate straight segment length
        straight_length = math.dist(points[i], points[i+1])
        total_length += straight_length
        
        # If this is a bend point (not first or last point)
        if i > 0 and i < n-1 and radii[i] and radii[i] > 0:
            # Calculate vectors for incoming and outgoing segments
            v1 = tuple(b-a for a, b in zip(points[i-1], points[i]))
            v2 = tuple(b-a for a, b in zip(points[i], points[i+1]))
            
            # Calculate magnitudes
            mag1 = math.sqrt(sum(x*x for x in v1))
            mag2 = math.sqrt(sum(x*x for x in v2))
            
            if mag1 == 0 or mag2 == 0:
                continue
                
            # Calculate dot product and angle
            dot_product = sum(a*b for a,b in zip(v1, v2))
            cos_theta = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
            theta = math.acos(cos_theta)
            
            if theta == 0:
                continue
                
            # Calculate bend adjustments
            R = radii[i]
            tangent_length = R * math.tan(theta / 2)
            arc_length = R * theta
            
            # Subtract the overlap of tangent lines and add the arc length
            total_length -= 2 * tangent_length
            total_length += arc_length
    
    return round(total_length, 2)

# --- Safe Dimension Processing ---
def safe_dimension_processing(ai_results):
    """Safely process dimensions with default values to prevent KeyError"""
    dimensions = {
        "id1": "Not Found",
        "id2": "Not Found", 
        "od1": "Not Found",
        "od2": "Not Found",
        "thickness": "Not Found",
        "centerline_length": "Not Found",
        "radius": "Not Found",
        "angle": "Not Found"
    }
    
    # Safely map AI results to dimensions
    if ai_results:
        dimension_map = {
            "id": "id1",
            "od": "od1", 
            "thickness": "thickness",
            "centerline_length": "centerline_length",
            "radius": "radius",
            "angle": "angle"
        }
        
        for ai_key, dim_key in dimension_map.items():
            if ai_key in ai_results and ai_results[ai_key] != "Not Found":
                dimensions[dim_key] = ai_results[ai_key]
    
    return dimensions

# --- Function to generate Excel sheet with all details ---
def generate_excel_sheet(analysis_results, dimensions, development_length):
    """
    Generate Excel sheet with all fields matching the exact required format.
    Includes input validation, calculated fields, and comprehensive error handling.
    """
    try:
        # Initialize column structure exactly as required
        columns = [
            'child part',                                         # Row 1: Original format
            'child quantity',                                     # Row 1: Original format
            'CHILD PART',                                        # Row 2: Uppercase format
            'CHILD PART DESCRIPTION',                            # Description
            'CHILD PART QTY',                                    # Quantity
            'SPECIFICATION',                                     # Combined standard+grade
            'MATERIAL',                                         # From database lookup
            'REINFORCEMENT',                                     # Additional info
            'VOLUME AS PER 2D',                                 # Volume calculation
            'ID1 AS PER 2D (MM)',                              # First ID measurement
            'ID2 AS PER 2D (MM)',                              # Second ID measurement
            'OD1 AS PER 2D (MM)',                              # First OD measurement
            'OD2 AS PER 2D (MM)',                              # Second OD measurement
            'THICKNESS AS PER 2D (MM)',                        # Direct thickness
            'THICKNESS AS PER ID OD DIFFERENCE',               # Calculated thickness
            'CENTERLINE LENGTH AS PER 2D (MM)',                # From drawing
            'DEVELOPMENT LENGTH AS PER CO-ORDINATE (MM)',      # Calculated length
            'BURST PRESSURE AS PER 2D (BAR)',                  # From drawing
            'BURST PRESSURE AS PER WORKING PRESSURE (4XWP) (BAR)', # Calculated
            'VOLUME AS PER 2D MM3',                            # Volume in mm³
            'WEIGHT AS PER 2D KG',                             # Weight from drawing
            'COLOUR AS PER DRAWING',                           # Color specification
            'ADDITIONAL REQUIREMENT',                          # Notes and requirements
            'OUTSOURCE',                                       # Outsourcing details
            'REMARK'                                          # Generated remarks
        ]

        # Get part number and clean it
        part_number = str(analysis_results.get('part_number', '')).strip()
        if not part_number or part_number == 'Not Found':
            part_number = "Unknown Part"

        # Get description and clean it
        description = str(analysis_results.get('description', '')).strip()
        if not description or description == 'Not Found':
            description = "No Description Available"

        # Format specification string
        standard = analysis_results.get('standard', 'Not Found')
        grade = analysis_results.get('grade', 'Not Found')
        specification = f"{standard}"
        if grade != 'Not Found':
            specification += f" {grade}"

        # Calculate thickness from ID/OD
        thickness_calculated = "Not Found"
        try:
            od1 = dimensions.get('od1', 'Not Found')
            id1 = dimensions.get('id1', 'Not Found')
            if od1 != 'Not Found' and id1 != 'Not Found':
                od_val = float(str(od1).replace(',', '.'))
                id_val = float(str(id1).replace(',', '.'))
                if od_val > id_val:
                    thickness_calculated = f"{((od_val - id_val) / 2):.2f}"
        except (ValueError, TypeError) as e:
            logging.warning(f"Error calculating thickness: {e}")

        # Calculate burst pressure if working pressure is available
        burst_pressure_calc = "Not Found"
        if 'working_pressure' in analysis_results and analysis_results['working_pressure'] != "Not Found":
            try:
                wp = float(str(analysis_results['working_pressure']).replace(',', '.'))
                burst_pressure_calc = f"{(wp * 4):.1f}"
            except (ValueError, TypeError):
                pass

        # Build the row data dictionary
        row_data = {
            'child part': part_number.lower(),                    # Row 1 format
            'child quantity': "",                                # Usually blank
            'CHILD PART': part_number.upper(),                   # Row 2 format
            'CHILD PART DESCRIPTION': description,
            'CHILD PART QTY': "1",                              # Default to 1
            'SPECIFICATION': specification,
            'MATERIAL': analysis_results.get('material', 'Not Found'),
            'REINFORCEMENT': "Not Found",                        # Usually not in drawing
            'VOLUME AS PER 2D': analysis_results.get('volume', 'Not Found'),
            'ID1 AS PER 2D (MM)': dimensions.get('id1', 'Not Found'),
            'ID2 AS PER 2D (MM)': dimensions.get('id2', 'Not Found'),
            'OD1 AS PER 2D (MM)': dimensions.get('od1', 'Not Found'),
            'OD2 AS PER 2D (MM)': dimensions.get('od2', 'Not Found'),
            'THICKNESS AS PER 2D (MM)': dimensions.get('thickness', 'Not Found'),
            'THICKNESS AS PER ID OD DIFFERENCE': thickness_calculated,
            'CENTERLINE LENGTH AS PER 2D (MM)': dimensions.get('centerline_length', 'Not Found'),
            'DEVELOPMENT LENGTH AS PER CO-ORDINATE (MM)': development_length,
            'BURST PRESSURE AS PER 2D (BAR)': analysis_results.get('burst_pressure', 'Not Found'),
            'BURST PRESSURE AS PER WORKING PRESSURE (4XWP) (BAR)': burst_pressure_calc,
            'VOLUME AS PER 2D MM3': analysis_results.get('volume_mm3', 'Not Found'),
            'WEIGHT AS PER 2D KG': analysis_results.get('weight', 'Not Found'),
            'COLOUR AS PER DRAWING': analysis_results.get('color', 'Not Found'),
            'ADDITIONAL REQUIREMENT': "CUTTING & CHECKING FIXTURE COST TO BE ADDED. Marking cost to be added.",
            'OUTSOURCE': "",
            'REMARK': ""
        }

        # Generate remarks
        remarks = []
        
        # Check for specification conversion
        if standard.startswith('MPAPS F 1'):
            remarks.append('Drawing specifies MPAPS F 1, considered as MPAPS F 30.')

        # Check for ID mismatch
        id1 = dimensions.get('id1', 'Not Found')
        id2 = dimensions.get('id2', 'Not Found')
        if id1 != 'Not Found' and id2 != 'Not Found' and id1 != id2:
            remarks.append('THERE IS MISMATCH IN ID 1 & ID 2')

        # Check for OD mismatch
        od1 = dimensions.get('od1', 'Not Found')
        od2 = dimensions.get('od2', 'Not Found')
        if od1 != 'Not Found' and od2 != 'Not Found' and od1 != od2:
            remarks.append('THERE IS MISMATCH IN OD 1 & OD 2')

        # Check thickness consistency
        if thickness_calculated != "Not Found" and dimensions.get('thickness', 'Not Found') != "Not Found":
            try:
                thickness_drawing = float(str(dimensions['thickness']).replace(',', '.'))
                thickness_calc = float(thickness_calculated)
                if abs(thickness_drawing - thickness_calc) > 0.1:  # 0.1mm tolerance
                    remarks.append(f'THICKNESS MISMATCH: Drawing={thickness_drawing}mm, Calculated={thickness_calc}mm')
            except (ValueError, TypeError):
                pass

        row_data['REMARK'] = ' '.join(remarks) if remarks else 'No specific remarks.'

        # Create DataFrame
        df = pd.DataFrame([row_data], columns=columns)

        # Format the Excel sheet
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='FETCH FROM DRAWING', index=False)
            
            # Get the worksheet
            worksheet = writer.sheets['FETCH FROM DRAWING']
            
            # Auto-fit column widths
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.column_dimensions[openpyxl.utils.get_column_letter(idx + 1)].width = max_length + 2

        output.seek(0)
        return output

    except Exception as e:
        logging.error(f"Error generating Excel sheet: {e}", exc_info=True)
        # Create a minimal error sheet
        error_df = pd.DataFrame([{
            'child part': 'ERROR',
            'REMARK': f'Excel generation failed: {str(e)}'
        }], columns=columns)
        error_output = BytesIO()
        error_df.to_excel(error_output, sheet_name='FETCH FROM DRAWING', index=False)
        error_output.seek(0)
        return error_output

# --- Text Cleanup Functions ---
def assess_text_quality(text):
    """
    Assess the quality of extracted text using multiple metrics.
    Returns a score between 0 (poor) and 1 (excellent).
    """
    if not text or len(text.strip()) == 0:
        return 0.0
        
    try:
        # Initialize score components
        metrics = {
            'length_score': 0.0,
            'ascii_score': 0.0,
            'word_score': 0.0,
            'whitespace_score': 0.0,
            'special_char_score': 0.0,
            'technical_score': 0.0  # For engineering-specific content
        }
        
        # 1. Length assessment (0.15 weight)
        text_length = len(text.strip())
        metrics['length_score'] = min(text_length / 100, 1.0) * 0.15
        
        # 2. ASCII character ratio (0.15 weight)
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        metrics['ascii_score'] = (ascii_chars / len(text)) * 0.15
        
        # 3. Valid word ratio (0.2 weight)
        words = text.split()
        valid_word_pattern = re.compile(r'^[A-Za-z0-9\-\.]+$')
        valid_words = sum(1 for word in words if valid_word_pattern.match(word))
        metrics['word_score'] = (valid_words / len(words) if words else 0) * 0.2
        
        # 4. Whitespace distribution (0.15 weight)
        space_ratio = text.count(' ') / len(text) if len(text) > 0 else 0
        metrics['whitespace_score'] = (min(space_ratio * 5, 1.0)) * 0.15
        
        # 5. Special character assessment (0.15 weight)
        special_chars = sum(1 for c in text if c in '.,()-_/')
        special_ratio = special_chars / len(text) if len(text) > 0 else 0
        metrics['special_char_score'] = (min(special_ratio * 10, 1.0)) * 0.15
        
        # 6. Technical content indicators (0.2 weight)
        technical_indicators = {
            r'\b\d+(\.\d+)?(mm|cm|m)\b': 0.4,  # Measurements
            r'\bMPAPS\s+F-\d+\b': 0.4,         # Standards
            r'\bGRADE\s+[A-Z0-9]+\b': 0.4,     # Grades
            r'\b(STRAIGHT|ANGLE|DATUM)\b': 0.3, # Common technical terms
            r'\b(INSIDE|OUTSIDE)\b': 0.2,       # Position terms
            r'\bSEE NOTE\b': 0.2,              # Reference indicators
            r'[<>±]\d+': 0.3                   # Tolerances and comparisons
        }
        
        tech_score = 0.0
        for pattern, value in technical_indicators.items():
            if re.search(pattern, text, re.IGNORECASE):
                tech_score += value
        metrics['technical_score'] = min(tech_score, 1.0) * 0.2
        
        # Calculate final score
        total_score = sum(metrics.values())
        
        # Log detailed metrics for debugging
        logging.debug(f"Text quality metrics: {metrics}")
        logging.debug(f"Final quality score: {total_score:.2f}")
        
        return min(max(total_score, 0.0), 1.0)
        
    except Exception as e:
        logging.error(f"Error in text quality assessment: {str(e)}", exc_info=True)
        return 0.0

def clean_text_encoding(text):
    """
    Clean and normalize text with focus on engineering and technical content.
    """
    if not text:
        return ""
        
    try:
        # Common OCR error replacements (expanded for technical content)
        replacements = {
            # Russian OCR errors
            'ВТРАР': 'STRAIGHT',
            'АК': 'AK',
            'ОЛГЮЛЕ': 'ANGLE',
            'ГРАД': 'GRADE',
            'ТУПЕ': 'TYPE',
            
            # Grade format variations
            '1В': '1B',
            'IВ': '1B',
            'IБ': '1B',
            '1Б': '1B',
            'GRADE IB': '1B',
            'GRADE I': '1B',
            'I B': '1B',
            'GRADE 1B': '1B',
            'GRADE1B': '1B',
            
            # Standard format variations
            'MPAPS F30': 'MPAPS F-30/F-1',
            'MPAPS F 30': 'MPAPS F-30/F-1',
            'MPAPS F-30': 'MPAPS F-30/F-1',
            'F-30': 'MPAPS F-30/F-1',
            
            # Common phrase errors
            'ВТРАР LOCATION MARK': 'STRAIGHT LOCATION MARK',
            'АК ОЛГЮЛЕ DATUM': 'ANGLE DATUM',
            'IN-SIDE': 'INSIDE',
            'SEE NOTE': 'SEE NOTE',
            
            # Measurement and unit fixes
            'MM': 'mm',
            'Mm': 'mm',
            'mM': 'mm',
            'СМ': 'cm',
            'CM': 'cm',
            
            # Symbol normalizations
            '×': 'x',
            '–': '-',
            '—': '-',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '°': 'deg',
            '±': '+/-'
        }
        
        # 1. Basic cleanup
        text = ''.join(char if char.isprintable() or char in '\n\t' else ' ' for char in text)
        
        # 2. Apply common replacements
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        # 3. Remove zero-width and invisible characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # 4. Normalize numbers and measurements
        # Fix common number OCR errors
        text = re.sub(r'l(?=\d)', '1', text)  # Letter 'l' before numbers
        text = re.sub(r'O(?=\d)', '0', text)  # Letter 'O' before numbers
        text = re.sub(r'(?<=\d)O', '0', text) # Letter 'O' after numbers
        
        # Standardize measurement formats
        text = re.sub(r'(\d+)\s*[xX]\s*(\d+)', r'\1x\2', text)  # Dimensions
        text = re.sub(r'(\d+)\s*(?:mm|MM|Mm|mM)', r'\1mm', text)  # Millimeters
        
        # 5. Normalize whitespace while preserving structure
        lines = [' '.join(line.split()) for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)
        
        # 6. Final cleanup
        text = text.strip()
        
        return text
        
    except Exception as e:
        logging.error(f"Error cleaning text encoding: {str(e)}", exc_info=True)
        return text if text else ""

# --- Function to convert PDF to images ---
def convert_pdf_to_images(pdf_content):
    """Convert PDF bytes to list of PIL images with error handling and optimized memory usage."""
    try:
        # Lower DPI to reduce memory usage while maintaining readable quality
        images = convert_from_bytes(pdf_content, dpi=150, fmt='png')
        logger.info(f"Converted PDF to {len(images)} images at 150 DPI")
        return images
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {str(e)}")
        raise

def extract_text_from_image(image):
    """
    Extract text from an image using OCR with advanced preprocessing and adaptive techniques.
    Includes both OpenCV and PIL-based preprocessing methods with quality validation.
    """
    try:
        # Attempt 1: Original image direct OCR
        text = pytesseract.image_to_string(image, config='--psm 1')  # Auto page segmentation
        base_text = clean_text_encoding(text)
        
        # Initial quality check
        quality_score = assess_text_quality(base_text)
        logging.info(f"Initial text quality score: {quality_score}")
        
        if quality_score >= 0.7:  # Good quality threshold
            return base_text.strip()
        
        logging.info("Initial text quality insufficient, attempting preprocessing...")
        best_text = base_text
        best_score = quality_score
        
        # Try OpenCV preprocessing if available
        if HAS_OPENCV:
            logging.info("Using OpenCV for advanced preprocessing...")
            try:
                if not HAS_OPENCV:
                    raise ImportError("OpenCV is not available")
                    
                # Convert to numpy array for OpenCV processing
                img_array = np.array(image)
                
                # Color image processing
                if len(img_array.shape) == 3:
                    # Try different color channels
                    channels = cv2.split(img_array)
                    gray_versions = [
                        cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY),  # Standard grayscale
                        channels[0],  # Red channel
                        channels[1],  # Green channel
                        channels[2]   # Blue channel
                    ]
                else:
                    gray_versions = [img_array]  # Already grayscale
            
                # Enhanced preprocessing pipeline
                preprocessed_images = []
                
                for gray in gray_versions:
                    # 1. Basic preprocessing
                    preprocessed_images.extend([
                        gray,  # Original
                        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Otsu's method
                        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive
                    ])
                    
                    # 2. Noise reduction
                    denoised = cv2.fastNlMeansDenoising(gray)
                    preprocessed_images.extend([
                        denoised,
                        cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    ])
                    
                    # 3. Edge enhancement
                    edges = cv2.Canny(gray, 100, 200)
                    kernel = np.ones((2,2), np.uint8)
                    dilated = cv2.dilate(edges, kernel, iterations=1)
                    preprocessed_images.append(255 - dilated)  # Inverted edge enhancement
                    
                    # 4. Contrast enhancement
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    preprocessed_images.append(enhanced)
                
                # Try OCR on each preprocessed image
                for processed in preprocessed_images:
                    # Convert back to PIL Image for OCR
                    pil_image = Image.fromarray(processed)
                    new_text = pytesseract.image_to_string(pil_image, config='--psm 1')
                    new_text = clean_text_encoding(new_text)
                    new_score = assess_text_quality(new_text)
                    
                    if new_score > best_score:
                        best_text = new_text
                        best_score = new_score
                
                logging.info(f"Best score after OpenCV preprocessing: {best_score}")
                
            except Exception as e:
                logging.warning(f"OpenCV preprocessing failed: {e}")
                # Continue to PIL fallback
                
                for processed in preprocessed_images:
                    # Convert back to PIL Image for OCR
                    pil_image = Image.fromarray(processed)
                    new_text = pytesseract.image_to_string(pil_image, config='--psm 1')
                    new_text = clean_text_encoding(new_text)
                    new_score = assess_text_quality(new_text)
                    
                    if new_score > best_score:
                        best_text = new_text
                        best_score = new_score
                        
                logging.info(f"Best score after OpenCV preprocessing: {best_score}")
                
            except Exception as cv_error:
                logging.warning(f"OpenCV preprocessing failed: {cv_error}")
                # Continue to PIL fallback
        else:
            logging.info("OpenCV not available, using PIL-based preprocessing")
        
        # PIL-based preprocessing fallback
        try:
            # Convert to grayscale
            gray_image = image.convert('L')
            
            # Try different PIL enhancement methods
            enhanced_images = [
                gray_image,  # Original grayscale
                gray_image.filter(ImageFilter.SHARPEN),  # Sharpen
                gray_image.filter(ImageFilter.EDGE_ENHANCE),  # Edge enhancement
                gray_image.filter(ImageFilter.DETAIL),  # Enhance details
                gray_image.filter(ImageFilter.SMOOTH)  # Smoothing
            ]
        except Exception as e:
            logging.error(f"PIL enhancement failed: {e}")
            return ""  # Return empty string on complete failure
            
            # Try different threshold values with PIL
            for enhanced in enhanced_images:
                for threshold in [100, 127, 150]:  # Different threshold values
                    binary = enhanced.point(lambda x: 0 if x < threshold else 255, '1')
                    new_text = pytesseract.image_to_string(binary, config='--psm 1')
                    new_text = clean_text_encoding(new_text)
                    new_score = assess_text_quality(new_text)
                    
                    if new_score > best_score:
                        best_text = new_text
                        best_score = new_score
            
            logging.info(f"Best score after PIL preprocessing: {best_score}")
            
            # Add contrast-enhanced versions
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                preprocessed_images.extend([
                    enhanced,
                    cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                ])
            except Exception as e:
                logging.warning(f"OpenCV preprocessing failed, falling back to PIL: {e}")
                # Fallback to PIL preprocessing
                try:
                    gray_image = image.convert('L')
                    
                    # Try different PIL-based preprocessing methods
                    processed_images = [
                        gray_image,  # Original grayscale
                        gray_image.point(lambda x: 0 if x < 128 else 255, '1'),
                        gray_image.filter(ImageFilter.EDGE_ENHANCE),
                        gray_image.filter(ImageFilter.SHARPEN),
                        gray_image.filter(ImageFilter.DETAIL)
                    ]
                    
                    for img in processed_images:
                        for threshold in [100, 127, 150]:
                            binary = img.point(lambda x: 0 if x < threshold else 255, '1')
                            text = pytesseract.image_to_string(binary, config='--psm 1')
                            cleaned_text = clean_text_encoding(text)
                            score = assess_text_quality(cleaned_text)
                            
                            if score > best_score:
                                best_text = cleaned_text
                                best_score = score
                                
                    logging.info(f"Best score after PIL preprocessing: {best_score}")
                    
                except Exception as pil_error:
                    logging.error(f"PIL preprocessing failed: {pil_error}")
                    
            return best_text.strip()
        
        # Try OCR with different PSM modes and preprocessing
        psm_modes = [1, 3, 4, 6]  # Different page segmentation modes
        
        for idx, processed_img in enumerate(preprocessed_images):
            for psm in psm_modes:
                try:
                    config = f'--psm {psm}'
                    processed_text = pytesseract.image_to_string(processed_img, config=config)
                    cleaned_text = clean_text_encoding(processed_text)
                    
                    # Evaluate text quality
                    current_score = assess_text_quality(cleaned_text)
                    logging.debug(f"Method {idx}, PSM {psm}: Quality score {current_score}")
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_text = cleaned_text
                        logging.info(f"New best text found (score: {best_score}) with method {idx}, PSM {psm}")
                        
                        if best_score >= 0.8:  # Excellent quality threshold
                            return best_text.strip()
                            
                except Exception as ocr_error:
                    logging.warning(f"OCR failed for method {idx}, PSM {psm}: {ocr_error}")
                    continue
        
        logging.info(f"Using best found text with score {best_score}")
        return best_text.strip()
        
    except Exception as e:
        logging.error(f"Error in OCR text extraction: {str(e)}", exc_info=True)
        return ""

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Enhanced drawing analysis using Gemini multimodal capabilities ---
def analyze_drawing(pdf_bytes):
    """
    Analyze engineering drawing using Gemini's multimodal capabilities.
    Performs OCR, text extraction, and structured data parsing in a single pipeline.
    
    Args:
        pdf_bytes: Raw PDF file content in bytes
    
    Returns:
        dict: Structured data containing drawing information including:
            - part_number
            - description
            - standard
            - grade
            - dimensions (id, od, thickness, etc.)
            - coordinates for development length
            - other metadata
            
    Raises:
        ValueError: If pdf_bytes is None or empty
        pdf2image.exceptions.PDFPageCountError: If PDF has no pages or is invalid
        json.JSONDecodeError: If AI response cannot be parsed as JSON
        genai.types.generation_types.BlockedPromptException: If content violates AI policy
    """
    if not pdf_bytes:
        raise ValueError("PDF content cannot be empty")
    results = {
        "part_number": "Not Found",
        "description": "Not Found",
        "standard": "Not Found",
        "grade": "Not Found",
        "material": "Not Found",
        "dimensions": {
            "id1": "Not Found",
            "id2": "Not Found",
            "od1": "Not Found",
            "od2": "Not Found",
            "thickness": "Not Found",
            "centerline_length": "Not Found"
        },
        "coordinates": [],
        "error": None
    }
    
    try:
        # 1. Convert PDF to images
        logger.info("Converting PDF to images...")
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
            
        try:
            page_images = convert_from_path(temp_pdf_path, dpi=150)
            logger.info(f"Converted PDF to {len(page_images)} images")
        finally:
            os.remove(temp_pdf_path)
        
        # 2. Process each page with Gemini Vision
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        all_data = []
        
        for i, page in enumerate(page_images):
            logger.info(f"Processing page {i+1} with Gemini Vision...")
            
            # Prepare image for Gemini
            img_byte_arr = io.BytesIO()
            page.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            prompt = """Analyze this engineering drawing and extract all relevant information.
            Return a JSON object with the following structure:
            {
                "part_number": "The drawing number or part number",
                "description": "Description or title of the part",
                "standard": "Material standard specification",
                "grade": "Material grade",
                "dimensions": {
                    "id1": "First inside diameter value",
                    "id2": "Second inside diameter value (if exists)",
                    "od1": "First outside diameter value",
                    "od2": "Second outside diameter value (if exists)",
                    "thickness": "Wall thickness",
                    "centerline_length": "Centerline length"
                },
                "coordinates": [
                    {"point": "P0", "x": 0.0, "y": 0.0, "z": 0.0},
                    {"point": "P1", "x": 1.0, "y": 2.0, "z": 3.0}
                ]
            }
            Include all measurement values in millimeters (mm).
            If a value is not found, use "Not Found".
            Focus on text near dimension lines and in the title block."""
            
            # Process with Gemini
            image_part = [{"mime_type": "image/png", "data": img_base64}]
            response = model.generate_content([prompt, *image_part])
            
            if response and response.text:
                try:
                    cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                    page_data = json.loads(cleaned_text)
                    all_data.append(page_data)
                    logger.info(f"Successfully parsed data from page {i+1}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from page {i+1}: {e}")
                    continue
        
        # 3. Combine data from all pages
        if all_data:
            # Use the most complete data set as base
            results = max(all_data, key=lambda x: sum(1 for v in x.values() if v != "Not Found"))
            
            # Merge coordinates from all pages
            all_coordinates = []
            for data in all_data:
                if "coordinates" in data and isinstance(data["coordinates"], list):
                    all_coordinates.extend(data["coordinates"])
            results["coordinates"] = all_coordinates
        
        logger.info("Drawing analysis completed successfully")
        return results
        
    except (pdf2image.exceptions.PDFPageCountError, pdf2image.exceptions.PDFSyntaxError) as e:
        logger.error(f"PDF conversion error: {str(e)}")
        results["error"] = f"Failed to process PDF: {str(e)}"
        return results
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        results["error"] = "Failed to parse AI response"
        return results
    except genai.types.generation_types.BlockedPromptException as e:
        logger.error(f"Gemini API content policy violation: {str(e)}")
        results["error"] = "Content policy violation"
        return results
    except Exception as e:
        logger.error(f"Unexpected error in drawing analysis: {str(e)}")
        results["error"] = f"Analysis failed: {str(e)}"
        return results
    logging.info("Starting data parsing with Gemini...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # --- UPDATED PROMPT ---
    prompt = f"""
    Analyze the following text from an engineering drawing and return the information as a clean JSON object.
    Do not include any text before or after the JSON object.

    Instructions for extraction:
    - "child_part": Find the value explicitly labeled "PART NO". This is the primary drawing number, which often looks like a 7-digit number followed by 'C' and another digit (e.g., 4721473C1).
    - "description": Find the main title of the part, usually starting with "HOSE,...".
    - "standard": Find the specification document, which looks like "MPAPS F-XXXX" or "SAE JXXXX".
    - "grade": Find the grade associated with the standard, which looks like "GRADE XX" or "TYPE X".
    - "id": Find the value explicitly labeled "TUBING ID", "HOSE ID", or "Inside Diameter".
    - "thickness": Find the value explicitly labeled "WALL THICKNESS".
    - "centerline_length": Find the value explicitly labeled "APPROX CTRLINE LENGTH".
    - "coordinates": Carefully parse the main coordinate table into an array of objects. Each object must have "point", "x", "y", and "z" keys.

    If any value is not explicitly found on the drawing, use the string "Not Found".

    Text to analyze:
    ---
    {full_text}
    ---

    Required JSON format:
    {{
        "child_part": "...",
        "description": "...",
        "standard": "...",
        "grade": "...",
        "id": "...",
        "thickness": "...",
        "centerline_length": "...",
        "coordinates": [
            {{"point": "P0", "x": 0.0, "y": 0.0, "z": 0.0}},
            {{"point": "P1", "x": 1.0, "y": 2.0, "z": 3.0}}
        ]
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed_data = json.loads(cleaned_text)
        logging.info("Successfully parsed Gemini response into JSON.")
        return parsed_data
    except Exception as e:
        logging.error(f"Failed to parse text with Gemini: {e}")
        # Return a default error structure that matches the expected output
        return {
            "child_part": "Error", "description": "Error", "standard": "Error",
            "grade": "Error", "id": "Error", "thickness": "Error",
            "centerline_length": "Error", "coordinates": [], "error": str(e)
        }

def analyze_drawing_with_gemini(pdf_bytes):
    """
    Analyze drawing using Google Gemini with improved prompt and image analysis.
    Extracts dimensions, specifications, grades, and material properties using AI vision model.
    """
    results = {
        "part_number": "Not Found",
        "description": "Not Found",
        "standard": "Not Found",
        "grade": "Not Found",
        "material": "Not Found",
        "od": "Not Found",
        "thickness": "Not Found",
        "centerline_length": "Not Found",
        "development_length": "Not Found",
        "burst_pressure": "Not Found",
        "working_temperature": "Not Found",
        "working_pressure": "Not Found",
        "coordinates": [],
        "dimensions": {},
        "error": None
    }
    
    try:
        # Convert PDF to images for better analysis
        images = convert_pdf_to_images(pdf_bytes)
        
        # Extract text using both PyMuPDF and OCR if needed
        pdf_document = fitz.open("pdf", pdf_bytes)
        full_text = []
        block_texts = []
        
        # First pass: Extract text and structure from PDF
        for page_num, page in enumerate(pdf_document):
            page_text = page.get_text()
            full_text.append(page_text)
            
            # Get text blocks for structural analysis
            blocks = page.get_text("blocks")
            for block in blocks:
                block_text = block[4].strip()
                if block_text:
                    block_texts.append(block_text)
                    logger.debug(f"Block: {block_text}")
        
        # Combine all text
        combined_text = "\n".join(full_text)
        logger.info(f"Initial text extraction length: {len(combined_text)}")
        
        # Use OCR if text content seems insufficient
        if len(combined_text.strip()) < 100 or not re.search(r'\d{7}[A-Z]\d', combined_text):
            logger.info("Text content appears insufficient, applying OCR")
            ocr_texts = []
            for img in images:
                # First try with default settings
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    ocr_texts.append(ocr_text)
                    
            if ocr_texts:
                combined_text = combined_text + "\n" + "\n".join(ocr_texts)
                logger.info(f"After OCR text length: {len(combined_text)}")
        
        # Clean and normalize text
        combined_text = clean_text_encoding(combined_text)
        combined_text = re.sub(r'\s+', ' ', combined_text)
        logger.info("Text cleaned and normalized")
        
        # Extract part number (multiple patterns)
        part_patterns = [
            # 7-digit+C+1-digit format with better context (e.g., 4582819C2)
            r'(?i)(?:PART|DWG|DRAWING)[-\s]*(?:NO\.?|NUM\.?|NUMBER)?[-\s.:]*?(\d{7}C\d)\b',
            r'(?i)CHILD\s*PART[-\s.:]*?(\d{7}C\d)\b',
            r'(?i)REV[\s.-]*[A-Z][-\s.]*?(\d{7}C\d)\b',  # Common format with revision
            r'(?i)\b(\d{7}C\d)[-_]?(?:S\d+)?[-_]?R[-_]?[A-Z][-_]?F\d+\b',  # Matches 4582819C2_S001-_R-A_F00
            r'(?:^|\s)(\d{7}C\d)(?=\s|$)',  # Standalone 7-digit+C+1
            
            # 9-digit format with better context (e.g., 439461604)
            r'(?i)(?:PART|DWG|DRAWING)[-\s]*(?:NO\.?|NUM\.?|NUMBER)?[-\s.:]*?(\d{9})\b',
            r'(?i)CHILD\s*PART[-\s.:]*?(\d{9})\b',
            r'(?i)REV[\s.-]*[A-Z][-\s.]*?(\d{9})\b',
            r'(?:^|\s)(\d{9})(?=\s|$)'  # Standalone 9-digit
        ]
        
        logger.info("Starting part number extraction...")
        
        # Search for part numbers and validate format
        found_numbers = []
        for pattern in part_patterns:
            matches = re.finditer(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                part_num = match.group(1)
                # Log the match details
                logger.debug(f"Found potential part number: {part_num}")
                logger.debug(f"  Context: {match.group(0)}")
                logger.debug(f"  Pattern: {pattern}")
                
                # Validate the format
                if re.match(r'\d{7}C\d', part_num) or re.match(r'\d{9}', part_num):
                    logger.info(f"Valid part number found: {part_num}")
                    found_numbers.append(part_num)
                else:
                    logger.warning(f"Invalid part number format: {part_num}")
                if re.match(r'\d{7}C\d', part_num) or re.match(r'\d{9}', part_num):
                    found_numbers.append(part_num)
                    logger.info(f"Found potential part number: {part_num}")
        
        # If we found any valid part numbers, use the first one
        if found_numbers:
            results["part_number"] = found_numbers[0]
            if len(found_numbers) > 1:
                logger.warning(f"Multiple part numbers found: {found_numbers}")
        else:
            logger.warning("No valid part number found in document")
                
        # Extract description (multiple patterns)
        desc_patterns = [
            r'CHILD\s*PART\s*DESCRIPTION\s*:?\s*([\w\s,\-\.]+?)(?=\n|\s{2,}|$)',
            r'(?:PART\s+)?DESCRIPTION\s*:?\s*([\w\s,\-\.]+?)(?=\n|\s{2,}|$)',
            r'(?<![\w,])\s*HOSE\s*,\s*([\w\s,\-\.]+?)(?=\n|\s{2,}|$)'
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                desc = match.group(1).strip()
                desc = re.sub(r'\s+', ' ', desc)  # Normalize spaces
                desc = desc.strip(' ,.;:-')  # Clean edges
                desc = desc.upper()  # Standardize case
                results["description"] = desc
                logger.info(f"Found description: {desc}")
                break
        
        # Extract dimensions using pattern matching
        results["dimensions"] = extract_dimensions_from_text(combined_text)
        logger.info(f"Extracted dimensions: {results['dimensions']}")
        
        # Extract coordinates
        results["coordinates"] = extract_coordinates_from_text(combined_text)
        logger.info(f"Extracted coordinates: {len(results['coordinates'])} points")
        
        # Use Gemini for advanced analysis
        gemini_results = analyze_drawing(pdf_bytes)
        
        if gemini_results:
            # Use safe dimension processing
            results["dimensions"] = safe_dimension_processing(gemini_results)
            
            # Update main results with AI findings
            results["part_number"] = gemini_results.get("part_number", "Not Found")
            results["description"] = gemini_results.get("description", "Not Found")
            results["standard"] = gemini_results.get("standard", "Not Found")
            results["grade"] = gemini_results.get("grade", "Not Found")
                
        logger.info("------------------------------------------")
        
        # Extract coordinate points
        points = []
        point_pattern = r'P\d+\s*[\(\[]?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,?\s*(-?\d+\.?\d*)?\s*[\)\]]?\s*(?:R\s*=?\s*(\d+\.?\d*))?'
        point_matches = re.finditer(point_pattern, full_text)
        
        for match in point_matches:
            point = {
                'x': float(match.group(1)),
                'y': float(match.group(2)),
                'z': float(match.group(3)) if match.group(3) else 0,
                'r': float(match.group(4)) if match.group(4) else None
            }
            points.append(point)
        
        if points:
            results["coordinates"] = points
        
        # Enhanced prompt for Gemini
        prompt = """
You are an expert in analyzing Navistar engineering drawings for hose components.
Carefully extract the following information:

1. Child Part Number: Look for 7-8 digit numbers ending in 'C' followed by 1-2 digits
2. Description: The component description in title block or notes
3. Specification: Look for the PRIMARY material specification:
   - Focus on hose material specs like MPAPS F-6032 for fuel/oil resistant hoses
   - Don't confuse with tolerance specs like F-30 or assembly specs like F-1
   - If multiple specs found, prioritize the one defining hose material type
4. Grade: Look for Type designation (e.g., Type I) or grade code
5. Material: Only report if explicitly stated in drawing, don't infer
6. Dimensions:
   - OD (outer diameter) in mm
   - Wall thickness in mm
   - Centerline length in mm
7. Operating Conditions:
   - Working pressure in bar
   - Burst pressure in bar (if specified)
   - Working temperature range (if specified)
8. Notes: Any special requirements or restrictions

Respond with ONLY a JSON object containing these exact keys:
{
    "part_number": string,
    "description": string,
    "standard": string,
    "grade": string,
    "material": string,
    "od": string,
    "thickness": string,
    "centerline_length": string,
    "burst_pressure": string,
    "working_temperature": string,
    "working_pressure": string
}

For any value not found in drawing, use "Not Found" (not null or empty string).
Pay special attention to distinguishing primary material specs from reference specs.
"""
        model = genai.GenerativeModel('gemini-1.5-flash') # Use a fast and capable model
        
        prompt = f"""
        Analyze the following text extracted from a technical engineering drawing. Your task is to find three specific pieces of information: the part number, the material standard, and the grade.

        Respond ONLY with a valid JSON object containing these keys: "part_number", "standard", "grade".

        - The 'part_number' is typically a 7-digit number followed by a 'C' and another digit (e.g., 4403886C2).
        - The 'standard' is likely to be 'MPAPS F-30'.
        - The 'grade' is a short code that often follows the word 'GRADE' (e.g., 1B).

        If you cannot find a value for any key, the value in the JSON should be "Not Found".

        Here is the text to analyze:
        ---
        {full_text}
        ---
        """

        # --- Step 3: Call Gemini API with vision model ---
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content([prompt, full_text])
            
            if response and response.text:
                # Clean the response to ensure it's valid JSON
                cleaned_response = re.sub(r'```json\s*|\s*```', '', response.text.strip())
                
                try:
                    # Parse the JSON response
                    gemini_results = json.loads(cleaned_response)
                    
                    # Update results with Gemini's findings
                    for key in gemini_results:
                        if key in results:
                            results[key] = gemini_results[key]
                    
                    # Log successful analysis
                    logger.info(f'Successfully analyzed drawing with part number: {results["part_number"]}')
                    
                except json.JSONDecodeError as je:
                    logger.error(f'Failed to parse Gemini response as JSON: {str(je)}')
                    results['error'] = 'Failed to parse analysis results'
                    print("Raw response:", response.text)  # Debug logging
            else:
                logger.warning('Received empty response from Gemini')
                results['error'] = 'No analysis results received'
                
        except Exception as e:
            logger.error(f'Error during Gemini API call: {str(e)}')
            results['error'] = f'Analysis failed: {str(e)}'
            
        # --- Step 4: Look up the material in the DataFrame ---
        if results["standard"] != "Not Found" and results["grade"] != "Not Found":
            standard_key = results["standard"]
            grade_key = results["grade"]
            
            match = material_df[
                material_df['STANDARD'].str.contains(standard_key, na=False, case=False) &
                (material_df['GRADE'] == grade_key)
            ]
            
            if not match.empty:
                results["material"] = match.iloc[0]['MATERIAL']

    except json.JSONDecodeError:
        results["error"] = "AI model returned a non-JSON response. Please try again."
    except Exception as e:
        results["error"] = f"An unexpected error occurred: {str(e)}"
    
    return results

# --- Data Validation Function ---
def validate_extracted_data(data):
    """
    Validates extracted data for completeness and logical consistency.
    Returns a list of validation issues found.
    """
    issues = []
    
    try:
        # Check critical fields with safe gets
        if data.get('standard') == 'Not Found':
            issues.append("Standard specification not found in drawing")
        if data.get('grade') == 'Not Found':
            issues.append("Grade/Type not found in drawing")
        if data.get('material') == 'Not Found':
            issues.append("Material could not be identified from standard and grade")
        
        # Validate dimensions with safe access
        dimensions = data.get('dimensions', {})
        if not isinstance(dimensions, dict):
            dimensions = {}
            issues.append("Invalid dimensions format")
            return issues
        
        # Safely get dimension values
        id1 = dimensions.get('id1', 'Not Found')
        od1 = dimensions.get('od1', 'Not Found')
        thickness = dimensions.get('thickness', 'Not Found')
        
        # Helper function to safely convert dimension strings to float
        def safe_dim_to_float(value):
            try:
                if value == 'Not Found':
                    return None
                return float(str(value).replace('mm', '').strip())
            except (ValueError, TypeError):
                return None
        
        # Check ID/OD relationship
        id1_val = safe_dim_to_float(id1)
        od1_val = safe_dim_to_float(od1)
        
        if id1_val is not None and od1_val is not None:
            if od1_val <= id1_val:
                issues.append(f"Invalid dimensions: OD ({od1_val}mm) should be greater than ID ({id1_val}mm)")
        
        # Check wall thickness consistency
        thickness_val = safe_dim_to_float(thickness)
        if thickness_val is not None:
            if thickness_val <= 0:
                issues.append(f"Invalid wall thickness: {thickness_val}mm")
            
            # Cross-validate thickness with ID/OD if available
            if id1_val is not None and od1_val is not None:
                calculated_thickness = (od1_val - id1_val) / 2
                if abs(calculated_thickness - thickness_val) > 0.1:  # Allow 0.1mm tolerance
                    issues.append(f"Thickness inconsistency: Specified {thickness_val}mm vs calculated {calculated_thickness}mm")
                    
    except Exception as e:
        logging.error(f"Error during data validation: {str(e)}")
        issues.append(f"Validation error: {str(e)}")
    except ValueError as e:
        issues.append(f"Error validating dimensions: {str(e)}")
    
    # Validate coordinates if present
    coordinates = data.get('coordinates', [])
    if coordinates:
        if len(coordinates) < 2:
            issues.append("Insufficient coordinate points for path calculation")
        else:
            try:
                for point in coordinates:
                    if not all(isinstance(point.get(k), (int, float)) for k in ['x', 'y', 'z']):
                        issues.append("Invalid coordinate values found")
                        break
            except Exception as e:
                issues.append(f"Error validating coordinates: {str(e)}")
    
    return issues

# --- API endpoint for file analysis ---
@app.route('/api/analyze', methods=['POST'])
def upload_and_analyze():
    # 1. Basic request validation
    if 'file' not in request.files:
        logging.warning("No file part in request")
        return jsonify({'error': 'No file part in request'}), 400
        
    file = request.files['file']
    if not file or not file.filename:
        logging.warning("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    # 2. File type validation
    if not file.filename.lower().endswith('.pdf'):
        logging.warning(f"Invalid file type: {file.filename}")
        return jsonify({"error": "Invalid file type. Please upload a PDF file."}), 400

    logging.info(f"Processing analysis request for file: {file.filename}")
    
    try:
        # 3. Read file contents
        pdf_bytes = file.read()
        if not pdf_bytes:
            logging.warning("Uploaded file is empty")
            return jsonify({"error": "Uploaded file is empty"}), 400
            
        # 4. Analyze drawing
        final_results = analyze_drawing(pdf_bytes)
        
        # 5. Response validation and return
        if not isinstance(final_results, dict):
            logging.error(f"Invalid analyzer response type: {type(final_results)}")
            return jsonify({"error": "Internal error: Invalid response format"}), 500
            
        if final_results.get("error"):
            error_msg = final_results["error"]
            if "PDF conversion error" in error_msg:
                logging.warning(f"PDF conversion failed: {error_msg}")
                return jsonify({"error": "Invalid or corrupted PDF file"}), 400
            elif "Content policy violation" in error_msg:
                logging.warning(f"Content policy violation: {error_msg}")
                return jsonify({"error": "Content policy violation"}), 403
            else:
                logging.error(f"Analysis error: {error_msg}")
                return jsonify({"error": error_msg}), 500
        
        # Process the results further
        part_number = final_results.get('part_number', 'Unknown')
        logging.info(f"Successfully analyzed drawing for part {part_number}")

        # Initialize dimensions with safe defaults
        final_results["dimensions"] = safe_dimension_processing(final_results)

        # Look up material based on standard and grade
        try:
            standard = final_results.get("standard", "Not Found")
            grade = final_results.get("grade", "Not Found")
            final_results["material"] = get_material_from_standard(standard, grade)
        except Exception as e:
            logging.error(f"Error in material lookup: {e}")
            final_results["material"] = "Not Found"

        # Validate the extracted data
        try:
            validation_issues = validate_extracted_data(final_results)
            if validation_issues:
                final_results["validation_warnings"] = validation_issues
                logging.warning(f"Validation issues found: {validation_issues}")
        except Exception as e:
            logging.error(f"Error in data validation: {e}")
            final_results["validation_warnings"] = [f"Validation error: {str(e)}"]

        # Initialize development length
        dev_length = 0
        
        # Calculate development length
        try:
            coordinates = final_results.get("coordinates", [])
            if coordinates:
                dev_length = calculate_development_length(coordinates)
                final_results["development_length_mm"] = f"{dev_length:.2f}" if dev_length > 0 else "Not Found"
            else:
                final_results["development_length_mm"] = "Not Found"
                logging.info("No coordinates found for development length calculation")
        except Exception as e:
            logging.error(f"Error calculating development length: {e}")
            final_results["development_length_mm"] = "Not Found"
        
        # Generate Excel report if helper function exists
        try:
            if 'generate_excel_sheet' in globals():
                excel_file = generate_excel_sheet(final_results, final_results.get("dimensions", {}), dev_length)
                excel_b64 = base64.b64encode(excel_file.getvalue()).decode('utf-8')
                final_results["excel_data"] = excel_b64
        except Exception as e:
            logging.warning(f"Excel generation skipped: {e}")

        logging.info(f"Successfully analyzed drawing: {final_results.get('part_number', 'Unknown')}")
        return jsonify(final_results)

    except Exception as e:
        logging.error(f"Error analyzing drawing: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"Error analyzing drawing: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# --- Route for the main webpage (no change) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Run the application (no change) ---
def analyze_image_with_gemini_vision(pdf_bytes):
    """Process PDF using Gemini Vision API for OCR"""
    logger.info("Starting Gemini Vision OCR analysis...")
    full_text = ""
    temp_pdf_path = None

    try:
        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name

        # Convert PDF to images
        page_images = convert_from_path(temp_pdf_path, dpi=150)
        logger.info(f"Converted PDF to {len(page_images)} images at 150 DPI")

        # Process each page with Gemini Vision
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        for i, page in enumerate(page_images):
            logger.info(f"Processing page {i+1} with Gemini Vision...")
            
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            page.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Convert to base64
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            # Create Gemini-compatible image object
            image_parts = [
                {
                    'mime_type': 'image/png',
                    'data': img_base64
                }
            ]
            
            # Process with Gemini
            response = model.generate_content(["Extract all text from this engineering drawing.", *image_parts])
            if response and response.text:
                full_text += response.text + "\n"

        logger.info(f"OCR complete. Total characters extracted: {len(full_text)}")
        return full_text

    except Exception as e:
        logger.error(f"Error in Gemini Vision processing: {e}")
        return ""

    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")

def extract_text_from_pdf(pdf_bytes):
    """
    Enhanced text extraction from PDF using multiple methods and layout preservation.
    Tries different extraction techniques and combines results for best output.
    
    Args:
        pdf_bytes: Raw PDF file content in bytes
        
    Returns:
        str: Combined text from all extraction methods
    """
    logger.info("Starting enhanced text extraction process...")
    logger.debug(f"Input PDF size: {len(pdf_bytes)} bytes")
    texts = []
    
    try:
        # Method 1: PyMuPDF with layout preservation
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            # Try different text extraction modes
            for page in doc:
                # Get text with layout preservation
                layout_text = page.get_text("text", sort=True)
                texts.append(layout_text)
                
                # Get text blocks with position information
                blocks = page.get_text("blocks")
                structured_text = "\n".join([block[4] for block in blocks])
                texts.append(structured_text)
                
                # Get raw text as fallback
                raw_text = page.get_text("text", sort=False)
                texts.append(raw_text)
                
        combined_text = "\n".join(filter(None, texts))
        logger.info(f"PyMuPDF extraction found {len(combined_text)} characters with layout preservation")
        
        # If extracted text seems insufficient, try OCR
        if len(combined_text.strip()) < 100 or not any(char.isdigit() for char in combined_text):
            logger.info("Initial extraction insufficient, falling back to OCR...")
            ocr_text = analyze_image_with_gemini_vision(pdf_bytes)
            if ocr_text:
                texts.append(ocr_text)
                
        # Combine all extracted text, remove duplicates while preserving order
        seen = set()
        final_text = []
        for text in texts:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    final_text.append(line)
        
        result = '\n'.join(final_text)
        logger.info(f"Final extracted text length: {len(result)} characters")
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced text extraction: {e}")
        # Fall back to OCR as last resort
        return analyze_image_with_gemini_vision(pdf_bytes)

if __name__ == '__main__':
    app.run(debug=True)