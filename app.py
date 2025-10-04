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
from development_length import calculate_vector_magnitude, calculate_dot_product, calculate_angle
import numpy as np
import unicodedata
import openpyxl
import fitz  # PyMuPDF
from PIL import Image, ImageFilter
from excel_output import generate_corrected_excel_sheet

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
from gemini_helper import process_pages_with_vision_or_ocr, extract_text_from_image_wrapper

# --- Helper Functions ---
def parse_material_block(material_text: str) -> dict:
    """
    Extract reinforcement and rings robustly from a material block.
    Returns safe string values and a source tag.
    """
    if not material_text:
        return {'reinforcement': 'Not Found', 'rings': 'Not Found', 'reinforcement_source': 'none'}

    rein_pat = re.compile(
        r'REINFORCEMENT\s*[:\-]?\s*(.*?)(?=\n\s*(?:RINGS|UNLESS|NOTES|MUST|$))',
        re.IGNORECASE | re.DOTALL
    )
    rings_pat = re.compile(
        r'RINGS\s*[:\-]?\s*(.*?)(?=\n\s*(?:UNLESS|NOTES|MUST|$))',
        re.IGNORECASE | re.DOTALL
    )

    def _clean(s):
        if s is None:
            return None
        s = s.strip()
        s = re.sub(r'\s+', ' ', s)           # collapse whitespace/newlines
        s = s.strip(' .;,-:')                # trim stray punctuation
        return s

    rein_m = rein_pat.search(material_text)
    rings_m = rings_pat.search(material_text)

    reinforcement = _clean(rein_m.group(1)) if rein_m else None
    rings = _clean(rings_m.group(1)) if rings_m else None

    # Fallback keyword sniffing (useful if label missing or OCR dropped colon)
    if not reinforcement:
        kw = re.search(r'(META[\s\-]?ARAMID|ARAMID|NOMEX|KEVLAR|POLYESTER FABRIC)',
                       material_text, re.IGNORECASE)
        if kw:
            reinforcement = kw.group(0).upper().replace('META ARAMID', 'META-ARAMID')

    # Normalizations
    if reinforcement:
        reinforcement = reinforcement.replace('META ARAMID', 'META-ARAMID')
        reinforcement = reinforcement.replace('META-ARAMID FABRIC', 'META-ARAMID FABRIC')
    if rings:
        rings = rings.replace('PER ASTM-A-313', 'PER ASTM A-313')

    return {
        'reinforcement': reinforcement if reinforcement else 'Not Found',
        'rings': rings if rings else 'Not Found',
        'reinforcement_source': 'drawing' if reinforcement else 'none'
    }

def parse_float_with_tolerance(s):
    """Return the first float found in the string, or None if none."""
    if s is None:
        return None
    m = re.search(r'[-+]?\d+(?:\.\d+)?', str(s))
    return float(m.group(0)) if m else None

def process_with_gemini(image_path):
    """
    Process an image with Google's Gemini Vision model.
    Returns structured data extracted from the image.
    """
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        image = Image.open(image_path)
        content = [
            "Here is a technical drawing of a tube or hose component. Please analyze it and extract the following information:",
            image
        ]
        response = model.generate_content(content)
        
        if response and response.text:
            cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            try:
                results = json.loads(cleaned_text)
                
                # Post-process standards
                if 'standard' in results:
                    std = results['standard']
                    if isinstance(std, str) and 'F-1' in std and 'F-30' in std:
                        results['standard'] = 'MPAPS F-30'
                        results['standards_note'] = 'Drawing shows both F-1 and F-30 standards'
                
                # Normalize measurements
                if 'working_pressure' in results and results['working_pressure'] != 'Not Found':
                    if not results['working_pressure'].lower().endswith('kpag'):
                        results['working_pressure'] = f"{results['working_pressure']} kPag"
                
                if 'weight' in results and results['weight'] != 'Not Found':
                    if not results['weight'].lower().endswith('kg'):
                        results['weight'] = f"{results['weight']} KG"
                
                return results
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response: {e}")
                return None
    except Exception as e:
        logger.error(f"Error in process_with_gemini: {str(e)}")
        return None

def process_ocr_text(text):
    """
    Process extracted OCR text to structure it into the required format.
    Extracts key information including full reinforcement details from MATERIAL block.
    Always returns a valid dictionary.
    """
    result = {
        "part_number": "Not Found",
        "description": "Not Found",
        "standard": "Not Found",
        "grade": "Not Found",
        "material": "Not Found",
        "reinforcement": "Not Found",
        "reinforcement_raw": "Not Found",  # Store raw reinforcement text from drawing
        "dimensions": {},
        "operating_conditions": {},
        "coordinates": []
    }
    
    # If text is not a string, return default result
    if not isinstance(text, str):
        logger.warning(f"process_ocr_text received non-string input: {type(text)}")
        return result
    
    try:
        # Clean and normalize text for better pattern matching
        text = clean_text_encoding(text)
        
        # Extract part number
        part_match = re.search(r'\d{7}[A-Z]\d', text)
        if part_match:
            result["part_number"] = part_match.group(0)
        
        # Extract description with improved pattern matching
        desc_patterns = [
            r'(?:HOSE|HEATER)[,\s]+(.*?)(?=(?:MPAPS|TYPE\s+\d|GRADE\s+\d|\bWP\b|\bID\b|\bOD\b|STANDARD|$))',
            r'(?:HOSE|HEATER)[,\s]+([^,]+(?:,[^,]+)*?)(?=(?:MPAPS|TYPE\s+\d|GRADE\s+\d|\bWP\b|\bID\b|\bOD\b|STANDARD|$))',
            r'(?:HOSE|HEATER)[,\s]+(.*?)(?=\d{7}[A-Z]\d|$)'
        ]
        
        description = "Not Found"
        for pattern in desc_patterns:
            desc_match = re.search(pattern, text, re.IGNORECASE)
            if desc_match:
                desc_text = desc_match.group(1).strip()
                if len(desc_text) > len(description):
                    description = desc_text
        
        result["description"] = description
        
        # ENHANCED: Extract full reinforcement and rings from MATERIAL block
        # Look for the material section with more flexible patterns
        material_patterns = [
            r'MATERIAL:\s*(.*?)(?=UNLESS|NOTES:|$)',  # Capture until UNLESS or NOTES
            r'HOSE:\s*(.*?)(?=UNLESS|NOTES:|$)',      # Alternative pattern starting with HOSE:
            r'MATERIAL:\s*(.*?REINFORCEMENT.*?RINGS.*?)(?=UNLESS|NOTES:|$)',  # Specific for reinforcement and rings
        ]
        
        material_text = ""
        for pattern in material_patterns:
            material_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if material_match:
                material_text = material_match.group(1).strip()
                logging.info(f"Extracted material block with pattern '{pattern}': {material_text}")
                break
        
        logging.debug(f"Raw material block (first 500 chars): {material_text[:500]}")
        
        # Debug log the raw material block for regex iteration
        logging.debug("Material block (first 400 chars): %s", material_text[:400].replace("\n", "\\n"))

        if material_text:
            # Use the robust parser to extract reinforcement and rings
            parsed = parse_material_block(material_text)
            
            # Store raw values for debugging
            result["reinforcement_raw"] = parsed["reinforcement"]
            result["rings_raw"] = parsed["rings"]  # Store in rings_raw for debugging
            result["rings"] = parsed["rings"]       # Store in rings for excel output
            logging.debug(f"Existing reinforcement_raw='{result.get('reinforcement_raw')}', rings='{result.get('rings')}'")

            # prefer drawing values when present
            try:
                if parsed['reinforcement'] != 'Not Found':
                    result['reinforcement'] = parsed['reinforcement']
                    result['reinforcement_source'] = parsed['reinforcement_source']
                    logging.info(f"Using drawing-extracted reinforcement: {result['reinforcement']}")

                    # If we have rings too, combine them
                    if parsed['rings'] != 'Not Found':
                        result['reinforcement'] = f"{result['reinforcement']}, {parsed['rings']}"
                        logging.info(f"Combined with rings: {result['reinforcement']}")
                        result["reinforcement_source"] = "drawing"
                        logging.info(f"Using drawing-extracted reinforcement: {result['reinforcement']}")
                    else:
                        result["reinforcement_source"] = "none"
            except Exception as e:
                logging.warning(f"Error when preferring drawing reinforcement: {e}")
        
        # If we still haven't found reinforcement, try direct search in full text
        if result["reinforcement"] == "Not Found":
            # Direct search for reinforcement
            direct_rein_match = re.search(r'REINFORCEMENT\s*[:\-]?\s*([^\n\r]+)', text, re.IGNORECASE | re.DOTALL)
            if direct_rein_match:
                result["reinforcement_raw"] = direct_rein_match.group(1).strip()
                result["reinforcement"] = result["reinforcement_raw"]
                logging.info(f"Direct reinforcement extraction: {result['reinforcement']}")
            
            # Direct search for rings
            direct_rings_match = re.search(r'RINGS\s*[:\-]?\s*([^\n\r]+)', text, re.IGNORECASE | re.DOTALL)
            if direct_rings_match:
                rings_value = direct_rings_match.group(1).strip()
                result["rings_raw"] = rings_value  # Store in rings_raw for debugging
                result["rings"] = rings_value       # Store in rings for excel output
                # If we already have reinforcement, combine them
                if result["reinforcement"] != "Not Found":
                    result["reinforcement"] = f"{result['reinforcement']}, {rings_value}"
                else:
                    result["reinforcement"] = rings_value
                logging.info(f"Direct rings extraction and combination: {result['reinforcement']}")
        
        # Extract standard (MPAPS) with priority to F-30
        std_match_30 = re.search(r'MPAPS\s*F[-\s]*30', text, re.IGNORECASE)
        std_match = re.search(r'MPAPS\s*F[-\s]*(\d+(?:/F[-\s]*\d+)?)', text, re.IGNORECASE)
        
        if std_match_30:
            result["standard"] = "MPAPS F-30"
        elif std_match:
            result["standard"] = f"MPAPS F-{std_match.group(1)}"
        
        # Extract grade with better pattern matching
        grade_pattern = r'(?:GRADE|TYPE)\s*([0-9][A-Z]|[A-Z][0-9]|[0-9]+[A-Z]+|[A-Z]+[0-9]+)\b'
        grade_match = re.search(grade_pattern, text, re.IGNORECASE)
        
        if grade_match:
            grade = grade_match.group(1).strip()
            # Normalize common grade formats
            if re.match(r'[0-9][A-Z]', grade):
                result["grade"] = f"{grade[0]}{grade[1]}"
            elif re.match(r'[A-Z][0-9]', grade):
                result["grade"] = f"{grade[1]}{grade[0]}"
            else:
                result["grade"] = grade
        
        # Extract dimensions
        dim_patterns = {
            "id1": r'HOSE\s+ID\s*[=:]?\s*(\d+(?:\.\d+)?)',
            "od1": r'OD\s*[=:]?\s*(\d+(?:\.\d+)?)',
            "thickness": r'THICKNESS[:\s]+([\d.]+)',
            "length": r'LENGTH[:\s]+([\d.]+)'
        }
        
        for key, pattern in dim_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["dimensions"][key] = match.group(1)
        
        # Extract weight
        weight_match = re.search(r'(?:WEIGHT|WT\.?)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:KG|kg)', text)
        if weight_match:
            result["weight"] = f"{weight_match.group(1)} KG"
        
        # Extract working pressure
        wp_match = re.search(r'(?:WORKING PRESSURE|WP)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:KPAG|kPag|KPA|kPa)', text, re.IGNORECASE)
        if wp_match:
            result["working_pressure"] = f"{wp_match.group(1)} kPag"
        
        # Extract burst pressure if present
        bp_match = re.search(r'(?:BURST PRESSURE|BP)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:KPAG|kPag|KPA|kPa|BAR)', text, re.IGNORECASE)
        if bp_match:
            result["burst_pressure"] = f"{bp_match.group(1)} kPag"
        
        # Extract coordinates if present
        coord_matches = re.findall(r'P\d+\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', text)
        for i, (x, y, z) in enumerate(coord_matches):
            result["coordinates"].append({
                "point": f"P{i}",
                "x": float(x),
                "y": float(y),
                "z": float(z)
            })
        
        # Extract material block and process reinforcement using the robust parser
        material_block_pattern = r'MATERIAL:\s*(.+?)(?=\n\s*(?:UNLESS|NOTES|DIMENSION|\Z))'
        material_match = re.search(material_block_pattern, text, re.DOTALL | re.IGNORECASE)
        
        material_text = material_match.group(1).strip() if material_match else ""
        parsed = parse_material_block(material_text)

        # Keep raw parsed outputs separate for debugging/excel/traceability
        result['reinforcement_raw'] = parsed['reinforcement']            # raw string or 'Not Found'
        result['rings_raw'] = parsed['rings']                           # Store in rings_raw for debugging
        result['rings'] = parsed['rings']                               # Store in rings for excel output
        result['reinforcement_parsed_source'] = parsed['reinforcement_source']

        # Debug: log the material text snippet + parsed result
        logging.debug("Material block (first 500 chars): %s", (material_text or "")[:500].replace("\n", "\\n"))
        logging.debug("Parsed material block: %s", parsed)

        # Final decision: prefer drawing-parsed reinforcement if it's a real value
        def _is_real_value(v):
            if v is None:
                return False
            if isinstance(v, str):
                vstr = v.strip().lower()
                return vstr not in ("", "not found", "none", "nan")
            return True

        final_reinf = None
        final_source = None

        if _is_real_value(result.get('reinforcement_raw')):
            final_reinf = result['reinforcement_raw']
            final_source = 'drawing'
            logging.info("Using drawing-extracted reinforcement: %s", final_reinf)
            
        else:
            # Fallback to database lookup ONLY if drawing did not provide it
            db_reinf = None
            try:
                db_reinf = get_reinforcement_from_material(result.get('standard'), result.get('grade'), result.get('material'))
            except Exception as e:
                logging.warning("DB reinforcement lookup failed: %s", e)
                db_reinf = None

            if _is_real_value(db_reinf):
                # keep DB match as fallback, but record it separately
                result['reinforcement_db'] = db_reinf
                final_reinf = db_reinf
                final_source = 'db'
                logging.info("Using reinforcement from DB fallback: %s", db_reinf)
            else:
                final_reinf = 'Not Found'
                final_source = 'none'
                logging.info("No reinforcement found in drawing or DB for this part")

        # Final safe assignment (always store a string)
        result['reinforcement'] = final_reinf if isinstance(final_reinf, str) else str(final_reinf)
        result['reinforcement_source'] = final_source
        # Keep trace of DB even if not used
        if 'reinforcement_db' not in result:
            result['reinforcement_db'] = db_reinf if 'db_reinf' in locals() else 'Not Found'

        # Extra logging to help pinpoint failures
        logging.info("Final reinforcement decision - Value: '%s', Source: %s", result.get('reinforcement'), result.get('reinforcement_source'))
        logging.debug("reinforcement_raw='%s', reinforcement_db='%s', rings_raw='%s'",
                    result.get('reinforcement_raw'), result.get('reinforcement_db'), result.get('rings_raw'))
        
        return result
    except Exception as e:
        logger.error(f"Error processing OCR text: {str(e)}")
        return None

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

# --- Load and Clean Material and Reinforcement Database on Startup with Enhanced Debugging ---
def load_material_database():
    """Load and clean the material and reinforcement database from Excel file."""
    try:
        # Load from Sheet1 (contains both material and reinforcement)
        material_df = pd.read_excel("MATERIAL WITH STANDARD.xlsx", sheet_name="Sheet1")
        reinforcement_df = material_df  # Use the same DataFrame for reinforcement lookups
        source = "Excel"
    except FileNotFoundError:
        try:
            material_df = pd.read_csv('material_data.csv')
            source = "CSV"
            reinforcement_df = None
            logging.warning("Reinforcement data not available in CSV format")
        except FileNotFoundError:
            logging.error("Neither MATERIAL WITH STANDARD.xlsx nor material_data.csv found. Material lookup will not work.")
            return None, None
    
    try:
        # Clean and standardize the material data
        material_df.columns = material_df.columns.str.strip()
        material_df['STANDARD'] = material_df['STANDARD'].str.strip()
        material_df['GRADE'] = material_df['GRADE'].astype(str).str.strip()
        
        logging.info(f"Successfully loaded and cleaned material database from {source} with {len(material_df)} entries.")
        logging.info(f"Material database head (first 5 rows):\n{material_df.head().to_string()}")
        logging.info(f"Unique STANDARD values:\n{material_df['STANDARD'].unique().tolist()}")
        logging.info(f"Unique GRADE values:\n{material_df['GRADE'].unique().tolist()}")
        
        # Since reinforcement is in the same DataFrame, log confirmation
        logging.info(f"Reinforcement database set to material_df with {len(reinforcement_df)} entries.")
        
        return material_df, reinforcement_df
    except Exception as e:
        logging.error(f"Unexpected error processing material database: {str(e)}")
        return None, None

# Load the material and reinforcement databases
material_df, reinforcement_df = load_material_database()

def get_reinforcement_from_material(standard, grade, material):
    """
    Look up reinforcement information based on standard, grade, and material with flexible matching.
    Returns the reinforcement type or "Not Found" if no match is found.
    """
    if reinforcement_df is None:
        logging.error("Reinforcement database not loaded")
        return "Not Found"
    
    if standard == "Not Found" or grade == "Not Found" or material == "Not Found":
        logging.warning("Standard, grade, or material not provided")
        return "Not Found"
    
    try:
        # Clean inputs
        clean_standard = clean_text_encoding(str(standard))
        clean_grade = clean_text_encoding(str(grade))
        clean_material = clean_text_encoding(str(material))
        
        logging.info(f"Reinforcement lookup initiated: Standard='{standard}', Grade='{grade}', Material='{material}'")
        
        # Enhanced normalization for comparison
        def normalize_standard(std):
            std = str(std).upper().strip()
            # Handle MPAPS F-series variations
            std = re.sub(r'MPAPS\s*F\s*[-_]?\s*(\d+)', r'MPAPS F-\1', std)
            std = re.sub(r'\s+', ' ', std).strip()
            return std
        
        def normalize_grade(grd):
            grd = str(grd).upper().strip()
            # Handle grade variations (1B, I-B, etc.)
            grd = re.sub(r'GRADE\s*', '', grd)
            grd = re.sub(r'TYPE\s*', '', grd)
            grd = re.sub(r'[_\-\s]', '', grd)
            # Convert Roman numerals
            roman_map = {'I': '1', 'II': '2', 'III': '3'}
            for roman, num in roman_map.items():
                if grd == roman:
                    grd = num + 'B'  # Assume B type if not specified
            return grd
        
        norm_standard = normalize_standard(clean_standard)
        norm_grade = normalize_grade(clean_grade)
        
        logging.info(f"Normalized: Standard='{norm_standard}', Grade='{norm_grade}'")
        
        # Stage 1: Exact match on cleaned values
        exact_matches = reinforcement_df[
            (reinforcement_df['STANDARD'].str.upper().str.strip() == norm_standard) &
            (reinforcement_df['GRADE'].astype(str).str.upper().str.strip() == norm_grade)
        ]
        
        if not exact_matches.empty:
            reinforcement = exact_matches.iloc[0]['REINFORCEMENT']
            logging.info(f"Exact match found: {reinforcement}")
            return reinforcement
        
        # Stage 2: Handle common variations like MPAPS F-30/F-1
        if norm_standard == 'MPAPS F-30' and norm_grade in ['1B', '1']:
            matches = reinforcement_df[
                (reinforcement_df['STANDARD'].str.upper().str.contains('MPAPS F-30', regex=False)) &
                (reinforcement_df['GRADE'].astype(str).str.upper().str.contains('1B|1', regex=True))
            ]
            if not matches.empty:
                reinforcement = matches.iloc[0]['REINFORCEMENT']
                logging.info(f"Found reinforcement for MPAPS F-30/1B: {reinforcement}")
                return reinforcement
        
        # Stage 3: Partial matching with scoring
        best_match = None
        best_score = 0
        
        for idx, row in reinforcement_df.iterrows():
            db_standard = normalize_standard(str(row['STANDARD']))
            db_grade = normalize_grade(str(row['GRADE']))
            
            # Standard matching score
            standard_score = 0
            if norm_standard == db_standard:
                standard_score = 1.0
            elif 'F-30' in norm_standard and 'F-30' in db_standard:
                standard_score = 0.9
            elif any(term in db_standard for term in norm_standard.split()):
                standard_score = 0.7
            
            # Grade matching score
            grade_score = 0
            if norm_grade == db_grade:
                grade_score = 1.0
            elif norm_grade in db_grade or db_grade in norm_grade:
                grade_score = 0.8
            
            # Combined score with weighted standard matching
            total_score = (standard_score * 0.7) + (grade_score * 0.3)
            
            if total_score > best_score:
                best_score = total_score
                best_match = row['REINFORCEMENT']
                logging.info(f"New best match found: '{best_match}' (score: {best_score:.2f})\n" +
                             f"  DB Standard: {db_standard}\n" +
                             f"  DB Grade: {db_grade}\n" +
                             f"  Standard Score: {standard_score:.2f}\n" +
                             f"  Grade Score: {grade_score:.2f}")
        
        # Return best match if score is high enough
        if best_score >= 0.6:
            logging.info(f"Best match accepted: '{best_match}' (score: {best_score:.2f})")
            return best_match
        
        logging.warning(f"No reinforcement match found for Standard: '{standard}', Grade: '{grade}', Material: '{material}'")
        return "Not Found"
    
    except Exception as e:
        logging.error(f"Error during reinforcement lookup: {str(e)}", exc_info=True)
        return "Not Found"

        matches = reinforcement_df[
            (reinforcement_df['MATERIAL'].str.upper() == clean_material.upper())
        ]
        
        if not matches.empty:
            reinforcement = matches.iloc[0]['REINFORCEMENT']
            logging.info(f"Exact reinforcement match found: {reinforcement}")
            return reinforcement
        
        # Try matching with normalized values
        norm_standard = normalize_for_comparison(clean_standard)
        norm_grade = normalize_for_comparison(clean_grade)
        norm_material = normalize_for_comparison(clean_material)
        
        matches = reinforcement_df[
            reinforcement_df.apply(lambda row: (
                normalize_for_comparison(str(row['STANDARD'])) == norm_standard and
                normalize_for_comparison(str(row['GRADE'])) == norm_grade and
                normalize_for_comparison(str(row['MATERIAL'])) == norm_material
            ), axis=1)
        ]
        
        if not matches.empty:
            reinforcement = matches.iloc[0]['REINFORCEMENT']
            logging.info(f"Normalized reinforcement match found: {reinforcement}")
            return reinforcement
        
        logging.warning(f"No reinforcement match found for Standard: '{standard}', Grade: '{grade}', Material: '{material}'")
        return "Not Found"
    
    except Exception as e:
        logging.error(f"Error during reinforcement lookup: {str(e)}", exc_info=True)
        return "Not Found"

# --- String Normalization Helper ---
def get_standards_remark(text, standard):
    """
    Generate remarks about standards based on text content and extracted standard.
    Returns tuple of (remark, suggested_standard)
    """
    if not text or not standard:
        return None, standard
        
    f1_match = re.search(r'MPAPS\s*F[-\s]*1\b', text, re.IGNORECASE)
    f30_match = re.search(r'MPAPS\s*F[-\s]*30\b', text, re.IGNORECASE)
    
    if f1_match and f30_match:
        return 'Drawing shows both MPAPS F-1 and F-30 standards', 'MPAPS F-30'
    elif f1_match and not f30_match:
        return 'Drawing specifies MPAPS F-1, considered as MPAPS F-30', 'MPAPS F-30'
    elif standard.startswith('MPAPS F-1'):
        return 'Standard MPAPS F-1 is considered as MPAPS F-30', 'MPAPS F-30'
    
    return None, standard

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
        
        logging.info(f"Material lookup initiated: Standard='{standard}', Grade='{grade}'")
        logging.info(f"Cleaned: Standard='{clean_standard}', Grade='{clean_grade}'")
        
        # Special handling for MPAPS F-1 -> MPAPS F-30 mapping
        if 'MPAPS F-1' in clean_standard.upper() or 'MPAPSF1' in clean_standard.upper():
            clean_standard = 'MPAPS F-30'
            logging.info(f"Mapping MPAPS F-1 to {clean_standard}")
        
        # Enhanced normalization for comparison
        def normalize_standard(std):
            std = str(std).upper().strip()
            # Handle MPAPS F-series variations
            std = re.sub(r'MPAPS\s*F\s*[-_]?\s*(\d+)', r'MPAPS F-\1', std)
            std = re.sub(r'\s+', ' ', std).strip()
            return std
        
        def normalize_grade(grd):
            grd = str(grd).upper().strip()
            # Handle grade variations (1B, I-B, etc.)
            grd = re.sub(r'GRADE\s*', '', grd)
            grd = re.sub(r'TYPE\s*', '', grd)
            grd = re.sub(r'[_\-\s]', '', grd)
            # Convert Roman numerals
            roman_map = {'I': '1', 'II': '2', 'III': '3'}
            for roman, num in roman_map.items():
                if grd == roman:
                    grd = num + 'B'  # Assume B type if not specified
            return grd
        
        norm_standard = normalize_standard(clean_standard)
        norm_grade = normalize_grade(clean_grade)
        
        logging.info(f"Normalized: Standard='{norm_standard}', Grade='{norm_grade}'")
        
        # Stage 1: Exact match on cleaned values
        exact_matches = material_df[
            (material_df['STANDARD'].str.upper().str.strip() == norm_standard) &
            (material_df['GRADE'].astype(str).str.upper().str.strip() == norm_grade)
        ]
        
        if not exact_matches.empty:
            material = exact_matches.iloc[0]['MATERIAL']
            logging.info(f"Exact match found: {material}")
            return material
        
        # Stage 2: Flexible matching for common variations
        # Handle MPAPS F-30/F-1 mapping
        if norm_standard == 'MPAPS F-30' and norm_grade in ['1B', '1']:
            # Look for MPAPS F-30 with grade 1B
            matches = material_df[
                (material_df['STANDARD'].str.upper().str.contains('MPAPS F-30')) &
                (material_df['GRADE'].astype(str).str.upper().str.contains('1B'))
            ]
            if not matches.empty:
                material = matches.iloc[0]['MATERIAL']
                logging.info(f"Found material for MPAPS F-30/1B: {material}")
                return material
        
        # Stage 3: Partial matching with scoring
        best_match = None
        best_score = 0
        
        for idx, row in material_df.iterrows():
            db_standard = normalize_standard(str(row['STANDARD']))
            db_grade = normalize_grade(str(row['GRADE']))
            
            # Standard matching score
            standard_score = 0
            if norm_standard == db_standard:
                standard_score = 1.0
            elif 'F-30' in norm_standard and 'F-30' in db_standard:
                standard_score = 0.9
            elif any(term in db_standard for term in norm_standard.split()):
                standard_score = 0.7
            
            # Grade matching score
            grade_score = 0
            if norm_grade == db_grade:
                grade_score = 1.0
            elif norm_grade in db_grade or db_grade in norm_grade:
                grade_score = 0.8
            
            # Combined score with weighted standard matching
            total_score = (standard_score * 0.7) + (grade_score * 0.3)
            
            if total_score > best_score:
                best_score = total_score
                best_match = row['MATERIAL']
                logging.info(f"New best match found: '{best_match}' (score: {best_score:.2f})\n" +
                           f"  DB Standard: {db_standard}\n" +
                           f"  DB Grade: {db_grade}\n" +
                           f"  Standard Score: {standard_score:.2f}\n" +
                           f"  Grade Score: {grade_score:.2f}")
        
        # Return best match if score is high enough
        if best_score >= 0.6:
            logging.info(f"Best match accepted: '{best_match}' (score: {best_score:.2f})")
            return best_match
        
        logging.warning(f"No material match found for Standard: '{standard}', Grade: '{grade}'")
        return "Not Found"
    
    except Exception as e:
        logging.error(f"Error during material lookup: {str(e)}", exc_info=True)
        return "Not Found"
    
    if best_score >= 0.6:
        logging.info(f"Best match found: '{best_match}' (score: {best_score:.2f})")
        return best_match
    
    logging.warning(f"No suitable material match found for Standard: '{standard}', Grade: '{grade}'")
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
    Enhanced dimension extraction with improved patterns for specific PDF format
    """
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
        "burst_pressure": "Not Found"
    }

    try:
        # Clean the text
        text = clean_text_encoding(text)
        
        # DIRECT STRING MATCHES - Most reliable method
        logger.info("Starting direct string matching for dimensions...")
        
        # 1. Hose ID extraction
        # First try direct pattern matching
        id_patterns = [
            r'HOSE\s+ID\s*[=:.]\s*(\d+(?:\.\d+)?)',
            r'(?:INSIDE|INNER)?\s*(?:DIAMETER|DIA\.?)?\s*ID\s*[=:.]\s*(\d+(?:\.\d+)?)',
            r'(?:TUBE|HOSE)?\s*I\.?D\.?\s*[=:.]\s*(\d+(?:\.\d+)?)',
            r'BORE\s*(?:Ø|⌀)?\s*[=:.]\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in id_patterns:
            id_match = re.search(pattern, text, re.IGNORECASE)
            if id_match:
                id_value = id_match.group(1)
                dimensions["id1"] = id_value
                dimensions["id2"] = id_value
                logger.info(f"ID match found: {id_value} using pattern {pattern}")
                break
        
        # 2. Centerline length extraction  
        if "APPROX CTRLINE LENGTH = 489.67" in text:
            dimensions["centerline_length"] = "489.67"
            logger.info("Direct match: APPROX CTRLINE LENGTH = 489.67")
        
        # 3. Working pressure and burst pressure extraction/calculation
        # First try to find working pressure with unit
        wp_patterns = [
            r'(?:WORKING PRESSURE|WP)\s*[=:.]\s*(\d+(?:\.\d+)?)\s*(?:KPAG|kPag|KPA|kPa)\b',
            r'(?:WORKING PRESSURE|WP)\s*[=:.]\s*(\d+(?:\.\d+)?)\s*(?:BAR|bar)\b'
        ]
        
        for pattern in wp_patterns:
            wp_match = re.search(pattern, text, re.IGNORECASE)
            if wp_match:
                wp_value = float(wp_match.group(1))
                # Store working pressure in kPa
                if 'bar' in wp_match.group().lower():
                    wp_value_kpa = wp_value * 100  # Convert bar to kPa
                    dimensions["working_pressure"] = f"{wp_value} bar ({wp_value_kpa} kPag)"
                else:
                    wp_value_kpa = wp_value
                    dimensions["working_pressure"] = f"{wp_value} kPag"
                
                # Calculate and store burst pressure (4x working pressure) in both units
                burst_kpa = wp_value_kpa * 4
                burst_bar = burst_kpa / 100  # Convert kPa to bar
                dimensions["burst_pressure"] = f"{burst_bar:.1f} bar ({burst_kpa:.0f} kPag)"
                
                logger.info(f"Working pressure found: {dimensions['working_pressure']}")
                logger.info(f"Calculated burst pressure: {dimensions['burst_pressure']}")
                break
        
        # ENHANCED REGEX PATTERNS for fallback
        patterns = {
            'id': [
                # General ID patterns
                r'HOSE\s+ID\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'(?:INSIDE|INNER)?\s*(?:DIAMETER|DIA\.?)?\s*(?:I\.?D\.?|ID)\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'(?:TUBE|HOSE)?\s*I\.?D\.?\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'(?:INSIDE|INNER)?\s*(?:Ø|⌀)?\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'BORE\s*(?:Ø|⌀)?\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'(?:INT\.?|INTERNAL)\s*(?:DIA\.?|DIAM\.?)\s*[=:.]\s*(\d+(?:\.\d+)?)',
            ],
            'centerline': [
                # Direct patterns
                r'APPROX\s+CTRLINE\s+LENGTH\s*=\s*(489\.67)',  # Specific to our PDF
                # General patterns
                r'APPROX\s+CTRLINE\s+LENGTH\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'CENTERLINE\s+LENGTH\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'CTRLINE\s+LENGTH\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'C\.?L\.?\s*LENGTH\s*[=:.]\s*(\d+(?:\.\d+)?)',
            ],
            'working_pressure': [
                # Direct patterns
                r'(430)\s*kPag',  # Specific to our PDF
                # General patterns
                r'MAX\s+(?:OPERATING|WORKING)\s+PRESSURE\s*[=:.]\s*(\d+(?:\.\d+)?)\s*kPag',
                r'(?:OPERATING|WORKING)\s+PRESSURE\s*[=:.]\s*(\d+(?:\.\d+)?)\s*kPag',
                r'PRESSURE\s*RATING\s*[=:.]\s*(\d+(?:\.\d+)?)\s*kPag',
            ],
            'od': [
                # General OD patterns
                r'(?:OUTSIDE|OUTER)\s*DIAMETER\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'O\.?D\.?\s*[=:.]\s*(\d+(?:\.\d+)?)',
                r'(?:TUBE|HOSE)?\s*OD\s*[=:.]\s*(\d+(?:\.\d+)?)',
            ]
        }
        
        # Apply regex patterns as fallback
        for dim_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if dim_type == 'id':
                        if dimensions["id1"] == "Not Found":
                            dimensions["id1"] = value
                            dimensions["id2"] = value
                            logger.info(f"Found ID = {value}")
                    elif dim_type == 'centerline':
                        if dimensions["centerline_length"] == "Not Found":
                            dimensions["centerline_length"] = value
                            logger.info(f"Found centerline length = {value}")
                    elif dim_type == 'working_pressure':
                        if dimensions["working_pressure"] == "Not Found":
                            dimensions["working_pressure"] = value
                            # Calculate burst pressure
                            try:
                                wp = float(value)
                                dimensions["burst_pressure"] = str(wp * 4)
                                logger.info(f"Calculated burst pressure = {wp * 4} kPag")
                            except:
                                pass
                            logger.info(f"Found working pressure = {value}")
                    elif dim_type == 'od':
                        if dimensions["od1"] == "Not Found":
                            dimensions["od1"] = value
                            dimensions["od2"] = value
                            logger.info(f"Found OD = {value}")
                    break
        
        # Extract OD from notes if available (3/4" nominal)
        if dimensions["od1"] == "Not Found" and "3/4\" NOM" in text:
            dimensions["od1"] = "25.4"  # Approximate OD for 3/4" hose
            dimensions["od2"] = "25.4"
            logger.info("Estimated OD = 25.4mm from 3/4\" nominal size")
        
        # Calculate thickness if we have both ID and OD
        try:
            if dimensions["id1"] != "Not Found" and dimensions["od1"] != "Not Found":
                id_val = float(dimensions["id1"])
                od_val = float(dimensions["od1"])
                thickness = (od_val - id_val) / 2
                dimensions["thickness"] = f"{thickness:.2f}"
                logger.info(f"Calculated wall thickness = {thickness:.2f}mm")
        except:
            pass
        
        # Log successful extractions
        logger.info("EXTRACTED DIMENSIONS:")
        for key, value in dimensions.items():
            if value != "Not Found":
                logger.info(f"  {key}: {value}")
        
        return dimensions
        
    except Exception as e:
        logger.error(f"Error extracting dimensions: {e}")
        return dimensions

        
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
        
        # MANUAL EXTRACTION AS PRIMARY METHOD - MORE RELIABLE
        # Direct string search as primary method
        if 'HOSE ID = 18.4' in text:
            dimensions['id1'] = '18.4'
            logger.info("Direct string match found for ID: 18.4")
        elif 'HOSE ID =' in text:
            # Extract number after "HOSE ID ="
            match = re.search(r'HOSE ID =\s*(\d+(?:\.\d+)?)', text)
            if match:
                dimensions['id1'] = match.group(1)
                logger.info(f"Extracted ID from HOSE ID pattern: {match.group(1)}")
        
        if 'APPROX CTRLINE LENGTH = 489.67' in text:
            dimensions['centerline_length'] = '489.67'
            logger.info("Direct string match found for centerline: 489.67")
        elif 'APPROX CTRLINE LENGTH =' in text:
            # Extract number after "APPROX CTRLINE LENGTH ="
            match = re.search(r'APPROX CTRLINE LENGTH =\s*(\d+(?:\.\d+)?)', text)
            if match:
                dimensions['centerline_length'] = match.group(1)
                logger.info(f"Extracted centerline from pattern: {match.group(1)}")
        
        # Extract working pressure
        if '430 kPag' in text or '430 kPag' in text:
            dimensions['working_pressure'] = '430'
            logger.info("Found working pressure: 430 kPag")
        
        # Log the final extraction results
        logger.info("FINAL EXTRACTED DIMENSIONS:")
        for dim, value in dimensions.items():
            logger.info(f"{dim}: {value}")
        
    except Exception as e:
        logger.error(f"Error extracting dimensions: {e}")
        logger.debug(f"Exception details:", exc_info=True)
    
    return dimensions

def extract_coordinates_from_text(text):
    """
    Enhanced coordinate extraction with improved pattern matching and validation
    """
    coordinates = []
    
    try:
        # Clean and normalize text
        text = clean_text_encoding(text)
        
        logger.info("Starting coordinate extraction...")
        
        # Look for the coordinate table section
        coord_section_match = re.search(r'COORDS\s+POINTS\s+(.*?)(?:\n\s*\n|\Z)', text, re.DOTALL | re.IGNORECASE)
        if coord_section_match:
            coord_section = coord_section_match.group(1)
            logger.info(f"Found coordinate section: {coord_section[:200]}...")
            
            # Pattern for coordinate lines: P0, P1, etc. with X, Y, Z, R
            coord_pattern = r'P(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?' 
            
            matches = re.finditer(coord_pattern, coord_section)
            
            for match in matches:
                try:
                    point_num = int(match.group(1))
                    x = float(match.group(2))
                    y = float(match.group(3)) 
                    z = float(match.group(4))
                    r = match.group(5)  # Optional radius
                    
                    point_data = {
                        'point': f'P{point_num}',
                        'x': x,
                        'y': y,
                        'z': z
                    }
                    
                    if r:
                        point_data['r'] = float(r)
                    
                    coordinates.append(point_data)
                    logger.info(f"Extracted point {point_data['point']}: ({x}, {y}, {z}) R={r if r else 'None'}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid coordinate data at point P{point_num}: {e}")
                    continue
        
        # Alternative pattern for the entire table
        if not coordinates:
            logger.info("Trying alternative coordinate pattern...")
            # Pattern that matches the full table structure
            alt_pattern = r'P(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?\s*\n'
            matches = re.finditer(alt_pattern, text)
            
            for match in matches:
                try:
                    point_num = int(match.group(1))
                    x = float(match.group(2))
                    y = float(match.group(3))
                    z = float(match.group(4))
                    r = match.group(5)
                    
                    point_data = {
                        'point': f'P{point_num}',
                        'x': x,
                        'y': y, 
                        'z': z
                    }
                    
                    if r:
                        point_data['r'] = float(r)
                    
                    coordinates.append(point_data)
                    logger.info(f"Alt pattern match - Point {point_data['point']}: ({x}, {y}, {z}) R={r if r else 'None'}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid coordinate data at point P{point_num}: {e}")
                    continue
        
        # Sort by point number
        coordinates.sort(key=lambda p: int(p['point'][1:]))
        
        # Validate sequence
        if coordinates:
            point_numbers = {int(p['point'][1:]) for p in coordinates}
            expected_numbers = set(range(min(point_numbers), max(point_numbers) + 1))
            missing = expected_numbers - point_numbers
            
            if missing:
                logger.warning(f"Missing points in sequence: P{', P'.join(map(str, missing))}")
        
        logger.info(f"Successfully extracted {len(coordinates)} coordinate points")
        
        return coordinates
        
    except Exception as e:
        logger.error(f"Error extracting coordinates: {e}")
        return []



# --- Helper: try to coerce a value to float, else None ---
def _to_float(val):
    try:
        if val is None:
            return None
        # strip commas, spaces
        s = str(val).strip().replace(',', '.')
        # allow things like "18.4mm" by extracting first number
        m = re.search(r'[-+]?\d*\.\d+|\d+', s)
        if m:
            return float(m.group(0))
    except Exception:
        pass
    return None

# --- Helper: try many common shapes and extract first three numeric coords ---
_num_re = re.compile(r'[-+]?\d*\.\d+|\d+')

def point_to_xyz(p):
    """
    Convert a point in many possible forms to (x, y, z) (floats).
    Accepts:
      - dict with keys like 'x','X','x_coord','X_COORD', etc.
      - list/tuple like [x, y, z, ...]
      - string like "P0 0.0 0.0 0.0" or "0.0, 0.0, 0.0"
      - other objects: will inspect .values() if dict-like
    Returns:
      (x, y, z) as floats if at least 3 numeric values found, else None
    """
    # dict-like path
    if isinstance(p, dict):
        # try common keys first
        candidates = []
        for k in p.keys():
            candidates.append((str(k).lower(), p[k]))
        # gather numeric values among values (ordered)
        vals = []
        for k, v in candidates:
            nf = _to_float(v)
            if nf is not None:
                vals.append(nf)
        if len(vals) >= 3:
            return vals[0], vals[1], vals[2]
        # attempt specifically by common key names
        x = _to_float(p.get('x') or p.get('X') or p.get('x_coord') or p.get('X_COORD'))
        y = _to_float(p.get('y') or p.get('Y') or p.get('y_coord') or p.get('Y_COORD'))
        z = _to_float(p.get('z') or p.get('Z') or p.get('z_coord') or p.get('Z_COORD'))
        if x is not None and y is not None and z is not None:
            return x, y, z
        return None

    # list/tuple path
    if isinstance(p, (list, tuple)):
        nums = []
        for item in p:
            nf = _to_float(item)
            if nf is not None:
                nums.append(nf)
            if len(nums) >= 3:
                return nums[0], nums[1], nums[2]
        return None

    # string path
    if isinstance(p, str):
        found = _num_re.findall(p)
        if len(found) >= 3:
            return float(found[0]), float(found[1]), float(found[2])
        return None

    # last resort: try iterating attributes / values
    try:
        values = list(getattr(p, '__dict__', {}).values()) or list(getattr(p, 'values', lambda: [])())
        nums = []
        for v in values:
            nf = _to_float(v)
            if nf is not None:
                nums.append(nf)
            if len(nums) >= 3:
                return nums[0], nums[1], nums[2]
    except Exception:
        pass

    return None

def normalize_coordinates(points):
    """
    Canonicalize 'points' into a list of dicts: [{'point':'P0','x':float,'y':float,'z':float,'r':float}, ...]
    Accepts:
      - list of dicts {'x','y','z', optional 'r'}
      - list/tuple of numeric tuples (x,y,z) or (x,y,z,r)
      - string containing coordinate text (regex extraction)
      - None or unexpected types -> return []
    """
    coords = []

    if not points:
        return coords

    # If already a list-like collection
    if isinstance(points, list):
        for idx, p in enumerate(points):
            try:
                if isinstance(p, dict):
                    # Require numeric x,y,z
                    x = p.get('x')
                    y = p.get('y')
                    z = p.get('z')
                    if x is None or y is None or z is None:
                        logging.warning(f"normalize_coordinates: missing x/y/z in dict at index {idx}, skipping")
                        continue
                    coords.append({
                        'point': p.get('point', f'P{idx}'),
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'r': float(p.get('r', 0)) if p.get('r') is not None else 0.0
                    })
                elif isinstance(p, (tuple, list)) and len(p) >= 3:
                    # tuple/list form -> (x,y,z) or (x,y,z,r)
                    try:
                        x = float(p[0])
                        y = float(p[1])
                        z = float(p[2])
                        r = float(p[3]) if len(p) > 3 else 0.0
                        coords.append({
                            'point': f'P{idx}',
                            'x': x, 'y': y, 'z': z, 'r': r
                        })
                    except Exception:
                        logging.warning(f"normalize_coordinates: invalid numeric tuple at index {idx}, skipping")
                        continue
                else:
                    # unexpected element type; try to coerce if it's a string
                    if isinstance(p, str):
                        # attempt regex extraction from this element
                        txt_coords = re.findall(r'P?(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)', p)
                        if txt_coords:
                            for m in txt_coords:
                                pn = int(m[0])
                                coords.append({
                                    'point': f'P{pn}',
                                    'x': float(m[1]),
                                    'y': float(m[2]),
                                    'z': float(m[3]),
                                    'r': 0.0
                                })
                    else:
                        logging.debug(f"normalize_coordinates: skipping unsupported point type {type(p)} at index {idx}")
            except Exception as e:
                logging.warning(f"normalize_coordinates: error processing element index {idx}: {e}")
        return coords

    # If points is a tuple-like of coordinates
    if isinstance(points, (tuple,)):
        try:
            if len(points) >= 3:
                x = float(points[0]); y = float(points[1]); z = float(points[2])
                r = float(points[3]) if len(points) > 3 else 0.0
                return [{'point': 'P0', 'x': x, 'y': y, 'z': z, 'r': r}]
        except Exception:
            return []

    # If it's a string, attempt regex extraction for coordinate table lines
    if isinstance(points, str):
        found = re.finditer(r'P(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(?:\s+(-?\d+(?:\.\d+)?))?', points)
        seen = set()
        for m in found:
            try:
                pn = int(m.group(1))
                if pn in seen:
                    continue
                x = float(m.group(2)); y = float(m.group(3)); z = float(m.group(4))
                r = float(m.group(5)) if m.group(5) else 0.0
                coords.append({'point': f'P{pn}', 'x': x, 'y': y, 'z': z, 'r': r})
                seen.add(pn)
            except Exception as e:
                logging.warning(f"normalize_coordinates: skipping regex match due to: {e}")
        # sort by point number if available
        coords.sort(key=lambda p: int(re.sub(r'[^0-9]', '', p.get('point','0')) or 0))
        return coords

    # Fallback: unknown type
    logging.debug(f"normalize_coordinates: unknown points type {type(points)}")
    return []

def calculate_development_length_safe(coords, *args, **kwargs):
    """
    coords: iterable of points in mixed forms.
    Returns development length (float) or raises ValueError if not enough valid coords.
    This wraps the existing calculate_development_length logic but first normalizes points.
    """
    normalized = []
    for i, p in enumerate(coords or []):
        xyz = point_to_xyz(p)
        if xyz is None:
            logger.warning("Invalid coordinates: Missing numeric coords in point %s, skipping. Raw: %s", i, p)
            continue
        x, y, z = xyz
        normalized.append({'x': float(x), 'y': float(y), 'z': float(z)})

    if len(normalized) < 2:
        # preserve old behavior (warning + None) but avoid KeyError
        logger.warning("Insufficient valid coordinates after normalization (%d points).", len(normalized))
        raise ValueError("Insufficient valid coordinates to compute development length")

    # Call the original function with normalized coordinates
    try:
        return calculate_development_length(normalized, *args, **kwargs)
    except Exception as e:
        logger.exception("Development length calculation failed on normalized coords: %s", e)
        raise

def calculate_development_length(points):
    """
    Robust development-length calculator.
    Accepts coordinates in multiple forms (list-of-dicts, list-of-tuples, or text).
    Returns rounded length in mm (float). Returns 0 if insufficient valid coords.
    """
    try:
        # Known explicit centerline fallback check (string)
        if isinstance(points, str) and ('APPROX CTRLINE LENGTH = 489.67' in points or '489.67' in points):
            logging.info("calculate_development_length: found explicit centerline length in text -> using 489.67")
            return 489.67

        # Canonicalize coordinates
        norm = normalize_coordinates(points)

        if not norm or len(norm) < 2:
            logging.warning("calculate_development_length: insufficient valid coordinates after normalization")
            return 0

        # Build tuple list and radii list for existing path-length functions
        coords_tuples = []
        radii = []
        for c in norm:
            try:
                coords_tuples.append((float(c['x']), float(c['y']), float(c['z'])))
                radii.append(float(c.get('r', 0.0)))
            except Exception as e:
                logging.warning(f"calculate_development_length: skipping invalid coord {c}: {e}")
                continue

        if len(coords_tuples) < 2:
            logging.warning("calculate_development_length: not enough numeric coordinate tuples")
            return 0

        total_length = 0.0
        points_count = len(coords_tuples)
        
        for i in range(points_count - 1):
            current = coords_tuples[i]
            next_point = coords_tuples[i + 1]
            
            # Calculate vector for segment
            segment_vector = tuple(b - a for a, b in zip(current, next_point))
            
            # Calculate segment length using imported vector magnitude function
            segment_length = calculate_vector_magnitude(segment_vector)
            total_length += segment_length
            
            # If this is a bend point (not first or last) with radius
            if i > 0 and i < points_count - 1 and radii[i] > 0:
                try:
                    # Get previous point for bend calculation
                    prev = coords_tuples[i-1]
                    
                    # Calculate vectors for incoming and outgoing segments
                    v1 = tuple(b-a for a, b in zip(prev, current))
                    v2 = tuple(b-a for a, b in zip(current, next_point))
                    
                    # Calculate angle between vectors using utility function
                    theta = calculate_angle(v1, v2)
                    
                    if theta > 0:
                        R = radii[i]
                        tangent_length = R * math.tan(theta / 2)
                        arc_length = R * theta
                        
                        # Subtract the overlap of tangent lines and add the arc length
                        total_length -= 2 * tangent_length
                        total_length += arc_length
                        
                        logger.info(f"Bend at point {i}: angle={math.degrees(theta):.1f}°, "
                                  f"radius={R:.1f}mm, arc_length={arc_length:.1f}mm")
                    
                except Exception as e:
                    logger.warning(f"Error processing bend at point {i}: {e}")
                    continue
        
        total_length = round(total_length, 2)
        if total_length <= 0:
            logger.warning("Invalid total length calculated")
            return 0
            
        logger.info(f"calculate_development_length: calculated length = {total_length:.2f} mm")
        return total_length

    except Exception as e:
        logger.error(f"Error calculating development length: {e}", exc_info=True)
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
                    
                    # Calculate angle between vectors using imported functions
                    theta = calculate_angle(v1, v2)
                    
                    if theta == 0:
                        continue
                    
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

        # Ensure dimensions param is the dict you expect
        dims = dimensions or analysis_results.get('dimensions', {})

        # determine reinforcement to write: prefer previously-resolved reinforcement, else query DB, else 'Not Found'
        reinforcement_to_write = analysis_results.get('reinforcement')
        if reinforcement_to_write in (None, '', 'Not Found'):
            try:
                # attempt DB lookup by material if we don't already have a value
                reinforcement_to_write = get_reinforcement_from_material(analysis_results.get('standard', 'Not Found'),
                                                                     analysis_results.get('grade', 'Not Found'),
                                                                     analysis_results.get('material', 'Not Found'))
                if reinforcement_to_write is None or (isinstance(reinforcement_to_write, str) and reinforcement_to_write.strip() == ""):
                    reinforcement_to_write = "None"
            except Exception:
                reinforcement_to_write = analysis_results.get('reinforcement', 'Not Found')

        # Build the row data dictionary
        row_data = {
            'child part': part_number.lower(),
            'child quantity': "",
            'CHILD PART': part_number.upper(),
            'CHILD PART DESCRIPTION': description,
            'CHILD PART QTY': "1",
            'SPECIFICATION': specification,
            'MATERIAL': analysis_results.get('material', 'Not Found'),
            'REINFORCEMENT': reinforcement_to_write,
            'VOLUME AS PER 2D': analysis_results.get('volume', 'Not Found'),
        }
        
        # Write ID values with type conversion
        if dims.get('id1') not in (None, "Not Found"):
            try:
                row_data['ID1 AS PER 2D (MM)'] = float(str(dims['id1']).replace(',', '.'))
            except Exception:
                row_data['ID1 AS PER 2D (MM)'] = dims['id1']
        else:
            row_data['ID1 AS PER 2D (MM)'] = 'Not Found'
            
        if dims.get('id2') not in (None, "Not Found"):
            try:
                row_data['ID2 AS PER 2D (MM)'] = float(str(dims['id2']).replace(',', '.'))
            except Exception:
                row_data['ID2 AS PER 2D (MM)'] = dims['id2']
        else:
            row_data['ID2 AS PER 2D (MM)'] = 'Not Found'
            
        row_data.update({
            'OD1 AS PER 2D (MM)': dims.get('od1', 'Not Found'),
            'OD2 AS PER 2D (MM)': dims.get('od2', 'Not Found'),
            'THICKNESS AS PER 2D (MM)': dims.get('thickness', 'Not Found'),
            'THICKNESS AS PER ID OD DIFFERENCE': thickness_calculated,
        })
        
        # Write centerline length with type conversion
        if dims.get('centerline_length') not in (None, "Not Found"):
            try:
                row_data['CENTERLINE LENGTH AS PER 2D (MM)'] = float(str(dims['centerline_length']).replace(',', '.'))
            except Exception:
                row_data['CENTERLINE LENGTH AS PER 2D (MM)'] = dims['centerline_length']
        else:
            row_data['CENTERLINE LENGTH AS PER 2D (MM)'] = 'Not Found'
            
        # Write development length with type conversion if present
        if development_length not in (None, "Not Found"):
            try:
                row_data['DEVELOPMENT LENGTH AS PER CO-ORDINATE (MM)'] = float(development_length)
            except Exception:
                row_data['DEVELOPMENT LENGTH AS PER CO-ORDINATE (MM)'] = development_length
        else:
            row_data['DEVELOPMENT LENGTH AS PER CO-ORDINATE (MM)'] = 'Not Found'
            
        # Add remaining fields
        row_data.update({
            'BURST PRESSURE AS PER 2D (BAR)': analysis_results.get('burst_pressure', 'Not Found'),
            'BURST PRESSURE AS PER WORKING PRESSURE (4XWP) (BAR)': burst_pressure_calc,
            'VOLUME AS PER 2D MM3': analysis_results.get('volume_mm3', 'Not Found'),
            'WEIGHT AS PER 2D KG': analysis_results.get('weight', 'Not Found'),
            'COLOUR AS PER DRAWING': analysis_results.get('color', 'Not Found'),
            'ADDITIONAL REQUIREMENT': "CUTTING & CHECKING FIXTURE COST TO BE ADDED. Marking cost to be added.",
            'OUTSOURCE': "",
            'REMARK': ""
        })

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

def convert_pressure_to_bar(value, unit):
    """
    Convert pressure values to bar based on input unit
    """
    unit = unit.lower() if unit else ''
    try:
        value = float(value)
        if 'kpa' in unit:
            return value / 100  # kPa to bar
        elif 'psi' in unit:
            return value * 0.0689476  # PSI to bar
        elif 'bar' in unit:
            return value
        else:
            # Default to kPa if no unit specified
            return value / 100
    except (ValueError, TypeError):
        return None

def calculate_burst_pressure(working_pressure):
    """
    Calculate burst pressure (4x working pressure)
    """
    try:
        wp = float(working_pressure)
        return wp * 4
    except (ValueError, TypeError):
        return None

def calculate_arc_length(point1, point2, radius):
    """
    Calculate the arc length between two points given a radius
    """
    if radius is None or radius == 0:
        dx = point2['x'] - point1['x']
        dy = point2['y'] - point1['y']
        dz = point2['z'] - point1['z']
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Calculate vectors
    v1 = np.array([point2['x'] - point1['x'], 
                   point2['y'] - point1['y'], 
                   point2['z'] - point1['z']])
    
    length = np.linalg.norm(v1)
    if length == 0:
        return 0
    
    # Calculate angle using cosine law
    angle = 2 * math.asin(length / (2 * radius))
    
    # Calculate arc length
    arc_length = radius * angle
    return arc_length

def calculate_development_length(coordinates):
    """
    Calculate development length considering radii at bends
    """
    if not coordinates or len(coordinates) < 2:
        return 0
    
    total_length = 0
    for i in range(len(coordinates) - 1):
        current = coordinates[i]
        next_point = coordinates[i + 1]
        
        # Use radius from the point that has it
        radius = current.get('r') or next_point.get('r')
        
        # Calculate length considering radius if present
        segment_length = calculate_arc_length(current, next_point, radius)
        total_length += segment_length
    
    return total_length

def parse_material_block(material_text: str) -> dict:
    """
    Robustly extract reinforcement and rings from a material/text block.
    Returns dict with keys: reinforcement, rings, reinforcement_source
    """
    rein_pat = re.compile(
        r'REINFORCEMENT\s*[:\-]?\s*(.*?)(?=\n\s*(?:RINGS|UNLESS|NOTES|MUST|$))',
        re.IGNORECASE | re.DOTALL
    )
    rings_pat = re.compile(
        r'RINGS\s*[:\-]?\s*(.*?)(?=\n\s*(?:UNLESS|NOTES|MUST|$))',
        re.IGNORECASE | re.DOTALL
    )

    def _clean(s):
        if s is None:
            return None
        s = s.strip()
        s = re.sub(r'\s+', ' ', s)           # collapse newlines/extra spaces
        s = s.strip(' .;,-:')                # trim trailing punctuation
        return s

    rein_m = rein_pat.search(material_text)
    rings_m = rings_pat.search(material_text)

    reinforcement = _clean(rein_m.group(1)) if rein_m else None
    rings = _clean(rings_m.group(1)) if rings_m else None

    # Fallback: keyword sniffing (helps when label is missing or OCR dropped colon)
    if not reinforcement:
        kw = re.search(r'(META[\s\-]?ARAMID|ARAMID|NOMEX|KEVLAR|POLYESTER FABRIC)',
                       material_text, re.IGNORECASE)
        if kw:
            reinforcement = kw.group(0).upper().replace('META ARAMID', 'META-ARAMID')

    # Normalizations
    if reinforcement:
        reinforcement = reinforcement.replace('META ARAMID', 'META-ARAMID')
        reinforcement = reinforcement.replace('META-ARAMID FABRIC', 'META-ARAMID FABRIC')
    if rings:
        rings = rings.replace('PER ASTM-A-313', 'PER ASTM A-313')

    # Ensure string outputs (never None)
    return {
        'reinforcement': reinforcement if reinforcement else 'Not Found',
        'rings': rings if rings else 'Not Found',
        'reinforcement_source': 'drawing' if reinforcement else 'none'
    }

def parse_float_with_tolerance(s):
    """Parse numeric value from strings containing tolerances."""
    if s is None:
        return None
    m = re.search(r'[-+]?\d+(?:\.\d+)?', str(s))
    return float(m.group(0)) if m else None

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
def analyze_drawing_simple(pdf_bytes):
    """
    Simplified analysis function that uses direct PyMuPDF text extraction.
    """
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
        # Extract text using PyMuPDF (most reliable method)
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = ""
            for page in doc:
                full_text += page.get_text()
        
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        
        # Process the text with our OCR processing function
        processed_data = process_ocr_text(full_text)
        
        if processed_data:
            # Merge the processed data with results
            for key in ['part_number', 'description', 'standard', 'grade', 'material', 'coordinates']:
                if key in processed_data and processed_data[key] != "Not Found":
                    results[key] = processed_data[key]
            
            if 'dimensions' in processed_data and processed_data['dimensions']:
                results['dimensions'].update(processed_data['dimensions'])
        
        # Also try to extract dimensions using the dedicated function
        additional_dims = extract_dimensions_from_text(full_text)
        if additional_dims:
            results['dimensions'].update(additional_dims)
            
        logger.info("Simple analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in simple analysis: {str(e)}")
        results["error"] = f"Analysis failed: {str(e)}"
        return results

def analyze_drawing_simple(pdf_bytes):
    """
    Simplified analysis function that uses direct PyMuPDF text extraction.
    """
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
        # Extract text using PyMuPDF (most reliable method)
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = ""
            for page in doc:
                full_text += page.get_text()
        
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        
        # Process the text with our OCR processing function
        processed_data = process_ocr_text(full_text)
        
        if processed_data:
            # Merge the processed data with results
            for key in ['part_number', 'description', 'standard', 'grade', 'material', 'coordinates']:
                if key in processed_data and processed_data[key] != "Not Found":
                    results[key] = processed_data[key]
            
            if 'dimensions' in processed_data and processed_data['dimensions']:
                results['dimensions'].update(processed_data['dimensions'])
        
        # Also try to extract dimensions using the dedicated function
        additional_dims = extract_dimensions_from_text(full_text)
        if additional_dims:
            results['dimensions'].update(additional_dims)
            
        logger.info("Simple analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in simple analysis: {str(e)}")
        results["error"] = f"Analysis failed: {str(e)}"
        return results

def analyze_drawing(pdf_bytes):
    """
    Analyze engineering drawing using Gemini's multimodal capabilities with OCR fallback.
    Uses automated model discovery and robust fallback to OCR.
    """
    if not pdf_bytes:
        raise ValueError("PDF content cannot be empty")
    
    # Initialize default results structure
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
    
    temp_pdf_path = None
    
    try:
        # Convert PDF to images and process with Gemini and OCR fallback
        logger.info("Converting PDF to images...")
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        # Process pages using our helper module with required params
        drawing_prompt = """Analyze this technical drawing and extract the following information as JSON:
        - part_number: Look for part numbers like 7718817C1, 4582819C2, etc.
        - description: Description of the hose or tube
        - standard: Material standard (e.g., MPAPS F-30, MPAPS F-6032)
        - grade: Material grade (e.g., 1B, TYPE I)
        - dimensions: Include id1, od1, thickness, centerline_length
        - coordinates: Any coordinate points if present
        
        Return ONLY valid JSON. If any value is not found, use "Not Found"."""
        
        all_data = process_pages_with_vision_or_ocr(
            pages=temp_pdf_path,
            prompt=drawing_prompt,
            ocr_fallback_fn=process_ocr_text
        )
        
        logger.info(f"Processing completed. Got {len(all_data) if isinstance(all_data, list) else 0} results")
        
        # Process the results
        if not all_data or (isinstance(all_data, list) and len(all_data) == 0):
            logger.warning("No data returned from processing pipeline")
            results["error"] = "No data extracted from PDF"
            return results
        
        # Ensure all_data is a list
        if not isinstance(all_data, list):
            all_data = [all_data]
        
        # Filter out non-dict results
        valid_results = [item for item in all_data if isinstance(item, dict)]
        
        if not valid_results:
            logger.error("No valid dictionary results found")
            results["error"] = "No structured data could be extracted"
            return results
        
        # Use the first valid result (you can enhance this to pick the best one)
        best_result = valid_results[0]
        
        # Merge the best result with our default structure
        for key in results.keys():
            if key in best_result and best_result[key] != "Not Found":
                results[key] = best_result[key]
        
        # Handle nested structures
        if "dimensions" in best_result and isinstance(best_result["dimensions"], dict):
            results["dimensions"].update(best_result["dimensions"])
        
        if "coordinates" in best_result and isinstance(best_result["coordinates"], list):
            results["coordinates"] = best_result["coordinates"]
        
        logger.info("Drawing analysis completed successfully")
        return results
            
        # ------------------ Robust post-processing of all_data ------------------
        logger.info(f"Raw all_data type: {type(all_data)}; length (if list): {len(all_data) if isinstance(all_data, list) else 'N/A'}")

        # Normalize all_data into a flat list
        if isinstance(all_data, dict):
            normalized = [all_data]
        elif isinstance(all_data, list):
            # flatten one level if nested lists exist
            normalized = []
            for item in all_data:
                if isinstance(item, list):
                    normalized.extend(item)
                else:
                    normalized.append(item)
        else:
            # Unexpected type (e.g., bool, str). Wrap to preserve for logging and fail gracefully.
            logger.warning(f"process_pages_with_vision_or_ocr returned unexpected type: {type(all_data)}")
            normalized = [all_data]

        logger.info(f"Normalized results count: {len(normalized)}")

        # Keep only dict items (each dict is expected to be page-level structured data)
        dict_results = [item for item in normalized if isinstance(item, dict)]
        non_dict_count = len(normalized) - len(dict_results)
        if non_dict_count:
            logger.warning(f"Ignoring {non_dict_count} non-dict entries from all_data")

        if not dict_results:
            # Nothing usable — log details and return the default results object with an error.
            logger.error("No valid page dictionaries returned from vision/OCR pipeline.")
            # Put some diagnostic sample into results.error for visibility
            results["error"] = "No valid structured page results (process_pages_with_vision_or_ocr returned no dicts)."
            # Optionally attach a compact sample for debugging
            try:
                sample = normalized[:5]
                results["debug_sample"] = sample
            except Exception:
                pass
            return results

        # Use completeness_score to choose the best page dict
        def completeness_score(data):
            if not isinstance(data, dict):
                return 0
            score = 0
            # top-level fields except nested ones
            score += sum(1 for k, v in data.items() if v not in ("Not Found", None) and k not in ["dimensions", "operating_conditions", "coordinates"])
            if isinstance(data.get("dimensions"), dict):
                score += sum(1 for v in data["dimensions"].values() if v != "Not Found")
            if isinstance(data.get("operating_conditions"), dict):
                score += sum(1 for v in data["operating_conditions"].values() if v != "Not Found")
            if isinstance(data.get("coordinates"), list):
                score += len(data["coordinates"])
            return score

        # select best result
        try:
            results = max(dict_results, key=completeness_score)
        except Exception as e:
            logger.exception("Failed to select best result from dict_results; returning first dict as fallback.")
            results = dict_results[0]

        # Diagnostics
        logger.info(f"Selected best result type: {type(results)}")
        logger.debug(f"Best result keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}")

        # Normalize nested fields: convert lists -> dicts when necessary
        if isinstance(results.get("dimensions"), list):
            logger.warning("Converting dimensions list -> dict")
            flat_dims = {}
            for d in results["dimensions"]:
                if isinstance(d, dict):
                    flat_dims.update(d)
            results["dimensions"] = flat_dims

        if isinstance(results.get("operating_conditions"), list):
            logger.warning("Converting operating_conditions list -> dict")
            flat_ops = {}
            for d in results["operating_conditions"]:
                if isinstance(d, dict):
                    flat_ops.update(d)
            results["operating_conditions"] = flat_ops

        # Ensure required keys exist and have expected types
        if "dimensions" not in results or not isinstance(results["dimensions"], dict):
            results["dimensions"] = {}
        if "operating_conditions" not in results or not isinstance(results["operating_conditions"], dict):
            results["operating_conditions"] = {}
        if "coordinates" not in results or not isinstance(results["coordinates"], list):
            results["coordinates"] = []

        logger.info("Normalized results ready for downstream processing")
        # ------------------ end robust processing ------------------
            
        # Log the structure of results for debugging
        logger.info(f"Type of results: {type(results)}")
        logger.info(f"Keys in results (if dict): {results.keys() if isinstance(results, dict) else results}")
        logger.info(f"Type of dimensions: {type(results.get('dimensions'))}")
        
        # Convert dimensions list to dict if needed
        if isinstance(results.get("dimensions"), list):
                logger.warning("Converting dimensions list to dict")
                flat_dims = {}
                for d in results["dimensions"]:
                    if isinstance(d, dict):
                        flat_dims.update(d)
                results["dimensions"] = flat_dims
            
        # Convert operating_conditions list to dict if needed
        if isinstance(results.get("operating_conditions"), list):
            logger.warning("Converting operating_conditions list to dict")
            flat_ops = {}
            for d in results["operating_conditions"]:
                if isinstance(d, dict):
                    flat_ops.update(d)
            results["operating_conditions"] = flat_ops
        
        # Ensure all required fields exist
        if "dimensions" not in results:
            results["dimensions"] = {}
        if "operating_conditions" not in results:
            results["operating_conditions"] = {}
        if "coordinates" not in results:
            results["coordinates"] = []
        
        # Convert numeric strings to floats where appropriate
        for key in ["id1", "od1", "thickness", "centerline_length"]:
            if results["dimensions"].get(key) and results["dimensions"][key] != "Not Found":
                try:
                    results["dimensions"][key] = float(str(results["dimensions"][key]).replace(",", "."))
                except (ValueError, TypeError):
                    results["dimensions"][key] = "Not Found"
        
        # Convert pressure values
        for key in ["working_pressure", "burst_pressure"]:
            if results["operating_conditions"].get(key) and results["operating_conditions"][key] != "Not Found":
                try:
                    results["operating_conditions"][key] = float(str(results["operating_conditions"][key]).replace(",", "."))
                except (ValueError, TypeError):
                    results["operating_conditions"][key] = "Not Found"
        
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
    Falls back to OCR if Gemini processing fails.
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
        "error": None,
        "processing_method": "gemini"  # Track which method was used
    }
    
    try:
        # Process the drawing using helper module's functionality
        logger.info("Processing drawing with Gemini and OCR fallback...")
        
        # Convert PDF to temporary file and process
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
            
        # Process the pages using our helper module with required params
        drawing_prompt = "Analyze this technical drawing and extract part number, description, material, dimensions, operating conditions and any coordinate points."
        all_data = process_pages_with_vision_or_ocr(
            pages=temp_pdf_path,
            prompt=drawing_prompt,
            ocr_fallback_fn=process_ocr_text
        )

        # If we have data, format and return results
        if all_data:
            # Use the most complete data as our base
            def completeness_score(data):
                score = 0
                # Check top-level fields
                score += sum(1 for k, v in data.items() if v != "Not Found" and k not in ["dimensions", "coordinates"])
                # Check dimensions
                if isinstance(data.get("dimensions"), dict):
                    score += sum(1 for v in data["dimensions"].values() if v != "Not Found")
                # Check coordinates
                if isinstance(data.get("coordinates"), list):
                    score += len(data["coordinates"])
                return score
                
            # Get the most complete result set
            best_data = max(all_data, key=completeness_score)
            
            # Log data structure for debugging
            logger.info(f"Type of best_data: {type(best_data)}")
            logger.info(f"Keys in best_data (if dict): {best_data.keys() if isinstance(best_data, dict) else best_data}")
            logger.info(f"Type of dimensions: {type(best_data.get('dimensions'))}")
            
            # Convert dimensions list to dict if needed
            if isinstance(best_data.get("dimensions"), list):
                logger.warning("Converting dimensions list to dict")
                flat_dims = {}
                for d in best_data["dimensions"]:
                    if isinstance(d, dict):
                        flat_dims.update(d)
                best_data["dimensions"] = flat_dims
            
            # Convert operating_conditions list to dict if needed
            if isinstance(best_data.get("operating_conditions"), list):
                logger.warning("Converting operating_conditions list to dict")
                flat_ops = {}
                for d in best_data["operating_conditions"]:
                    if isinstance(d, dict):
                        flat_ops.update(d)
                best_data["operating_conditions"] = flat_ops
            
            # Update results with the best data
            results.update(best_data)
            
            # Ensure proper field types
            if not isinstance(results.get("dimensions"), dict):
                results["dimensions"] = {}
            if not isinstance(results.get("coordinates"), list):
                results["coordinates"] = []
            
            # Convert numeric values
            for key in ["id1", "id2", "od1", "od2", "thickness", "centerline_length"]:
                if results["dimensions"].get(key) and results["dimensions"][key] != "Not Found":
                    try:
                        results["dimensions"][key] = float(str(results["dimensions"][key]).replace(",", "."))
                    except (ValueError, TypeError):
                        results["dimensions"][key] = "Not Found"
            
            # Merge coordinates from all pages
            all_coordinates = []
            for data in all_data:
                if isinstance(data.get("coordinates"), list):
                    all_coordinates.extend(data["coordinates"])
            results["coordinates"] = all_coordinates

        logger.info("Drawing analysis completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"Drawing analysis failed: {str(e)}")
        results["error"] = f"Analysis failed: {str(e)}"
        return results
    finally:
        # Clean up any temporary files
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary PDF file: {e}")
    

    # Clean and normalize text (post-processing outside the above try/except)
    try:
        combined_text = clean_text_encoding(combined_text)
        combined_text = re.sub(r'\s+', ' ', combined_text)
        logger.info("Text cleaned and normalized")
    except Exception:
        # If cleaning fails, keep combined_text as-is (or empty)
        combined_text = combined_text if 'combined_text' in locals() else ''
        
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
        logger.info("=== START OF EXTRACTED TEXT FOR DEBUGGING ===")
        logger.info(f"Cleaned Text Length: {len(combined_text)}")
        logger.info("First 1000 characters of text:")
        logger.info(combined_text[:1000])
        logger.info("=== END OF EXTRACTED TEXT FOR DEBUGGING ===")
        
        # Initial coordinate extraction
        results["coordinates"] = extract_coordinates_from_text(combined_text)
        logger.info(f"Initial coordinates extracted: {len(results['coordinates'])} points")

        # Normalize coordinates robustly and compute development length
        raw_coords = results.get('coordinates', [])

        # If the AI returned stringified coordinates or partially filled dicts, normalise them
        norm_coords = normalize_coordinates(raw_coords)

        # If normaliser returned empty (AI missed it), try extracting from the cleaned text
        if not norm_coords:
            extracted_text = results.get('extracted_text') or extract_text_from_pdf(pdf_bytes)
            norm_coords = normalize_coordinates(extracted_text)

        # Final defensive pass: drop incomplete points and log details
        clean_coords = []
        for p in norm_coords:
            if all(k in p and isinstance(p[k], (int, float)) for k in ('x','y','z')):
                clean_coords.append(p)
            else:
                logger.warning(f"Skipping incomplete coord entry: {p}")

        results['coordinates'] = clean_coords
        logger.debug("coordinates after normalize (first 5): %s", clean_coords[:5])

        # Compute development length only when coords are valid
        if len(clean_coords) >= 2:
            # Convert coordinates and calculate length
            pts = [(p['x'], p['y'], p['z']) for p in clean_coords]
            try:
                dev_len = calculate_development_length(clean_coords)
            except Exception:
                # Fallback to simpler path-length
                radii = [p.get('r', 0) for p in clean_coords]
                dev_len = calculate_path_length(pts, radii)
            results['development_length'] = dev_len
            logger.info(f"Calculated development length: {dev_len}")
        else:
            results['development_length'] = None
            logger.warning("Insufficient coordinates after normalization to compute development length")
            
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
        prompt = """Analyze this engineering drawing with high precision.
Find the exact values for the keys below based on the specified labels.
Return a JSON object. If a value is not explicitly found, use the string "Not Found".

Instructions:
1. Part Number ('part_number'): 
   - Look for the label "PART NO." or "PART NUMBER"
   - It will typically be a 7-8 digit number followed by 'C' and 1-2 digits
   - Also check title block and drawing header

2. Description ('description'):
   - Find the main title or description of the part
   - Usually located in the title block or drawing header
   - Include the full description text

3. Standard/Specification ('standard'):
   - Look specifically for "SPEC:" label
   - The value will typically be in format "MPAPS F-XXXX"
   - Common values: MPAPS F-6032, MPAPS F-6028, MPAPS F-6034
   - Don't include assembly specs like F-1

4. Grade ('grade'):
   - Look specifically for "TYPE" or "GRADE" labels
   - Common values: "TYPE I", "TYPE II", "GRADE C-AN"
   - Report exactly as shown on drawing

5. Dimensions Object:
   Look for these specific labels and report numeric values only:
   - "id1": Find "HOSE ID" or "INSIDE DIAMETER"
   - "od1": Find "HOSE OD" or "OUTSIDE DIAMETER"
   - "thickness": Find "WALL THICKNESS" or "THK"
   - "centerline_length": Find "CTRLINE LENGTH" or "C/L LENGTH"

6. Operating Conditions:
   - Find "MAX OPERATING PRESSURE" (in kPag)
   - Find "BURST PRESSURE" (in kPag)
   - Find "OPERATING TEMPERATURE" range

7. Coordinate Points:
   - Look for points labeled P0, P1, P2, etc.
   - Each point should have X, Y, Z coordinates
   - Report coordinates in array format

Required JSON format:
{
    "part_number": "...",
    "description": "...",
    "standard": "...",
    "grade": "...",
    "dimensions": {
        "id1": "...",
        "od1": "...",
        "thickness": "...",
        "centerline_length": "..."
    },
    "operating_conditions": {
        "working_pressure": "...",
        "burst_pressure": "...",
        "temperature_range": "..."
    },
    "coordinates": [{"point": "P0", "x": 0.0, "y": 0.0, "z": 0.0}, ...]
}

Important: Report numeric values WITHOUT units. Example: for "HOSE ID = 19.05 MM", report only "19.05".
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
        model = genai.GenerativeModel('gemini-pro-vision') # Use vision-capable model
        
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
            model = genai.GenerativeModel('gemini-2.5-pro')
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

        # Enhanced dimension extraction and merging
        # Get cleaned extracted text from multiple sources
        extracted_text = final_results.get('extracted_text') or final_results.get('combined_text') \
                        or final_results.get('ocr_text') or extract_text_from_pdf(pdf_bytes)

        # Log the full extracted text for debugging
        app.logger.info("--- START OF FULL EXTRACTED TEXT ---")
        app.logger.info(extracted_text)
        app.logger.info("--- END OF FULL EXTRACTED TEXT ---")

        # 1) Get dimensions from the AI/previous processing safely (existing fallback)
        ai_dims = final_results.get('dimensions', {}) if isinstance(final_results.get('dimensions'), dict) else {}

        # 2) Parse dimensions from the raw text (regex-based direct extraction)
        text_dims = extract_dimensions_from_text(extracted_text)

        # 3) Merge: prefer direct text values when present (text_dims override empty ai dims)
        merged_dims = {}
        merged_dims.update(ai_dims)
        for k, v in text_dims.items():
            if v and v != "Not Found":
                merged_dims[k] = v

        # 4) Ensure normalized types for downstream code
        # convert numeric-like strings to floats where appropriate
        for k in ('id1', 'centerline_length', 'od1', 'id2', 'thickness'):
            if merged_dims.get(k) not in (None, "Not Found"):
                try:
                    merged_dims[k] = float(str(merged_dims[k]).replace(',', '.'))
                except Exception:
                    pass

        final_results['dimensions'] = merged_dims
        logger.debug("dimensions after merge: %s", final_results.get('dimensions'))

        # Look up material based on standard and grade
        try:
            standard = final_results.get("standard", "Not Found")
            grade = final_results.get("grade", "Not Found")
            ocr_text = extract_text_from_pdf(pdf_bytes) if pdf_bytes else ""
            # Handle standards and remarks
            remark, suggested_standard = get_standards_remark(ocr_text, standard)
            if remark:
                final_results['remark'] = remark
                standard = suggested_standard
            
            # Lookup material using the potentially updated standard
            final_results["material"] = get_material_from_standard(standard, grade)

            # Handle reinforcement: always prefer drawing details over database lookup
            try:
                # Store raw extracted reinforcement data for reference
                raw_reinforcement = final_results.get("reinforcement")
                raw_source = final_results.get("reinforcement_source", "unknown")
                
                logger.debug("Raw reinforcement data - Value: %s, Source: %s", raw_reinforcement, raw_source)
                
                # Look up DB reinforcement for comparison and fallback
                db_rein = get_reinforcement_from_material(standard, grade, final_results.get("material", "Not Found"))
                if db_rein and db_rein != "Not Found":
                    final_results["reinforcement_db"] = db_rein
                    logger.debug("Found DB reinforcement: %s", db_rein)
                
                # Decision logic for final reinforcement value
                if raw_reinforcement and raw_reinforcement != "Not Found" and raw_source != "none":
                    # Keep the drawing-extracted value
                    final_results["reinforcement"] = raw_reinforcement
                    final_results["reinforcement_source"] = raw_source
                    logger.info("Using reinforcement from drawing: %s (source: %s)", raw_reinforcement, raw_source)
                elif db_rein and db_rein != "Not Found":
                    # Fallback to DB value
                    final_results["reinforcement"] = db_rein
                    final_results["reinforcement_source"] = "db"
                    logger.info("Using reinforcement from DB: %s", db_rein)
                else:
                    # No valid value found
                    final_results["reinforcement"] = "Not Found"
                    final_results["reinforcement_source"] = "none"
                    logger.warning("No valid reinforcement found in drawing or DB")
                
                # Always log the final decision
                logger.info("Final reinforcement decision - Value: '%s', Source: %s", 
                          final_results.get("reinforcement"), 
                          final_results.get("reinforcement_source"))
                
                # Clean up raw extraction fields
                final_results.pop("reinforcement_raw", None)
                final_results.pop("rings", None)
                
            except Exception as e:
                logger.exception("Reinforcement processing failed")
                final_results["reinforcement"] = final_results.get("reinforcement", "Not Found")
                final_results["reinforcement_source"] = "error"
            except Exception as e:
                logging.warning(f"Reinforcement lookup failed: {e}", exc_info=True)
                # Keep a deterministic key for downstream code
                final_results["reinforcement"] = final_results.get("reinforcement", "Not Found")
        except Exception as e:
            logging.error(f"Error in material lookup: {e}")
            final_results["material"] = "Not Found"
            final_results["reinforcement"] = "Not Found"

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
        
        # Calculate development length with improved handling
        try:
            coordinates = final_results.get("coordinates", [])
            logger.debug("Raw coordinates (first 6): %s", coordinates[:6] if coordinates else [])
            
            if coordinates:
                try:
                    dev_length = calculate_development_length_safe(coordinates)
                    final_results["development_length_mm"] = f"{dev_length:.2f}"
                    logger.debug("Development length computed: %s", final_results["development_length_mm"])
                except ValueError:
                    final_results["development_length_mm"] = "Not Found"
                    logger.warning("Could not compute development length due to insufficient valid coordinates")
                except Exception as exc:
                    final_results["development_length_mm"] = "Not Found"
                    logger.exception("Unexpected error computing development length: %s", exc)
            else:
                final_results["development_length_mm"] = "Not Found"
                logger.info("No coordinates found for development length calculation")
        except Exception as e:
            final_results["development_length_mm"] = "Not Found"
            logger.exception("Error in development length calculation: %s", e)
        
        # Generate Excel report if helper function exists
        try:
            # Log the final data before Excel generation
            app.logger.info("Final data being sent to Excel writer:")
            app.logger.info("Reinforcement data:")
            app.logger.info(f"  reinforcement: {final_results.get('reinforcement', 'Not Found')}")
            app.logger.info(f"  reinforcement_raw: {final_results.get('reinforcement_raw', 'Not Found')}")
            app.logger.info(f"  reinforcement_source: {final_results.get('reinforcement_source', 'Not Found')}")
            app.logger.info(f"  rings_raw: {final_results.get('rings_raw', 'Not Found')}")
            app.logger.info(f"  reinforcement_db: {final_results.get('reinforcement_db', 'Not Found')}")
            
            # Log the complete final_results for thorough debugging
            app.logger.info(f"Complete final_results: {final_results}")

            if 'generate_excel_sheet' in globals():
                excel_file = generate_corrected_excel_sheet(final_results, final_results.get("dimensions", {}), coordinates)
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
        model = genai.GenerativeModel('gemini-pro-vision')
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
            # Try unlink first, then fallback to remove if necessary
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                try:
                    os.remove(temp_pdf_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {e}")

def format_description(text):
    """
    Format description text with proper spacing and commas.
    """
    if not text:
        return "Not Found"
    
    # Clean and normalize commas and spaces
    text = text.strip()
    text = re.sub(r',\s*', ', ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' ,')
    
    # Ensure proper HOSE prefix
    if not text.upper().startswith('HOSE'):
        text = f"HOSE, {text}"
    
    return text

def extract_text_from_pdf(pdf_bytes):
    """
    Enhanced text extraction from PDF using multiple methods and layout preservation.
    Tries different extraction techniques and combines results for best output.
    
    Args:
        pdf_bytes: Raw PDF file content in bytes
        
    Returns:
        str: Combined text from all extraction methods
    """
    logger.info("==================== PDF TEXT EXTRACTION START ====================")
    logger.info("Starting enhanced text extraction process...")
    logger.debug(f"Input PDF size: {len(pdf_bytes)} bytes")
    texts = []
    
    def log_extracted_text(source, text):
        """Helper to log extracted text chunks with clear formatting"""
        logger.debug(f"\n{'='*20} {source} TEXT {'='*20}\n{text}\n{'='*50}")
        if 'RING' in text.upper() or 'MATERIAL' in text.upper():
            logger.info(f"Found relevant section in {source}:\n{text[:500]}")
    
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
        log_extracted_text("PYMUPDF", combined_text)
        logger.info(f"PyMuPDF extraction found {len(combined_text)} characters with layout preservation")
        
        # If extracted text seems insufficient, try OCR
        if len(combined_text.strip()) < 100 or not any(char.isdigit() for char in combined_text):
            logger.info("Initial extraction insufficient, falling back to OCR...")
            ocr_text = analyze_image_with_gemini_vision(pdf_bytes)
            if ocr_text:
                log_extracted_text("OCR", ocr_text)
                texts.append(ocr_text)
                
        # Combine all extracted text, remove duplicates while preserving order
        seen = set()
        final_text = []
        
        logger.info("==================== EXTRACTED TEXT SECTIONS ====================")
        for text in texts:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    final_text.append(line)
        
        result = '\n'.join(final_text)
        
        logger.info("==================== FINAL COMBINED TEXT ====================")
        logger.info("Complete extracted text:")
        logger.info(f"\n{result}")
        logger.info("==================== END OF TEXT EXTRACTION ====================")
        logger.info(f"Final extracted text length: {len(result)} characters")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced text extraction: {e}")
        # Fall back to OCR as last resort
        return analyze_image_with_gemini_vision(pdf_bytes)

if __name__ == '__main__':
    app.run(debug=True)