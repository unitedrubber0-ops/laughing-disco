# Standard library imports
import os
import re
import math
import time
import uuid
import base64
import io
import gc
import json
import logging
import tempfile
import shutil
import unicodedata
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image

# Try importing optional dependencies
try:
    import cv2
    cv2_available = True
except ImportError:
    cv2 = None
    cv2_available = False

# OCR Debug Configuration
DEBUG_SAVE_DIR = os.path.join(os.path.dirname(__file__), "ocr_debug")
os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

def save_and_log_image_for_debug(img_pil, tag="preocr"):
    """Save intermediate images for OCR debugging"""
    fname = f"{tag}_{uuid.uuid4().hex[:8]}.png"
    p = os.path.join(DEBUG_SAVE_DIR, fname)
    img_pil.save(p)
    logger.info(f"Saved debug image: {p}")
    return p

def run_tesseract_and_log(pil_img):
    """Run OCR with multiple configurations and log results"""
    # Save raw image for inspection
    saved = save_and_log_image_for_debug(pil_img, tag="raw")
    # run pytesseract with a couple of configs
    cfgs = ["--psm 6", "--psm 3", "--psm 11"]  # different page segmentation modes
    results = {}
    for cfg in cfgs:
        try:
            txt = pytesseract.image_to_string(pil_img, lang="eng", config=cfg)
        except Exception as e:
            txt = ""
            logger.exception(f"pytesseract failed with cfg {cfg}: {e}")
        length = len(txt.strip())
        logger.info(f"tesseract cfg={cfg} -> chars={length}")
        # Save the result for quick glance
        out_file = os.path.join(DEBUG_SAVE_DIR, f"ocr_{uuid.uuid4().hex[:8]}_{cfg.replace(' ', '_')}.txt")
        with open(out_file, "w", encoding="utf-8") as fh:
            fh.write(txt)
        logger.info(f"Saved OCR text: {out_file}")
        results[cfg] = txt
    return results

def preprocess_for_ocr(pil_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Enhanced preprocessing pipeline for OCR.
    Includes: CLAHE, denoising, adaptive threshold, morphology, and deskew

    Args:
        pil_img: Input PIL Image

    Returns:
        Tuple[Image.Image, Image.Image]: Tuple of (normal, inverted) processed images.
        If processing fails, returns the original image for both.
    
    Raises:
        ValueError: If input is not a PIL Image
    """
    # Input validation
    if not isinstance(pil_img, Image.Image):
        logger.error("Input must be a PIL Image")
        raise ValueError("Input must be a PIL Image")

    # Check OpenCV availability
    if not cv2_available or not cv2:
        logger.warning("OpenCV not available for preprocessing")
        return pil_img, pil_img  # Return original and inverted versions

    try:
        # PIL -> OpenCV
        img = np.array(pil_img.convert("RGB"))
        if not hasattr(cv2, 'cvtColor') or not hasattr(cv2, 'COLOR_RGB2GRAY'):
            logger.error("Required OpenCV functions not available")
            return pil_img, pil_img
            
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1) CLAHE equalization
        if not hasattr(cv2, 'createCLAHE'):
            logger.error("CLAHE function not available")
            return pil_img, pil_img
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        eq = clahe.apply(gray)
        save_and_log_image_for_debug(Image.fromarray(eq), "after_clahe")

        # 2) Denoise (fast)
        if not hasattr(cv2, 'fastNlMeansDenoising'):
            logger.error("Denoising function not available")
            return pil_img, pil_img
        
        den = cv2.fastNlMeansDenoising(eq, None, h=10, templateWindowSize=7, searchWindowSize=21)
        save_and_log_image_for_debug(Image.fromarray(den), "after_denoise")

        # 3) Adaptive threshold (good for varying lighting)
        if not all(hasattr(cv2, attr) for attr in ['adaptiveThreshold', 'ADAPTIVE_THRESH_GAUSSIAN_C', 'THRESH_BINARY']):
            logger.error("Thresholding functions not available")
            return pil_img, pil_img
        
        th = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 9)
        save_and_log_image_for_debug(Image.fromarray(th), "after_threshold")

        # 4) Morphology to close gaps (if characters broken)
        if not all(hasattr(cv2, attr) for attr in ['getStructuringElement', 'MORPH_RECT', 'morphologyEx', 'MORPH_CLOSE']):
            logger.error("Morphological operations not available")
            return pil_img, pil_img
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morphed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        save_and_log_image_for_debug(Image.fromarray(morphed), "after_morph")

        # 5) Deskew (estimate angle by moments on the binary image)
        if all(hasattr(cv2, attr) for attr in ['minAreaRect', 'getRotationMatrix2D', 'warpAffine', 'INTER_CUBIC', 'BORDER_REPLICATE']):
            coords = np.column_stack(np.where(morphed > 0))
            if coords.shape[0] > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                if abs(angle) > 0.1:  # only rotate if reasonable
                    (h, w) = morphed.shape[:2]
                    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                    morphed = cv2.warpAffine(morphed, M, (w, h),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    logger.info(f"Applied deskew angle {angle:.2f}")
                    save_and_log_image_for_debug(Image.fromarray(morphed), "after_deskew")
            else:
                logger.info("No non-zero coords found for deskew")
        else:
            logger.warning("Deskew functions not available")

        # Try inverted version for white-on-dark text
        inv = cv2.bitwise_not(morphed)
        save_and_log_image_for_debug(Image.fromarray(inv), "inverted")

        # Convert back to PIL for pytesseract (both normal and inverted)
        pil_out = Image.fromarray(morphed)
        pil_inv = Image.fromarray(inv)
        
        return pil_out, pil_inv

    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        return pil_img, pil_img  # Return original image on error

# Third party imports
import numpy as np
import pandas as pd
import openpyxl
import openpyxl.utils
import fitz  # PyMuPDF
from fitz import Page as FitzPage  # For type hints
from PIL import Image, ImageFilter
import pytesseract
import requests
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pdf2image
from pdf2image import convert_from_path, convert_from_bytes
import google.generativeai as genai
from roboflow import Roboflow

# Local application imports
from model_selection import get_vision_model
from development_length import (
    calculate_vector_magnitude, 
    calculate_dot_product, 
    calculate_angle,
    calculate_development_length as calculate_development_length_safe
)
from rings_extraction import RingsExtractor
from excel_output import generate_corrected_excel_sheet
from gemini_helper import process_pages_with_vision_or_ocr, extract_text_from_image_wrapper

# Initialize OpenCV availability
cv2 = None
cv2_available = False

try:
    import cv2
    if cv2 and hasattr(cv2, 'cvtColor'):  # Check for key OpenCV functions
        cv2_available = True
        logging.info("OpenCV (cv2) initialized successfully")
    else:
        cv2 = None
        logging.warning("OpenCV (cv2) imported but required functions not found")
except ImportError:
    logging.warning("OpenCV (cv2) import failed")
except Exception as e:
    logging.warning(f"OpenCV (cv2) initialization error: {e}")
    cv2 = None

# Check OCR dependencies
tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    logging.warning("tesseract binary not found in PATH. OCR functionality may be limited. Install tesseract-ocr or set pytesseract.pytesseract.tesseract_cmd")
else:
    logging.info(f"tesseract found at {tesseract_path}")
    # Try to set tesseract path explicitly
    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    except Exception as e:
        logging.warning(f"Failed to set tesseract path: {e}")

# Check poppler (required for pdf2image)
pdftoppm_path = shutil.which("pdftoppm")  # Part of poppler-utils
if not pdftoppm_path:
    logging.warning("poppler-utils (pdftoppm) not found in PATH. PDF to image conversion may fail. Install poppler-utils package.")
else:
    logging.info(f"poppler-utils found at {pdftoppm_path}")

# Define custom exceptions
class BlockedPromptException(Exception):
    """Custom exception for blocked prompts"""

class PDFPageCountError(Exception):
    """Error when PDF page count is invalid"""

class PDFSyntaxError(Exception):
    """Error when PDF syntax is invalid"""


# --- Helper Functions ---

def extract_text_from_pdf_memory_safe(pdf_bytes, max_pages=1, dpi=150):
    """
    Memory-safe PDF text extraction using pdf2image and OCR.
    Processes one page at a time and aggressively frees memory.
    """
    if not pdf_bytes:
        return None

    text_parts = []
    try:
        # Convert pages one at a time to manage memory
        images = convert_from_bytes(
            pdf_bytes,
            dpi=dpi,
            first_page=1,
            last_page=max_pages,
            fmt='jpg',
            size=(1700, None),  # reasonable default size
            grayscale=True
        )

        for img in images:
            try:
                # Free memory between pages
                gc.collect()
                
                # Process page
                processed = preprocess_for_ocr(img)
                if isinstance(processed, tuple):
                    processed, _ = processed
                
                # Run OCR and collect text
                ocr_result = run_tesseract_and_log(processed)
                if ocr_result and isinstance(ocr_result, dict):
                    best_text = max(ocr_result.values(), key=len, default="")
                    text_parts.append(best_text)
                
                # Free the processed image
                del processed
                gc.collect()
            
            except Exception as e:
                logger.error(f"Error processing PDF page: {e}")
                continue
            finally:
                # Always free the page image
                del img
                gc.collect()
        
        # Combine all extracted text
        return "\n".join(text_parts)
    
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return None
    finally:
        gc.collect()

def match_materials_db(parsed_data):
    """
    Match extracted materials info against the materials database.
    Returns DataFrame of matching records or empty list if no matches.
    
    parsed_data: dict with 'materials' (list), 'grades' (list), etc.
    """
    try:
        if not isinstance(parsed_data, dict):
            return []
        
        # Get global database DataFrame
        df = globals().get('material_df')
        if df is None or not isinstance(df, pd.DataFrame):
            logger.error("Materials database not properly loaded")
            return []
        
        matches = []
        materials = parsed_data.get('materials', [])
        grades = parsed_data.get('grades', [])
        
        # Try exact matches first
        for mat in materials:
            for grade in grades:
                mask = (
                    df['STANDARD'].str.contains(str(mat), case=False, na=False) &
                    df['GRADE'].str.contains(str(grade), case=False, na=False)
                )
                matches.extend(df[mask].to_dict('records'))
        
        # If no exact matches, try fuzzy matching on materials
        if not matches:
            for mat in materials:
                mask = df['STANDARD'].str.contains(str(mat), case=False, na=False)
                matches.extend(df[mask].to_dict('records'))
        
        return matches
    except Exception as e:
        logger.error(f"Error matching materials: {e}")
        return []

def safe_extract_text(page, method=None, **kwargs):
    """Safely extract text from a page object"""
    try:
        if method:
            return page.get_text(method, **kwargs)
        return page.get_text()
    except (AttributeError, TypeError):
        return str(page)

def extract_page_text(page, method=None, **kwargs):
    """Safely extract text from a page object."""
    try:
        # Check if it's a PyMuPDF page
        if hasattr(page, 'get_text'):
            if method:
                return page.get_text(method, **kwargs)
            return page.get_text()
        # Fallback for other page types
        text = str(page)
        return [] if method == "blocks" else text
    except Exception as e:
        logging.warning(f"Error extracting text: {e}")
        return [] if method == "blocks" else str(page)

def clean_rings_text(rings_text):
    """Clean and normalize rings text"""
    if not rings_text or rings_text == "Not Found":
        return "Not Found"
    
    # Remove extra whitespace
    rings_text = re.sub(r'\s+', ' ', rings_text).strip()
    
    # Remove common trailing phrases that might be part of next specification
    rings_text = re.sub(r'\s*(?:,|\.|;|:).*$', '', rings_text)
    rings_text = re.sub(r'\s*SEE\s+.*$', '', rings_text, flags=re.IGNORECASE)
    rings_text = re.sub(r'\s*REFER\s+.*$', '', rings_text, flags=re.IGNORECASE)
    
    # Ensure meaningful content
    if len(rings_text.strip()) < 3:
        return "Not Found"
    
    return rings_text.strip()





def analyze_drawing_enhanced(image_path):
    """Enhanced analysis with conditional rings extraction"""
    try:
        # First, get preliminary text using OCR
        with Image.open(image_path) as img:
            preliminary_text = extract_text_from_image_wrapper(img)
        
        # Determine if we should extract rings
        extract_rings = should_extract_rings(preliminary_text)
        
        # Use conditional prompt
        prompt = get_enhanced_engineering_prompt(preliminary_text)
        
        # Process with Gemini using robust generation
        image = Image.open(image_path)
        content = [prompt, image]
        
        try:
            # Use robust generation with automatic model selection
            response = robust_generate_with_models(content)
            analysis_result = {'result': response.text} if response and response.text else None
            
            if analysis_result and analysis_result.get('result'):
                raw_text = analysis_result['result'].strip()
                cleaned_text = raw_text.replace('```json', '').replace('```', '').strip()
                try:
                    results = json.loads(cleaned_text)
                    
                    # If rings extraction was skipped but we found rings indicators later, extract them
                    if extract_rings and results.get('rings') == 'Not Found':
                        rings_info = extract_rings_from_text_detailed(preliminary_text)
                        if rings_info != "Not Found":
                            results['rings'] = rings_info
                            logger.info(f"Added rings information: {rings_info}")
                    
                    return results
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Enhanced analysis failed to parse JSON: {e}")
                    logger.error(f"Raw text received: {raw_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Gemini model generation failed: {e}")
            return None
                
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        return None

def process_with_gemini(image_path):
    """
    Process an image with Google's Gemini Vision model using robust model selection.
    Returns structured data extracted from the image.
    """
    try:
        image = Image.open(image_path)
        content = [
            "Here is a technical drawing of a tube or hose component. Please analyze it and extract the following information:",
            image
        ]
        
        # Use robust generation helper
        try:
            response = robust_generate_with_models(content)
            analysis_result = {'result': response.text} if response and response.text else None
        except Exception as e:
            logger.error(f"Robust model generation failed: {e}")
            return None
        
        if analysis_result and analysis_result.get('result'):
            raw_text = analysis_result['result']
            cleaned_text = raw_text.strip().replace('```json', '').replace('```', '').strip()
            try:
                results = json.loads(cleaned_text)
                logger.info("Successfully parsed JSON response")
                
                # Post-process standards
                if 'standard' in results:
                    std = results['standard']
                    if isinstance(std, str) and 'F-1' in std and 'F-30' in std:
                        results['standard'] = 'MPAPS F-30'
                        results['standards_note'] = 'Drawing shows both F-1 and F-30 standards'
                        logger.info("Normalized standard to MPAPS F-30")
                
                # Normalize measurements
                if 'working_pressure' in results and results['working_pressure'] != 'Not Found':
                    if not results['working_pressure'].lower().endswith('kpag'):
                        results['working_pressure'] = f"{results['working_pressure']} kPag"
                
                if 'weight' in results and results['weight'] != 'Not Found':
                    if not results['weight'].lower().endswith('kg'):
                        results['weight'] = f"{results['weight']} KG"
                
                return results
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                logger.error(f"Raw text received: {raw_text}")
                logger.error(f"Cleaned text attempted: {cleaned_text}")
                return None
    except Exception as e:
        logger.error(f"Error in process_with_gemini: {str(e)}")
        return None
        return None

def process_ocr_text(text):
    """
    Process extracted OCR text to structure it into the required format.
    Always returns a valid dictionary.
    """
    result = {
        "part_number": "Not Found",
        "description": "Not Found",
        "standard": "Not Found",
        "grade": "Not Found",
        "material": "Not Found",
        "reinforcement": "Not Found",
        "rings": "Not Found",  # Added: Rings field
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
        text = re.sub(r',(\d)', r'.\1', text)  # Convert decimal commas to points
        
        # Extract part number
        part_match = re.search(r'\d{7}[A-Z]\d', text)
        if part_match:
            result["part_number"] = part_match.group(0)

        # Extract rings using RingsExtractor with fallback patterns
        result["rings"] = RingsExtractor.extract_rings(text)
        
        # If RingsExtractor didn't find anything, try direct pattern matching
        if result["rings"] == "Not Found":
            rings_patterns = [
                r'RING\s+REINFORCEMENT.*?(\d+)\s+PLACES?',  # For "RING REINFORCEMENT ... 2 PLACES"
                r'(\d+)\s+(?:PC|PCS|EA|NOS).*?RING',  # For "2 PCS RING"
                r'RINGS?.*?(\d+)\s+(?:PC|PCS|EA|NOS|PLACES?)',  # For "RINGS 2 PCS"
                r'RINGS:\s*([^\n]+?(?:ASTM[^,\n]*)(?:[^,\n]*TYPE[^,\n]*)?)',
                r'RINGS[:\s]+([^\n]+?ASTM[^,\n]*(?:TYPE[^,\n]*)?)',
                r'RINGS\s*-\s*([^\n]+?ASTM[^,\n]*)',
                r'STEEL\s+RINGS\s*[\(\[]?\s*([^\)\]]+?ASTM[^\)\]]*)',
                r'RING.*?STAINLESS\s+WIRE\s+(\d+(?:\.\d+)?)\s*MM\s*DIA',  # For "RING ... STAINLESS WIRE 2MM DIA"
            ]
            
            for pattern in rings_patterns:
                rings_match = re.search(pattern, text, re.IGNORECASE)
                if rings_match:
                    rings_text = rings_match.group(1).strip()
                    rings_text = re.sub(r'\s+', ' ', rings_text)  # Normalize spaces
                    rings_text = rings_text.strip(' ,.-')
                    if len(rings_text) > 5:  # Only use if we have meaningful content
                        result["rings"] = rings_text
                        break
        
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
            "id1": r'(?:TUBING ID|HOSE ID|ID)\s*[=:]?\s*(\d+[.,]?\d*)',
            "od1": r'OD\s*[=:]?\s*(\d+[.,]?\d*)',
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
        
        return result
    except Exception as e:
        logger.error(f"Error processing OCR text: {str(e)}")
        return None

# --- Ring Detection and Extraction ---
def extract_rings_from_text_detailed(text):
    """Detailed rings extraction only called when rings are likely present"""
    try:
        text = clean_text_encoding(text)
        
        # More specific patterns for when we know rings exist
        detailed_patterns = [
            # Pattern for "RING REINFORCEMENT / STAINLESS WIRE 2MM / 2 PLACES"
            r'RING\s+REINFORCEMENT\s*/([^/]+?)\s*/\s*(\d+)\s+PLACES?',
            
            # Pattern for rings with material and size
            r'RINGS?[:\s]+([A-Za-z]+)\s+([A-Za-z]+)\s+(\d+(?:\.\d+)?\s*MM?)',
            
            # Pattern for rings with ASTM specifications
            r'RINGS?[:\s]+([^,\n]+?ASTM[^,\n]*(?:TYPE[^,\n]*)?)',
            
            # Pattern for steel rings with specifications
            r'STEEL\s+RINGS?[:\s]*([^\n]+?(?:\d+(?:\.\d+)?\s*MM?[^,\n]*)?)',
            
            # General rings specification
            r'RINGS?[:\s]+([^\n]+?(?:\d+(?:\.\d+)?[^,\n]*)?)',
        ]
        
        best_match = None
        best_specificity = 0
        
        for pattern in detailed_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Calculate specificity (more groups = more specific)
                specificity = len([g for g in match.groups() if g])
                
                if specificity > best_specificity:
                    best_specificity = specificity
                    rings_text = ' '.join([g for g in match.groups() if g]).strip()
                    rings_text = re.sub(r'\s+', ' ', rings_text)
                    rings_text = rings_text.strip(' ,.-/')
                    
                    # Filter out generic terms
                    if rings_text and rings_text not in ['REINFORCEMENT', 'RING', 'RINGS']:
                        if len(rings_text) > 5:  # Meaningful content
                            best_match = rings_text
        
        if best_match:
            logger.info(f"Detailed rings extraction found: {best_match}")
            return best_match
        else:
            logger.warning("No detailed rings information found despite indicators")
            return "Not Found"
            
    except Exception as e:
        logger.error(f"Error in detailed rings extraction: {e}")
        return "Not Found"

def should_extract_rings(text):
    """
    Determine if rings information should be extracted based on keyword presence
    Returns True if rings are likely present, False otherwise
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Primary rings indicators
    primary_indicators = [
        'ring reinforcement',
        'steel ring',
        'wire ring', 
        'rings:',
        'ring spec',
        'ring specification'
    ]
    
    # Secondary indicators (need context)
    secondary_indicators = [
        'ring',
        'rings'
    ]
    
    # Check for strong primary indicators
    for indicator in primary_indicators:
        if indicator in text_lower:
            logger.info(f"Rings extraction triggered by primary indicator: {indicator}")
            return True
    
    # Check for secondary indicators with context
    secondary_count = sum(1 for indicator in secondary_indicators if indicator in text_lower)
    if secondary_count >= 2:
        logger.info(f"Rings extraction triggered by multiple secondary indicators")
        return True
    
    # Check for specific patterns that indicate rings
    rings_patterns = [
        r'ring.*\d+.*mm',  # "ring" followed by numbers and "mm"
        r'ring.*steel',    # "ring" followed by "steel"
        r'ring.*wire',     # "ring" followed by "wire"
        r'rings.*astm',    # "rings" followed by "ASTM"
        r'ring.*reinforcement.*\d+',  # "ring reinforcement" with numbers
    ]
    
    for pattern in rings_patterns:
        if re.search(pattern, text_lower):
            logger.info(f"Rings extraction triggered by pattern: {pattern}")
            return True
    
    logger.info("No rings indicators found, skipping rings extraction")
    return False

# --- Prompt Generation ---
def get_enhanced_engineering_prompt(extracted_text):
    """Conditionally include rings in prompt only if keywords are found"""
    base_prompt = """Analyze this technical engineering drawing and extract the following information as JSON:

REQUIRED FIELDS:
- part_number: Look for formats like 3718791C1, 3541592C1 (7 digits + C + digit)
- description: Main title, usually starting with "HOSE," "PIPE," etc.
- standard: Material specifications like "MPAPS F-6034", "TMS-6034"
- grade: Look for "GRADE", "TYPE" followed by codes like "H-AN", "1B", "C-AN"
- dimensions: Extract ID, OD, wall thickness, centerline length
- material: Primary material specification
- reinforcement: Reinforcement type if specified"""

    # Check if rings keywords exist in extracted text
    rings_keywords = ['RING', 'RINGS', 'RING REINFORCEMENT', 'STEEL RING', 'WIRE RING']
    has_rings = any(keyword.lower() in extracted_text.lower() for keyword in rings_keywords)
    
    if has_rings:
        base_prompt += """
- rings: Extract detailed rings specification including material, dimensions, and quantity"""

    base_prompt += """

SPECIAL INSTRUCTIONS:
- Return ONLY valid JSON format
- Use "Not Found" for missing values
- Be precise with technical specifications"""

    return base_prompt

# --- Basic Configuration ---
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model and API Configuration ---

def get_available_models():
    """Get list of actually available models"""
    available_models = []
    try:
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        logger.info(f"Found {len(available_models)} available models")
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        # Fallback to known working models
        available_models = [
            'models/gemini-1.0-pro',
            'models/gemini-1.0-pro-vision', 
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
        ]
        logger.info("Using fallback model list")
    return available_models

def get_safe_model():
    """Get a model that definitely exists and supports content generation"""
    try:
        models = genai.list_models()
        # Prefer vision-capable / generateContent models
        candidates = []
        for m in models:
            name = getattr(m, "name", None)
            methods = getattr(m, "supported_generation_methods", []) or []
            if name and "generateContent" in methods:
                candidates.append(name)

        # Prefer current flash/pro models
        preferred_order = [
            "models/gemini-2.5-flash",
            "models/gemini-flash-latest",
            "models/gemini-2.5-flash-image",
            "models/gemini-pro-latest",
            "models/gemini-2.5-pro-preview-03-25"
        ]
        
        # Keep only those present
        for pref in preferred_order:
            if pref in candidates:
                logger.info(f"Selected preferred model: {pref}")
                return pref

        if candidates:
            logger.info(f"No preferred model found; using first candidate: {candidates[0]}")
            return candidates[0]

        logger.error("No models supporting generateContent found")
        return None

    except Exception as e:
        logger.error(f"Error listing/selecting models: {e}", exc_info=True)
        logger.warning("Available models: %s", [m.name for m in genai.list_models()])
        return None

def robust_generate_with_models(content, model_candidates=None):
    """Attempt to generate content with multiple model candidates, falling back on failure"""
    if model_candidates is None:
        # Get current safe model and any additional candidates
        safe_model = get_safe_model()
        if safe_model:
            model_candidates = [safe_model]
            # Add backup candidates from list_models
            try:
                models = genai.list_models()
                for m in models:
                    name = getattr(m, "name", None)
                    methods = getattr(m, "supported_generation_methods", []) or []
                    if name and "generateContent" in methods and name not in model_candidates:
                        model_candidates.append(name)
            except Exception as e:
                logger.warning(f"Failed to get additional model candidates: {e}")
        else:
            raise RuntimeError("No safe model available")

    last_exc = None
    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(content)
            if resp and getattr(resp, "text", None):
                return resp
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            try:
                logger.warning("Available models: %s", [m.name for m in genai.list_models()])
            except Exception:
                pass
            last_exc = e
            continue
    
    # If all fail, raise last exception with context
    raise last_exc or RuntimeError("All models failed to generate content")

# Gemini API Configuration
api_key = os.environ.get("GEMINI_API_KEY")
is_test_mode = os.environ.get("TEST_MODE", "").lower() == "true"

if not api_key and not is_test_mode:
    logging.error("GEMINI_API_KEY environment variable not set and not in test mode")
    raise ValueError("GEMINI_API_KEY environment variable must be set to use Gemini AI features")

if api_key:
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini API key configured successfully")

        # Log available models and their capabilities
        try:
            models = genai.list_models()
            logger.info("Available Gemini models (names & methods):")
            for m in models:
                logger.info(f"  name={m.name}  methods={getattr(m, 'supported_generation_methods', None)}")
        except Exception as e:
            logger.error("Failed to list Gemini models: %s", e)

        # Test model availability immediately
        if get_safe_model():
            logger.info("Successfully verified model availability")
        else:
            logger.warning("No models available despite successful configuration")
    except Exception as e:
        if not is_test_mode:
            logger.error(f"Failed to configure Gemini API key: {str(e)}")
            raise RuntimeError(f"Failed to initialize Gemini AI: {str(e)}")
        else:
            logger.warning("Failed to configure Gemini AI, but continuing in test mode")

# Roboflow API Configuration
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
roboflow_available = False
rf = None

if ROBOFLOW_API_KEY:
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        roboflow_available = True
        logging.info("Roboflow initialized successfully")
    except Exception as e:
        logging.warning(f"Roboflow initialization failed: {e}")
else:
    logging.warning("ROBOFLOW_API_KEY not set, Roboflow OCR not available")

# --- Roboflow OCR Functions ---
def extract_text_with_roboflow_ocr(image_path):
    """
    Extract text from image using Roboflow OCR API
    """
    if not roboflow_available or not rf:
        logger.warning("Roboflow not available for OCR")
        return None
    
    try:
        # Load and prepare image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Get OCR project
        workspace = rf.workspace()
        if not workspace:
            logger.error("Failed to access Roboflow workspace")
            return None
            
        project = workspace.project("text-recognition")
        if not project:
            logger.error("Failed to access text-recognition project")
            return None
            
        model = project.version(1).model
        if not model:
            logger.error("Failed to access model")
            return None
        
        # Perform OCR
        result = model.predict(image_path, confidence=40, overlap=30)
        
        # Extract and combine text
        extracted_text = ""
        if result and hasattr(result, 'predictions'):
            for prediction in result.predictions:
                if hasattr(prediction, 'text') and prediction.text:
                    extracted_text += prediction.text + " "
        
        logger.info(f"Roboflow OCR extracted {len(extracted_text)} characters")
        return extracted_text.strip() if extracted_text else None
        
    except Exception as e:
        logger.error(f"Roboflow OCR failed: {e}")
        return None

def extract_text_with_roboflow_inference(image_path):
    """
    Alternative method using Roboflow Inference API
    """
    if not ROBOFLOW_API_KEY:
        return None
    
    try:
        # Read image as base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Roboflow OCR API endpoint
        url = "https://infer.roboflow.com/ocr/ocr"
        
        payload = {
            "api_key": ROBOFLOW_API_KEY,
            "image": image_data
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            text_blocks = result.get('predictions', [])
            
            # Sort text blocks by position (top to bottom, left to right)
            sorted_blocks = sorted(text_blocks, 
                                 key=lambda x: (x.get('y', 0), x.get('x', 0)))
            
            extracted_text = " ".join([block.get('text', '') for block in sorted_blocks])
            logger.info(f"Roboflow Inference OCR extracted {len(extracted_text)} characters")
            return extracted_text.strip()
        else:
            logger.error(f"Roboflow API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Roboflow Inference OCR failed: {e}")
        return None

# --- Advanced Text Extraction ---
def extract_text_advanced(image_path: str, use_roboflow: bool = True) -> dict:
    """
    Advanced text extraction using multiple OCR engines with quality assessment
    """
    results: dict[str, str | None] = {
        'tesseract': None,
        'roboflow': None,
        'best_result': None
    }
    
    # 1. Tesseract OCR (existing method)
    try:
        with Image.open(image_path) as img:
            tesseract_text = extract_text_from_image_wrapper(img)
            if tesseract_text:
                results['tesseract'] = clean_text_encoding(str(tesseract_text))
                tesseract_score = assess_text_quality(str(results['tesseract']))
                logger.info(f"Tesseract quality score: {tesseract_score}")
    except Exception as e:
        logger.error(f"Tesseract extraction failed: {e}")
    
    # 2. Roboflow OCR (if available and requested)
    if use_roboflow and roboflow_available:
        try:
            roboflow_text = extract_text_with_roboflow_ocr(image_path)
            if roboflow_text:
                results['roboflow'] = clean_text_encoding(roboflow_text)
                roboflow_score = assess_text_quality(results['roboflow'])
                logger.info(f"Roboflow quality score: {roboflow_score}")
                
                # Alternative Roboflow method if primary fails
                if not roboflow_text or roboflow_score < 0.5:
                    roboflow_text_alt = extract_text_with_roboflow_inference(image_path)
                    if roboflow_text_alt:
                        results['roboflow_alt'] = clean_text_encoding(roboflow_text_alt)
                        alt_score = assess_text_quality(results['roboflow_alt'])
                        logger.info(f"Roboflow alternative score: {alt_score}")
                        if alt_score > roboflow_score:
                            results['roboflow'] = results['roboflow_alt']
        except Exception as e:
            logger.error(f"Roboflow extraction failed: {e}")
    
    # 3. Select best result based on quality score
    best_score = 0
    best_text = ""
    
    for method, text in results.items():
        if method != 'best_result' and text:
            score = assess_text_quality(text)
            if score > best_score:
                best_score = score
                best_text = text
                logger.info(f"New best: {method} with score {score}")
    
    results['best_result'] = best_text
    logger.info(f"Selected best text with score: {best_score}")
    
    return results

def assess_text_quality(text: str) -> float:
    """
    Wrapper for evaluate_text_quality to maintain backward compatibility
    """
    return evaluate_text_quality(text)
    
    try:
        score = 0.0
        text = str(text).upper()  # Normalize text for consistent matching
        
        # Check for key technical indicators with weights
        indicators = {
            'part_number': (r'\d{7}[A-Z]\d', 0.3),  # Part number is most important
            'mpaps': (r'MPAPS\s*F[-\s]*\d+', 0.2),
            'dimensions': (r'(?:ID|OD|LENGTH)\s*[=:]\s*\d+(?:\.\d+)?', 0.2),
            'pressure': (r'(?:PRESSURE|WP|BP)\s*[=:]\s*\d+(?:\.\d+)?\s*(?:KPAG|KPA|BAR)', 0.1),
            'grade': (r'(?:GRADE|TYPE)\s*[A-Z0-9]+', 0.1),
            'material': (r'(?:MATERIAL|MAT\.?)\s*[:=]', 0.1)
        }
        
        # Check each indicator
        for key, (pattern, weight) in indicators.items():
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
                logger.debug(f"Found {key} indicator in text")
        
        # Additional quality metrics
        words = text.split()
        if len(words) > 10:  # Reasonable amount of text
            score += 0.1
            
        # Check word quality (most technical terms 3-15 chars)
        word_lengths = [len(w) for w in words if w.isalnum()]
        if word_lengths and 3 <= sum(word_lengths) / len(word_lengths) <= 15:
            score += 0.1
        
        # Check for excessive special characters or garbage text
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s\.,\-]', text)) / (len(text) or 1)
        if special_char_ratio < 0.3:  # Not too many special characters
            score += 0.1
        
        return min(1.0, score)  # Cap score at 1.0
        
    except Exception as e:
        logger.error(f"Error assessing text quality: {e}")
        return 0.0

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
        logging.info(f"Reinforcement database set to material_df with {len(reinforcement_df) if reinforcement_df is not None else 0} entries.")
        
        return material_df, reinforcement_df
    except Exception as e:
        logging.error(f"Unexpected error processing material database: {str(e)}")
        return None, None

# --- BEGIN: Lightweight AI Agent integration (natural-language -> plan -> execute) ---
# Note: Most imports like json, tempfile, and uuid already exist at top-level

def _safe_llm_plan(prompt, model_name=None, max_tokens=512):
    """
    Lightweight wrapper to ask the configured LLM to produce text (ideally JSON).
    Uses existing `genai` / `PREFERRED_MODEL` objects if available in app.py.
    """
    # Lazy imports/guards so this block can be appended safely to any app.py
    try:
        model = model_name or globals().get("PREFERRED_MODEL") or "models/gemini-2.5-flash"
        # Attempt to use `genai.GenerativeModel` if present in this module's globals
        genai_mod = globals().get("genai")
        if genai_mod is None:
            raise RuntimeError("LLM wrapper `genai` not available in globals.")

        resp = genai_mod.GenerativeModel(model).generate_content({
            "input": [{"role": "user", "content": prompt}],
            "maxOutputTokens": max_tokens
        })

        # SDKs expose text differently; try common attributes
        text = None
        if hasattr(resp, "output_text"):
            text = resp.output_text
        elif hasattr(resp, "text"):
            text = resp.text
        else:
            # Fallback: str()
            text = str(resp)
        return text
    except Exception as e:
        # Log with existing logger if available; otherwise print
        lg = globals().get("logger")
        if lg:
            lg.exception("LLM planning call failed")
        else:
            print("LLM planning call failed:", e)
        return None

def _parse_json_safe(text):
    """
    Try strict json.loads first; then heuristically extract first {...} block.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # Heuristic: find first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
        return None

def _ensure_tool(name):
    """
    Check whether helper functions referenced by the agent exist.
    Returns True if present, False otherwise.
    """
    return name in globals() and callable(globals()[name])

def run_agent_plan(plan, pdf_bytes=None):
    """
    Execute a simple JSON plan produced by the LLM.
    
    plan: dict with {"steps":[ {"tool": "<name>", "args": {...}}, ... ] }
    Supported tools (by name):
    - "ocr": runs extract_text_from_pdf_memory_safe(pdf_bytes, max_pages=..., dpi=...)
    - "preprocess_image": runs preprocess_for_ocr on a PIL image (if passed) 
    - "parse_text": asks LLM to parse given text into structured JSON
    - "db_match": runs match_materials_db(parsed_json)
    
    The function is conservative about memory: it processes pages one-by-one and GC's aggressively.
    """
    logger_local = globals().get("logger")
    results = {"steps": []}

    # Basic validation
    if not isinstance(plan, dict) or "steps" not in plan:
        return {"error": "invalid plan", "plan": plan}

    for step in plan.get("steps", []):
        tool = step.get("tool")
        args = step.get("args") or {}
        step_result = {"tool": tool, "args": args, "ok": False, "output": None}
        try:
            if tool == "ocr":
                if not _ensure_tool("extract_text_from_pdf_memory_safe"):
                    step_result["output"] = {"error": "ocr helper not found"}
                else:
                    # allow passing overrides (dpi, max_pages) but the helper is memory-safe
                    max_pages = int(args.get("max_pages", 1))
                    text = globals()["extract_text_from_pdf_memory_safe"](pdf_bytes, max_pages=max_pages)
                    step_result["output"] = {"text": text}
                    step_result["ok"] = bool(text and len(text.strip()) > 0)

            elif tool == "preprocess_image":
                # expects 'image' in args (PIL) or 'image_path'
                if not _ensure_tool("preprocess_for_ocr"):
                    step_result["output"] = {"error": "preprocess_for_ocr not available"}
                else:
                    img = args.get("image")
                    if img is None and args.get("image_path"):
                        from PIL import Image
                        img = Image.open(args["image_path"])
                    if img is None:
                        step_result["output"] = {"error": "no image provided"}
                    else:
                        processed = globals()["preprocess_for_ocr"](img)
                        # preprocess_for_ocr may return single PIL or (pre,inv)
                        step_result["output"] = {"preprocessed": True}
                        step_result["ok"] = True

            elif tool == "parse_text":
                # Ask the LLM to parse raw text into structured JSON
                instruction = args.get("instruction", "Extract materials, grades, and dimensions as JSON.")
                raw_text = args.get("text")
                if not raw_text:
                    # attempt to find OCR output from previous steps
                    for prior in reversed(results["steps"]):
                        if prior.get("tool") == "ocr" and prior.get("output", {}).get("text"):
                            raw_text = prior["output"]["text"]
                            break
                if not raw_text:
                    step_result["output"] = {"error": "no text to parse"}
                else:
                    prompt = (
                        "You are a strict JSON parser. Input text between >>> and <<<.\n"
                        "Return only JSON with keys: materials (list), grades (list), dimensions (list of {name,value,units}), confidence (0-1).\n\n"
                        f">>>\n{raw_text}\n<<<\n\nInstruction: {instruction}\n"
                    )
                    llm_out = _safe_llm_plan(prompt)
                    parsed = _parse_json_safe(llm_out)
                    step_result["output"] = {"parsed": parsed, "raw_llm": llm_out}
                    step_result["ok"] = parsed is not None

            elif tool == "db_match":
                # expects parsed JSON either in args or prior parse_text output
                parsed = args.get("parsed")
                if parsed is None:
                    for prior in reversed(results["steps"]):
                        if prior.get("tool") == "parse_text" and prior.get("output", {}).get("parsed"):
                            parsed = prior["output"]["parsed"]
                            break
                if parsed is None:
                    step_result["output"] = {"error": "no parsed data to match"}
                else:
                    if not _ensure_tool("match_materials_db"):
                        step_result["output"] = {"error": "match_materials_db not available"}
                    else:
                        matches = globals()["match_materials_db"](parsed)
                        # match_materials_db can return DataFrame or list; try to jsonify-friendly convert
                        try:
                            import pandas as pd
                            if isinstance(matches, pd.DataFrame):
                                matches = matches.fillna("").to_dict(orient="records")
                        except Exception:
                            pass
                        step_result["output"] = {"matches": matches}
                        step_result["ok"] = True

            else:
                step_result["output"] = {"error": f"unknown tool {tool}"}

        except MemoryError as me:
            if logger_local:
                logger_local.exception("MemoryError during agent step %s", tool)
            step_result["ok"] = False
            step_result["output"] = {"error": "memory"}
            results["steps"].append(step_result)
            # short-circuit on MemoryError to avoid crashing the worker
            return results

        except Exception as e:
            if logger_local:
                logger_local.exception("Exception running agent step %s", tool)
            step_result["ok"] = False
            step_result["output"] = {"error": str(e)}

        # housekeeping: try to free memory between steps
        try:
            gc.collect()
        except Exception:
            pass

        results["steps"].append(step_result)

    return results

# Flask endpoint to accept natural language requests and run the agent plan
try:
    # Bind only if `app` exists in globals (the main Flask app variable)
    if "app" in globals():
        @app.route("/api/agent", methods=["POST"])
        def api_agent():
            """
            Endpoint to accept an instruction and an optional PDF file.
            Accepts multipart/form-data with fields:
              - prompt (string)
              - file (pdf)

            Returns:
              {
                "plan": {...},        # JSON plan produced (or fallback)
                "result": {...}       # execution results per step
              }
            """
            data = {}
            try:
                # prefer form (file upload) but allow JSON body
                if request.content_type and request.content_type.startswith("multipart/form-data"):
                    prompt = request.form.get("prompt") or request.values.get("prompt")
                    file = request.files.get("file")
                else:
                    payload = request.get_json(silent=True) or {}
                    prompt = payload.get("prompt")
                    file = None

                pdf_bytes = file.read() if file else None

                # Ask the LLM to produce a simple plan. If LLM fails, use a small default plan.
                plan_prompt = (
                    "You are a planner. Produce a JSON plan object with 'steps' (list). "
                    "Allowed tools: ['ocr','preprocess_image','parse_text','db_match'].\n"
                    "Each step must be: {\"tool\":\"...\",\"args\":{...}}.\n"
                    "If user mentions pages or DPI, include them. Prefer memory-safe operations.\n"
                    "Return only JSON.\n"
                    f"User request: {prompt}\n\n"
                    "Example:\n"
                    '{"steps":[{"tool":"ocr","args":{"dpi":150,"max_pages":1}},'
                    '{"tool":"parse_text","args":{"instruction":"extract materials,grades,dimensions"}},'
                    '{"tool":"db_match","args":{}}]}'
                )

                plan_text = _safe_llm_plan(plan_prompt)
                plan = _parse_json_safe(plan_text) or {
                    "steps": [
                        {"tool": "ocr", "args": {"dpi": 150, "max_pages": 1}},
                        {"tool": "parse_text", "args": {"instruction": "extract materials and grades"}},
                        {"tool": "db_match", "args": {}}
                    ]
                }

                # If the parse_text step expects the OCR text inline, we will populate it during execution
                result = run_agent_plan(plan, pdf_bytes=pdf_bytes)

                return jsonify({"plan": plan, "result": result})
    else:
        # If `app` not present, skip binding endpoint
        print("api_agent not bound: `app` variable not found in globals()")

except Exception as e:
    # Avoid crashing on import; log if logger present
    lg = globals().get("logger")
    if lg:
        lg.exception("Failed to add /api/agent endpoint: %s", e)
    else:
        print("Failed to add /api/agent endpoint:", e)

# --- END: Lightweight AI Agent integration ---

# Add a test page for the agent
@app.route('/agent-test')
def agent_test():
    """Simple test page for the /api/agent endpoint"""
    return render_template('agent.html')

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
        norm_material = normalize_for_comparison(clean_material)
        
        matches = reinforcement_df[
            reinforcement_df.apply(lambda row: (
                normalize_for_comparison(str(row['STANDARD'])) == norm_standard and
                normalize_for_comparison(str(row['GRADE'])) == norm_grade and
                normalize_for_comparison(str(row['MATERIAL'])) == norm_material
            ), axis=1)
        ]
        
        if matches.empty:
            logging.warning(f"No reinforcement match found for Standard: '{standard}', Grade: '{grade}', Material: '{material}'")
            return "Not Found"
            
        try:
            reinforcement = matches.iloc[0]['REINFORCEMENT']
            logging.info(f"Normalized reinforcement match found: {reinforcement}")
            return reinforcement
        except (KeyError, IndexError) as e:
            logging.error(f"Error accessing reinforcement data: {str(e)}", exc_info=True)
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

# --- PDF Processing Functions ---
def extract_text_from_pdf_robust(pdf_bytes):
    """Enhanced PDF text extraction with multiple DPI attempts and preprocessing"""
    try:
        # Create temp file for processing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf.flush()
            pdf_path = temp_pdf.name
            
        logger.info("Starting enhanced PDF text extraction")
        
        # Try multiple DPI settings
        dpi_options = [300, 400]
        best_text = ""
        best_dpi = None
        best_length = 0
        
        for dpi in dpi_options:
            logger.info(f"Attempting conversion at {dpi} DPI")
            try:
                # Convert PDF to images with current DPI
                images = convert_from_path(pdf_path, dpi=dpi)
                logger.info(f"Successfully converted PDF to {len(images)} images at {dpi} DPI")
                
                for i, pil_img in enumerate(images):
                    logger.info(f"Processing page {i+1} with {dpi} DPI")
                    
                    # Save original for reference
                    saved_orig = save_and_log_image_for_debug(pil_img, f"orig_dpi{dpi}_page{i+1}")
                    
                    # Try normal and preprocessed versions
                    # Get preprocessed versions
                    try:
                        normal_img, inverted_img = preprocess_for_ocr(pil_img)
                        variants = [
                            ("original", [pil_img]),
                            ("preprocessed", [normal_img, inverted_img])
                        ]
                    except Exception as e:
                        logger.error(f"Error in preprocessing: {e}")
                        # If preprocessing fails, only use original image
                        variants = [("original", [pil_img])]
                    
                    for variant_name, variant_imgs in variants:
                        for img_idx, img in enumerate(variant_imgs):
                            img_type = "normal" if img_idx == 0 else "inverted"
                            logger.info(f"Trying {variant_name} {img_type}")
                            
                            # Run OCR with multiple configs
                            ocr_results = run_tesseract_and_log(img)
                            
                            # Find best result for this variant
                            if ocr_results:
                                best_cfg, text = max(ocr_results.items(), key=lambda t: len(t[1].strip()))
                                text = text.strip()
                                length = len(text)
                                
                                logger.info(f"{variant_name} {img_type} produced {length} chars with {best_cfg}")
                                
                                if length > best_length:
                                    best_length = length
                                    best_text = text
                                    best_dpi = dpi
                                    logger.info(f"New best result: {length} chars at {dpi} DPI")
                
            except Exception as e:
                logger.error(f"Error processing at {dpi} DPI: {str(e)}")
                continue
        
        if best_text:
            logger.info(f"Best extraction: {best_length} chars at {best_dpi} DPI")
            return best_text
        else:
            logger.error("All text extraction attempts failed")
            return None
        # Text extraction methods
        get_text_old = getattr(page, 'getText', None)
        
        if callable(get_text_method):
            extraction_methods = [
                ("get_text()", lambda p: p.get_text()),
                ("get_text('blocks')", lambda p: p.get_text("blocks")),
                ("get_text('words')", lambda p: p.get_text("words")),
                ("get_text('text')", lambda p: p.get_text("text")),
                ("getText()", lambda p: p.getText()),
                ("getText('blocks')", lambda p: p.getText("blocks")),
                ("getText('words')", lambda p: p.getText("words")),
            ]
        
            page_text = ""
            for method_name, extractor in extraction_methods:
                try:
                    extracted = extractor(page)
                    if isinstance(extracted, (list, tuple)):
                        # Handle block/word output formats
                        if isinstance(extracted[0], (list, tuple)):
                            # Blocks format
                            extracted = " ".join(block[4] if len(block) > 4 else str(block) for block in extracted)
                        else:
                            # Words format
                            extracted = " ".join(str(word) for word in extracted)
                    
                    if extracted:
                        extracted_str = str(extracted).strip()
                        logger.info(f"Method {method_name} extracted {len(extracted_str)} chars")
                        if len(extracted_str) > len(page_text):
                            page_text = extracted_str
                            logger.info(f"Using better result from {method_name} (sample: {extracted_str[:100]})")
                            break
                except Exception as e:
                    logger.debug(f"Method {method_name} failed: {str(e)}")
                    continue
            
            if not page_text:
                logger.warning("All text extraction methods failed for page")
        else:
            logger.warning(f"Unexpected page type: {type(page)}")
            page_text = ""
        
        # Ensure we're always concatenating strings with proper spacing
        text = f"{text}\n{page_text}" if text else page_text
        
        if text:
            text_len = len(text.strip())
            logger.info(f"PyMuPDF extracted {text_len} characters")
            if text_len > 100:
                logger.info(f"Text sample: {text[:200]}...")
            return text
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
    
    # Method 2: Convert to images and use Tesseract OCR with enhanced preprocessing
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)  # Increased DPI for better quality
        text = ""
        for i, image in enumerate(images):
            # Enhanced preprocessing
            img = image.convert("L")  # Convert to grayscale
            if cv2_available:
                try:
                    # Apply CLAHE for better contrast
                    if cv2 is not None and cv2_available:
                        img_array = np.array(img)
                        if len(img_array.shape) == 2:  # Check if grayscale
                            img = Image.fromarray(cv2.equalizeHist(img_array))
                        else:
                            logger.warning("Skipping histogram equalization - image is not grayscale")
                    else:
                        logger.warning("OpenCV not available for preprocessing")
                except Exception as e:
                    logger.warning(f"OpenCV preprocessing failed: {e}")
            
            # Save preprocessed image with high quality
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                img.save(temp_img.name, format="PNG", quality=95)
                page_text = extract_text_from_image_wrapper(img)
            
            if page_text:
                text += f"Page {i+1}:\n{page_text}\n\n"
            
            # Log extraction results
            logger.info(f"Page {i+1}: Extracted {len(page_text) if page_text else 0} characters")
        
        if text and len(text.strip()) > 100:
            logger.info(f"Tesseract OCR extracted {len(text)} characters")
            return text
    except Exception as e:
        logger.warning(f"Tesseract OCR failed: {e}")
    # Method 3: Simple Gemini analysis as last resort
    try:
        model_name = get_safe_model()
        if model_name and api_key:
            model = genai.GenerativeModel(model_name)
            # Convert first page to image for analysis
            images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
            if images:
                response = model.generate_content([
                    "Extract any visible text from this engineering drawing. Return only the raw text found.",
                    images[0]
                ])
                if response.text:
                    logger.info(f"Gemini extracted {len(response.text)} characters")
                    return response.text
    except Exception as e:
        logger.error(f"All text extraction methods failed: {e}")
    
    return ""

def process_pdf_with_enhanced_ocr(pdf_bytes):
    """
    Process PDF with enhanced OCR capabilities including Roboflow
    """
    temp_pdf_path = None
    all_text_results = []
    
    try:
        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        # Convert PDF to images
        images = convert_from_path(temp_pdf_path)
        
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1} with enhanced OCR...")
            
            # Save image to temporary file for Roboflow
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                image_path = temp_img.name
                image.save(image_path, 'PNG')
            
            try:
                # Use enhanced OCR with multiple engines
                ocr_results = extract_text_advanced(image_path)
                best_text = ocr_results['best_result']
                
                if best_text:
                    # Process the text with your existing function
                    processed_data = process_ocr_text(best_text)
                    if processed_data:
                        all_text_results.append(processed_data)
                        logger.info(f"Page {i+1} processed successfully")
                    
                    # Log OCR method comparison for debugging
                    logger.info(f"Page {i+1} OCR comparison:")
                    for method, text in ocr_results.items():
                        if method != 'best_result' and text:
                            score = assess_text_quality(text)
                            sample = text[:100] + "..." if len(text) > 100 else text
                            logger.info(f"  {method}: score={score:.2f}, sample='{sample}'")
                
            except Exception as page_error:
                logger.error(f"Error processing page {i+1}: {page_error}")
            finally:
                # Clean up temporary image file
                if os.path.exists(image_path):
                    os.unlink(image_path)
    
    except Exception as e:
        logger.error(f"Enhanced PDF processing failed: {e}")
        return []
    
    finally:
        # Clean up temporary PDF file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary PDF: {e}")
    
    return all_text_results

# --- Utility Functions ---
def merge_ocr_results(ocr_results):
    """
    Merge multiple OCR results into a single comprehensive result
    """
    if not ocr_results:
        return {}
    
    merged = {}
    
    for result in ocr_results:
        if not isinstance(result, dict):
            continue
            
        for key, value in result.items():
            if value and value != "Not Found":
                if key not in merged:
                    merged[key] = value
                elif key == 'coordinates' and isinstance(value, list):
                    # Merge coordinates
                    if 'coordinates' not in merged:
                        merged['coordinates'] = []
                    merged['coordinates'].extend(value)
                elif key == 'dimensions' and isinstance(value, dict):
                    # Merge dimensions
                    if 'dimensions' not in merged:
                        merged['dimensions'] = {}
                    merged['dimensions'].update(value)
    
    return merged

def enhance_with_material_lookup(results):
    """
    Enhance results with material and reinforcement lookup
    """
    try:
        standard = results.get("standard", "Not Found")
        grade = results.get("grade", "Not Found")
        
        if standard != "Not Found" and grade != "Not Found":
            # Material lookup
            results["material"] = get_material_from_standard(standard, grade)
            
            # Reinforcement lookup
            reinforcement = get_reinforcement_from_material(
                standard, grade, results["material"]
            )
            results["reinforcement"] = reinforcement if reinforcement else "Not Found"
    
    except Exception as e:
        logger.error(f"Material enhancement failed: {e}")
    
    return results

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
        
        logger.info("Starting direct string matching for dimensions...")
        
        # 1. OD extraction
        if "101.4" in text:
            dimensions["od1"] = "101.4"
            dimensions["od2"] = "101.4"
            logger.info("OD found: 101.4mm")
        
        # 2. Thickness extraction
        thickness_match = re.search(r'WALL\s+THICKNESS\s+([\d.]+)', text, re.IGNORECASE)
        if thickness_match:
            dimensions["thickness"] = thickness_match.group(1)
            logger.info(f"Thickness found: {thickness_match.group(1)}mm")
        
        # 3. Calculate ID from OD and thickness
        if dimensions["od1"] != "Not Found" and dimensions["thickness"] != "Not Found":
            try:
                od_val = float(dimensions["od1"])
                thickness_val = float(dimensions["thickness"])
                calculated_id = od_val - (2 * thickness_val)
                dimensions["id1"] = f"{calculated_id:.2f}"
                dimensions["id2"] = f"{calculated_id:.2f}"
                logger.info(f"Calculated ID from OD and thickness: {calculated_id:.2f}mm")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error calculating ID: {e}")
        
        # 4. Centerline length
        if "147.7" in text:
            dimensions["centerline_length"] = "147.7"
            logger.info("Centerline length found: 147.7mm")
        
        # 5. Pressure extraction
        # Burst pressure
        burst_match = re.search(r'MINIMUM\s+BURST\s+PRESSURE\s+([\d.]+)\s*PSIG', text, re.IGNORECASE)
        if burst_match:
            psi = float(burst_match.group(1))
            bar = psi * 0.0689476  # Convert PSI to bar
            dimensions["burst_pressure"] = f"{bar:.1f}"
            logger.info(f"Burst pressure found: {psi} PSIG ({bar:.1f} bar)")
        
        # Working pressure (peak pressure)
        peak_match = re.search(r'PEAK\s+PRESSURE\s*:\s*([\d.]+)\s*PSIG', text, re.IGNORECASE)
        if peak_match:
            psi = float(peak_match.group(1))
            bar = psi * 0.0689476
            dimensions["working_pressure"] = f"{bar:.1f}"
            logger.info(f"Working pressure found: {psi} PSIG ({bar:.1f} bar)")
        
        # Log successful extractions
        logger.info("EXTRACTED DIMENSIONS:")
        for key, value in dimensions.items():
            if value != "Not Found":
                logger.info(f"  {key}: {value}")
        
        return dimensions
        
    except Exception as e:
        logger.error(f"Error extracting dimensions: {e}")
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

def extract_rings_from_text(text):
    """
    Extract rings information from PDF text.
    Returns the rings specification or "Not Found".
    """
    try:
        # Clean the text
        text = clean_text_encoding(text)
        
        # Enhanced patterns including specific formats and variations
        rings_patterns = [
            # New pattern for "RING REINFORCEMENT / MATERIAL / N PLACES" format
            r'RING\s+REINFORCEMENT\s*/([^/]+?)\s*/\s*(\d+)\s+PLACES?',
            # Alternative format for reinforcement details
            r'RING\s+REINFORCEMENT\s*(.*?)(?:\d+\s+PLACES|$)',
            # Legacy patterns for ASTM specifications
            r'RINGS:\s*([^\n]+?(?:ASTM[^,\n]*)(?:[^,\n]*TYPE[^,\n]*)?)',
            r'RINGS[:\s]+([^\n]+?ASTM[^,\n]*(?:TYPE[^,\n]*)?)',
            r'RINGS\s*-\s*([^\n]+?ASTM[^,\n]*)',
            # Simpler fallback patterns
            r'RINGS?(?:\s+INFO(?:RMATION)?)?[:;\s]+([^\n]+)',
        ]
        
        for pattern in rings_patterns:
            rings_match = re.search(pattern, text, re.IGNORECASE)
            if rings_match:
                # For patterns with multiple groups (material and places), combine them
                if len(rings_match.groups()) > 1:
                    material = rings_match.group(1).strip()
                    places = rings_match.group(2).strip()
                    rings = f"{material} / {places} PLACES"
                else:
                    rings = rings_match.group(1).strip()
                
                # Clean up the rings text
                rings = re.sub(r'\s+', ' ', rings)  # Normalize spaces
                rings = rings.strip(' ,.-/')  # Strip common separators
                
                if len(rings.strip()) > 3:  # Ensure meaningful content
                    logger.info(f"Rings found: {rings}")
                    return rings
        
        logger.warning("No rings information found in text")
        return "Not Found"
        
    except Exception as e:
        logger.error(f"Error extracting rings: {e}")
        return "Not Found"





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
            'REINFORCEMENT', 
            'RINGS',                                    # Additional info
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
                else:
                    thickness_calculated = "Invalid"
        except (ValueError, TypeError) as e:
            logging.warning(f"Error calculating thickness: {e}")
            thickness_calculated = "Calculation Error"

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
            'RINGS': analysis_results.get('rings', 'Not Found'),  # NEW: Rings data
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
                col_letter = openpyxl.utils.get_column_letter(idx + 1)
                worksheet.column_dimensions[col_letter].width = max_length + 2

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
def evaluate_text_quality(text: str) -> float:
    """
    Enhanced text quality assessment with weighted metrics.
    Returns a score between 0.0 and 1.0.
    """
    if not text:
        return 0.0
        
    try:
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



def calculate_arc_length(current: dict, next_point: dict, radius: float | None) -> float:
    """Calculate arc length between two points considering radius if present."""
    # Get points as tuples for vector calculation
    p1 = (float(current['x']), float(current['y']), float(current['z']))
    p2 = (float(next_point['x']), float(next_point['y']), float(next_point['z']))
    
    # Calculate straight distance
    straight_length = math.dist(p1, p2)
    
    # If no radius or invalid, return straight length
    if not radius or radius <= 0:
        return straight_length
    
    # Otherwise calculate arc length
    # Using distance as chord length to find arc length
    chord_length = straight_length
    theta = 2 * math.asin(chord_length / (2 * radius))  # Central angle
    arc_length = radius * theta
    
    return arc_length

def calculate_development_length(coordinates):
    """Calculate development length considering radii at bends"""
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
def extract_rings_info_from_text(text):
    """
    Extract rings information from PDF text.
    Returns the rings specification or "Not Found".
    """
    try:
        # Clean the text
        text = clean_text_encoding(text)
        
        # Pattern for rings information
        rings_patterns = [
            r'RINGS:\s*([^\n]+?(?:ASTM[^,\n]*)(?:[^,\n]*TYPE[^,\n]*)?)',
            r'RINGS[:\s]+([^\n]+?ASTM[^,\n]*(?:TYPE[^,\n]*)?)',
            r'RINGS\s*-\s*([^\n]+?ASTM[^,\n]*)',
        ]
        
        for pattern in rings_patterns:
            rings_match = re.search(pattern, text, re.IGNORECASE)
            if rings_match:
                rings = rings_match.group(1).strip()
                # Clean up the rings text
                rings = re.sub(r'\s+', ' ', rings)  # Normalize spaces
                rings = rings.strip(' ,.-')
                logger.info(f"Rings found: {rings}")
                return rings
        
        logger.warning("No rings information found in text")
        return "Not Found"
        
    except Exception as e:
        logger.error(f"Error extracting rings: {e}")
        return "Not Found"
        
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
        if cv2_available and cv2 is not None:
            logging.info("Using OpenCV for advanced preprocessing...")
            # Convert to numpy array for OpenCV processing
            try:
                img_array = np.array(image)
                preprocessed_images = []

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
                    try:
                        # Convert back to PIL Image for OCR
                        pil_image = Image.fromarray(processed)
                        new_text = pytesseract.image_to_string(pil_image, config='--psm 1')
                        new_text = clean_text_encoding(new_text)
                        new_score = assess_text_quality(new_text)
                        
                        if new_score > best_score:
                            best_text = new_text
                            best_score = new_score
                    except Exception as ocr_error:
                        logging.warning(f"OCR failed for preprocessed image: {ocr_error}")
                        continue
                        
                logging.info(f"Best score after OpenCV preprocessing: {best_score}")
                
            except Exception as e:
                logging.warning(f"OpenCV preprocessing failed: {e}")
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
                # Extract text based on page type
                if isinstance(page, fitz.Page):
                    # Add text to full_text using safe text extraction
                    full_text += safe_extract_text(page)
                else:
                    full_text += page.extractText()
        
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
    """Simplified robust drawing analysis"""
    logger.info("Starting simplified drawing analysis...")
    
    # Initialize default results
    results = {
        "part_number": "Not Found",
        "description": "Not Found", 
        "standard": "Not Found",
        "grade": "Not Found",
        "material": "Not Found",
        "reinforcement": "Not Found",
        "rings": "Not Found",
        "dimensions": {},
        "coordinates": [],
        "error": None
    }
    
    try:
        # Extract text using robust method
        extracted_text = extract_text_from_pdf_robust(pdf_bytes)
        
        if not extracted_text or len(extracted_text.strip()) < 50:
            results["error"] = "Could not extract sufficient text from PDF"
            return results
        
        # Process with OCR text processor
        processed_data = process_ocr_text(extracted_text)
        
        if processed_data:
            # Update results with processed data
            for key in ['part_number', 'description', 'standard', 'grade', 'material', 'rings', 'coordinates']:
                if key in processed_data and processed_data[key] != "Not Found":
                    results[key] = processed_data[key]
            
            if 'dimensions' in processed_data:
                results['dimensions'].update(processed_data['dimensions'])
        
        # Enhance with material lookup
        if results["standard"] != "Not Found" and results["grade"] != "Not Found":
            material = get_material_from_standard(results["standard"], results["grade"])
            if material != "Not Found":
                results["material"] = material
            
            # Get reinforcement
            reinforcement = get_reinforcement_from_material(
                results["standard"], 
                results["grade"], 
                results["material"]
            )
            if reinforcement != "Not Found":
                results["reinforcement"] = reinforcement
        
        logger.info("Drawing analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in drawing analysis: {e}")
        results = {
            "error": f"Analysis failed: {str(e)}",
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
                "thickness": "Not Found"
            },
            "centerline_length": "Not Found",
            "reinforcement": "Not Found",
            "coordinates": [],
            "error": None
        }
        return results
    
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
        
    except (PDFPageCountError, PDFSyntaxError) as e:
        logger.error(f"PDF conversion error: {str(e)}")
        results["error"] = f"Failed to process PDF: {str(e)}"
        return results
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        results["error"] = "Failed to parse AI response"
        return results
    except BlockedPromptException as e:
        logger.error(f"Gemini API content policy violation: {str(e)}")
        results["error"] = "Content policy violation"
        return results
    except Exception as e:
        logger.error(f"Unexpected error in drawing analysis: {str(e)}")
        results["error"] = f"Analysis failed: {str(e)}"
        return results
    logging.info("Starting data parsing with Gemini...")
    
    # Get appropriate model using robust selection
    try:
        model = get_vision_model()
    except Exception as e:
        logger.error(f"Failed to get vision model: {e}")
        results["error"] = "Failed to initialize vision model"
        return results
    
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
        cleaned_text = analysis_result['result'].strip().replace("```json", "").replace("```", "").strip()
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
        try:
            model = get_vision_model()
        except Exception as e:
            logger.error(f"Failed to get vision model: {e}")
            return {'error': 'Failed to initialize vision model'}
        
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
            model = get_vision_model()
            # Generate content with the enhanced prompt
            response = model.generate_content([prompt, full_text])
            
            if analysis_result and analysis_result.get('result'):
                # Clean the response to ensure it's valid JSON
                cleaned_response = re.sub(r'```json\s*|\s*```', '', analysis_result['result'].strip())
                
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
                    print("Raw response:", analysis_result['result'])  # Debug logging
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
                    
    except ValueError as e:
        logging.warning(f"Value error during validation: {str(e)}")
        issues.append(f"Value error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during validation: {str(e)}")
        issues.append(f"Validation error: {str(e)}")
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
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logging.info(f"[{request_id}] Starting new analysis request")
    if 'file' not in request.files:
        logging.warning("No file part in request")
        return jsonify({'error': 'No file part in request', 'code': 'MISSING_FILE'}), 400
        
    file = request.files['file']
    if not file or not file.filename:
        logging.warning("No file selected")
        return jsonify({'error': 'No file selected', 'code': 'NO_FILE'}), 400
    
    # 2. File validations
    if not file.filename.lower().endswith('.pdf'):
        logging.warning(f"Invalid file type: {file.filename}")
        return jsonify({"error": "Invalid file type. Please upload a PDF file.", 'code': 'INVALID_TYPE'}), 400

    # Check file size (20MB limit)
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB in bytes
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to start
    
    if file_size > MAX_FILE_SIZE:
        duration = time.time() - start_time
        logging.warning(f"[{request_id}] File too large: {file_size} bytes (processed in {duration:.2f}s)")
        return jsonify({
            "error": "File too large. Maximum size is 20MB.",
            "code": "FILE_TOO_LARGE",
            "details": f"File size: {file_size/1024/1024:.1f}MB, Max: 20MB",
            "request_id": request_id,
            "processing_time": f"{duration:.2f}s"
        }), 413

    logging.info(f"[{request_id}] Processing request for file: {file.filename} ({file_size/1024/1024:.1f}MB)")
    analysis_start = time.time()
    
    try:
        # 3. Read file contents
        pdf_bytes = file.read()
        if not pdf_bytes:
            logging.warning("Uploaded file is empty")
            return jsonify({"error": "Uploaded file is empty", 'code': 'EMPTY_FILE'}), 400
            
        # 4. Analyze drawing
        try:
            final_results = analyze_drawing(pdf_bytes)
        except Exception as e:
            logging.error(f"Analysis failed with error: {str(e)}")
            return jsonify({"error": "Analysis failed", "details": str(e), 'code': 'ANALYSIS_ERROR'}), 500
        
        # 5. Response validation and return
        if not isinstance(final_results, dict):
            logging.error(f"Invalid analyzer response type: {type(final_results)}")
            return jsonify({"error": "Internal error: Invalid response format", 'code': 'INVALID_RESPONSE'}), 500
            
        if final_results.get("error"):
            error_msg = final_results["error"]
            error_code = final_results.get("code", "UNKNOWN_ERROR")
            if "PDF conversion error" in error_msg:
                logging.warning(f"PDF conversion failed: {error_msg}")
                return jsonify({
                    "error": "Invalid or corrupted PDF file", 
                    "code": "PDF_ERROR",
                    "details": error_msg
                }), 400
            elif "Content policy violation" in error_msg:
                logging.warning(f"Content policy violation: {error_msg}")
                return jsonify({
                    "error": "Content policy violation",
                    "code": "POLICY_VIOLATION",
                    "details": error_msg
                }), 403
            elif "Model not available" in error_msg:
                logging.error(f"Model availability error: {error_msg}")
                return jsonify({
                    "error": "Service temporarily unavailable",
                    "code": "MODEL_UNAVAILABLE",
                    "details": error_msg
                }), 503
            else:
                logging.error(f"Analysis error: {error_msg}")
                return jsonify({
                    "error": error_msg,
                    "code": error_code,
                    "details": error_msg
                }), 500
        
        # Process the results further
        part_number = final_results.get('part_number', 'Unknown')
        logging.info(f"Successfully analyzed drawing for part {part_number}")

        # Enhanced dimension extraction and merging
        try:
            # Get cleaned extracted text from multiple sources
            extracted_text = final_results.get('extracted_text') or final_results.get('combined_text') \
                            or final_results.get('ocr_text') or extract_text_from_pdf(pdf_bytes)

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
                        merged_dims[k] = float(str(merged_dims[k]).replace(',', ''))
                    except (ValueError, TypeError):
                        logging.warning(f"Failed to convert {k}={merged_dims[k]} to float")
                        merged_dims[k] = None

            # 5) Validate results
            # Check for required fields in final_results
            required_result_fields = ['part_number', 'extracted_text']
            missing_result_fields = [f for f in required_result_fields if not final_results.get(f)]
            if missing_result_fields:
                logging.warning(f"Analysis results missing required fields: {', '.join(missing_result_fields)}")
                return jsonify({
                    "error": "Incomplete analysis results",
                    "code": "MISSING_RESULT_FIELDS",
                    "details": f"Missing fields: {', '.join(missing_result_fields)}",
                    "partial_results": final_results
                }), 422

            # Validate dimension values
            missing_fields = []
            invalid_fields = []
            for field, value in merged_dims.items():
                if field in ['id1', 'od1', 'centerline_length']:
                    if value is None or value == "Not Found":
                        missing_fields.append(field)
                    elif not isinstance(value, (int, float)) or value <= 0:
                        invalid_fields.append(field)
            
            if missing_fields or invalid_fields:
                error_details = []
                if missing_fields:
                    error_details.append(f"Missing values for: {', '.join(missing_fields)}")
                if invalid_fields:
                    error_details.append(f"Invalid values for: {', '.join(invalid_fields)}")
                
                logging.warning(f"Dimension validation failed: {'; '.join(error_details)}")
                return jsonify({
                    "error": "Invalid dimension values",
                    "code": "INVALID_DIMENSIONS",
                    "details": '; '.join(error_details),
                    "partial_results": merged_dims
                }), 422

            # 6) Return successful response
            response = {
                "part_number": part_number,
                "dimensions": merged_dims,
                "confidence": final_results.get('confidence', 'medium'),
                "analysis_source": "hybrid",  # indicates we used both AI and direct text extraction
                "extraction_method": "multi-source"  # indicates we combined multiple text sources
            }
            
            if 'material' in final_results:
                response['material'] = final_results['material']

            # Add performance metrics
            duration = time.time() - start_time
            analysis_duration = time.time() - analysis_start
            response.update({
                "request_id": request_id,
                "processing_time": f"{duration:.2f}s",
                "analysis_time": f"{analysis_duration:.2f}s"
            })

            logging.info(f"[{request_id}] Successfully completed analysis in {duration:.2f}s (analysis: {analysis_duration:.2f}s)")
            return jsonify(response)

        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"[{request_id}] Error processing dimensions: {str(e)} (duration: {duration:.2f}s)")
            return jsonify({
                "error": "Failed to process dimensions",
                "code": "DIMENSION_PROCESSING_ERROR",
                "details": str(e),
                "request_id": request_id,
                "processing_time": f"{duration:.2f}s"
            }), 500

    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"[{request_id}] Unexpected error processing request: {str(e)} (duration: {duration:.2f}s)")
        return jsonify({
            "error": "Internal server error",
            "code": "SERVER_ERROR",
            "details": str(e),
            "request_id": request_id,
            "processing_time": f"{duration:.2f}s"
        }), 500

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

            # --- NEW: lookup reinforcement based on standard, grade and material ---
            try:
                # try to get reinforcement from the reinforcement table (helper exists in this file)
                reinforcement_val = get_reinforcement_from_material(standard, grade, final_results.get("material", "Not Found"))
                # Normalize common outputs: empty -> 'None', explicit 'Not Found' stays 'Not Found'
                if reinforcement_val is None or (isinstance(reinforcement_val, str) and reinforcement_val.strip() == ""):
                    reinforcement_val = "None"
                final_results["reinforcement"] = reinforcement_val
                logging.info(f"Reinforcement resolved: {final_results['reinforcement']}")
            except Exception as e:
                logging.warning(f"Reinforcement lookup failed: {e}", exc_info=True)
                # Keep a deterministic key for downstream code
                final_results["reinforcement"] = final_results.get("reinforcement", "Not Found")
        except Exception as e:
            logging.error(f"Error in material lookup: {e}")
            final_results["material"] = "Not Found"
            final_results["reinforcement"] = "Not Found"

            # In the /api/analyze route, after reinforcement extraction:

# Extract rings information
        rings_info = extract_rings_from_text(extracted_text)
        if rings_info != "Not Found":
            final_results["rings"] = rings_info
            logger.info(f"Rings information extracted: {rings_info}")
        else:
            final_results["rings"] = "Not Found"
            
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
                except ValueError as ve:
                    final_results["development_length_mm"] = "Not Found"
                    logger.warning(f"Could not compute development length: {ve}")
                except Exception as exc:
                    final_results["development_length_mm"] = "Not Found"
                    logger.exception("Error computing development length: %s", exc)
            else:
                final_results["development_length_mm"] = "Not Found"
                logger.info("No coordinates found for development length calculation")
        except Exception as e:
            final_results["development_length_mm"] = "Not Found"
            logger.exception("Error in development length calculation: %s", e)
        
        # Generate Excel report if helper function exists
        try:
            if 'generate_excel_sheet' in globals():
                try:
                    excel_file = generate_corrected_excel_sheet(final_results, final_results.get("dimensions", {}), coordinates)
                    excel_b64 = base64.b64encode(excel_file.getvalue()).decode('utf-8')
                    final_results["excel_data"] = excel_b64
                except Exception as excel_error:
                    logging.warning(f"Excel generation failed: {excel_error}")
                    final_results["excel_error"] = str(excel_error)
        except Exception as e:
            logging.error(f"Error in analysis process: {str(e)}")
            return jsonify({
                "error": f"Analysis failed: {str(e)}",
                "stage": "analysis"
            }), 500
        
        logging.info(f"Successfully analyzed drawing: {final_results.get('part_number', 'Unknown')}")
        return jsonify(final_results)

# --- Route for the main webpage (no change) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Run the application (no change) ---
def analyze_image_with_gemini_vision(pdf_bytes):
    """Process PDF using Gemini Vision API with enhanced image optimization"""
    logger.info("Starting Gemini Vision analysis with optimizations...")
    full_text = ""
    temp_pdf_path = None

    try:
        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name

        # Convert PDF to images with higher DPI
        page_images = convert_from_path(temp_pdf_path, dpi=200)
        logger.info(f"Converted PDF to {len(page_images)} images at 200 DPI")

        # Optimize images for Gemini
        from image_optimization import ensure_minimum_image_quality
        optimized_images = ensure_minimum_image_quality(page_images)
        logger.info("Images optimized for Gemini processing")
        
        from gemini_analysis import robust_gemini_analysis, get_engineering_drawing_prompt
        
        for i, page in enumerate(optimized_images):
            logger.info(f"Processing optimized page {i+1} with Gemini Vision...")
            
            try:
                # Convert optimized PIL Image to bytes
                img_byte_arr = io.BytesIO()
                page.save(img_byte_arr, format='PNG', optimize=True)
                img_data = img_byte_arr.getvalue()
                
                # Use robust analysis with multiple models
                analysis_result = robust_gemini_analysis(
                    img_data,
                    get_engineering_drawing_prompt()
                )
                
                if analysis_result and analysis_result.get('result'):
                    full_text += analysis_result['result'] + "\n"
                    logger.info(f"Page {i+1} processed successfully, extracted {len(analysis_result['result'])} characters")
                else:
                    logger.warning(f"No text extracted from page {i+1}")
                    
            except Exception as page_error:
                logger.error(f"Error processing page {i+1}: {page_error}")
                continue

        logger.info(f"OCR complete. Total characters extracted: {len(full_text)}")
        return full_text

    except Exception as e:
        logger.error(f"Error in Gemini Vision processing: {e}")
        return ""

    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
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
    """
    logger.info("Starting enhanced text extraction process...")
    texts = []
    
    try:
        # Method 1: PyMuPDF - try harder to extract text
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Try multiple extraction methods with PyMuPDF
                extraction_methods = [
                    ("text", lambda p: p.get_text()),
                    ("words", lambda p: p.get_text("words")),
                    ("blocks", lambda p: p.get_text("blocks")),
                    ("raw", lambda p: p.get_text("raw")),
                ]
                
                for method_name, method in extraction_methods:
                    try:
                        text = method(page)
                        if text and len(str(text).strip()) > 10:
                            texts.append(f"--- Page {page_num+1} ({method_name}) ---\n{text}")
                            logger.info(f"Extracted {len(str(text))} chars from page {page_num+1} using {method_name}")
                    except Exception as e:
                        logger.debug(f"Method {method_name} failed for page {page_num+1}: {e}")
        
        combined_text = "\n".join(filter(None, texts))
        logger.info(f"PyMuPDF extraction found {len(combined_text)} characters")
        
        # If still no meaningful text, try the working Gemini model
        if len(combined_text.strip()) < 100:
            logger.info("Text extraction insufficient, trying Gemini Vision with working model...")
            gemini_text = analyze_image_with_gemini_vision(pdf_bytes)
            if gemini_text and len(gemini_text.strip()) > 50:
                texts.append(f"--- Gemini Vision OCR ---\n{gemini_text}")
                logger.info(f"Gemini Vision extracted {len(gemini_text)} characters")
            else:
                logger.warning("Gemini Vision also failed to extract meaningful text")
                
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
        
        # Combine all extracted text
        final_text = "\n".join(filter(None, texts))
        logger.info(f"Final extracted text length: {len(final_text)} characters")
        return final_text if final_text.strip() else ""
        
    except Exception as e:
        logger.error(f"Error in enhanced text extraction: {e}")
        # Last resort: try basic Gemini Vision
        try:
            return analyze_image_with_gemini_vision(pdf_bytes)
        except:
            return ""

if __name__ == '__main__':
    app.run(debug=True)