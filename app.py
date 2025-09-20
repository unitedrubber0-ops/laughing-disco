import os
import re
import json
import logging
import pandas as pd
import fitz  # PyMuPDF
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, current_app
from pdf2image import convert_from_bytes
from PIL import Image
import io
import tempfile
from werkzeug.exceptions import RequestTimeout, ServiceUnavailable
from functools import wraps
import threading
import time
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log separation line
logger.info("------------------------------------------")
            
            # Calculate and log success rates
            total_fields = 0
            regex_found = 0
            ai_found = 0
            both_matched = 0
            
            comparison_fields = {
                "id": ("id1", "id"),
                "od": ("od1", "od"),
                "thickness": ("thickness", "thickness"),
                "centerline_length": ("centerline_length", "centerline_length"),
                "radius": ("radius", "radius"),
                "angle": ("angle", "angle")
            }
            
            for regex_key, ai_key in comparison_fields.values():
                total_fields += 1
                regex_val = regex_dimensions.get(regex_key, "Not Found")
                ai_val = ai_results.get(ai_key, "Not Found")
                
                if regex_val != "Not Found":
                    regex_found += 1
                if ai_val != "Not Found":
                    ai_found += 1
                if regex_val != "Not Found" and regex_val == ai_val:
                    both_matched += 1
            
            logger.info("\n----------- EXTRACTION STATISTICS -----------")
            logger.info(f"Total fields checked: {total_fields}")
            logger.info(f"Regex success rate: {(regex_found/total_fields)*100:.1f}%")
            logger.info(f"AI success rate: {(ai_found/total_fields)*100:.1f}%")
            logger.info(f"Match rate when both found value: {(both_matched/total_fields)*100:.1f}%")
            logger.info("------------------------------------------")
            
        # Extract coordinate points

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_dimensions_from_text(text):
    """
    Extract dimensions from the PDF text using regex patterns with detailed logging
    """
    # Normalize text by replacing newlines with spaces
    text = text.replace('\n', ' ')
    
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

    # Log the full text for pattern analysis
    logger.info("\n----------- DIMENSION EXTRACTION TEXT -----------")
    logger.info(f"Full text being processed: {text}")
    logger.info("------------------------------------------")
    
    # Define all patterns we're looking for with their descriptions
    patterns = {
        "part_number": (r'(\d{7}C\d)', "Part number (7 digits + C + digit)"),
        "id_2d": (r'(?:ID|TUBING ID)\s*[:=]\s*(\d+[.,]?\d*)\s*[±]\s*(\d+[.,]?\d*)', "ID from drawing with tolerance"),
        "id_3d": (r'(?:ID|TUBING ID)\s*[:=]\s*(\d+[.,]?\d*)', "ID value"),
        "thickness": (r'WALL THICKNESS\s*[:=]\s*([\d.,]+)', "Wall thickness"),
        "centerline": (r'(?:APPROX\s*)?CTRLINE LENGTH\s*=\s*(\d+[.,]?\d*)', "Centerline length"),
        "radius": (r'(?:RADIUS|R)\s*[:=]?\s*\(?\s*(\d+[.,]?\d*)\)?', "Radius value"),
        "angle": (r'(?:ANGLE|ANG)?\s*(\d+)\s*°(?:\s*[±+]\s*\d+\s*°)?', "Bend angle"),
        "tubing_od": (r'TUBING\s+OD\s*[:=]\s*([<>]?)\s*(\d+[.,]?\d*)', "Tubing outer diameter"),
        "description": (r'(?:HOSE|TUBE),\s*([\s\w,\-\.]+?)(?:\s*\n|\s*$)', "Part description"),
        "specification": (r'(?:MPAPS|SAE)\s*(?:F-)?(\d+(?:\.\d+)?)', "Material specification"),
        "grade": (r'(?:GRADE|TYPE)\s*([\w\d]+)', "Material grade")
    }
    
    # Log each pattern matching attempt
    logger.info("\n----------- PATTERN MATCHING ATTEMPTS -----------")
    
    # Try each pattern and log results
    for pattern_name, (pattern, description) in patterns.items():
        logger.info(f"\nTrying to match {description} with pattern: {pattern}")
        # First try case-sensitive match
        matches = list(re.finditer(pattern, text))
        if not matches:
            # If no matches, try case-insensitive
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                logger.info("Found matches using case-insensitive search")
        
        found = False
        for match in matches:
            found = True
            logger.info(f"Found match for {description}:")
            logger.info(f"Full match: {match.group(0)}")
            logger.info(f"Captured groups: {match.groups()}")
            
            # Extract larger context around the match
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            logger.info(f"Context around match: \n...{context}...")
            
            # Process the match based on pattern type
            value = None
            if pattern_name == "id_2d":
                value = match.group(1).replace(',', '.')
                dimensions["id1"] = value
            elif pattern_name == "id_3d":
                value = match.group(1).replace(',', '.')
                dimensions["id2"] = value
            elif pattern_name == "thickness":
                value = match.group(1).replace(',', '.')
                dimensions["thickness"] = value
            elif pattern_name == "centerline":
                value = match.group(1).replace(',', '.')
                dimensions["centerline_length"] = value
            elif pattern_name == "radius":
                value = match.group(1).replace(',', '.')
                dimensions["radius"] = value
            elif pattern_name == "angle":
                value = match.group(1)
                dimensions["angle"] = value
            elif pattern_name == "tubing_od":
                value = match.group(2).replace(',', '.')
                dimensions["od1"] = value
            
            logger.info(f"Extracted value for {pattern_name}: {value}")
            
            # Validate numeric values
            if value and value != "Not Found":
                try:
                    float(value.replace(',', '.'))
                except ValueError:
                    logger.warning(f"Invalid numeric value extracted for {pattern_name}: {value}")
                    # Keep the value as "Not Found"
                    if pattern_name in dimensions:
                        dimensions[pattern_name] = "Not Found"
        
        if not found:
            logger.info(f"No matches found for {description}")
    
    logger.info("\n----------- FINAL EXTRACTED DIMENSIONS -----------")
    logger.info(json.dumps(dimensions, indent=2))
    logger.info("------------------------------------------")
    
    return dimensions
    
    # Extract ID1 (look for patterns like "As per 2D : 43.5±0.5")
    id_match = re.search(r'As per 2D\s*:\s*(\d+[.,]?\d*)\s*[±]\s*(\d+[.,]?\d*)', text)
    if id_match:
        dimensions["id1"] = id_match.group(1).replace(',', '.')
        print(f"Found ID1: {dimensions['id1']}")
    
    # Extract ID2 (look for patterns like "As per 3D : 44.85")
    id2_match = re.search(r'As per 3D\s*:\s*(\d+[.,]?\d*)', text)
    if id2_match:
        dimensions["id2"] = id2_match.group(1).replace(',', '.')
        print(f"Found ID2: {dimensions['id2']}")
    
    # Extract thickness (look for patterns like "WALL THICKNESS - 4,050")
    thickness_match = re.search(r'WALL THICKNESS\s*[-\s]*\s*(\d+[.,]?\d*)', text)
    if thickness_match:
        dimensions["thickness"] = thickness_match.group(1).replace(',', '.')
        print(f"Found thickness: {dimensions['thickness']}")
    
    # Extract centerline length
    centerline_match = re.search(r'APPROX CTRLINE LENGTH\s*=\s*(\d+[.,]?\d*)', text, re.IGNORECASE)
    if centerline_match:
        dimensions["centerline_length"] = centerline_match.group(1).replace(',', '.')
        print(f"Found centerline length: {dimensions['centerline_length']}")
    
    # Extract radius (look for patterns like (40))
    radius_matches = re.findall(r'\((\d+)\)', text)
    if radius_matches:
        # Use the first radius found
        dimensions["radius"] = radius_matches[0]
        print(f"Found radius: {dimensions['radius']}")
    
    # Extract angle (look for patterns like 90° +5°)
    angle_match = re.search(r'(\d+)\s*°\s*[+]\s*\d+\s*°', text)
    if angle_match:
        dimensions["angle"] = angle_match.group(1)
        print(f"Found angle: {dimensions['angle']}")
    
    # Extract OD from tubing information
    od_match = re.search(r'TUBING OD[^\d]*([<>]?)\s*(\d+[.,]?\d*)', text, re.IGNORECASE)
    if od_match:
        dimensions["od1"] = od_match.group(2).replace(',', '.')
        print(f"Found OD: {dimensions['od1']}")
    
    # Debug logging
    print("Extracted dimensions:", json.dumps(dimensions, indent=2))
    
    return dimensions
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
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"API Key not found in environment variables. Please set GEMINI_API_KEY. Error: {e}")
    # For quick testing, you can uncomment the line below and paste your key
    # genai.configure(api_key="YOUR_API_KEY_HERE")


# Initialize the Flask application
app = Flask(__name__, static_url_path='/static')

# Enable CORS with more permissive settings
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://laughing-disco-docker.onrender.com",
            "http://localhost:5000",
            "http://127.0.0.1:5000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure static files with correct MIME types
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MIME_TYPES'] = {
    '.js': 'application/javascript',
    '.css': 'text/css'
}

# --- Load the material database from the CSV file ---
try:
    material_df = pd.read_csv('material_data.csv')
    material_df.columns = material_df.columns.str.strip()
    material_df['STANDARD'] = material_df['STANDARD'].str.strip()
    material_df['GRADE'] = material_df['GRADE'].astype(str).str.strip()
    print("Material database loaded successfully.")
except FileNotFoundError:
    print("Error: material_data.csv not found. Please ensure the file exists.")
    material_df = pd.DataFrame()

# --- Function to extract dimensions from PDF text ---
def extract_dimensions_from_text(text):
    """
    Extract dimensions from the PDF text using regex patterns
    """
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
    
    try:
        # Extract ID1 (look for patterns like "43.5±0.5" or "43.5 ± 0.5")
        id_match = re.search(r'(\d+\.?\d*)\s*[±]\s*(\d+\.?\d*)', text)
        if id_match:
            dimensions["id1"] = id_match.group(1)
        
        # Extract thickness (look for patterns like "4.050" after "WALL THICKNESS")
        thickness_match = re.search(r'WALL THICKNESS[^\d]*(\d+\.?\d*)', text)
        if thickness_match:
            dimensions["thickness"] = thickness_match.group(1)
        
        # Extract centerline length with multiple patterns
        centerline_match = re.search(r'CTRLINE LENGTH\s*=\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if centerline_match:
            dimensions["centerline_length"] = centerline_match.group(1)
        else:
            # Try alternative patterns
            centerline_match2 = re.search(r'APPROX CTRLINE LENGTH\s*=\s*(\d+\.?\d*)', text, re.IGNORECASE)
            if centerline_match2:
                dimensions["centerline_length"] = centerline_match2.group(1)
        
        # Extract radius (look for patterns like (40) which might indicate radius)
        radius_match = re.search(r'\((\d+)\)', text)
        if radius_match:
            dimensions["radius"] = radius_match.group(1)
        
        # Extract angle (look for patterns like 90°)
        angle_match = re.search(r'(\d+)\s*°', text)
        if angle_match:
            dimensions["angle"] = angle_match.group(1)
        
        # Try to extract OD from tubing information
        od_match = re.search(r'TUBING OD[^\d]*(\d+\.?\d*)', text, re.IGNORECASE)
        if od_match:
            dimensions["od1"] = od_match.group(1)
        
        # Debug logging
        print("Extracted dimensions:", json.dumps(dimensions, indent=2))
    
    except Exception as e:
        print(f"Error extracting dimensions: {e}")
    
    return dimensions

# --- Function to calculate development length using vector geometry ---
def calculate_development_length(dimensions, points=None):
    """
    Calculate development length using vector geometry for accurate bend calculations.
    Can handle both extracted dimensions and 3D coordinate points.
    
    Args:
        dimensions: Dictionary containing extracted dimensions
        points: Optional list of {x,y,z,r} dictionaries for precise calculation
    
    Returns:
        float: Calculated development length in mm
        str: Error message if calculation fails
    """
    try:
        # If we have 3D points with radii, use precise calculation
        if points and len(points) >= 2:
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
                except (ValueError, TypeError):
                    continue
            
            if len(coordinates) >= 2:
                return calculate_path_length(coordinates, radii)
        
        # Fall back to dimension-based calculation if no valid points
        radius = dimensions.get("radius", "Not Found")
        angle = dimensions.get("angle", "Not Found")
        
        # Try using radius and angle
        if (radius != "Not Found" and angle != "Not Found" and
            str(radius).replace('.', '', 1).replace('-', '', 1).isdigit() and
            str(angle).replace('.', '', 1).replace('-', '', 1).isdigit()):
            
            radius = float(radius)
            angle = float(angle)
            return round(2 * math.pi * radius * (angle / 360), 2)
        
        # Fall back to centerline length
        centerline = dimensions.get("centerline_length", "Not Found")
        if centerline != "Not Found" and str(centerline).replace('.', '', 1).replace('-', '', 1).isdigit():
            return round(float(centerline), 2)
        
        # Use default values if all else fails
        return round(2 * math.pi * 40 * (90 / 360), 2)  # 40mm radius, 90° angle default
        
    except Exception as e:
        print(f"Error calculating development length: {e}")
        return "Calculation error"

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

# --- Function to generate Excel sheet with all details ---
def generate_excel_sheet(analysis_results, dimensions, development_length):
    # Create a DataFrame with the structure of FETCH FROM DRAWING worksheet
    columns = [
        'child part', 'child quantity', 'CHILD PART', 'CHILD PART DESCRIPTION', 
        'CHILD PART QTY', 'SPECIFICATION', 'MATERIAL', 'REINFORCEMENT', 
        'VOLUME AS PER 2D', 'ID1 AS PER 2D (MM)', 'ID2 AS PER 2D (MM)', 
        'OD1 AS PER 2D (MM)', 'OD2 AS PER 2D (MM)', 'THICKNESS AS PER 2D (MM)', 
        'THICKNESS AS PER ID OD DIFFERENCE', 'CENTERLINE LENGTH AS PER 2D (MM)', 
        'DEVELOPMENT LENGTH AS PER CO-ORDINATE (MM)', 'BURST PRESSURE AS PER 2D (BAR)',
        'BURST PRESSURE AS PER WORKING PRESSURE (4XWP) (BAR)', 'VOLUME AS PER 2D MM3', 
        'WEIGHT AS PER 2D KG', 'COLOUR AS PER DRAWING', 'ADDITIONAL REQUIREMENT', 
        'OUTSOURCE', 'REMARK'
    ]
    
    # Create a row with the extracted data
    row_data = {
        'SPECIFICATION': f"{analysis_results.get('standard', 'Not Found')} {analysis_results.get('grade', 'Not Found')}",
        'MATERIAL': analysis_results.get('material', 'Not Found'),
        'ID1 AS PER 2D (MM)': dimensions.get('id1', 'Not Found'),
        'ID2 AS PER 2D (MM)': dimensions.get('id2', 'Not Found'),
        'OD1 AS PER 2D (MM)': dimensions.get('od1', 'Not Found'),
        'OD2 AS PER 2D (MM)': dimensions.get('od2', 'Not Found'),
        'THICKNESS AS PER 2D (MM)': dimensions.get('thickness', 'Not Found'),
        'CENTERLINE LENGTH AS PER 2D (MM)': dimensions.get('centerline_length', 'Not Found'),
        'DEVELOPMENT LENGTH AS PER CO-ORDINATE (MM)': development_length,
        'ADDITIONAL REQUIREMENT': 'CUTTING & CHECKING FIXTURE COST TO BE ADDED. Marking cost to be added.',
        'REMARK': ''
    }
    
    # Add remarks based on the data
    remarks = []
    if analysis_results.get('standard', '').startswith('MPAPS F 1'):
        remarks.append('The drawing specifies MPAPS F 1, but we have considered the specification as MPAPS F 30.')
    if dimensions.get('id1') != dimensions.get('id2') and dimensions.get('id1') != 'Not Found' and dimensions.get('id2') != 'Not Found':
        remarks.append('THERE IS MISMATCH IN ID 1 & ID 2')
    row_data['REMARK'] = ' '.join(remarks) if remarks else 'No specific remarks.'
    
    # Calculate thickness from ID/OD if available
    od1 = dimensions.get('od1', 'Not Found')
    id1 = dimensions.get('id1', 'Not Found')
    if (od1 != 'Not Found' and id1 != 'Not Found' and
        isinstance(od1, str) and isinstance(id1, str) and
        od1.replace('.', '', 1).replace('-', '', 1).isdigit() and 
        id1.replace('.', '', 1).replace('-', '', 1).isdigit()):
        try:
            thickness = abs(float(od1) - float(id1)) / 2
            row_data['THICKNESS AS PER ID OD DIFFERENCE'] = round(thickness, 3)
        except Exception as e:
            print(f"Error calculating thickness difference: {e}")
            row_data['THICKNESS AS PER ID OD DIFFERENCE'] = 'Calculation error'
    
    # Create DataFrame with the row data
    df = pd.DataFrame([row_data], columns=columns)
    
    # Create an in-memory Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='FETCH FROM DRAWING', index=False)
    
    output.seek(0)
    return output

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

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Analyze drawing using improved Gemini vision model ---
def parse_text_with_gemini(full_text):
    """
    Uses Gemini to parse the text and extract structured data.
    This complements the existing regex-based system.
    """
    logging.info("Starting AI-powered text parsing with Gemini...")
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = f"""
        Analyze this engineering drawing text and extract the following information in JSON format.
        Return ONLY the JSON object, no other text.

        Required fields (use "Not Found" if not present):
        - part_number: 7 digits + 'C' + digit (e.g., 4403886C2)
        - id: Inner diameter with tolerances
        - thickness: Wall thickness value
        - centerline_length: The centerline measurement
        - radius: Any radius value (often in parentheses)
        - angle: Any angle measurement (usually with ° symbol)
        - od: Outer diameter value
        - description: Part description (usually after "HOSE," or "TUBE,")
        - specification: Material spec like MPAPS F-30
        - grade: Material grade or type

        Text to analyze:
        ---
        {full_text}
        ---
        """

        response = model.generate_content(prompt)
        if not response or not response.text:
            logging.warning("Received empty response from Gemini")
            return None

        # Clean response and parse JSON
        cleaned_response = re.sub(r'```json\s*|\s*```', '', response.text.strip())
        parsed_data = json.loads(cleaned_response)
        
        logging.info("Successfully parsed text with Gemini AI")
        logging.info(f"AI Extracted Data: {json.dumps(parsed_data, indent=2)}")
        
        return parsed_data

    except Exception as e:
        logging.error(f"Error in AI parsing: {str(e)}")
        return None

def analyze_drawing_with_gemini(pdf_bytes):
    """
    Analyze drawing using Google Gemini with improved prompt and image analysis.
    Now includes extraction of specifications, grades, and material properties.
    Uses both regex-based and AI-powered parsing for comparison.
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
        
        # Prepare content for Gemini
        content = []
        for img in images:
            content.append({
                'mime_type': 'image/png',
                'data': image_to_base64(img)
            })
        
        # Extract text for coordinate and dimension parsing
        pdf_document = fitz.open("pdf", pdf_bytes)
        full_text = ""
        for page_num, page in enumerate(pdf_document):
            page_text = page.get_text()
            full_text += page_text
            # Log each page's text separately for better debugging
            logger.info(f"\n----------- RAW EXTRACTED TEXT (PAGE {page_num + 1}) -----------")
            logger.info(page_text)
            logger.info("------------------------------------------")
            
            # Also log text blocks with their positions for structure analysis
            blocks = page.get_text("blocks")
            logger.info(f"\n----------- TEXT BLOCKS STRUCTURE (PAGE {page_num + 1}) -----------")
            for block in blocks:
                logger.info(f"Block at position {block[:4]}: {block[4]}")
            logger.info("------------------------------------------")
        
        # Log the complete extracted text
        logger.info("\n----------- COMPLETE EXTRACTED TEXT -----------")
        logger.info(full_text)
        logger.info("------------------------------------------")
        
        # Extract dimensions using both regex and AI methods
        regex_dimensions = extract_dimensions_from_text(full_text)
        ai_results = parse_text_with_gemini(full_text)
        
        # Use regex results as primary, but log comparison with AI results
        results["dimensions"] = regex_dimensions
        
        if ai_results:
            logger.info("\n----------- COMPARING REGEX VS AI RESULTS -----------")
            for key in ["id", "thickness", "centerline_length", "radius", "angle", "od"]:
                regex_val = regex_dimensions.get(f"{key}1" if key in ["id", "od"] else key, "Not Found")
                ai_val = ai_results.get(key, "Not Found")
                logger.info(f"{key}: Regex='{regex_val}' vs AI='{ai_val}'")
                
                # If regex didn't find it but AI did, use AI's value as backup
                if regex_val == "Not Found" and ai_val != "Not Found":
                    logger.info(f"Using AI value for {key} as regex failed to find it")
                    if key in ["id", "od"]:
                        regex_dimensions[f"{key}1"] = ai_val
                    else:
                        regex_dimensions[key] = ai_val
            
            # Update analysis results with any additional AI findings
            if results["part_number"] == "Not Found" and ai_results.get("part_number", "Not Found") != "Not Found":
                results["part_number"] = ai_results["part_number"]
            if results["description"] == "Not Found" and ai_results.get("description", "Not Found") != "Not Found":
                results["description"] = ai_results["description"]
                
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
            response = model.generate_content([prompt, *content])
            
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

# --- API endpoint for file analysis (now uses the Gemini function) ---
@app.route('/api/analyze', methods=['POST'])
def upload_and_analyze():
    if 'drawing' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['drawing']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        pdf_bytes = file.read()
        analysis_results = analyze_drawing_with_gemini(pdf_bytes)
        
        if analysis_results.get("error"):
            return jsonify({"error": analysis_results["error"]}), 400
        
        # Calculate development length using coordinates if available, otherwise use dimensions
        dimensions = analysis_results.get("dimensions", {})
        coordinates = analysis_results.get("coordinates", [])
        
        development_length = calculate_development_length(
            dimensions=dimensions,
            points=coordinates if coordinates else None
        )
        
        # Store the calculated length in the results
        analysis_results["development_length_mm"] = development_length
        
        # Generate Excel sheet with all details
        excel_file = generate_excel_sheet(analysis_results, dimensions, development_length)
        
        # Convert Excel file to base64 for sending in response
        excel_b64 = base64.b64encode(excel_file.getvalue()).decode('utf-8')
        
        # Add Excel data to response
        analysis_results["excel_data"] = excel_b64
        
        return jsonify(analysis_results)
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

# --- Route for the main webpage (no change) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Run the application (no change) ---
if __name__ == '__main__':
    app.run(debug=True)