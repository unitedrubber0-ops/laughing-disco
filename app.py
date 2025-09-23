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
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import fitz  # PyMuPDF
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import google.generativeai as genai

# --- Basic Configuration ---
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Key Configuration ---
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    logging.info("Gemini API key configured successfully.")
except Exception as e:
    logging.error(f"Failed to configure Gemini API key: {e}")

# --- Load and Clean Material Database on Startup with Enhanced Debugging ---
try:
    # Read from Excel file directly, specifying Sheet1
    material_df = pd.read_excel("MATERIAL WITH STANDARD.xlsx", sheet_name="Sheet1")
    
    # Clean the data by stripping whitespace from the key columns
    material_df['STANDARD'] = material_df['STANDARD'].str.strip()
    material_df['GRADE'] = material_df['GRADE'].astype(str).str.strip()
    logging.info(f"V2 CODE RUNNING: Successfully loaded and cleaned material database with {len(material_df)} entries.")
    
    # Enhanced debugging: Show the first few rows of the database
    logging.info(f"Material database head (first 5 rows):\n{material_df.head().to_string()}")
    
    # Additional debug info: Show unique standards and grades
    logging.info(f"Unique STANDARD values:\n{material_df['STANDARD'].unique().tolist()}")
    logging.info(f"Unique GRADE values:\n{material_df['GRADE'].unique().tolist()}")
except FileNotFoundError:
    logging.error("MATERIAL WITH STANDARD.xlsx not found. Material lookup will not work.")
    material_df = None

# --- String Normalization Helper ---
def normalize_for_comparison(text):
    """Converts text to a standardized format for reliable comparison."""
    # Converts to lowercase and removes all non-alphanumeric characters
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

# --- Material Lookup Function ---
def get_material_from_standard(standard, grade):
    """
    Looks up the material from the database using a flexible matching approach.
    Tries exact match first, then falls back to more flexible matching strategies.
    """
    if material_df is None or standard == "Not Found" or grade == "Not Found":
        return "Not Found"
    
    try:
        # First attempt: Exact canonical match
        norm_standard_from_pdf = normalize_for_comparison(standard)
        norm_grade_from_pdf = normalize_for_comparison(grade)

        result = material_df[
            (material_df['STANDARD'].apply(normalize_for_comparison) == norm_standard_from_pdf) &
            (material_df['GRADE'].apply(normalize_for_comparison) == norm_grade_from_pdf)
        ]
        
        if not result.empty:
            material = result.iloc[0]['MATERIAL']
            logging.info(f"Material lookup successful (exact match): Found '{material}' for Standard='{standard}', Grade='{grade}'")
            return material
            
        # Second attempt: More flexible matching for standards like "MPAPS F-6032" vs "MPAPS F6032"
        clean_standard = re.sub(r'[^A-Z0-9]', '', standard.upper())
        matches = []
        
        for idx, row in material_df.iterrows():
            db_standard = re.sub(r'[^A-Z0-9]', '', str(row['STANDARD']).upper())
            db_grade = str(row['GRADE']).upper().strip()
            
            # Check if the cleaned standard is contained in the database entry
            if clean_standard in db_standard and normalize_for_comparison(grade) == normalize_for_comparison(str(row['GRADE'])):
                matches.append({
                    'material': row['MATERIAL'],
                    'standard': row['STANDARD'],
                    'grade': row['GRADE'],
                    'match_quality': len(clean_standard) / len(db_standard)  # Higher ratio = better match
                })
        
        if matches:
            # Sort by match quality and take the best match
            best_match = sorted(matches, key=lambda x: x['match_quality'], reverse=True)[0]
            logging.info(f"Material lookup successful (flexible match): Found '{best_match['material']}' "
                        f"for Standard='{standard}' (matched with '{best_match['standard']}'), Grade='{grade}'")
            return best_match['material']
        
        logging.warning(f"Material lookup failed: No match found for Standard='{standard}', Grade='{grade}' "
                       f"(tried both exact and flexible matching)")
        return "Not Found"
            
    except Exception as e:
        logging.error(f"Error during material lookup: {e}")
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



# --- Function to calculate development length using vector geometry ---
def calculate_development_length(points):
    """
    Calculate development length using vector geometry for accurate bend calculations.
    
    Args:
        points: List of coordinate points from the drawing
    
    Returns:
        float: Calculated development length in mm
        float: 0 if calculation fails
    """
    try:
        if not points or len(points) < 2:
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
        
        if len(coordinates) >= 2:
            return calculate_path_length(coordinates, radii)
            
        return 0
            
    except Exception as e:
        logging.error(f"Error calculating development length: {e}")
        return 0
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
    Uses Gemini to parse the entire text block and extract structured data.
    This version has a much more specific prompt for higher accuracy.
    """
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
        
        # Extract dimensions using AI-only approach
        ai_results = parse_text_with_gemini(full_text)
        
        if ai_results:
            # Map AI results to dimension structure
            results["dimensions"] = {
                "id1": ai_results.get("id", "Not Found"),
                "id2": "Not Found",  # Currently AI doesn't differentiate between id1 and id2
                "od1": ai_results.get("od", "Not Found"),
                "od2": "Not Found",  # Currently AI doesn't differentiate between od1 and od2
                "thickness": ai_results.get("thickness", "Not Found"),
                "centerline_length": ai_results.get("centerline_length", "Not Found"),
                "radius": ai_results.get("radius", "Not Found"),
                "angle": ai_results.get("angle", "Not Found")
            }
            
            # Update main results with AI findings
            results["part_number"] = ai_results.get("part_number", "Not Found")
            results["description"] = ai_results.get("description", "Not Found")
            results["standard"] = ai_results.get("standard", "Not Found")
            results["grade"] = ai_results.get("grade", "Not Found")
                
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

# --- Data Validation Function ---
def validate_extracted_data(data):
    """
    Validates extracted data for completeness and logical consistency.
    Returns a list of validation issues found.
    """
    issues = []
    
    # Check critical fields
    if data.get('standard') == 'Not Found':
        issues.append("Standard specification not found in drawing")
    if data.get('grade') == 'Not Found':
        issues.append("Grade/Type not found in drawing")
    if data.get('material') == 'Not Found':
        issues.append("Material could not be identified from standard and grade")
    
    # Validate dimensions
    dimensions = data.get('dimensions', {})
    try:
        # Check ID/OD relationship
        if dimensions.get('id1') != 'Not Found' and dimensions.get('od1') != 'Not Found':
            id_val = float(str(dimensions['id1']).replace('mm', '').strip())
            od_val = float(str(dimensions['od1']).replace('mm', '').strip())
            if od_val <= id_val:
                issues.append(f"Invalid dimensions: OD ({od_val}mm) should be greater than ID ({id_val}mm)")
        
        # Check wall thickness consistency
        if dimensions.get('thickness') != 'Not Found':
            thickness = float(str(dimensions['thickness']).replace('mm', '').strip())
            if thickness <= 0:
                issues.append(f"Invalid wall thickness: {thickness}mm")
            
            # Cross-validate thickness with ID/OD if available
            if dimensions.get('id1') != 'Not Found' and dimensions.get('od1') != 'Not Found':
                calculated_thickness = (od_val - id_val) / 2
                if abs(calculated_thickness - thickness) > 0.1:  # Allow 0.1mm tolerance
                    issues.append(f"Thickness inconsistency: Specified {thickness}mm vs calculated {calculated_thickness}mm")
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
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    logging.info(f"New analysis request for file: {file.filename}")
    try:
        # Read and analyze PDF
        pdf_bytes = file.read()
        full_text = extract_text_from_pdf(pdf_bytes)
        if not full_text:
            return jsonify({"error": "Failed to extract text from PDF"}), 500

        # Parse text with Gemini
        final_results = parse_text_with_gemini(full_text)
        if "error" in final_results:
            return jsonify(final_results), 500

        # Look up material based on standard and grade
        standard = final_results.get("standard", "Not Found")
        grade = final_results.get("grade", "Not Found")
        final_results["material"] = get_material_from_standard(standard, grade)

        # Validate the extracted data
        validation_issues = validate_extracted_data(final_results)
        if validation_issues:
            final_results["validation_warnings"] = validation_issues
            logging.warning(f"Validation issues found: {validation_issues}")

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
                excel_file = generate_excel_sheet(final_results, final_results, dev_length)
                excel_b64 = base64.b64encode(excel_file.getvalue()).decode('utf-8')
                final_results["excel_data"] = excel_b64
        except Exception as e:
            logging.warning(f"Excel generation skipped: {e}")

        logging.info(f"Successfully analyzed drawing: {final_results.get('child_part', 'Unknown')}")
        return jsonify(final_results)

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
    """
    logger.info("Starting enhanced text extraction process...")
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