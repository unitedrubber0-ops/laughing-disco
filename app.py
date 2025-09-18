import os
import re
import json
import math
import base64
import pandas as pd
import fitz  # PyMuPDF
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pdf2image import convert_from_bytes
from PIL import Image
import io
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import psutil with fallback
try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil not found. Memory logging disabled.")

def get_memory_usage():
    """Returns current memory usage in MB, or None if psutil is not available."""
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception as e:
        logger.error(f"Error getting memory usage: {str(e)}")
        return None

def process_page_with_gemini(page_image):
    """Process a page image using Gemini Vision API.
    
    Args:
        page_image: A PIL Image object containing the page to process
        
    Returns:
        str: Extracted text from the image, or empty string if extraction fails
    """
    try:
        # Convert PIL Image to bytes
        buffered = io.BytesIO()
        page_image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        # Encode image for Gemini
        img_base64 = base64.b64encode(img_bytes).decode()
        
        # Configure and use Gemini model
        model = genai.GenerativeModel('gemini-pro-vision')
        prompt = """Extract all readable text from this engineering drawing image. Focus on:
        - Part numbers (e.g., 7 digits + C + digit)
        - Material standards (e.g., MPAPS F-30)
        - Grades
        - Coordinates (P0-Pn with X/Y/Z values)
        - Dimensions and measurements
        - Pressure specifications
        - Part descriptions
        Output as clean, structured text with clear labeling."""
        
        # Generate content with image
        response = model.generate_content([
            prompt,
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ])
        
        # Clean up
        del buffered, img_bytes, img_base64
        
        return response.text if response.text else ""
        
    except Exception as e:
        logger.error(f"Error in Gemini Vision processing: {str(e)}")
        return ""

# --- Helper Functions for Detailed Analysis ---
def load_feasibility_data():
    """Load data from FEASIBILITY 24251022.xlsx worksheet 'FETCH FROM DRAWING'"""
    try:
        # Read the Excel file
        df = pd.read_excel('FEASIBILITY 24251022.xlsx', sheet_name='FETCH FROM DRAWING')
        
        # Extract the relevant row (assuming it's the last row with data)
        material_data = df.iloc[-1] if len(df) > 0 else None
        
        if material_data is not None:
            return {
                'specification': material_data.get('SPECIFICATION', 'Not Found'),
                'material': material_data.get('MATERIAL', 'Not Found'),
                'reinforcement': material_data.get('REINFORCEMENT', 'Not Found'),
                'id': material_data.get('ID1 AS PER 2D (MM)', 'Not Found'),
                'centerline_length': material_data.get('CENTERLINE LENGTH AS PER 2D (MM)', 'Not Found'),
                'burst_pressure': material_data.get('BURST PRESSURE AS PER 2D (BAR)', 'Not Found')
            }
        return {}
    except Exception as e:
        logger.error(f"Error loading feasibility data: {str(e)}")
        return {}

def extract_specific_info(text):
    """Extracts key-value data with more flexible regex patterns."""
    info = {
        'child_part': "Not Found",
        'description': "Not Found",
        'specification': "Not Found",
        'material': "Not Found",
        'reinforcement': "Not Found",
        'id': "Not Found",
        'centerline_length': "Not Found",
        'burst_pressure_bar': "Not Found",
        'working_pressure_kpag': "Not Found",
        'development_length_mm': "Not Found",
        'od': "Not Found",
        'thickness': "Not Found"
    }

    # Load data from feasibility worksheet
    feasibility_data = load_feasibility_data()
    
    # Part Number: Find the specific C-number format directly
    part_num_match = re.search(r'(\d{7}[Cc]\d(?: Rev [A-Z])?)', text, re.IGNORECASE)
    if part_num_match:
        info['child_part'] = part_num_match.group(1)

    # Description: Find the "HOSE, ..." pattern
    desc_match = re.search(r'(HOSE,[\s\w,]+)', text, re.IGNORECASE)
    if desc_match:
        info['description'] = desc_match.group(1).strip()
        
    # Specification: Find MPAPS F-30 or similar
    spec_match = re.search(r'(MPAPS\s*F[- ]*\d+)', text, re.IGNORECASE)
    if spec_match:
        info['specification'] = spec_match.group(0).replace(" ", "")
    elif feasibility_data.get('specification'):
        info['specification'] = feasibility_data['specification']

    # Material: Find the Grade
    material_match = re.search(r'GRADE\s+([\w\d]+)', text, re.IGNORECASE)
    if material_match:
        grade = material_match.group(1)
        info['material'] = f"GRADE {grade}"
        
        # Try to find the material type based on standard and grade
        if info['specification'] != "Not Found":
            # Look up in material database
            match = material_df[
                material_df['STANDARD'].str.contains(info['specification'], na=False, case=False) &
                (material_df['GRADE'] == grade)
            ]
            if not match.empty:
                info['material'] = match.iloc[0]['MATERIAL']
    elif feasibility_data.get('material'):
        info['material'] = feasibility_data['material']

    # Reinforcement
    if feasibility_data.get('reinforcement'):
        info['reinforcement'] = feasibility_data['reinforcement']

    # ID: Look for "HOSE ID" with an equals sign
    id_match = re.search(r'HOSE ID\s*=\s*([\d\.\u00b1]+)', text, re.IGNORECASE)
    if id_match:
        info['id'] = id_match.group(1)
    elif feasibility_data.get('id'):
        info['id'] = feasibility_data['id']

    # Centerline Length: Handle various formats
    ctr_length_match = re.search(r'(?:APPROX\s+)?(?:CTRLINE\s+)?LENGTH\s*[=:]?\s*([\d\.]+)', text, re.IGNORECASE)
    if ctr_length_match:
        info['centerline_length'] = ctr_length_match.group(1)
    elif feasibility_data.get('centerline_length'):
        info['centerline_length'] = feasibility_data['centerline_length']

    # Burst pressure (looking for specific format)
    burst_match = re.search(r'Burst\s+Pressure\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:bar|BAR)', text, re.IGNORECASE)
    if burst_match:
        info['burst_pressure_bar'] = burst_match.group(1)
    elif feasibility_data.get('burst_pressure'):
        info['burst_pressure_bar'] = feasibility_data['burst_pressure']

    # Working pressure
    working_match = re.search(r'Working\s+Pressure\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:kPag|KPAG)', text, re.IGNORECASE)
    if working_match:
        info['working_pressure_kpag'] = working_match.group(1)

    # Additional measurements
    od_match = re.search(r'OD\s*[=:]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
    if od_match:
        info['od'] = od_match.group(1)

    thickness_match = re.search(r'THICKNESS\s*[=:]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
    if thickness_match:
        info['thickness'] = thickness_match.group(1)
        
    return info

def extract_coordinates(text):
    """Extracts P0, P1, P2... coordinates with stricter regex."""
    coords = {}
    valid_float_pattern = r'-?\d+\.?\d*'
    
    pattern = re.compile(
        r'^P(\d+)\s+(' + valid_float_pattern + r')\s+(' + valid_float_pattern + r')\s+(' + valid_float_pattern + r')(?:\s+' + valid_float_pattern + r')?$',
        re.MULTILINE
    )
    matches = pattern.findall(text)
    
    for match in matches:
        point_id = f"P{match[0]}"
        coords[point_id] = {
            'x': float(match[1]),
            'y': float(match[2]),
            'z': float(match[3])
        }
    
    # Sort by integer value of the point number
    sorted_keys = sorted(coords.keys(), key=lambda p: int(p[1:]))
    sorted_coords = [coords[key] for key in sorted_keys]
    return sorted_coords

def calculate_development_length(coords):
    """Calculates the total length by summing distances between consecutive points."""
    if len(coords) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        distance = math.sqrt(
            (p2['x'] - p1['x'])**2 +
            (p2['y'] - p1['y'])**2 +
            (p2['z'] - p1['z'])**2
        )
        total_length += distance
    
    return total_length

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF with fallback to Gemini Vision API."""
    logger.info("Starting PDF Text Extraction")
    
    # Step 1: Try direct text extraction
    logger.info("Attempting direct text extraction with PyMuPDF...")
    pdf_document = fitz.open("pdf", pdf_bytes)
    full_text = ""
    
    for page_num, page in enumerate(pdf_document):
        page_text = page.get_text()
        full_text += page_text
        logger.info(f"Page {page_num + 1}: Extracted {len(page_text)} characters")
    
    pdf_document.close()
    
    logger.info(f"Direct extraction found {len(full_text)} characters")
    
    # Step 2: If direct extraction yields little text, try Gemini Vision
    if len(full_text.strip()) < 100:
        logger.info("Direct extraction yielded limited text. Attempting Gemini Vision...")
        try:
            # Convert PDF to images one page at a time to manage memory
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf_path = temp_pdf.name
            
            # Initialize text collection
            full_text = ""
            pdf_document = fitz.open(temp_pdf_path)
            
            for page_num in range(len(pdf_document)):
                # Convert only the current page to an image
                images = convert_from_bytes(
                    pdf_bytes, 
                    first_page=page_num + 1, 
                    last_page=page_num + 1,
                    dpi=100,  # Lower DPI to reduce memory
                    fmt='jpeg',
                    thread_count=1  # Single thread to reduce memory usage
                )
                
                if images:
                    # Process with Gemini Vision
                    page_text = process_page_with_gemini(images[0])
                    if page_text:
                        full_text += page_text + "\n"
                    
                    # Clean up
                    for img in images:
                        img.close()
                    del images
            
            pdf_document.close()
            if 'temp_pdf_path' in locals():
                os.unlink(temp_pdf_path)
            
            logger.info(f"Text extraction complete. Found {len(full_text)} characters")
            return full_text
                
        except Exception as e:
            logger.error(f"Error during text extraction: {str(e)}")
            return full_text
    
    return full_text

# --- Configuration ---
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
except Exception as e:
    logger.error(f"API Key not found in environment variables: {e}")

# Initialize the Flask application
app = Flask(__name__, static_url_path='/static')
CORS(app)

# Configure static files with correct MIME types
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MIME_TYPES'] = {
    '.js': 'application/javascript',
    '.css': 'text/css'
}

# --- Helper function to initialize default results ---
def get_default_results():
    """Returns a dictionary with default values for analysis results."""
    return {
        "child_part": "Not Found",
        "description": "Not Found",
        "specification": "Not Found",
        "material": "Not Found",
        "od": "Not Found",
        "thickness": "Not Found",
        "centerline_length": "Not Found",
        "development_length_mm": "Not Found",
        "working_pressure_kpag": "Not Found",
        "burst_pressure_bar": "Not Found",
        "coordinates": [],
        "error": None
    }

# --- Load the material database from the CSV file ---
try:
    material_df = pd.read_csv('material_data.csv')
    material_df.columns = material_df.columns.str.strip()
    material_df['STANDARD'] = material_df['STANDARD'].str.strip()
    material_df['GRADE'] = material_df['GRADE'].astype(str).str.strip()
    logger.info("Material database loaded successfully.")
except FileNotFoundError:
    logger.error("material_data.csv not found. Please ensure the file exists.")
    material_df = pd.DataFrame()

# --- Enhanced function to analyze the PDF text using Gemini API ---
def analyze_drawing_with_gemini(pdf_bytes):
    logger.info("Starting Drawing Analysis")
    
    # Initialize results dictionary
    final_results = get_default_results()
    
    try:
        logger.info("Attempting text extraction...")
        
        # Extract text with detailed logging
        full_text = extract_text_from_pdf(pdf_bytes)
        
        if not full_text:
            logger.error("Text extraction failed")
            final_results["error"] = "Failed to extract text from PDF"
            return final_results
            
        logger.info(f"Text extraction successful. Length: {len(full_text)}")
        
        # Process the extracted text
        logger.info("Processing extracted text for specific information...")
        
        # Extract information using regex patterns
        regex_results = extract_specific_info(full_text)
        
        # Extract coordinates
        coordinates = extract_coordinates(full_text)
        
        # Merge the results, preferring regex results when available
        for key, value in regex_results.items():
            if value != "Not Found":
                final_results[key] = value
        
        # Add coordinates and calculate development length
        final_results["coordinates"] = coordinates
        if coordinates:
            final_results["development_length_mm"] = f"{calculate_development_length(coordinates):.2f}"
        
        # Use Gemini API only if essential information is missing
        if final_results["child_part"] == "Not Found" or final_results["specification"] == "Not Found":
            logger.info("Using Gemini API to extract missing information...")
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Analyze the following text extracted from a technical engineering drawing. Find:
            1. Part number (typically a 7-digit number followed by a 'C' and another digit)
            2. Standard (likely 'MPAPS F-30')
            3. Grade (short code that often follows the word 'GRADE')
            
            Respond ONLY with a valid JSON object containing these keys: "part_number", "standard", "grade".
            If you cannot find a value, use "Not Found".
            
            Text to analyze:
            ---
            {full_text[:4000]}  # Limit text size to avoid token limits
            ---
            """
            
            response = model.generate_content(prompt)
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response.text.strip())
            
            try:
                ai_results = json.loads(cleaned_response)
                
                # Update from AI results if regex didn't find them
                if final_results["child_part"] == "Not Found" and ai_results.get("part_number", "Not Found") != "Not Found":
                    final_results["child_part"] = ai_results["part_number"]
                    
                if final_results["specification"] == "Not Found" and ai_results.get("standard", "Not Found") != "Not Found":
                    final_results["specification"] = ai_results["standard"]
                    
                if final_results["material"] == "Not Found" and ai_results.get("grade", "Not Found") != "Not Found":
                    grade = ai_results["grade"]
                    standard = ai_results.get("standard", "")
                    
                    # Look up in material database
                    if standard and grade != "Not Found":
                        match = material_df[
                            material_df['STANDARD'].str.contains(standard, na=False, case=False) &
                            (material_df['GRADE'] == grade)
                        ]
                        if not match.empty:
                            final_results["material"] = match.iloc[0]['MATERIAL']
                        else:
                            final_results["material"] = f"GRADE {grade}"
            except json.JSONDecodeError:
                logger.warning("AI model returned a non-JSON response")
        
        logger.info("Analysis completed successfully")
        for key, value in final_results.items():
            if key != "coordinates":  # Skip coordinates to keep output clean
                logger.info(f"{key}: {value}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        final_results["error"] = f"An unexpected error occurred: {str(e)}"
    
    return final_results

# --- API endpoint for file analysis ---
@app.route('/api/analyze', methods=['POST'])
def upload_and_analyze():
    logger.info("New Analysis Request Started")
    mem_usage = get_memory_usage()
    if mem_usage is not None:
        logger.info(f"Initial memory usage: {mem_usage:.2f} MB")
    
    try:
        if 'drawing' not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['drawing']
        
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({"error": "No file selected"}), 400
            
        # Check file size before processing
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell() / (1024 * 1024)  # Size in MB
        file.seek(0)  # Reset file pointer
        
        if file_size > 5:
            logger.error(f"File too large ({file_size:.1f}MB)")
            return jsonify({
                "error": "File too large. Please upload a PDF smaller than 5MB"
            }), 400
            
        if file and file.filename.lower().endswith('.pdf'):
            logger.info(f"Processing file: {file.filename}")
            pdf_bytes = file.read()
            logger.info(f"File size: {len(pdf_bytes)} bytes")
            
            mem_before = get_memory_usage()
            if mem_before is not None:
                logger.info(f"Memory before analysis: {mem_before:.2f} MB")
            
            try:
                analysis_results = analyze_drawing_with_gemini(pdf_bytes)
                
                mem_after = get_memory_usage()
                if mem_after is not None:
                    logger.info(f"Final memory usage: {mem_after:.2f} MB")
                
                return jsonify(analysis_results)
            except MemoryError:
                mem_usage = get_memory_usage()
                if mem_usage is not None:
                    logger.error(f"Memory error occurred. Current usage: {mem_usage:.2f} MB")
                return jsonify({
                    "error": "Server memory limit reached. Please try a smaller or simpler PDF file."
                }), 507
            except ValueError as ve:
                logger.error(f"Validation error: {str(ve)}")
                return jsonify({"error": str(ve)}), 400
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                mem_usage = get_memory_usage()
                if mem_usage is not None:
                    logger.error(f"Memory at error: {mem_usage:.2f} MB")
                return jsonify({
                    "error": "An error occurred while processing the file",
                    "details": str(e)
                }), 500
        else:
            logger.error("Invalid file type")
            return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400
            
    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        return jsonify({"error": "Error processing file upload"}), 500

# --- Route for the main webpage ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Run the application ---
if __name__ == '__main__':
    app.run(debug=True, threaded=True)