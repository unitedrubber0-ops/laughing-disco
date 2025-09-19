import os
import re
import json
import math
import base64
import pandas as pd
import fitz  # PyMuPDF
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, send_from_directory, current_app
from flask_cors import CORS
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image
import io
import tempfile
import logging
from werkzeug.exceptions import RequestTimeout, ServiceUnavailable
import signal
from functools import wraps
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_static_url():
    """Get the static file URL based on environment"""
    if os.environ.get('STATIC_URL'):
        return os.environ.get('STATIC_URL')
    return '/static'

# Initialize Flask app with dynamic static url path
app = Flask(__name__, static_url_path=get_static_url())
CORS(app)

def timeout_handler(timeout=300):
    """Decorator to handle timeouts for long-running functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = []
            error = []
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    error.append(e)
                    
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {timeout} seconds")
                raise RequestTimeout(f"Operation timed out after {timeout} seconds")
            
            if error:
                logger.error(f"Error in {func.__name__}: {str(error[0])}")
                raise error[0]
                
            return result[0] if result else None
        return wrapper
    return decorator

def check_memory():
    """Check if memory usage is too high."""
    memory_usage = get_memory_usage()
    if memory_usage and memory_usage > 1024:  # More than 1GB
        logger.warning(f"High memory usage detected: {memory_usage:.2f} MB")
        raise ServiceUnavailable("Server is under high load, please try again later")

## --- Load the material database from the Excel file ---
try:
    material_df = pd.read_excel('MATERIAL WITH STANDARD.xlsx', sheet_name='Sheet1')
    
    # Clean up column names and data
    material_df.columns = material_df.columns.str.strip()
    material_df['STANDARD'] = material_df['STANDARD'].astype(str).str.strip()
    material_df['GRADE'] = material_df['GRADE'].astype(str).str.strip()
    material_df['MATERIAL'] = material_df['MATERIAL'].astype(str).str.strip()
    
    # Handle multi-line materials by replacing newlines with spaces
    material_df['MATERIAL'] = material_df['MATERIAL'].str.replace('\n', ' ', regex=False)
    
    logger.info("Material database loaded successfully.")
    logger.info(f"Found {len(material_df)} material entries")
except Exception as e:
    logger.error(f"Error loading material database: {str(e)}")
    material_df = pd.DataFrame()

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

@timeout_handler(timeout=180)  # 3 minutes timeout for image processing
def process_page_with_gemini(page_image):
    """Process a page image using Gemini Vision API.
    
    Args:
        page_image: A PIL Image object containing the page to process
        
    Returns:
        str: Extracted text from the image, or empty string if extraction fails
    
    Raises:
        RequestTimeout: If processing takes longer than timeout
        ServiceUnavailable: If server is under high load
    """
    try:
        # Check memory usage before processing
        check_memory()
        
        logger.info("Starting Gemini Vision processing...")
        
        # Convert PIL Image to bytes with compression
        buffered = io.BytesIO()
        page_image.save(buffered, format="JPEG", quality=85)  # Reduced quality for better performance
        img_bytes = buffered.getvalue()
        
        # Check image size and compress if too large
        img_size_mb = len(img_bytes) / (1024 * 1024)
        logger.info(f"Image size: {img_size_mb:.2f} MB")
        
        if img_size_mb > 4:  # If image is larger than 4MB
            logger.warning("Image too large, applying additional compression")
            img = Image.open(io.BytesIO(img_bytes))
            max_size = (2000, 2000)  # Maximum dimensions
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=80)
            img_bytes = buffered.getvalue()
            logger.info(f"Compressed image size: {len(img_bytes) / (1024 * 1024):.2f} MB")
        
        # Configure and use Gemini model
        logger.info("Initializing Gemini Vision model...")
        try:
            model = genai.GenerativeModel('gemini-pro-vision')  # Using pro-vision for better stability
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load Gemini model: {str(e)}")
            raise ServiceUnavailable("Model initialization failed")
        
        prompt = """Extract all readable text from this engineering drawing image. Focus on:
        - Part numbers (e.g., 7 digits + C + digit)
        - Material standards (e.g., MPAPS F-30)
        - Grades
        - Coordinates (P0-Pn with X/Y/Z values)
        - Dimensions and measurements
        - Pressure specifications
        - Part descriptions
        Output as clean, structured text with clear labeling."""
        
        # Prepare the content parts
        content_parts = [
            prompt,
            Image.open(io.BytesIO(img_bytes)) # Pass PIL Image object directly
        ]
        
        # Generate content with image
        logger.info("Sending request to Gemini Vision API...")
        try:
            response = model.generate_content(
                content_parts,
                generation_config={'timeout': 60}  # Increase timeout to 60 seconds
            )
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return ""
            
        # Clean up
        del buffered, img_bytes
        import gc
        gc.collect()  # Force garbage collection
        
        if response and response.text:
            logger.info(f"Gemini Vision processing successful. Extracted {len(response.text)} characters")
            return response.text
        else:
            logger.warning("Gemini Vision processing returned empty response")
            return ""
        
    except Exception as e:
        logger.error(f"Error in Gemini Vision processing: {str(e)}")
        return ""

# --- Helper Functions for Detailed Analysis ---
def load_material_data():
    """Load data from MATERIAL WITH STANDARD.xlsx worksheet 'Sheet1'"""
    try:
        # Read the Excel file
        df = pd.read_excel('MATERIAL WITH STANDARD.xlsx', sheet_name='Sheet1')
        
        # Return the entire dataframe for material lookup
        return df
    except Exception as e:
        logger.error(f"Error loading material data: {str(e)}")
        return pd.DataFrame()

def extract_specific_info(text):
    """Extracts key-value data with more flexible and robust regex patterns."""
    info = {
        'child_part': "Not Found",
        'description': "Not Found",
        'specification': "Not Found",
        'material': "Not Found",
        'od': "Not Found",
        'thickness': "Not Found",
        'centerline_length': "Not Found",
        'working_pressure_kpag': "Not Found", # Added this field
        'burst_pressure_bar': "Not Found" # Kept for consistency
        # development_length is calculated separately
    }
    
    # Part Number: Prioritize the main part number from the title block
    part_num_match = re.search(r'PART NO\.\s*([0-9A-Z]+X[0-9])', text, re.IGNORECASE)
    if part_num_match:
        info['child_part'] = part_num_match.group(1)
    else:
        # Fallback to the stamped part number if the main one isn't found
        part_num_match_fallback = re.search(r'(\d{7}[Cc]\d)', text, re.IGNORECASE)
        if part_num_match_fallback:
            info['child_part'] = part_num_match_fallback.group(1)

    # Description: Find the "HOSE, ..." pattern from the title block
    desc_match = re.search(r'NAME\s+(HOSE,[\s\w,]+TO[\s\w,]+)', text, re.IGNORECASE)
    if desc_match:
        info['description'] = desc_match.group(1).strip()
        
    # Specification: Look for the specific F-30 standard mentioned in the material block
    spec_match = re.search(r'PER\s+(MPAPS\s*F[- ]*30)', text, re.IGNORECASE)
    if spec_match:
        info['specification'] = spec_match.group(1).replace(" ", "")
    
    # Material: Find the Grade and look up in the database
    # This logic remains largely the same but will now work better with the correct specification
    material_match = re.search(r'GRADE\s+([\w\d]+)', text, re.IGNORECASE)
    if material_match:
        grade = material_match.group(1)
        if info['specification'] != "Not Found":
            normalized_spec = info['specification'].upper().replace("-", "").replace(" ", "")
            match = material_df[
                material_df['STANDARD'].apply(lambda x: str(x).upper().replace("-", "").replace(" ", "").replace("/", "")).str.contains(normalized_spec, na=False) &
                (material_df['GRADE'].astype(str).str.upper() == grade.upper())
            ]
            if not match.empty:
                info['material'] = match.iloc[0]['MATERIAL']
            else:
                info['material'] = f"GRADE {grade} (Material not found for {info['specification']})"
        else:
            info['material'] = f"GRADE {grade}"

    # OD and Thickness from the specific table
    od_match = re.search(r'TUBING\s+OD\s*=\s*([\d\.]+)', text, re.IGNORECASE)
    if od_match:
        info['od'] = od_match.group(1)

    thickness_match = re.search(r'WALL\s+THICKNESS\s*=\s*([\d\.]+)', text, re.IGNORECASE)
    if thickness_match:
        info['thickness'] = thickness_match.group(1)

    # Centerline Length from the table
    ctr_length_match = re.search(r'APPROX\s+CTRLINE\s+LENGTH\s*=\s*([\d\.]+)', text, re.IGNORECASE)
    if ctr_length_match:
        info['centerline_length'] = ctr_length_match.group(1)

    # Working pressure (looking for "WP" abbreviation)
    working_match = re.search(r'(?:WP|Working\s+Pressure)\s+(\d+)\s*kPag', text, re.IGNORECASE)
    if working_match:
        info['working_pressure_kpag'] = working_match.group(1)

    # ID: Look for "HOSE ID" (no change needed here, but kept for completeness)
    id_match = re.search(r'HOSE ID\s*=\s*([\d\.±]+)', text, re.IGNORECASE)
    if id_match:
        # Assuming you might want to add an 'id' field to your info dict
        info['id'] = id_match.group(1)

    # Burst pressure
    burst_match = re.search(r'Burst\s+Pressure\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:bar|BAR)', text, re.IGNORECASE)
    if burst_match:
        info['burst_pressure_bar'] = burst_match.group(1)

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
    """Extract text from PDF using Gemini Vision API."""
    logger.info("Starting PDF Text Extraction with Gemini Vision")
    
    full_text = ""
    temp_pdf_path = None  # Initialize path variable
    
    try:
        # Create a temporary file to work with pdf2image
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        # Initialize text collection for Gemini
        vision_text = ""
        
        # Convert PDF pages to images using the file path
        page_images = convert_from_path(
            temp_pdf_path,
            dpi=150,  # Higher DPI for better OCR
            fmt='jpeg',
            thread_count=1
        )
        
        for image in page_images:
            page_text = process_page_with_gemini(image)
            if page_text:
                vision_text += page_text + "\n"
            image.close()  # Clean up image object memory
        
        full_text = vision_text  # Use vision results

    except Exception as e:
        # Log the specific error during the Gemini process
        logger.error(f"Error during text extraction with Gemini Vision: {str(e)}")
        # Keep full_text empty if processing fails
    finally:
        # This block ALWAYS runs, ensuring the temporary file is deleted
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
            logger.info(f"Deleted temporary file: {temp_pdf_path}")
    
    logger.info(f"Text extraction complete. Found {len(full_text)} characters")
    return full_text

# --- Configuration ---
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")

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

# --- Enhanced function to analyze the PDF text using Gemini API ---
def analyze_drawing_with_gemini(pdf_bytes):
    logger.info("Starting Drawing Analysis")
    mem_usage = get_memory_usage()
    if mem_usage is not None:
        logger.info(f"Memory at start of analysis: {mem_usage:.2f} MB")
    
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
            
            # Using 'gemini-1.5-flash-latest' for text-only processing as well
            model = genai.GenerativeModel('gemini-1.5-flash-latest') 
            
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
                        # Normalize standard for better matching
                        normalized_standard_ai = str(standard).upper().replace("-", "").replace(" ", "").replace("/", "")
                        
                        match = material_df[
                            material_df['STANDARD'].apply(lambda x: str(x).upper().replace("-", "").replace(" ", "").replace("/", "")).str.contains(normalized_standard_ai, na=False) &
                            (material_df['GRADE'].astype(str).str.upper() == grade.upper())
                        ]
                        if not match.empty:
                            final_results["material"] = match.iloc[0]['MATERIAL']
                        else:
                            final_results["material"] = f"GRADE {grade}"
            except json.JSONDecodeError:
                logger.warning(f"AI model returned a non-JSON response: {cleaned_response}")
        
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
    """Handle PDF upload and analysis with comprehensive error handling and monitoring."""
    request_id = f"req_{threading.get_ident()}"
    
    try:
        logger.info(f"=== New Analysis Request Started (ID: {request_id}) ===")
        check_memory()  # Check memory before processing
        
        # Validate request
        if 'drawing' not in request.files:
            logger.error(f"[{request_id}] No file part in request")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['drawing']
        if file.filename == '':
            logger.error(f"[{request_id}] No file selected")
            return jsonify({"error": "No file selected"}), 400
        
        # Read and validate file content
        try:
            file_content = file.read()
            file_size_mb = len(file_content) / (1024 * 1024)
        except Exception as e:
            logger.error(f"[{request_id}] Failed to read file: {str(e)}")
            return jsonify({"error": "Failed to read uploaded file"}), 400
        
        # Size validation
        if file_size_mb > 5:
            logger.error(f"[{request_id}] File too large ({file_size_mb:.1f}MB)")
            return jsonify({
                "error": f"File too large ({file_size_mb:.1f}MB)",
                "limit": "5MB",
                "uploaded": f"{file_size_mb:.1f}MB"
            }), 413
            
        # File type validation and processing
        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"[{request_id}] Invalid file type: {file.filename}")
            return jsonify({
                "error": "Invalid file type. Please upload a PDF.",
                "uploaded": file.filename
            }), 400
            
        logger.info(f"[{request_id}] Processing file: {file.filename} ({file_size_mb:.1f}MB)")
        
        # Monitor memory throughout processing
        mem_before = get_memory_usage()
        if mem_before is not None:
            logger.info(f"[{request_id}] Memory before analysis: {mem_before:.2f} MB")
            if mem_before > 1024:  # 1GB limit
                raise ServiceUnavailable("Server is under high load")
        
        # Process the PDF with timeout handling
        try:
            with current_app.app_context():
                analysis_results = analyze_drawing_with_gemini(file_content)
        except RequestTimeout:
            logger.error(f"[{request_id}] Analysis timed out")
            return jsonify({
                "error": "Analysis timed out",
                "suggestion": "Try a smaller PDF or try again later"
            }), 504
        
        # Final memory check and cleanup
        mem_after = get_memory_usage()
        if mem_after is not None:
            logger.info(f"[{request_id}] Analysis complete. Final memory: {mem_after:.2f} MB")
            
        return jsonify(analysis_results)
            
    except MemoryError:
        logger.error(f"[{request_id}] MemoryError during analysis", exc_info=True)
        return jsonify({
            "error": "Memory limit exceeded",
            "suggestion": "Try a smaller PDF"
        }), 507
        
    except ServiceUnavailable as e:
        logger.error(f"[{request_id}] Service unavailable: {str(e)}")
        return jsonify({
            "error": str(e),
            "retry_after": "60 seconds"
        }), 503
        
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled error in /api/analyze: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "request_id": request_id
        }), 500
        
    finally:
        logger.info(f"=== Request {request_id} Completed ===")
        if 'file_content' in locals():
            del file_content  # Explicit cleanup of large objects
        mem_usage = get_memory_usage()
        if mem_usage:
            logger.error(f"Memory at crash: {mem_usage:.2f} MB")
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

# --- Route for static files ---
@app.route('/static/<path:filename>')
def send_static(filename):
    return send_from_directory('static', filename)

# --- Route for the main webpage ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Run the application ---
if __name__ == '__main__':
    app.run(debug=True, threaded=True)