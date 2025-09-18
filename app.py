import os
import re
import json
import math
import base64
import pandas as pd
import fitz  # PyMuPDF
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Added for CORS support
from pdf2image import convert_from_bytes
from PIL import Image
import io
import tempfile

# Try to import psutil with fallback
try:
    import psutil
except ImportError:
    psutil = None
    print("Warning: psutil not found. Memory logging disabled.")
import os

def get_memory_usage():
    """Returns current memory usage in MB, or None if psutil is not available."""
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception as e:
        print(f"Error getting memory usage: {str(e)}")
        return None

# --- Helper Functions for Detailed Analysis ---
def extract_specific_info(text):
    """
    Extracts key-value data with more flexible regex patterns.
    Now updated to handle formats from the 4403886C2 drawing.
    """
    info = {
        'child_part': "Not Found",
        'description': "Not Found",
        'specification': "Not Found",
        'material': "Not Found",
        'id': "Not Found",
        'centerline_length': "Not Found",
        'burst_pressure_bar': "Not Found",
        'working_pressure_kpag': "Not Found",
        'development_length_mm': "Not Found",
        'od': "Not Found",
        'thickness': "Not Found"
    }

    # Part Number: Find the specific C-number format directly
    part_num_match = re.search(r'(\d{7}[Cc]\d)', text, re.IGNORECASE)
    if part_num_match:
        info['child_part'] = part_num_match.group(1)

    # Description: Find the "HOSE, ..." pattern
    desc_match = re.search(r'(HOSE,[\s\w,]+)', text, re.IGNORECASE)
    if desc_match:
        info['description'] = desc_match.group(1).strip()
        
    # Specification: Find MPAPS F-30
    spec_match = re.search(r'(MPAPS\s+F-30)', text, re.IGNORECASE)
    if spec_match:
        info['specification'] = spec_match.group(0)

    # Material: Find the Grade
    material_match = re.search(r'GRADE\s+([\w\d]+)', text, re.IGNORECASE)
    if material_match:
        info['material'] = f"GRADE {material_match.group(1)}"

    # ID: Look for "HOSE ID" with an equals sign
    id_match = re.search(r'HOSE ID\s*=\s*([\d\.]+)', text, re.IGNORECASE)
    if id_match:
        info['id'] = id_match.group(1)

    # Centerline Length: Handle various formats
    ctr_length_match = re.search(r'(?:APPROX\s+)?(?:CTRLINE\s+)?LENGTH\s*[=:]?\s*([\d\.]+)', text, re.IGNORECASE)
    if ctr_length_match:
        info['centerline_length'] = ctr_length_match.group(1)

    # Burst pressure (looking for specific format)
    burst_match = re.search(r'Burst\s+Pressure\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:bar|BAR)', text, re.IGNORECASE)
    if burst_match:
        info['burst_pressure_bar'] = burst_match.group(1)

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
    """
    Extracts P0, P1, P2... coordinates.
    Updated with a stricter regex to only match valid floating-point numbers.
    """
    coords = {}
    # This pattern matches an optional hyphen ONLY at the start of a number.
    # It prevents matching strings with hyphens in the middle.
    valid_float_pattern = r'-?\d+\.?\d*'
    
    # The overall pattern now looks for P<num> followed by 3 or 4 valid numbers.
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

# Function to extract text with detailed logging
def extract_text_from_pdf(pdf_bytes):
    """Extract text with PyMuPDF first, fallback to Gemini Vision."""
    print("\n=== Starting PDF Text Extraction ===")
    
    # Step 1: Direct extraction
    print("\n1. Attempting direct text extraction with PyMuPDF...")
    pdf_document = fitz.open("pdf", pdf_bytes)
    full_text = ""
    page_count = 0
    
    for page in pdf_document:
        page_count += 1
        page_text = page.get_text()
        full_text += page_text
        print(f"  - Page {page_count}: Extracted {len(page_text)} characters")
    
    pdf_document.close()
    
    print(f"\nDirect Extraction Results:")
    print("------------------------")
    print(full_text)
    print("------------------------")
    print(f"Total characters extracted: {len(full_text)}")
    
    # Step 2: If direct extraction yields little text, use Gemini Vision
    if len(full_text.strip()) < 100:
        print("\n2. Direct extraction limited. Using Gemini Vision...")
        try:
            # Lightweight conversion (first page for simplicity)
            page_image = convert_from_bytes(pdf_bytes, 
                                         first_page=1, 
                                         last_page=1, 
                                         dpi=100, 
                                         fmt='jpeg')
            
            # Base64 encode
            buffered = io.BytesIO()
            page_image[0].save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Gemini model configuration
            model = genai.GenerativeModel('gemini-1.5-pro')  # Using pro for better accuracy
            
            # Prompt for extraction
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
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ])
            
            # Update full text with Gemini Vision results
            full_text = response.text
            
            # Cleanup
            del page_image, buffered
            
            print("\nGemini Vision Results:")
            print("------------------------")
            print(full_text)
            print("------------------------")
                
                for i, image in enumerate(images):
                    print(f"  - Processing page {i+1} with OCR...")
                    page_text = pytesseract.image_to_string(image)
                    ocr_text += page_text + "\n"
                    print(f"    Extracted {len(page_text)} characters")
                
                print("\nOCR Results:")
                print("-----------")
                print(ocr_text)
                print("-----------")
                print(f"Total OCR characters: {len(ocr_text)}")
                
                return ocr_text if ocr_text.strip() else full_text
                
        except Exception as e:
            print(f"\nError during OCR processing: {str(e)}")
            return full_text
    
    return full_text

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
CORS(app)  # Enable CORS for all routes

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
    print("Material database loaded successfully.")
except FileNotFoundError:
    print("Error: material_data.csv not found. Please ensure the file exists.")
    material_df = pd.DataFrame()

# --- NEW: Enhanced function to analyze the PDF text using Gemini API ---
def analyze_drawing_with_gemini(pdf_bytes):
    print("\n=== Starting Drawing Analysis ===")
    
    # Initialize results dictionary before the try block
    final_results = {
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
    
    try:
        print("\n----------- STARTING PDF ANALYSIS -----------")
        print("[1/3] Attempting text extraction...")
        
        # Extract text with detailed logging
        full_text = extract_text_from_pdf(pdf_bytes)
        
        print("\n----------- RAW EXTRACTED TEXT -----------")
        print(full_text if full_text else "[No text extracted]")
        print("------------------------------------------")
        
        if not full_text:
            print("[ERROR] Text extraction failed")
            return {
                "burst_pressure_bar": "Not Found",
                "error": "Failed to extract text from PDF",
                "coordinates": []
            }
            
        print("[2/3] Text extraction successful. Length:", len(full_text))
        
        # Process the extracted text
        print("[3/3] Processing extracted text for specific information...")
        # --- Step 1: Extract text from PDF ---
        print("Attempting pattern matching on extracted text...")
        pdf_document = fitz.open("pdf", pdf_bytes)
        full_text = ""
        for page in pdf_document:
            full_text += page.get_text()
        pdf_document.close()
        print(f"Direct extraction found {len(full_text)} characters.")

        # --- OCR Fallback Logic ---
        if not full_text.strip():
            print("No selectable text found. Attempting memory-efficient OCR fallback.")
            mem_usage = get_memory_usage()
            if mem_usage is not None:
                print(f"Initial memory usage: {mem_usage:.2f} MB")
            full_text = ""

            # Check file size before proceeding
            if len(pdf_bytes) > 5 * 1024 * 1024:  # 5MB limit
                raise ValueError("File too large for OCR processing (>5MB)")

            # Use a temporary file to avoid holding everything in memory
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
                temp.write(pdf_bytes)
                temp.flush()
                
                # Get the number of pages
                page_count = len(fitz.open(temp.name))
                
                # Process one page at a time
                for i in range(page_count):
                    print(f"Converting and processing OCR for page {i+1}/{page_count}...")
                    mem_usage = get_memory_usage()
                    if mem_usage is not None:
                        print(f"Memory before page {i+1}: {mem_usage:.2f} MB")
                    
                    try:
                        # Convert only the single page to an image with optimized parameters
                        page_image = convert_from_bytes(pdf_bytes, 
                                                      first_page=i+1, 
                                                      last_page=i+1, 
                                                      dpi=150,  # Lower DPI
                                                      fmt='jpeg',  # Use JPEG format
                                                      thread_count=1)  # Reduce thread usage
                        
                        if page_image:
                            # Preprocess image to reduce memory usage
                            img = page_image[0].convert('L')  # Convert to grayscale
                            # Resize to 75% while maintaining aspect ratio
                            new_width = int(img.width * 0.75)
                            new_height = int(img.height * 0.75)
                            img = img.resize((new_width, new_height))
                            
                            # OCR with optimized configuration
                            page_text = pytesseract.image_to_string(
                                img,
                                config='--oem 3 --psm 6'  # Assume uniform text block
                            )
                            full_text += page_text + "\n"
                            
                            # Clean up to free memory
                            img.close()
                            page_image[0].close()
                            del img
                            del page_image
                            
                            # Clean Tesseract temporary files
                            pytesseract.pytesseract.cleanup('/tmp/tess*')
                            
                    except Exception as page_e:
                        print(f"Could not process page {i+1}: {page_e}")
                        mem_usage = get_memory_usage()
                        if mem_usage is not None:
                            print(f"Memory at error: {mem_usage:.2f} MB")
                        continue  # Move to the next page

            print(f"OCR processing complete. Found {len(full_text)} characters.")
            if not full_text.strip():
                final_results["error"] = "No text could be extracted via OCR."
                return final_results

        # --- Step 2: Prepare the prompt for the Gemini API ---
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

        # --- Step 3: Call the Gemini API ---
        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's valid JSON
        # The model might sometimes wrap the JSON in ```json ... ```
        cleaned_response = re.sub(r'```json\s*|\s*```', '', response.text.strip())
        
        # Parse the JSON response from the model
        ai_results = json.loads(cleaned_response)

        # Get regex-based results
        print("\nExtracting information using regex patterns...")
        regex_results = extract_specific_info(full_text)
        
        # Extract coordinates
        print("Extracting coordinates...")
        coordinates = extract_coordinates(full_text)
        
        # Merge the results, preferring regex results when available
        final_results = get_default_results()
        
        # Update from regex results
        for key, value in regex_results.items():
            if value != "Not Found":
                final_results[key] = value
        
        # Update from AI results if regex didn't find them
        if final_results["child_part"] == "Not Found" and ai_results.get("part_number", "Not Found") != "Not Found":
            final_results["child_part"] = ai_results["part_number"]
            
        if final_results["specification"] == "Not Found" and ai_results.get("standard", "Not Found") != "Not Found":
            final_results["specification"] = ai_results["standard"]
            
        if final_results["material"] == "Not Found" and ai_results.get("grade", "Not Found") != "Not Found":
            grade = ai_results["grade"]
            standard = ai_results.get("standard", "")
            
            # Look up in material database if we have both standard and grade
            if standard and grade != "Not Found":
                match = material_df[
                    material_df['STANDARD'].str.contains(standard, na=False, case=False) &
                    (material_df['GRADE'] == grade)
                ]
                if not match.empty:
                    final_results["material"] = match.iloc[0]['MATERIAL']
                else:
                    final_results["material"] = f"GRADE {grade}"
        
        # Add coordinates and calculate development length
        final_results["coordinates"] = coordinates
        if coordinates:
            final_results["development_length_mm"] = f"{calculate_development_length(coordinates):.2f}"
            
        print("\nAnalysis Results:")
        for key, value in final_results.items():
            if key != "coordinates":  # Skip coordinates to keep output clean
                print(f"{key}: {value}")

    except json.JSONDecodeError:
        final_results["error"] = "AI model returned a non-JSON response. Please try again."
    except Exception as e:
        final_results["error"] = f"An unexpected error occurred: {str(e)}"
    
    return final_results

# --- API endpoint for file analysis (now uses the Gemini function) ---
@app.route('/api/analyze', methods=['POST'])
def upload_and_analyze():
    print("\n=== New Analysis Request Started ===")
    mem_usage = get_memory_usage()
    if mem_usage is not None:
        print(f"Initial memory usage: {mem_usage:.2f} MB")
    
    try:
        if 'drawing' not in request.files:
            print("Error: No file part in request")
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['drawing']
    except Exception as e:
        print(f"Error handling file upload: {str(e)}")
        return jsonify({"error": "Error processing file upload"}), 500
        
        if file.filename == '':
            print("Error: No file selected")
            return jsonify({"error": "No file selected"}), 400
            
        # Check file size before processing
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell() / (1024 * 1024)  # Size in MB
        file.seek(0)  # Reset file pointer
        
        if file_size > 5:
            print(f"Error: File too large ({file_size:.1f}MB)")
            return jsonify({
                "error": "File too large. Please upload a PDF smaller than 5MB"
            }), 400
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        print(f"\nProcessing file: {file.filename}")
        pdf_bytes = file.read()
        print(f"File size: {len(pdf_bytes)} bytes")
        print(f"Memory before analysis: {get_memory_usage():.2f} MB")
        
        try:
            analysis_results = analyze_drawing_with_gemini(pdf_bytes)
            print("\nAnalysis completed. Results:", json.dumps(analysis_results, indent=2))
            print(f"Final memory usage: {get_memory_usage():.2f} MB")
            return jsonify(analysis_results)
        except MemoryError:
            mem_usage = get_memory_usage()
            if mem_usage is not None:
                print(f"Memory error occurred. Current usage: {mem_usage:.2f} MB")
            return jsonify({
                "error": "Server memory limit reached. Please try a smaller or simpler PDF file."
            }), 507  # 507 Insufficient Storage
        except ValueError as ve:
            print(f"Validation error: {str(ve)}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            mem_usage = get_memory_usage()
            if mem_usage is not None:
                print(f"Memory at error: {mem_usage:.2f} MB")
            return jsonify({
                "error": "An error occurred while processing the file",
                "details": str(e)
            }), 500
    else:
        print("Error: Invalid file type")
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

# --- Route for the main webpage (no change) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Run the application (no change) ---
if __name__ == '__main__':
    app.run(debug=True)