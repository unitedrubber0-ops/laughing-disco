import os
import re
import json
import math
import pandas as pd
import fitz  # PyMuPDF
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Added for CORS support
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import tempfile

# --- Helper Functions for Detailed Analysis ---
def extract_specific_info(text):
    """Extracts key-value data from the drawing text using regex."""
    info = {
        'child_part': 'Not Found',
        'description': 'Not Found',
        'specification': 'Not Found',
        'material': 'Not Found',
        'od': 'Not Found',
        'thickness': 'Not Found',
        'centerline_length': 'Not Found',
        'working_pressure_kpag': 'Not Found'
    }

    # Part number pattern (e.g., 4717736X1)
    part_num_match = re.search(r'(\d{7}[Cc]\d)', text)
    if part_num_match:
        info['child_part'] = part_num_match.group(1)

    # Standard pattern (e.g., MPAPS F-30)
    spec_match = re.search(r'MPAPS\s+F-30', text)
    if spec_match:
        info['specification'] = spec_match.group(0)

    # Grade pattern
    grade_match = re.search(r'GRADE\s+(\w+)', text)
    if grade_match:
        info['material'] = f"GRADE {grade_match.group(1)}"

    # Additional measurements
    od_match = re.search(r'OD[:\s]+(\d+\.?\d*)', text)
    if od_match:
        info['od'] = od_match.group(1)

    thickness_match = re.search(r'THICKNESS[:\s]+(\d+\.?\d*)', text)
    if thickness_match:
        info['thickness'] = thickness_match.group(1)

    length_match = re.search(r'LENGTH[:\s]+(\d+\.?\d*)', text)
    if length_match:
        info['centerline_length'] = length_match.group(1)

    return info

def extract_coordinates(text):
    """Extracts P0, P1, P2... coordinates from the text."""
    coords = {}
    # Pattern for finding coordinates (P0 X Y Z format)
    pattern = re.compile(r'P(\d)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)')
    matches = pattern.findall(text)
    
    for match in matches:
        point_id = f"P{match[0]}"
        coords[point_id] = {
            'x': float(match[1]),
            'y': float(match[2]),
            'z': float(match[3])
        }
    
    return sorted([coords[f'P{i}'] for i in range(len(coords)) if f'P{i}' in coords], 
                 key=lambda x: int(next(k[1] for k in coords.keys() if coords[k] == x)))

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
    """Extract text from PDF with fallback to OCR and detailed logging."""
    print("\n=== Starting PDF Text Extraction ===")
    
    # Step 1: Try direct text extraction
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
    
    # Step 2: If direct extraction yields little text, try OCR
    if len(full_text.strip()) < 100:
        print("\n2. Direct extraction yielded limited text. Attempting OCR...")
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf.flush()
                
                # Convert PDF pages to images one at a time
                ocr_text = ""
                images = convert_from_bytes(pdf_bytes)
                
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
    results = {
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
        # Extract text with detailed logging
        full_text = extract_text_from_pdf(pdf_bytes)
        if not full_text:
            return {
                "burst_pressure_bar": "Not Found",
                "error": "Failed to extract text from PDF",
                "coordinates": []
            }
            
        # Process the extracted text
        # --- Step 1: Extract text from PDF ---
        print("Attempting direct text extraction...")
        pdf_document = fitz.open("pdf", pdf_bytes)
        full_text = ""
        for page in pdf_document:
            full_text += page.get_text()
        pdf_document.close()
        print(f"Direct extraction found {len(full_text)} characters.")

        # --- OCR Fallback Logic ---
        if not full_text.strip():
            print("No selectable text found. Attempting memory-efficient OCR fallback.")
            full_text = ""
            # Use a temporary file to avoid holding everything in memory
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
                temp.write(pdf_bytes)
                temp.flush()
                
                # Get the number of pages
                page_count = len(fitz.open(temp.name))
                
                # Process one page at a time
                for i in range(page_count):
                    print(f"Converting and processing OCR for page {i+1}/{page_count}...")
                    try:
                        # Convert only the single page to an image
                        page_image = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1)
                        if page_image:
                            page_text = pytesseract.image_to_string(page_image[0])
                            full_text += page_text + "\n"
                    except Exception as page_e:
                        print(f"Could not process page {i+1}: {page_e}")
                        continue # Move to the next page

            print(f"OCR processing complete. Found {len(full_text)} characters.")
            if not full_text.strip():
                results["error"] = "No text could be extracted via OCR."
                return results

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
        extracted_data = json.loads(cleaned_response)

        results["part_number"] = extracted_data.get("part_number", "Not Found")
        results["standard"] = extracted_data.get("standard", "Not Found")
        results["grade"] = extracted_data.get("grade", "Not Found")

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
    print("\n=== New Analysis Request Started ===")
    
    if 'drawing' not in request.files:
        print("Error: No file part in request")
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['drawing']
    
    if file.filename == '':
        print("Error: No file selected")
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        print(f"\nProcessing file: {file.filename}")
        pdf_bytes = file.read()
        print(f"File size: {len(pdf_bytes)} bytes")
        
        analysis_results = analyze_drawing_with_gemini(pdf_bytes)
        print("\nAnalysis completed. Results:", json.dumps(analysis_results, indent=2))
        return jsonify(analysis_results)
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