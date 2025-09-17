import os
import re
import json
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

# --- NEW: Function to analyze the PDF text using Gemini API ---
def analyze_drawing_with_gemini(pdf_bytes):
    print("--- New analysis request received ---")
    results = {
        "part_number": "Not Found",
        "standard": "Not Found",
        "grade": "Not Found",
        "material": "Not Found",
        "error": None
    }
    
    try:
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
    if 'drawing' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['drawing']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        pdf_bytes = file.read()
        analysis_results = analyze_drawing_with_gemini(pdf_bytes) # Use the new Gemini function
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