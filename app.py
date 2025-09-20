import os
import re
import json
import pandas as pd
import fitz  # PyMuPDF
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, send_file

def extract_od_from_text(text):
    dimensions = {}
    od_match = re.search(r'TUBING OD[^\d]*(\d+\.?\d*)', text, re.IGNORECASE)
    if od_match:
        dimensions["od1"] = od_match.group(1)
    
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

# --- Function to calculate development length based on COSTING TOOLS.xlsx ---
def calculate_development_length(dimensions):
    """
    Calculate development length based on the formula in COSTING TOOLS.xlsx
    Formula: Arc Length = 2 * π * radius * (angle_degrees / 360)
    """
    try:
        # Try to use radius and angle if available
        if (dimensions["radius"] != "Not Found" and 
            dimensions["angle"] != "Not Found" and
            dimensions["radius"].replace('.', '', 1).isdigit() and
            dimensions["angle"].replace('.', '', 1).isdigit()):
            
            radius = float(dimensions["radius"])
            angle = float(dimensions["angle"])
            return round(2 * math.pi * radius * (angle / 360), 2)
        
        # Fall back to centerline length if available
        elif (dimensions["centerline_length"] != "Not Found" and
              dimensions["centerline_length"].replace('.', '', 1).isdigit()):
            
            return round(float(dimensions["centerline_length"]), 2)
        
        # Default calculation if no specific dimensions found
        else:
            # Use typical values for hose bending
            return round(2 * math.pi * 40 * (90 / 360), 2)  # 40mm radius, 90° angle
    except Exception as e:
        print(f"Error calculating development length: {e}")
        return "Calculation error"

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
        od1.replace('.', '', 1).isdigit() and id1.replace('.', '', 1).isdigit()):
        try:
            thickness = (float(od1) - float(id1)) / 2
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

# --- NEW: Function to analyze the PDF text using Gemini API ---
def analyze_drawing_with_gemini(pdf_bytes):
    results = {
        "part_number": "Not Found",
        "standard": "Not Found",
        "grade": "Not Found",
        "material": "Not Found",
        "error": None
    }
    
    try:
        # --- Step 1: Extract text from PDF ---
        pdf_document = fitz.open("pdf", pdf_bytes)
        full_text = ""
        for page in pdf_document:
            full_text += page.get_text()
        
        if not full_text.strip():
            results["error"] = "Could not extract any text from the PDF."
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
        analysis_results = analyze_drawing_with_gemini(pdf_bytes)
        
        if analysis_results.get("error"):
            return jsonify({"error": analysis_results["error"]}), 400
        
        # Calculate development length using extracted dimensions
        dimensions = analysis_results.get("dimensions", {})
        development_length = calculate_development_length(dimensions)
        
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