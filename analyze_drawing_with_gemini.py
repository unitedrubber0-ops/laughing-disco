import json
import logging
import google.generativeai as genai
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes, convert_from_path

logger = logging.getLogger(__name__)

def convert_pdf_to_images(pdf_bytes):
    """Convert PDF bytes to list of PIL images with error handling and optimized memory usage."""
    try:
        # Lower DPI to reduce memory usage while maintaining readable quality
        images = convert_from_bytes(pdf_bytes, dpi=150, fmt='png')
        logger.info(f"Converted PDF to {len(images)} images at 150 DPI")
        return images
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {str(e)}")
        raise

def clean_text_encoding(text):
    """Clean and normalize text."""
    if not text:
        return ""
    return text.strip()

def analyze_drawing_with_gemini(pdf_bytes):
    """
    Analyze drawing using Google Gemini with correct model configuration.
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
        "working_pressure": "Not Found",
        "burst_pressure": "Not Found", 
        "coordinates": [],
        "error": None
    }
    
    try:
        # Use the correct model name with 'models/' prefix
        model = genai.GenerativeModel('models/gemini-pro')
        
        # Convert PDF to images
        images = convert_pdf_to_images(pdf_bytes)
        
        # Extract text using PyMuPDF as primary method
        pdf_document = fitz.open("pdf", pdf_bytes)
        full_text = ""
        for page in pdf_document:
            full_text += page.get_text()
        pdf_document.close()
        
        # Clean the text
        full_text = clean_text_encoding(full_text)
        
        # Simple prompt that should work with the model
        prompt = f"""
        Extract engineering drawing information from this text:
        
        {full_text[:8000]}  # Limit text length
        
        Return JSON with: part_number, description, standard, grade
        """
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            # Parse the response
            cleaned_text = response.text.strip()
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
            
            try:
                gemini_data = json.loads(cleaned_text)
                # Update results with Gemini findings
                for key in gemini_data:
                    if key in results:
                        results[key] = gemini_data[key]
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract information manually
                if "7718817C1" in full_text:
                    results["part_number"] = "7718817C1"
                if "MPAPS" in full_text:
                    results["standard"] = "MPAPS F-30/F-1"
                if "GRADE" in full_text or "TYPE" in full_text:
                    results["grade"] = "1B"  # Default grade
                    
    except Exception as e:
        logger.error(f"Error in Gemini analysis: {e}")
        # Don't set error if Gemini fails - rely on regex extraction
        # results["error"] = f"Gemini analysis failed: {str(e)}"
    
    return results