from typing import Optional, Dict, Any
import logging
import json
import google.generativeai as genai
from image_optimization import optimize_image_for_gemini

logger = logging.getLogger(__name__)

def get_engineering_drawing_prompt() -> str:
    """
    Returns a specialized prompt for engineering drawing analysis.
    """
    return """You are analyzing a technical engineering drawing. Extract the following information:

CRITICAL INFORMATION:
1. PART NUMBER: Look for formats like 3718791C1, 3541592C1, 4034580C1 (7 digits + C + digit)
2. DESCRIPTION: Main title, usually starting with "HOSE," "PIPE," etc.
3. STANDARD: Material specifications like "MPAPS F-6034", "TMS-6034", "SAE J..."
4. GRADE: Look for "GRADE", "TYPE" followed by codes like "H-AN", "1B", "C-AN"
5. DIMENSIONS: Extract ID, OD, wall thickness, centerline length
6. MATERIAL & REINFORCEMENT: Look for material specifications and reinforcement types

SPECIAL INSTRUCTIONS:
- Read text directly from the image, don't rely on OCR
- Understand the drawing layout and title blocks
- Look for tables and specification sections
- Handle both MPAPS and TMS standards
- If information is not clearly visible, return "Not Found"

Return the information in structured JSON format."""

def assess_result_confidence(result: Dict[str, Any]) -> float:
    """
    Score the completeness and confidence of Gemini results.
    
    Args:
        result (dict): Analysis results from Gemini
        
    Returns:
        float: Confidence score between 0 and 1
    """
    if not isinstance(result, dict):
        return 0.0
    
    # Key fields that should be present
    critical_fields = ['part_number', 'description', 'standard', 'grade']
    dimension_fields = ['id1', 'od1', 'thickness', 'centerline_length']
    
    score = 0.0
    total_weight = 0
    
    # Check critical fields (60% of total score)
    for field in critical_fields:
        weight = 15  # Each critical field worth 15%
        if result.get(field) and result[field] != "Not Found":
            score += weight
        total_weight += weight
    
    # Check dimension fields (40% of total score)
    dimensions = result.get('dimensions', {})
    if isinstance(dimensions, dict):
        for field in dimension_fields:
            weight = 10  # Each dimension field worth 10%
            if dimensions.get(field) and dimensions[field] != "Not Found":
                score += weight
            total_weight += weight
    
    return score / total_weight if total_weight > 0 else 0.0

def robust_gemini_analysis(image_data: bytes, prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Try multiple Gemini models for maximum accuracy with image analysis.
    
    Args:
        image_data (bytes): Raw image data to analyze
        prompt (str, optional): Custom prompt to use
        
    Returns:
        dict: Best analysis results from any model
    """
    models_to_try = [
        'gemini-2.0-flash-exp',  # Fast, good for text
        'gemini-1.5-pro',        # High context, better for complex images
        'gemini-1.5-flash'       # Balanced approach
    ]
    
    best_result = None
    best_confidence = 0
    
    if not prompt:
        prompt = get_engineering_drawing_prompt()
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': image_data}])
            
            if response and response.text:
                try:
                    # Try to parse as JSON first
                    parsed_result = json.loads(response.text)
                except json.JSONDecodeError:
                    # If not JSON, use the text as is
                    parsed_result = {'text': response.text}
                
                # Score result completeness
                confidence = assess_result_confidence(parsed_result)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = parsed_result
                
                if confidence > 0.8:  # Good enough result found
                    break
                    
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            continue
    
    if best_result is None:
        return {
            "error": "No model was able to successfully analyze the drawing",
            "confidence": 0.0
        }
    
    return {
        "result": best_result,
        "confidence": best_confidence,
        "model": models_to_try[0]  # Return which model was used
    }