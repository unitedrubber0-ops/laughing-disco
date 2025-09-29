import logging
import google.generativeai as genai
from typing import Optional, Dict, Any, List, Union
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def discover_vision_model() -> Optional[str]:
    """
    Discover the first available vision-capable model that supports generateContent.
    Result is cached to avoid repeated API calls.
    
    Returns:
        str: Model name to use, or None if no suitable model found
    """
    try:
        models = genai.list_models()
        for model in models:
            # Check if model supports generateContent and has vision/image capabilities
            methods = getattr(model, "supported_generation_methods", [])
            name = getattr(model, "name", "")
            description = getattr(model, "description", "").lower()
            
            if ("generateContent" in methods and 
                ("vision" in name.lower() or 
                 "vision" in description or 
                 "image" in description)):
                logger.info(f"Found suitable vision model: {name}")
                return name
                
        logger.warning("No vision-capable models found that support generateContent")
        return None
        
    except Exception as e:
        logger.error(f"Error discovering models: {e}")
        return None

def process_with_vision_model(
    model_name: str,
    prompt: str,
    image_data: Union[str, bytes],
    mime_type: str = "image/png"
) -> Optional[Dict[Any, Any]]:
    """
    Process an image with a specified Gemini model, with error handling.
    
    Args:
        model_name: Name of the model to use (from discover_vision_model)
        prompt: Text prompt for the model
        image_data: Base64 encoded image or raw bytes
        mime_type: MIME type of the image (default: image/png)
        
    Returns:
        Dict containing parsed response or None if processing failed
    """
    try:
        model = genai.GenerativeModel(model_name)
        image_part = {"mime_type": mime_type, "data": image_data}
        
        response = model.generate_content([prompt, image_part])
        
        if not response or not response.text:
            logger.warning("Empty response from model")
            return None
            
        # Clean and parse response
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        
        import json
        return json.loads(text)
        
    except Exception as e:
        logger.error(f"Vision model processing failed: {e}")
        return None

def extract_text_from_image_wrapper(image):
    """
    Wrapper function to extract text from image for OCR fallback.
    This should be imported from app.py or defined here.
    """
    try:
        import pytesseract
        from PIL import Image
        # Basic OCR extraction
        text = pytesseract.image_to_string(image, config='--psm 1')
        return text
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""

def process_pages_with_vision_or_ocr(
    pages: Union[str, List[Any]],
    prompt: str,
    ocr_fallback_fn: callable,
    mime_type: str = "image/png"
) -> List[Dict[Any, Any]]:
    """
    Process pages from a PDF with vision model, falling back to OCR if needed.
    
    Args:
        pages: Either a path to a PDF file or a list of PIL Image objects
        prompt: Prompt for the vision model
        ocr_fallback_fn: Function to call for OCR processing if vision fails
        mime_type: MIME type of the images
        
    Returns:
        List of parsed results from each page
    """
    results = []
    gemini_failed = False
    page_images = []
    
    # Convert PDF path to images if needed
    if isinstance(pages, str):
        try:
            from pdf2image import convert_from_path
            page_images = convert_from_path(pages, dpi=150)
            logger.info(f"Converted PDF to {len(page_images)} pages")
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []
    else:
        page_images = pages
    
    # Try vision model first
    model_name = discover_vision_model()
    if not model_name:
        logger.warning("No suitable vision model found, falling back to OCR")
        gemini_failed = True
    
    if not gemini_failed:
        try:
            for i, page in enumerate(page_images):
                # Convert page to expected format
                import io
                import base64
                img_byte_arr = io.BytesIO()
                page.save(img_byte_arr, format='PNG')
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                
                result = process_with_vision_model(model_name, prompt, img_base64, mime_type)
                if result:
                    results.append(result)
                    logger.info(f"Successfully processed page {i+1} with vision model")
                else:
                    logger.warning(f"Vision model failed on page {i+1}, switching to OCR")
                    gemini_failed = True
                    break
                    
        except Exception as e:
            logger.error(f"Error during vision processing: {e}")
            gemini_failed = True
    
    # Fall back to OCR if vision failed
    if gemini_failed or not results:
        logger.info("Processing all pages with OCR fallback")
        results = []  # Reset results
        for i, page in enumerate(page_images):
            try:
                # Extract text from image first
                extracted_text = extract_text_from_image_wrapper(page)
                # Then process the text with the OCR function
                result = ocr_fallback_fn(extracted_text)
                if result:
                    results.append(result)
                    logger.info(f"Successfully processed page {i+1} with OCR")
                else:
                    logger.warning(f"OCR processing failed on page {i+1}")
            except Exception as e:
                logger.error(f"Error in OCR fallback for page {i+1}: {e}")
                # Create a default result for this page
                results.append({
                    "part_number": "Not Found",
                    "description": "Not Found", 
                    "standard": "Not Found",
                    "grade": "Not Found",
                    "material": "Not Found",
                    "dimensions": {},
                    "coordinates": []
                })
    
    return results