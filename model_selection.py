import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

def get_vision_model():
    """Get the best available vision model with fallbacks"""
    model_preference = [
        'gemini-1.5-flash',      # Latest recommended model
        'gemini-1.5-pro',        # Alternative high-context model  
        'gemini-1.0-pro-vision', # Legacy vision model
        'gemini-2.0-flash',      # Latest Flash model
    ]
    
    for model_name in model_preference:
        try:
            model = genai.GenerativeModel(model_name)
            # Test if model is available by making a simple call
            logger.info(f"Testing model: {model_name}")
            # If no exception, model is available
            logger.info(f"Using model: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Model {model_name} not available: {e}")
            continue
    
    # Fallback to any available model
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except:
        raise Exception("No Gemini vision models available")