import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

def verify_gemini_models():
    """Verify which Gemini models are available"""
    try:
        available_models = genai.list_models()
        model_names = [model.name for model in available_models]
        logger.info(f"Available Gemini models: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error listing available models: {e}")
        return []

def get_vision_model():
    """Get the best available vision model with fallbacks"""
    # Only use models that we know are available from the logs
    model_preference = [
        'gemini-2.0-flash-exp',      # This one works (from gemini_helper)
        'gemini-2.0-pro-exp',        # Alternative experimental
        'gemini-2.0-flash',          # Stable Flash
        'gemini-2.0-pro',            # Stable Pro
        'gemini-1.0-pro-vision',     # Legacy vision model
    ]
    
    for model_name in model_preference:
        try:
            model = genai.GenerativeModel(model_name)
            logger.info(f"Successfully initialized model: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Model {model_name} not available: {e}")
            continue
    
    # If all else fails, try any model
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        logger.error(f"All model attempts failed: {e}")
        raise Exception("No Gemini vision models available")