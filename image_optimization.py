from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)

def optimize_image_for_gemini(image):
    """
    Enhance image quality before sending to Gemini.
    
    Args:
        image (PIL.Image): Input image to optimize
        
    Returns:
        PIL.Image: Optimized image with enhanced quality
    """
    try:
        # Increase resolution for better text recognition
        # Only upscale if image is too small
        if image.width < 1000 or image.height < 1000:
            new_width = max(1500, image.width)
            new_height = int(new_width * (image.height / image.width))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast
        image = ImageEnhance.Contrast(image).enhance(1.5)
        
        # Sharpen image
        image = ImageEnhance.Sharpness(image).enhance(1.2)
        
        # Enhance brightness slightly
        image = ImageEnhance.Brightness(image).enhance(1.1)
        
        return image
    except Exception as e:
        logger.warning(f"Image optimization failed: {e}")
        return image  # Return original image if optimization fails

def ensure_minimum_image_quality(images):
    """
    Ensure all images meet minimum quality standards for Gemini.
    
    Args:
        images (list): List of PIL.Image objects
        
    Returns:
        list: List of processed PIL.Image objects
    """
    processed_images = []
    
    for img in images:
        try:
            processed = optimize_image_for_gemini(img)
            processed_images.append(processed)
        except Exception as e:
            logger.warning(f"Failed to process image: {e}")
            processed_images.append(img)  # Keep original if processing fails
    
    return processed_images