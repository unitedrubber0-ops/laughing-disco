"""
Application startup module with environment verification
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_environment():
    """Verify all required modules and files are present"""
    logger.info("=== Environment Verification ===")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    logger.info(f"Python path: {sys.path}")
    
    required_files = [
        'pdf_processor.py',
        'app.py',
        'material_utils.py',
        'gemini_helper.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            logger.error(f"Required file missing: {file}")
        else:
            logger.info(f"Found required file: {file}")
    
    if missing_files:
        raise RuntimeError(f"Missing required files: {', '.join(missing_files)}")
    
    # Try importing required modules
    try:
        import pdf_processor
        logger.info("Successfully imported pdf_processor")
    except ImportError as e:
        logger.error(f"Failed to import pdf_processor: {e}")
        raise

def create_app():
    """Create and configure the Flask application"""
    verify_environment()
    
    from app import app
    return app

# This will be used by gunicorn
app = create_app()