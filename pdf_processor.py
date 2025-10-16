# pdf_processor.py
"""
Minimal pdf_processor module providing process_pdf_comprehensive and generate_output.
Designed to be safe, small, and dependency-light (uses PyMuPDF if available).
Drop into your project root so `import pdf_processor` succeeds.
"""

import io
import json
import logging

def _extract_text_with_pymupdf(pdf_bytes):
    try:
        import fitz  # PyMuPDF
    except Exception:
        logging.warning("PyMuPDF not available; falling back to naive text extract")
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for p in doc:
            pages.append(p.get_text())
        return {"page_count": doc.page_count, "pages": pages, "text": "\n".join(pages)}
    except Exception as e:
        logging.exception("PyMuPDF extraction failed")
        return None

def _naive_text_extract(pdf_bytes):
    # Very small fallback: return raw bytes length and empty text
    return {"page_count": None, "pages": [], "text": ""}

def process_pdf_comprehensive(pdf_bytes, output_type='json'):
    """
    Process PDF data using a comprehensive approach that combines text extraction,
    OCR, and field parsing.
    
    Args:
        pdf_bytes: Raw PDF file data
        output_type: Type of output ("json" or "excel")
        
    Returns:
        Tuple of (extracted data dict, error message if any)
    """
    try:
        # Load PDF
        doc = cast(Document, fitz.open(stream=pdf_bytes, filetype="pdf"))
        
        if not doc.page_count:
            return {}, "Invalid or empty PDF"
            
        # Stage 1: Direct Text Extraction
        text_content = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]  # Get page
            text = page.get_textpage().extractText()  # Use native method
            text_content += text
            
        # If we got meaningful text, process it
        if text_content and len(text_content.strip()) > 50:
            results = extract_all_fields_from_text(text_content, "PyMuPDF Text")
            logger.info("Successfully extracted fields from PyMuPDF text")
            return results, ""
            
        # Stage 2: OCR via Gemini Vision
        try:
            # Convert first page to image
            # Convert first page to PNG using direct rendering
            page = doc[0]  # Get first page
            zoom = 2  # Scale factor for better quality
            mat = fitz.Matrix(zoom, zoom)  # Create transformation matrix
            pix = page.get_pixmap(matrix=mat, alpha=False)  # type: ignore # PyMuPDF type stubs incomplete
            img_data = pix.tobytes(output="png")  # Convert to PNG bytes
            
            # Process with Gemini Vision
            ocr_result = process_with_vision_model(
                prompt="Please extract all text from this technical drawing image, including measurements, specifications, and notes.",
                image_data=img_data,
                model_name="gemini-pro-vision"  # Specify the model to use
            )
            ocr_text = ""
            
            if isinstance(ocr_result, dict):
                ocr_text = ocr_result.get('text', '')
            elif isinstance(ocr_result, str):
                ocr_text = ocr_result
            
            if ocr_text and len(ocr_text.strip()) > 50:
                results = extract_all_fields_from_text(ocr_text, "Gemini Vision OCR")
                logger.info("Successfully extracted fields from Gemini Vision OCR")
                return results, ""
                
        except Exception as e:
            logger.error(f"Error in Gemini Vision OCR: {e}")
            pass
            
        return {}, "Could not extract sufficient text from PDF"
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {}, str(e)
    finally:
        if 'doc' in locals():
            doc.close()
            
def generate_output(data: Dict[str, Any], output_type: str = "json") -> Any:
    """
    Generate output in requested format
    """
    if output_type == "excel":
        # TODO: Implement Excel output generation
        pass
    return data  # Default to JSON