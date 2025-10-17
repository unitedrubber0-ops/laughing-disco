import sys
import io
import logging
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

def ocr_pdf_first_page(pdf_path, dpi=200):
    """
    Perform OCR on the first page of a PDF file.
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering, higher values may improve accuracy (default: 200)
    """
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            print("No pages in PDF")
            return
            
        page = doc.load_page(0)
        # Render at specified DPI for better OCR quality
        matrix = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = page.get_pixmap(matrix=matrix, alpha=False)  # type: ignore
        
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        
        # Perform OCR
        text = pytesseract.image_to_string(img)
        
        print("----- OCR Output (first 800 chars) -----")
        print(text[:800])
        print("\n----- Tesseract Version -----")
        print(f"Tesseract Version: {pytesseract.get_tesseract_version()}")
        
    except Exception as e:
        logging.exception("Error during OCR processing")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_ocr.py <file.pdf> [dpi]")
        print("Example: python tests/test_ocr.py sample.pdf 300")
        sys.exit(2)
        
    pdf_path = sys.argv[1]
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    
    print(f"Processing {pdf_path} at {dpi} DPI...")
    ocr_pdf_first_page(pdf_path, dpi)