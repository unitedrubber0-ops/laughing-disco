"""
Utilities for material handling and text normalization.
"""
import re
import logging
import traceback
import math
import inspect
from typing import Any, List, Dict, Optional, Union

def safe_material_lookup_entry(standard_raw, grade_raw, material_df, lookup_fn):
    """
    Call lookup_fn with appropriate number of arguments based on its signature.
    Handles both 2-arg (standard, grade) and 3-arg (standard, grade, material_df) function styles.
    Also coerces dict inputs to strings and provides detailed error logging.
    
    NOTE: Authoritative mappings from material_mappings.py are consulted FIRST,
    before any fuzzy or CSV lookups, to ensure specifications match user table.
    """
    try:
        # Log types to find leak
        if isinstance(standard_raw, dict) or isinstance(grade_raw, dict):
            logging.warning("safe_material_lookup_entry: dict-like input detected. standard repr=%r grade repr=%r",
                            standard_raw, grade_raw)

        # Coerce with normalize helpers (safe for dicts)
        std = normalize_standard(standard_raw)
        grd = normalize_grade(grade_raw)

        logging.debug("safe_material_lookup_entry: normalized standard=%r grade=%r", std, grd)

        # ==== AUTHORITATIVE LOOKUP FIRST ====
        # Check authoritative mapping from material_mappings.py before fuzzy matching
        try:
            from material_mappings import get_material_by_standard_grade
            auth_material, auth_reinforce = get_material_by_standard_grade(std, grd)
            if auth_material and auth_material != (None, None):
                # Found in authoritative table, return immediately
                logging.info(f"safe_material_lookup_entry: Using authoritative mapping for {std}/{grd}: {auth_material}")
                return auth_material
        except (ImportError, Exception) as e:
            logging.debug(f"Authoritative lookup unavailable or failed: {e}")
        # ==== END AUTHORITATIVE LOOKUP ====

        # Check function signature and call appropriately
        try:
            sig = inspect.signature(lookup_fn)
            params = len(sig.parameters)
            logging.debug(f"safe_material_lookup_entry: lookup_fn expects {params} parameters")
            
            if params >= 3:
                result = lookup_fn(std, grd, material_df)
            else:
                result = lookup_fn(std, grd)
                
            # If result found, try to normalize it
            if result and result != "Not Found":
                normalized_result = normalize_material(result)
                logging.debug(f"Material normalization: {result} -> {normalized_result}")
                result = normalized_result or result  # Use normalized if available, else original
                
            return result if result else "Not Found"
            
        except Exception as e:
            logging.warning(f"safe_material_lookup_entry signature inspection failed: {e}, trying fallback calls")
            # Fallback attempts
            try:
                result = lookup_fn(std, grd)
                if result and result != "Not Found":
                    result = normalize_material(result) or result
                return result
            except TypeError:
                result = lookup_fn(std, grd, material_df)
                if result and result != "Not Found":
                    result = normalize_material(result) or result
                return result

    except Exception as e:
        logging.error("Error in material lookup: %s\nstandard_raw repr=%r\ngrade_raw repr=%r\nTraceback:\n%s",
                      e, standard_raw, grade_raw, traceback.format_exc())
        # return a safe fallback so rest of pipeline doesn't crash
        return "Not Found"

def _coerce_to_str_maybe_dict(val, keys_to_try=('STANDARD','standard','value','name')):
    """
    If val is a dict, try common keys; otherwise return str(val) (or '' if None).
    """
    if val is None:
        return ''
    if isinstance(val, dict):
        for k in keys_to_try:
            if k in val and val[k] is not None:
                return str(val[k])
        # fallback: return repr so logs show the dict content
        return repr(val)
    if not isinstance(val, str):
        try:
            return str(val)
        except Exception:
            return repr(val)
    return val

def map_astm_d2000_grade(type_class, grade_num):
    """
    Map ASTM D2000 grade numbers and Type-Class combinations to MPAPS grades.
    """
    # Default mappings based on Type-Class
    type_class_mappings = {
        'CA': '1B',    # EPDM standard grade
        'BA': '1B',    # EPDM high-temp grade
        'DA': '1B',    # EPDM special grade
        'BF': '1BF',   # NBR standard grade
        'CH': '1BF',   # NBR/ECH grade
        'BC': '1BC',   # Neoprene grade
        'BE': '1BC',   # Neoprene grade
    }
    
    # Grade number overrides (e.g., M7 -> 1B)
    grade_num_mappings = {
        '7': '1B',     # Standard EPDM grade
        '6': '1BF',    # Standard NBR grade
        '5': '1BC',    # Standard Neoprene grade
    }
    
    # Try grade number mapping first
    if grade_num and grade_num in grade_num_mappings:
        return grade_num_mappings[grade_num]
        
    # Fall back to Type-Class mapping
    if type_class in type_class_mappings:
        return type_class_mappings[type_class]
        
    return "1B"  # Default grade if no specific mapping found

def map_astm_d2000_to_mpaps(std):
    """
    Map ASTM D2000 specifications to MPAPS standards.
    Example: "ASTM D2000 M7 CA 807" -> ("MPAPS F-30/F-1", "1B")
    Returns tuple of (mapped_standard, mapped_grade)
    """
    if not std:
        return (std, "Not Found")
        
    std_upper = str(std).upper()
    if "ASTM" not in std_upper or "D2000" not in std_upper:
        return (std, "Not Found")
        
    try:
        # Extract components from ASTM D2000 spec
        match = re.search(r'(?:M?(\d*))?\s*([A-K]{2})\s*(\d{3})', std_upper)
        if not match:
            return (std, "Not Found")
            
        grade_num = match.group(1)  # M7 -> 7
        type_class = match.group(2)  # CA
        hardness = match.group(3)    # 807
        
        # Map based on Type-Class
        if type_class in ['CA', 'BA', 'DA']:  # EPDM types
            mapped_std = "MPAPS F-30/F-1"
        elif type_class in ['BF', 'CH']:      # NBR types
            mapped_std = "MPAPS F-48"
        elif type_class in ['BC', 'BE']:      # Neoprene types
            mapped_std = "MPAPS F-42"
        else:
            mapped_std = "MPAPS F-30/F-1"     # Default to EPDM spec
            
        # Get mapped grade
        mapped_grade = map_astm_d2000_grade(type_class, grade_num)
        
        logging.info(f"Mapped ASTM D2000 '{std}' to MPAPS standard='{mapped_std}' grade='{mapped_grade}'")
        return (mapped_std, mapped_grade)
        
    except Exception as e:
        logging.error(f"Error mapping ASTM D2000 spec '{std}': {e}")
        return (std, "Not Found")

def normalize_standard(std):
    try:
        std = _coerce_to_str_maybe_dict(std, keys_to_try=('STANDARD','standard','std'))
        std = std.upper().strip()
        
        # Map ASTM D2000 to MPAPS if needed
        if "ASTM" in std and "D2000" in std:
            mapped_std, _ = map_astm_d2000_to_mpaps(std)
            std = mapped_std
            
        # Apply existing cleanup rules
        std = re.sub(r'MPAPS\s*F\s*[-_]?\s*(\d+)', r'MPAPS F-\1', std)
        std = re.sub(r'\s+', ' ', std).strip()
        return std
    except Exception as e:
        logging.exception("normalize_standard failed for input: %r; returning empty string", std)
        return ''

def normalize_grade(grd, std=None):
    """
    Normalize grade values, with optional standard context for ASTM D2000 mapping.
    """
    try:
        grd = _coerce_to_str_maybe_dict(grd, keys_to_try=('GRADE','grade','grd'))
        grd = grd.upper().strip()
        
        # If we have an ASTM D2000 standard, try to map its grade
        if std:
            std_upper = str(std).upper()
            if "ASTM" in std_upper and "D2000" in std_upper:
                _, mapped_grade = map_astm_d2000_to_mpaps(f"{std} {grd}")
                if mapped_grade != "Not Found":
                    return mapped_grade
                    
        # Standard grade normalization
        grd = re.sub(r'[^A-Z0-9\-]', '', grd)       # remove weird punctuation
        grd = re.sub(r'GRADE|TYPE|CLASS', '', grd)
        grd = grd.replace(' ', '').replace('_', '').replace('/', '')
        grd = re.sub(r'([A-Z0-9\-]+)[XZ]$', r'\1', grd)
        return grd
    except Exception as e:
        logging.exception("normalize_grade failed for input: %r; returning empty string", grd)
        return ''

def normalize_material(material):
    """
    Normalize material names to handle common variations.
    """
    if not material:
        return ''
        
    try:
        # Convert to string and clean up
        mat = _coerce_to_str_maybe_dict(material, keys_to_try=('MATERIAL','material','name'))
        mat = mat.upper().strip()

        # If the material string already contains composite qualifiers, preserve it as-is
        # (prevents collapsing "INNER NBR OUTER:ECO" down to just "NBR")
        if 'INNER' in mat or 'OUTER' in mat or ':' in mat:
            return mat
        
        # Common material name variations
        variations = {
            'EPDM': ['P-EPDM', 'EPDM RUBBER', 'ETHYLENE PROPYLENE'],
            'NBR': ['P-NBR', 'NITRILE', 'BUNA-N', 'BUNA N'],
            'SILICONE': ['P-SILICONE', 'SI', 'VMQ'],
            'VITON': ['FKM', 'FLUOROELASTOMER'],
            'NEOPRENE': ['P-NEOPRENE', 'CR', 'CHLOROPRENE']
        }
        
        # Check for each variation
        for base, variants in variations.items():
            if base in mat:
                return base
            for variant in variants:
                if variant in mat:
                    return base
                    
        # Remove common prefixes/suffixes
        mat = re.sub(r'^P-', '', mat)
        mat = re.sub(r'\s*RUBBER$', '', mat)
        
        return mat
        
    except Exception as e:
        logging.exception("normalize_material failed for input: %r; returning empty string", material)
        return ''

# Optional: for PDF->image + OCR
# pip install pytesseract Pillow
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from PIL import Image
    import pytesseract
except Exception:
    pytesseract = None
    Image = None

# Mapping derived from ASTM D2000 Table X1.1 (Table of "Polymers Most Often Used").
_D2000_MATERIAL_TO_POLYMER = {
    "AA": "Natural rubber / SBR / butyl / polyisoprene (general-purpose)",
    "AK": "Polysulfides",
    "BA": "Ethylene-propylene (EPDM) / high-temp SBR / butyl",
    "BC": "Chloroprene polymers (neoprene)",
    "BE": "Chloroprene polymers (neoprene)",
    "BF": "NBR polymers (nitrile)",
    "BG": "NBR polymers, urethanes",
    "BK": "NBR",
    "CA": "Ethylene propylene (EPDM)",
    "CE": "Chlorosulfonated polyethylene (Hypalon) / CM",
    "CH": "NBR polymers, epichlorohydrin polymers",
    "DA": "Ethylene propylene polymers",
    "DE": "CM, CSM",
    "DF": "Polyacrylic (butyl-acrylate type)",
    "DH": "Polyacrylic polymers, HNBR",
    "EE": "AEM (ethylene acrylic elastomer)",
    "EH": "ACM (polyacrylate)",
    "EK": "FZ (special family)",
    "FC": "Silicones (high strength)",
    "FE": "Silicones",
    "FK": "Fluorinated silicones",
    "GE": "Silicones",
    "HK": "Fluorinated elastomers (Viton, Fluorel, etc.)",
    "KK": "Perfluoroelastomers",
}

# Multiple regex patterns to find D2000 callouts in free text
# Order matters - more specific patterns first
D2000_CALLOUT_PATTERNS = [
    # Pattern 1: Standard D2000 format (e.g., "ASTM D2000 M2BC507")
    r'(?:ASTM\s*D\s*2000[:\s]*)?'       # optional "ASTM D2000" prefix
    r'(?:M?\d*)\s*([A-K]{2})\s*([0-9]{3})',  # Grade (M2), Type-Class (BC), Hardness/Tensile (507)

    # Pattern 2: Type-Class between grade and numbers (e.g., "M7 CA 807")
    r'M?\d*\s*([A-K]{2})\s*\d{3}',

    # Pattern 3: Standalone Type-Class with numbers (e.g., "CA 807")
    r'\b([A-K]{2})\s*\d{3}\b'
]

D2000_CALLOUT_RE = re.compile('|'.join(f'({p})' for p in D2000_CALLOUT_PATTERNS), flags=re.IGNORECASE)

def parse_d2000_callouts_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse ASTM D2000 callouts from a chunk of text and map Type/Class to polymer family.
    Returns list of dicts: {raw, type_class, hardness_tensile, polymer, context_snippet}
    """
    results = []
    if not text:
        return results

    # Clean up the input text
    text = text.replace('\n', ' ').strip()
    
    for m in D2000_CALLOUT_RE.finditer(text):
        try:
            # Find the first non-None group after group 0 (full match)
            match_groups = [g for g in m.groups() if g is not None]
            if not match_groups:
                continue
                
            raw = match_groups[0]  # The matched pattern
            
            # Extract type_class using a targeted regex on the raw match
            type_class_match = re.search(r'([A-K]{2})', raw.upper())
            if not type_class_match:
                continue
                
            type_class = type_class_match.group(1)
            
            # Extract hardness/tensile using pattern after type_class
            hard_tens_match = re.search(f'{type_class}\\s*(\\d{{3}})', raw.upper())
            hard_tens = hard_tens_match.group(1) if hard_tens_match else "000"
            
            # Lookup polymer type
            polymer = _D2000_MATERIAL_TO_POLYMER.get(type_class, "Unknown / not in mapping")
            
            # Capture context for verification
            start, end = m.span()
            snippet = text[max(0, start-60):min(len(text), end+60)].strip()
            
            # Log successful parsing
            logging.debug(f"Successfully parsed ASTM D2000 callout: {raw} -> Type-Class: {type_class}, " 
                        f"Hardness/Tensile: {hard_tens}, Polymer: {polymer}")
            
            results.append({
                "raw": raw,
                "type_class": type_class,
                "hardness_tensile": hard_tens,
                "polymer": polymer,
                "context": snippet
            })
        except Exception as e:
            logging.warning(f"Failed to parse potential D2000 callout: {m.group(0)}, error: {e}")
            continue
    return results

def extract_text_from_pdf_with_ocr(pdf_path: str, dpi: int = 200) -> str:
    """
    Render each PDF page to an image and OCR it with pytesseract.
    Returns concatenated text for all pages.
    Requirements: PyMuPDF (fitz) + pytesseract + tesseract binary installed.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not available. Install it to enable PDF rendering.")
    if pytesseract is None:
        raise RuntimeError("pytesseract not available. Install pytesseract and tesseract binary.")
    doc = fitz.open(pdf_path)
    out_text = []
    for pageno in range(len(doc)):
        page = doc.load_page(pageno)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        try:
            # Try new API (PyMuPDF >= 1.19.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)  # type: ignore
        except AttributeError:
            # Fallback for older versions
            pix = page.getPixmap(matrix=mat, alpha=False)  # type: ignore
        try:
            img_bytes = pix.tobytes("png")
        except AttributeError:
            # Fallback for older versions
            img_bytes = pix.getPNGData()
        if Image is not None:  # Check if PIL is available
            from io import BytesIO
            img = Image.open(BytesIO(img_bytes))
            page_text = pytesseract.image_to_string(img) if pytesseract else ""
            out_text.append(page_text)
    return "\n\n".join(out_text)

def find_polymers_in_pdf_or_text(pdf_path: Optional[str] = None, text: Optional[str] = None, try_ocr: bool = True) -> List[Dict[str, Any]]:
    """
    If text is supplied, parse it. If pdf_path is supplied:
      - attempt to extract embedded text (via fitz.get_text("text"))
      - if embedded text is essentially empty and try_ocr=True -> run OCR
    Returns parse_d2000_callouts_from_text results.
    """
    doc_text = text or ""
    if pdf_path and not text:
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) required to read PDF. Install fitz first.")
        doc = fitz.open(pdf_path)
        extracted = []
        for p in doc:
            try:
                # Try new API (PyMuPDF >= 1.19.0)
                extracted.append(p.get_text("text"))  # type: ignore
            except AttributeError:
                # Fallback for older versions
                extracted.append(p.getText("text"))  # type: ignore
        doc_text = "\n\n".join(extracted).strip()
        # If embedded text is tiny/empty, fall back to OCR (useful for scanned drawings)
        if (not doc_text or len(doc_text) < 20) and try_ocr:
            try:
                doc_text = extract_text_from_pdf_with_ocr(pdf_path)
            except Exception as ocr_exc:
                logging.exception("OCR failed")
                # if OCR fails, return empty with an informative dict
                return [{"error": str(ocr_exc)}]
    # now parse text
    return parse_d2000_callouts_from_text(doc_text)

def get_polymer_type_from_astm_code(astm_callout_string: str) -> str:
    """
    Looks up the polymer type based on the Type-Class letters in an ASTM D2000 call-out.
    The mapping is based on ASTM D2000-18, Table X1.1.

    Args:
        astm_callout_string (str): The full ASTM D2000 specification string.

    Returns:
        str: The polymer type (e.g., "Ethylene propylene") or "Not Found".
    """
    if not astm_callout_string or not isinstance(astm_callout_string, str):
        return "Not Found"
        
    # Use our existing robust parsing logic
    results = parse_d2000_callouts_from_text(astm_callout_string)
    
    # Return first polymer type found, or "Not Found" if none found
    return results[0]["polymer"] if results else "Not Found"

    # Match pattern from ASTM D2000-18, Table X1.1
    match = re.search(r'M?\d?\s*([A-K]{2})', astm_callout_string.upper())
    
    if match:
        type_class_code = match.group(1)
        polymer = _D2000_MATERIAL_TO_POLYMER.get(type_class_code, "Not Found")
        logging.info(f"Found Type-Class code '{type_class_code}', mapped to polymer: '{polymer}'")
        return polymer
    else:
        logging.warning(f"Could not extract Type-Class code from string: '{astm_callout_string}'")
        return "Not Found"

def _process_extraction_result(val: Any) -> str:
    """Process an extraction result, converting to string or empty string if None."""
    if val is None:
        return ''
    return str(val)

# Diameter extraction helpers
_DIA_PATTERNS = [
    r'(?:DIA|DIAMETER|Ø|PHI|⌀)\s*[:=]?\s*([+-]?\d+\.?\d*)\s*(MM|IN|INCH|CM)?',  # e.g., DIA: 85 mm
    r'\bD\s*[:=]?\s*([+-]?\d+\.?\d*)\s*(MM|IN|CM)?\b'
]

def extract_diameter(text: str) -> tuple[float, str | None] | None:
    """Extract diameter value and unit from text, returning (value, unit) or None."""
    t = (text or "").upper()
    for p in _DIA_PATTERNS:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                unit = (m.group(2) or '').upper().strip() or None
                return val, unit
            except Exception:
                continue
    return None

def development_length_from_diameter(dia_value: float, unit: str | None = None) -> float:
    """Return circumference = pi * D. Unit left as-is."""
    return math.pi * dia_value

def are_rings_empty(rings_info: dict) -> bool:
    """Return True if both types and count are missing/empty."""
    if rings_info is None:
        return True
    return not rings_info.get('types') and not rings_info.get('count')

def safe_search(pattern: str, text: str | None, flags: int = 0) -> re.Match | None:
    """Safely perform regex search on text that might be None or non-string."""
    try:
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        return re.search(pattern, text, flags=flags)
    except Exception as e:
        logging.exception("safe_search failed for pattern %r on text %r: %s", pattern, text, e)
        return None