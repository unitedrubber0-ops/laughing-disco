"""
Helper functions for Google Cloud Vision API (distinct from Gemini).

Provides OCR, object detection, and document analysis capabilities.

Usage:
- Set up Google Cloud credentials:
  * Create a service account in Google Cloud Console.
  * Download JSON key and set GOOGLE_CLOUD_CREDENTIALS env var or GOOGLE_APPLICATION_CREDENTIALS.
- Call analyze_image_with_google_vision(image_path, feature_type) to perform analysis.

Note: This module safely checks for the google.cloud.vision package and credentials.
If not configured, it returns a clear error rather than failing silently.
"""
import os
import json
import base64
import logging
from typing import Dict, Any, Optional, List

try:
    from google.cloud import vision
    from google.api_core import exceptions as gcp_exceptions
except Exception:
    vision = None
    gcp_exceptions = None

logger = logging.getLogger(__name__)

# Try to initialize credentials from environment
GOOGLE_CREDENTIALS = None
try:
    # Check for explicit credentials path or use default Google Application Credentials
    creds_path = os.getenv("GOOGLE_CLOUD_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        GOOGLE_CREDENTIALS = True
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        GOOGLE_CREDENTIALS = True
except Exception:
    pass


def analyze_image_with_google_vision(
    image_path: str,
    feature_types: Optional[List[str]] = None,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Analyze an image using Google Cloud Vision API.

    Args:
        image_path: Path to image file (PNG, JPEG, GIF, BMP, WebP, PDF, TIFF).
        feature_types: List of feature types to detect.
                       Default: ['TEXT_DETECTION', 'DOCUMENT_TEXT_DETECTION']
                       Options: TEXT_DETECTION, DOCUMENT_TEXT_DETECTION, LABEL_DETECTION,
                                OBJECT_LOCALIZATION, SAFE_SEARCH_DETECTION, etc.
        max_results: Max results per feature type.

    Returns:
        Dict with keys: success (bool), response (str), parsed (dict/list if JSON),
                        features (dict with detected elements by type).
                        If error: includes 'error' key with explanation.
    """
    if vision is None:
        logger.error("google.cloud.vision package not installed. Install with `pip install google-cloud-vision`")
        return {"success": False, "error": "google.cloud.vision package not installed"}

    if not GOOGLE_CREDENTIALS:
        logger.error(
            "Google Cloud credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS env var "
            "or GOOGLE_CLOUD_CREDENTIALS pointing to your service account JSON."
        )
        return {"success": False, "error": "Google Cloud credentials not configured"}

    if feature_types is None:
        feature_types = ["DOCUMENT_TEXT_DETECTION"]  # Default to document OCR

    try:
        # Read image
        with open(image_path, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        logger.exception("Failed to read image %s: %s", image_path, e)
        return {"success": False, "error": f"Failed to read image: {str(e)}"}

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=img_bytes)

    # Build requests for each feature type
    features = []
    for ftype in feature_types:
        try:
            features.append(vision.Feature(type_=getattr(vision.Feature.Type, ftype), max_results=max_results))
        except AttributeError:
            logger.warning("Unknown feature type: %s", ftype)

    if not features:
        return {"success": False, "error": f"No valid feature types in {feature_types}"}

    try:
        logger.info("Calling Google Cloud Vision API for image %s with features %s", image_path, feature_types)
        request = vision.AnnotateImageRequest(image=image, features=features)
        response = client.annotate_image(request)

        result = {"success": True, "features": {}, "response": ""}

        # Extract TEXT_DETECTION results
        if response.text_annotations:
            text = response.text_annotations[0].description if response.text_annotations else ""
            result["response"] = text
            result["features"]["text"] = text
            logger.info("Extracted text from image: %d characters", len(text))

        # Extract DOCUMENT_TEXT_DETECTION results (more structured)
        if response.full_text_annotation:
            doc_text = response.full_text_annotation.text
            result["features"]["document_text"] = doc_text
            if not result["response"]:
                result["response"] = doc_text
            logger.info("Extracted document text: %d characters", len(doc_text))

        # Extract other annotation types
        if response.label_annotations:
            labels = [{"description": label.description, "score": label.score} for label in response.label_annotations]
            result["features"]["labels"] = labels

        if response.object_annotations:
            objects = [
                {"name": obj.name, "score": obj.score, "bounding_poly": obj.bounding_poly}
                for obj in response.object_annotations
            ]
            result["features"]["objects"] = objects

        if response.safe_search_annotation:
            safe_search = {
                "adult": response.safe_search_annotation.adult,
                "spoof": response.safe_search_annotation.spoof,
                "medical": response.safe_search_annotation.medical,
                "violence": response.safe_search_annotation.violence,
            }
            result["features"]["safe_search"] = safe_search

        # Try to parse response as JSON if it looks like JSON
        parsed = None
        try:
            # Check if response contains JSON-like content
            response_str = result.get("response", "")
            if response_str.strip().startswith(("{", "[")):
                parsed = json.loads(response_str)
        except json.JSONDecodeError:
            pass

        result["parsed"] = parsed
        return result

    except gcp_exceptions.GoogleAPICallError as e:
        logger.exception("Google Cloud Vision API error: %s", e)
        return {"success": False, "error": f"API error: {str(e)}"}
    except Exception as e:
        logger.exception("Unexpected error in Google Vision analysis: %s", e)
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


def extract_text_from_image(image_path: str) -> Optional[str]:
    """
    Convenience function to extract text from image using Google Vision API.

    Returns:
        Extracted text string, or None if extraction failed.
    """
    result = analyze_image_with_google_vision(image_path, feature_types=["DOCUMENT_TEXT_DETECTION"])
    if result.get("success"):
        return result.get("response") or result.get("features", {}).get("document_text")
    return None


def extract_coordinates_from_image(image_path: str) -> Optional[List[Dict[str, float]]]:
    """
    Extract coordinate points from an engineering drawing using Google Vision OCR.

    Parses text response to find patterns like "P0 (x, y, z)" or "POINT x y z".

    Returns:
        List of coordinate dicts with x, y, z keys, or None if extraction failed.
    """
    import re

    text = extract_text_from_image(image_path)
    if not text:
        return None

    # Use the same parser as existing coordinate extraction
    coords = []
    for line in text.splitlines():
        if not line or line.strip() == "":
            continue

        # Try to find patterns like P0, P1, POINT 0, etc.
        nums_raw = re.findall(r"-?\d+(?:[.,]\d+)?", line)
        nums = []
        for tok in nums_raw:
            s = str(tok).strip()
            if "," in s and "." in s:
                s = s.replace(",", "")
            elif "," in s:
                s = s.replace(",", ".")
            try:
                nums.append(float(s))
            except Exception:
                continue

        if len(nums) < 3:
            continue

        label_match = re.match(r"^(P\d+|POINT\s*\d+|P\s*\d+)\b", line, re.IGNORECASE)
        label = label_match.group(1).replace(" ", "") if label_match else None

        coords.append({"point": label, "x": nums[0], "y": nums[1], "z": nums[2], "r": nums[3] if len(nums) >= 4 else None})

    return coords if coords else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze an image with Google Cloud Vision API")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "-f",
        "--features",
        nargs="+",
        default=["DOCUMENT_TEXT_DETECTION"],
        help="Feature types to detect (default: DOCUMENT_TEXT_DETECTION)",
    )
    args = parser.parse_args()

    out = analyze_image_with_google_vision(args.image, feature_types=args.features)
    print(json.dumps(out, indent=2, default=str))
