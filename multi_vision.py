"""
Multi-vision orchestrator: intelligently routes image analysis to Claude or Google Vision.

This module provides a unified interface to extract coordinates, text, and specs from 
engineering drawings using either Claude Vision or Google Cloud Vision, with smart 
fallback and caching.

Usage:
    from multi_vision import extract_drawing_specs
    
    specs = extract_drawing_specs(image_path)  # Tries Claude, falls back to Google
    coords = extract_coordinates(image_path)    # Same fallback strategy
    text = extract_text(image_path)             # Same fallback strategy
"""
import os
import json
import logging
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Attempt imports for each vision provider
try:
    from claude_vision import analyze_image_with_claude
    CLAUDE_AVAILABLE = True
except Exception as e:
    CLAUDE_AVAILABLE = False
    logger.debug("Claude Vision not available: %s", e)

try:
    from google_vision import analyze_image_with_google_vision, extract_text_from_image as google_extract_text
    GOOGLE_VISION_AVAILABLE = True
except Exception as e:
    GOOGLE_VISION_AVAILABLE = False
    logger.debug("Google Vision not available: %s", e)

# Load environment flags
ENABLE_CLAUDE = os.getenv("ENABLE_CLAUDE_VISION", "true").lower() in ("true", "1", "yes")
ENABLE_GOOGLE = os.getenv("ENABLE_GOOGLE_VISION", "true").lower() in ("true", "1", "yes")

CACHE_DIR = Path(".cache/vision")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _image_hash(image_path: str) -> str:
    """Compute SHA256 hash of image file for caching."""
    try:
        with open(image_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return None


def _cache_path(image_hash: str, provider: str, feature: str) -> Path:
    """Build cache file path."""
    return CACHE_DIR / f"{image_hash}_{provider}_{feature}.json"


def _get_cached(image_path: str, provider: str, feature: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached result if available."""
    img_hash = _image_hash(image_path)
    if not img_hash:
        return None
    
    cache_file = _cache_path(img_hash, provider, feature)
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
                logger.debug("Using cached %s result for %s", provider, feature)
                return cached
        except Exception as e:
            logger.debug("Failed to read cache for %s: %s", feature, e)
    return None


def _set_cache(image_path: str, provider: str, feature: str, result: Dict[str, Any]) -> None:
    """Store result in cache."""
    img_hash = _image_hash(image_path)
    if not img_hash:
        return
    
    cache_file = _cache_path(img_hash, provider, feature)
    try:
        with open(cache_file, "w") as f:
            json.dump(result, f)
            logger.debug("Cached %s result for %s", provider, feature)
    except Exception as e:
        logger.debug("Failed to cache result: %s", e)


def extract_text(image_path: str, prefer_provider: Optional[str] = None) -> Optional[str]:
    """
    Extract text from image using Claude or Google Vision.
    
    Args:
        image_path: Path to image file.
        prefer_provider: 'claude' or 'google' to prefer one provider. If None, tries both.
    
    Returns:
        Extracted text or None if all providers failed.
    """
    # Try cache first
    cached = _get_cached(image_path, "any", "text")
    if cached:
        return cached.get("text")

    text = None
    used_provider = None

    # Determine provider order
    providers = []
    if prefer_provider == "claude" or (prefer_provider is None and ENABLE_CLAUDE):
        providers.append(("claude", CLAUDE_AVAILABLE))
    if prefer_provider == "google" or (prefer_provider is None and ENABLE_GOOGLE):
        providers.append(("google", GOOGLE_VISION_AVAILABLE))

    for provider, available in providers:
        if not available:
            logger.debug("%s not available, skipping", provider)
            continue

        try:
            if provider == "claude":
                prompt = "Extract all text from this engineering drawing. Return the raw text."
                resp = analyze_image_with_claude(image_path, prompt)
                if resp.get("success"):
                    text = resp.get("response") or resp.get("parsed", {}).get("text")
                    used_provider = "claude"
            elif provider == "google":
                text = google_extract_text(image_path)
                used_provider = "google"

            if text:
                logger.info("Extracted text using %s (%d chars)", used_provider, len(text))
                break
        except Exception as e:
            logger.warning("Error extracting text with %s: %s", provider, e)

    # Cache and return
    if text:
        _set_cache(image_path, used_provider or "unknown", "text", {"text": text})
    return text


def extract_coordinates(image_path: str, prefer_provider: Optional[str] = None) -> Optional[List[Dict[str, float]]]:
    """
    Extract coordinate points from engineering drawing.
    
    Args:
        image_path: Path to image file.
        prefer_provider: 'claude' or 'google' to prefer one provider.
    
    Returns:
        List of coordinate dicts (x, y, z, r keys) or None if extraction failed.
    """
    # Try cache first
    cached = _get_cached(image_path, "any", "coordinates")
    if cached:
        return cached.get("coordinates")

    coords = None
    used_provider = None

    # Determine provider order
    providers = []
    if prefer_provider == "claude" or (prefer_provider is None and ENABLE_CLAUDE):
        providers.append(("claude", CLAUDE_AVAILABLE))
    if prefer_provider == "google" or (prefer_provider is None and ENABLE_GOOGLE):
        providers.append(("google", GOOGLE_VISION_AVAILABLE))

    for provider, available in providers:
        if not available:
            logger.debug("%s not available, skipping", provider)
            continue

        try:
            if provider == "claude":
                prompt = (
                    "Extract coordinate points from this engineering drawing. "
                    "Return a JSON array of objects with keys x, y, z (optional r). "
                    "Example: [{\"x\": 0, \"y\": 100, \"z\": 200}, ...]"
                )
                resp = analyze_image_with_claude(image_path, prompt)
                if resp.get("success"):
                    parsed = resp.get("parsed")
                    if isinstance(parsed, list):
                        coords = [c for c in parsed if isinstance(c, dict) and "x" in c and "y" in c]
                    elif isinstance(parsed, dict):
                        coords = parsed.get("coordinates") or parsed.get("points")
                    if coords:
                        used_provider = "claude"

            elif provider == "google":
                from google_vision import extract_coordinates_from_image
                coords = extract_coordinates_from_image(image_path)
                if coords:
                    used_provider = "google"

            if coords:
                logger.info("Extracted %d coordinates using %s", len(coords), used_provider)
                break
        except Exception as e:
            logger.warning("Error extracting coordinates with %s: %s", provider, e)

    # Cache and return
    if coords:
        _set_cache(image_path, used_provider or "unknown", "coordinates", {"coordinates": coords})
    return coords


def extract_drawing_specs(image_path: str, prefer_provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract complete drawing specifications (part number, standard, grade, dimensions, coords).
    
    Args:
        image_path: Path to image file.
        prefer_provider: 'claude' or 'google' to prefer one provider.
    
    Returns:
        Dict with keys: part_number, standard, grade, dimensions, coordinates, material, etc.
        Empty or partially filled dict if extraction failed.
    """
    # Try cache first
    cached = _get_cached(image_path, "any", "specs")
    if cached:
        return cached.get("specs", {})

    specs = {
        "part_number": None,
        "description": None,
        "standard": None,
        "grade": None,
        "material": None,
        "dimensions": {},
        "coordinates": [],
    }
    used_provider = None

    # Determine provider order
    providers = []
    if prefer_provider == "claude" or (prefer_provider is None and ENABLE_CLAUDE):
        providers.append(("claude", CLAUDE_AVAILABLE))
    if prefer_provider == "google" or (prefer_provider is None and ENABLE_GOOGLE):
        providers.append(("google", GOOGLE_VISION_AVAILABLE))

    for provider, available in providers:
        if not available:
            logger.debug("%s not available, skipping", provider)
            continue

        try:
            if provider == "claude":
                prompt = """Extract engineering drawing specifications and return as JSON.
Required fields: part_number, standard, grade, material, dimensions (id, od, thickness, centerline_length), coordinates.
Example format:
{
  "part_number": "1234567C1",
  "description": "Hose assembly",
  "standard": "MPAPS F-6032",
  "grade": "TYPE I",
  "material": "P-EPDM",
  "dimensions": {
    "id": 12.7,
    "od": 25.4,
    "thickness": 6.35,
    "centerline_length": 100.0
  },
  "coordinates": [
    {"x": 0, "y": 100, "z": 200},
    {"x": 50, "y": 150, "z": 250}
  ]
}
Return only JSON, no other text."""
                resp = analyze_image_with_claude(image_path, prompt)
                if resp.get("success"):
                    parsed = resp.get("parsed") or {}
                    if isinstance(parsed, dict):
                        specs.update(parsed)
                        used_provider = "claude"

            elif provider == "google":
                # Google Vision: extract text and manually parse
                text = google_extract_text(image_path)
                if text:
                    # For now, Google Vision helps with text extraction; spec parsing could be added here
                    specs["_raw_text_from_google"] = text
                    used_provider = "google"

            if used_provider:
                logger.info("Extracted specs using %s", used_provider)
                break

        except Exception as e:
            logger.warning("Error extracting specs with %s: %s", provider, e)

    # Cache and return
    _set_cache(image_path, used_provider or "unknown", "specs", {"specs": specs})
    return specs


def get_provider_status() -> Dict[str, bool]:
    """Return availability status of each vision provider."""
    return {
        "claude": CLAUDE_AVAILABLE and ENABLE_CLAUDE,
        "google_vision": GOOGLE_VISION_AVAILABLE and ENABLE_GOOGLE,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-vision extractor for engineering drawings")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("-t", "--text", action="store_true", help="Extract text only")
    parser.add_argument("-c", "--coords", action="store_true", help="Extract coordinates only")
    parser.add_argument("-s", "--specs", action="store_true", help="Extract full specifications")
    parser.add_argument("-p", "--provider", choices=["claude", "google"], help="Prefer specific provider")

    args = parser.parse_args()

    print("Vision Provider Status:", json.dumps(get_provider_status(), indent=2))

    if args.text or (not args.text and not args.coords and not args.specs):
        print("\n=== Text Extraction ===")
        text = extract_text(args.image, prefer_provider=args.provider)
        print(text or "No text extracted")

    if args.coords:
        print("\n=== Coordinate Extraction ===")
        coords = extract_coordinates(args.image, prefer_provider=args.provider)
        print(json.dumps(coords, indent=2) if coords else "No coordinates extracted")

    if args.specs:
        print("\n=== Full Specifications ===")
        specs = extract_drawing_specs(args.image, prefer_provider=args.provider)
        print(json.dumps(specs, indent=2, default=str))
