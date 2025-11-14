"""
Helper functions for using Claude Vision (Anthropic) to analyze images.

Usage:
- Put your Claude/Anthropic API key in a .env file as ANTHROPIC_API_KEY
- Call analyze_image_with_claude(image_path, prompt) to send an image + prompt

Note: This module does not call the API unless ANTHROPIC_API_KEY is set. The sample
function returns a helpful error if the key is missing.
"""
import os
import json
import base64
import logging
from typing import Dict, Any, Optional

try:
    import anthropic
except Exception:
    anthropic = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Example model name - replace with the relevant image-enabled Claude model available to your account
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


def analyze_image_with_claude(image_path: str, prompt_text: str, model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> Dict[str, Any]:
    """
    Send an image with a text prompt to Claude (Anthropic) and return the parsed JSON response.

    Returns a dict with keys: success (bool), response (raw text) and parsed (if JSON parseable).

    Note: This requires an Anthropic API key set in environment as ANTHROPIC_API_KEY.
    """
    if anthropic is None:
        logger.error("anthropic package not installed. Install with `pip install anthropic`")
        return {"success": False, "error": "anthropic package not installed"}

    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not found in environment. Create a .env with ANTHROPIC_API_KEY=... or set env var")
        return {"success": False, "error": "ANTHROPIC_API_KEY not set"}

    # Read and base64 encode image
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.exception("Failed to read image %s: %s", image_path, e)
        return {"success": False, "error": str(e)}

    client = anthropic.Client(api_key=ANTHROPIC_API_KEY)

    # Build a message containing image and prompt. The exact shape supported may vary by SDK/model; adjust if needed.
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ]

        logger.info("Sending image to Claude model %s", model)
        response = client.messages.create(
            model=model,
            messages=messages,
            max_tokens_to_sample=max_tokens,
        )

        # The SDK returns a structure; attempt to extract text
        raw_text = None
        try:
            # Response shape may be response.content or response.output
            raw_text = None
            if hasattr(response, 'content'):
                # content can be list-like or string
                raw = response.content
                raw_text = json.dumps(raw) if not isinstance(raw, str) else raw
            elif hasattr(response, 'output'):
                raw_text = response.output.get('text') if isinstance(response.output, dict) else str(response.output)
            else:
                raw_text = str(response)
        except Exception:
            raw_text = str(response)

        parsed = None
        try:
            parsed = json.loads(raw_text)
        except Exception:
            # Not JSON, leave parsed None but return raw
            parsed = None

        return {"success": True, "response": raw_text, "parsed": parsed}

    except Exception as e:
        logger.exception("Claude API call failed: %s", e)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze an image with Claude Vision (Anthropic)")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("-p", "--prompt", default="Extract coordinate points and return JSON.", help="Text prompt to send with image")
    args = parser.parse_args()

    out = analyze_image_with_claude(args.image, args.prompt)
    print(json.dumps(out, indent=2))
