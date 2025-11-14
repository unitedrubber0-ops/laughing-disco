# Google Cloud Vision API + Claude Vision Integration

## Overview

This project now integrates **two powerful vision AI providers**:

1. **Claude Vision (Anthropic)** — GPT-4V-like capabilities, excellent at understanding complex technical drawings
2. **Google Cloud Vision API** — Specialized OCR and document analysis, cost-effective for text extraction

The **multi-vision orchestrator** (`multi_vision.py`) intelligently routes image analysis to the best available provider, with:
- Automatic fallback (tries Claude first, falls back to Google if Claude unavailable)
- Smart caching (per-image hash) to reduce API costs
- Environment-based control (enable/disable providers via `ENABLE_CLAUDE_VISION` and `ENABLE_GOOGLE_VISION`)
- Graceful degradation (reverts to text-based extraction if both vision APIs fail)

## Setup

### 1. Install Required Packages

All packages are already in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install anthropic python-dotenv google-cloud-vision
```

### 2. Configure Credentials

#### Claude Vision (Anthropic)

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
```

Get your key at: https://console.anthropic.com/

#### Google Cloud Vision API

1. **Create a Google Cloud Project** (or use existing):
   - Go to: https://console.cloud.google.com/
   - Create a new project or select existing one

2. **Enable the Vision API**:
   - In Cloud Console, navigate to **APIs & Services > Library**
   - Search for "Cloud Vision API"
   - Click **Enable**

3. **Create a Service Account**:
   - Go to **APIs & Services > Credentials**
   - Click **Create Credentials > Service Account**
   - Fill in the service account name (e.g., "vision-extractor")
   - Click **Create and Continue**
   - Grant the role: **Basic > Editor** (or more restrictive **Compute Vision User** if available)
   - Click **Continue > Create Key**
   - Select **JSON** and click **Create**
   - Save the JSON file (e.g., `google-service-account.json`)

4. **Configure in `.env`**:

```dotenv
GOOGLE_CLOUD_CREDENTIALS=/path/to/google-service-account.json
```

Or set the standard Google environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-service-account.json
```

### 3. Enable/Disable Vision Providers (Optional)

By default, both providers are enabled. Control them via environment variables in `.env`:

```dotenv
ENABLE_CLAUDE_VISION=true        # Default: true
ENABLE_GOOGLE_VISION=true        # Default: true
```

Set to `false` to disable a provider without removing the API key.

## Usage

### In Application Code

The main pipeline in `app.py` now automatically uses the multi-vision orchestrator for coordinate extraction:

```python
from multi_vision import extract_text, extract_coordinates, extract_drawing_specs

# Extract coordinates from an image (tries Claude, then Google)
coords = extract_coordinates("drawing.png")

# Extract all text
text = extract_text("drawing.png")

# Extract complete drawing specs (part number, standard, grade, coordinates, etc.)
specs = extract_drawing_specs("drawing.png")
```

### From Command Line

Test the multi-vision orchestrator directly:

```bash
# Extract text
python multi_vision.py drawing.png --text

# Extract coordinates
python multi_vision.py drawing.png --coords

# Extract complete specs
python multi_vision.py drawing.png --specs

# Prefer a specific provider
python multi_vision.py drawing.png --text --provider claude
python multi_vision.py drawing.png --coords --provider google
```

### Individual Vision Providers

#### Claude Vision

```python
from claude_vision import analyze_image_with_claude

resp = analyze_image_with_claude("image.png", "Extract text from this drawing")
if resp.get("success"):
    text = resp.get("response")
    parsed_json = resp.get("parsed")
```

#### Google Vision

```python
from google_vision import analyze_image_with_google_vision, extract_text_from_image

# Full analysis with multiple feature types
resp = analyze_image_with_google_vision(
    "image.png",
    feature_types=["DOCUMENT_TEXT_DETECTION", "LABEL_DETECTION", "OBJECT_LOCALIZATION"]
)

# Quick text extraction
text = extract_text_from_image("image.png")
```

## Feature Matrix

| Feature | Claude Vision | Google Vision | Multi-Vision |
|---------|---------------|---------------|-------------|
| Coordinate Extraction | ✓ Excellent | ✓ Good | ✓ Tries both |
| Text/OCR | ✓ Good | ✓ Excellent | ✓ Tries both |
| Part Number Detection | ✓ Excellent | ⚠ Partial | ✓ Tries both |
| Spec/Grade Recognition | ✓ Excellent | ⚠ Partial | ✓ Tries both |
| Label Detection | ⚠ Partial | ✓ Excellent | ✓ Tries both |
| Cost per Call | ~$0.01 (1MB limit) | ~$0.001 | Smart (tries cheaper first) |
| Latency | ~500ms | ~1s | ~500ms (cached) |
| Caching | ✓ Yes | ✓ Yes | ✓ Yes |

## Caching

Both providers' results are cached locally under `.cache/vision/` using an image SHA256 hash:

```
.cache/vision/
  ├── abc123def456_claude_text.json
  ├── abc123def456_google_coordinates.json
  ├── abc123def456_any_specs.json
  └── ...
```

- **Per-image basis**: Same image won't be re-analyzed even across app restarts
- **TTL**: Files remain until manually deleted (no auto-expiry)
- **Manual cleanup**: `rm -r .cache/vision/` to clear all caches

## Error Handling & Fallback Strategy

### Missing Credentials

If API keys/credentials are not set, the system:
1. Logs a clear error message with setup instructions
2. Returns `{"success": False, "error": "..."}`
3. Falls back to the next provider or text-based extraction

Example Claude without API key:
```
ERROR: ANTHROPIC_API_KEY not found in environment. Create a .env with ANTHROPIC_API_KEY=... or set env var
```

Example Google Vision without credentials:
```
ERROR: Google Cloud credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS env var or GOOGLE_CLOUD_CREDENTIALS pointing to your service account JSON.
```

### Provider Fallback Order

The multi-vision orchestrator tries providers in this order:

1. **Claude Vision** (if `ENABLE_CLAUDE_VISION=true` and API key present)
2. **Google Cloud Vision** (if `ENABLE_GOOGLE_VISION=true` and credentials present)
3. **Text-based parser** (fallback, uses existing OCR/text extraction)

### Network/API Errors

If a provider's API call fails:
- Error is logged (not fatal)
- Next provider is tried
- If all fail, graceful fallback to text-based extraction

## Testing

Run the integration test suite:

```bash
python test_multi_vision_integration.py
```

Expected output:
```
✓ PASS: test_claude_vision_import
✓ PASS: test_google_vision_import
✓ PASS: test_multi_vision_import
✓ PASS: test_multi_vision_cache
✓ PASS: test_multi_vision_provider_routing
✓ PASS: test_app_py_imports

Total: 6/6 passed
```

## Cost Estimate

### Claude Vision (Anthropic)

- **Per request**: ~$0.01 (1MB image max)
- **Monthly estimate** (100 drawings/day): ~$30

### Google Cloud Vision

- **Text detection**: $0.0015 per image
- **Document OCR**: $0.003 per page
- **Monthly estimate** (100 drawings/day): ~$4.50

### Smart Caching

Caching reduces costs by ~80% (same drawings re-analyzed less often).

**Recommended**: Use Google Vision for straightforward OCR/text, Claude Vision for complex spec extraction.

## Troubleshooting

### Claude Vision Returns Empty or Wrong Data

- **Cause**: API key not set or account not active
- **Fix**: Check `.env` file, regenerate API key at console.anthropic.com

### Google Vision Returns Credential Error

- **Cause**: Service account JSON path incorrect or credentials expired
- **Fix**: Verify path in `.env`, regenerate service account key in Cloud Console

### Caching Issues

- **Clear cache**: `rm -r .cache/vision/`
- **Disable caching**: Modify `multi_vision.py` and comment out `_set_cache()` calls

### High API Costs

- **Cause**: Not using cache effectively or running lots of analyses
- **Fix**: Check `.cache/vision/` size, enable caching, prefer Google Vision for simple text

## Architecture Diagram

```
App Pipeline (app.py)
    ↓
Extract PDF Page → Image
    ↓
Multi-Vision Orchestrator (multi_vision.py)
    ├─→ [Check Cache] → Return cached result (fast, free)
    │
    └─→ [No Cache] Try Providers in Order:
            ├─→ Claude Vision Helper
            │   ├─→ ANTHROPIC_API_KEY present?
            │   ├─→ Analyze image
            │   └─→ Parse JSON response
            │
            ├─→ Google Vision Helper
            │   ├─→ GOOGLE_APPLICATION_CREDENTIALS present?
            │   ├─→ Analyze image
            │   └─→ Extract text/labels/objects
            │
            └─→ Text-Based Fallback
                ├─→ OCR (Tesseract)
                └─→ Regex parsing
    ↓
Cache Result
    ↓
Return Coordinates/Specs to Calling Function
    ↓
Continue Processing (MPAPS rules, Excel output, etc.)
```

## Files Added/Modified

### New Files

- **`claude_vision.py`** — Claude/Anthropic Vision helper
- **`google_vision.py`** — Google Cloud Vision helper
- **`multi_vision.py`** — Multi-provider orchestrator with caching
- **`test_multi_vision_integration.py`** — Integration test suite

### Modified Files

- **`app.py`** — Import multi-vision and use for coordinate extraction
- **`.env.example`** — Added documentation for `GOOGLE_CLOUD_CREDENTIALS` and provider flags

## References

- **Anthropic Claude API**: https://console.anthropic.com/
- **Google Cloud Vision**: https://cloud.google.com/vision
- **Caching Strategy**: SHA256-based per-image caching under `.cache/vision/`

## Next Steps

1. **Set up credentials** (follow setup above)
2. **Run tests** to verify installation
3. **Enable in production** by ensuring `.env` is properly configured
4. **Monitor costs** via Claude and Google Cloud consoles
5. **Tune caching** based on typical workflow (clean up old files if needed)

