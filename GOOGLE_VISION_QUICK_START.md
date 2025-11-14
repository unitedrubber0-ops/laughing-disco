# Quick Start: Google Vision AI + Claude Vision

## 5-Minute Setup (No API Keys Required)

The system **works without any API keys** — it gracefully falls back to text-based extraction:

```bash
# Just run the app
python app.py
```

✓ Done. Coordinates will be extracted from PDF text if vision APIs aren't configured.

---

## With Claude Vision (Recommended for Specs)

Get your **Anthropic API key** from: https://console.anthropic.com/

```bash
# 1. Copy .env.example
cp .env.example .env

# 2. Edit .env and add your API key
# ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE

# 3. Run the app
python app.py
```

✓ Now Claude Vision will extract coordinates and specs from drawings.

---

## With Google Vision (Recommended for OCR)

Set up Google Cloud:

```bash
# 1. Create a Google Cloud project (free tier available)
#    → https://console.cloud.google.com/

# 2. Enable Vision API:
#    → APIs & Services > Library > "Cloud Vision API" > Enable

# 3. Create service account:
#    → APIs & Services > Credentials > Create Service Account
#    → Name: "vision-extractor"
#    → Grant role: "Editor" (or "Compute Vision User")
#    → Create Key > JSON > Save as "google-service-account.json"

# 4. Edit .env
echo "GOOGLE_CLOUD_CREDENTIALS=$(pwd)/google-service-account.json" >> .env

# 5. Run the app
python app.py
```

✓ Now Google Vision will be fallback for text extraction if Claude unavailable.

---

## With Both (Production-Ready)

Set both API keys in `.env`:

```dotenv
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
GOOGLE_CLOUD_CREDENTIALS=/path/to/google-service-account.json
```

Then:
```bash
python app.py
```

✓ Claude tries first (specs/coordinates), Google provides fallback (OCR), caching reduces costs.

---

## Test It

```bash
# Run the integration tests
python test_multi_vision_integration.py

# Expected output: 6/6 tests PASS
```

---

## Direct Usage

```python
from multi_vision import extract_coordinates, extract_text

# Extract coordinates from a drawing image
coords = extract_coordinates("my_drawing.png")
print(coords)
# Output: [{'point': 'P0', 'x': 0.0, 'y': 100.0, 'z': 200.0, 'r': None}, ...]

# Extract text
text = extract_text("my_drawing.png")
print(text)
```

---

## Cost Estimate

| Setup | Monthly Cost (100 drawings/day) | Notes |
|-------|----------------------------------|-------|
| No APIs (text fallback) | $0 | Slower, less reliable for specs |
| Claude only | ~$30 | Great for specs, slower |
| Google only | ~$4.50 | Great for OCR, less reliable for specs |
| **Both + caching** | ~$6 | **Recommended**: Claude for specs, Google for OCR, 80% cached |

---

## Environment Variables

```dotenv
# API Keys (optional, but recommended)
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_CLOUD_CREDENTIALS=/path/to/service-account.json

# Enable/Disable providers (default: auto-detect)
ENABLE_CLAUDE_VISION=true
ENABLE_GOOGLE_VISION=true
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "ANTHROPIC_API_KEY not found" | Set in `.env` or `export ANTHROPIC_API_KEY=...` |
| "Google Cloud credentials not found" | Set `GOOGLE_CLOUD_CREDENTIALS` in `.env` or export `GOOGLE_APPLICATION_CREDENTIALS` |
| Cache is stale | Delete `.cache/vision/` directory |
| High API costs | Caching is enabled by default; clear old cache files if needed |

---

## What Happens (Without Config)

```
PDF Page 1
    ↓
Try: Claude Vision → ❌ No API key
    ↓
Try: Google Vision → ❌ No credentials
    ↓
Fallback: Text-based parser (Tesseract/Regex)
    ↓
Extract coordinates → ✓ Works (slower, less accurate)
```

## What Happens (With Both APIs)

```
PDF Page 1
    ↓
Check Cache → Hit ✓ Return instantly (free)
    ↓
Try: Claude Vision → ✓ Success (specs, coordinates)
    ↓
Cache result
    ↓
Extract coordinates → ✓ Works perfectly
```

---

## Next: Full Documentation

For detailed setup, feature matrix, architecture, and advanced usage:
→ See `GOOGLE_VISION_INTEGRATION.md`

For implementation details and internal API:
→ See docstrings in `multi_vision.py`, `claude_vision.py`, `google_vision.py`

---

## Files Added

- `claude_vision.py` — Claude Vision wrapper
- `google_vision.py` — Google Vision wrapper
- `multi_vision.py` — Multi-provider orchestrator (main integration point)
- `test_multi_vision_integration.py` — Test suite (6 tests, all passing)
- `GOOGLE_VISION_INTEGRATION.md` — Full documentation
- `.env.example` — Configuration template

## Status

✓ All tests passing (6/6)
✓ Graceful fallback (no broken pipelines)
✓ Caching enabled (cost reduction)
✓ Documentation complete
✓ Committed to `update-mpaps-tables` branch

