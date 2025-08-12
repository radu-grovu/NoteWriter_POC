# ocr.py — Hybrid auto‑interpret OCR (fast + concise)
# Strategy (max 2 calls/image):
#   1) JPEG‑normalize and send
#   2) Quick bottom crop and resend (tables often below charts)
# Behavior:
#   • If image is mostly chart/dashboard, extract labels/units/numbers AND add a brief interpretation.
#   • If image is mostly text/notes/tables, return raw text (tables as TSV) with NO interpretation.
#   • Returns '[[NO_TEXT_FOUND]]' if nothing readable; never raises.
#   • Falls back to NOTEWRITER_FALLBACK model if the selected model is unavailable.

from __future__ import annotations
import base64, os, io
from pathlib import Path
from openai import OpenAI
from PIL import Image, ImageOps, ImageEnhance

# ------------------- config via environment -------------------
_DEF_MODEL   = os.environ.get("NOTEWRITER_MODEL", "gpt-5")
_FALLBACK    = os.environ.get("NOTEWRITER_FALLBACK", "gpt-4.1")
_MAXTOK      = int(os.environ.get("NOTEWRITER_OCR_TOKENS", "600"))   # 400–800 is a good range
_JPEG_W      = int(os.environ.get("NOTEWRITER_OCR_WIDTH",  "1400"))  # resize width before send

_client: OpenAI | None = None

def _client_or_init() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

# Single hybrid prompt (auto decides whether to add interpretation)
_PROMPT_HYBRID = (
    "Task: Read this clinical screenshot. First, extract ALL readable on‑screen text in natural reading order. "
    "Format tables as TSV: each row on one line, with cells separated by a single TAB. "
    "If the image is primarily a chart/dash/graph with minimal prose, after extracting any labels/units/legends and key numbers, "
    "append 1–3 short bullet(s) interpreting the trend/outliers. "
    "If there is substantial paragraph or tabular text (e.g., provider notes, lab tables), do NOT add interpretation—just return the text. "
    "If nothing readable, return [[NO_TEXT_FOUND]]. Output ONLY text."
)

# ------------------- image helpers -------------------

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def _data_url_jpeg(jpeg_bytes: bytes) -> str:
    return f"data:image/jpeg;base64,{_b64(jpeg_bytes)}"


def _preprocess_jpeg(p: Path) -> bytes:
    """Normalize: RGB, resize to ~_JPEG_W, light sharpen/contrast, autocontrast, JPEG @ q=88."""
    im = Image.open(p).convert("RGB")
    if im.width > _JPEG_W:
        im = im.resize((_JPEG_W, int(_JPEG_W * im.height / im.width)), Image.LANCZOS)
    im = ImageEnhance.Sharpness(im).enhance(1.2)
    im = ImageEnhance.Contrast(im).enhance(1.08)
    im = ImageOps.autocontrast(im)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


def _bottom_crop(jpeg_bytes: bytes) -> bytes:
    im = Image.open(io.BytesIO(jpeg_bytes))
    w, h = im.size
    crop = im.crop((0, int(h*0.45), w, h))  # bottom 55% — tends to capture tables under charts
    out = io.BytesIO()
    crop.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue()

# ------------------- network helpers -------------------

def _try_once(data_url: str, model: str, prompt: str, max_tokens: int) -> str | None:
    r = _client_or_init().responses.create(
        model=model,
        input=[{"role":"user","content":[
            {"type":"input_text","text": prompt},
            {"type":"input_image","image_url": data_url}
        ]}],
        max_output_tokens=max_tokens,
        temperature=0,
    )
    t = getattr(r, "output_text", "").strip()
    return t if t and t != "[[NO_TEXT_FOUND]]" else None


def _try_with_fallback(data_url: str, prompt: str, max_tokens: int) -> str | None:
    model = os.environ.get("NOTEWRITER_MODEL", _DEF_MODEL)
    try:
        return _try_once(data_url, model, prompt, max_tokens)
    except Exception as e:
        # Quick model fallback on capability/availability errors
        if "model" in str(e).lower() or "not found" in str(e).lower():
            try:
                return _try_once(data_url, os.environ.get("NOTEWRITER_FALLBACK", _FALLBACK), prompt, max_tokens)
            except Exception:
                return None
        # For other errors (e.g., transient network), bail fast
        return None

# ------------------- public API -------------------

def ocr_image(path: Path, max_tokens: int | None = None) -> str:
    """
    ChatGPT‑style, fast OCR with small fallback:
      1) JPEG‑normalize & send
      2) Quick bottom‑crop & send
    Auto‑interprets charts; raw OCR for text‑heavy screens. Returns '[[NO_TEXT_FOUND]]' if nothing readable.
    """
    tok = max_tokens or _MAXTOK

    # Pass 1 — normalized full frame
    jpg = _preprocess_jpeg(path)
    text = _try_with_fallback(_data_url_jpeg(jpg), _PROMPT_HYBRID, tok)
    if text:
        return text

    # Pass 2 — quick bottom crop (tables often live below charts)
    crop = _bottom_crop(jpg)
    text = _try_with_fallback(_data_url_jpeg(crop), _PROMPT_HYBRID, tok)
    return text or "[[NO_TEXT_FOUND]]"
