# hnp.py — build the Admission H&P from prompt + combined text
from pathlib import Path
from openai import OpenAI
import os

_DEF_MODEL = os.environ.get("NOTEWRITER_MODEL", "gpt-5")

_client = None

def _client_or_init():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def build_hnp(prompt_text: str, combined_text: str, max_tokens: int = 1800) -> str:
    client = _client_or_init()
    model = os.environ.get("NOTEWRITER_MODEL", _DEF_MODEL)
    content = [
        {"type":"input_text","text": prompt_text},
        {"type":"input_text","text": "\n\n---\nRAW EXTRACTED DATA (from images/text files):\n" + combined_text}
    ]
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role":"user","content": content}],
            max_output_tokens=max_tokens,
        )
        txt = getattr(resp, "output_text", "").strip()
        if txt:
            return txt
        # fallback
        resp = client.responses.create(
            model=os.environ.get("NOTEWRITER_FALLBACK","gpt-4.1"),
            input=[{"role":"user","content": content}],
            max_output_tokens=max_tokens,
        )
        return getattr(resp, "output_text", "").strip()
    except Exception as e:
        raise RuntimeError(f"H&P generation failed: {e}")