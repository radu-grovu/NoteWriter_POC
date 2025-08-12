# app.py — Streamlit UI for NoteWriter (runtime, paste‑first, inbox‑only)
import os, time
from pathlib import Path
import streamlit as st
from ocr import ocr_image
from hnp import build_hnp

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT/"inbox"
OUT = ROOT/"out"
PROMPT_FILE = ROOT/"prompt_hnp.txt"

INBOX.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

st.set_page_config(page_title="NoteWriter", page_icon="🩺", layout="wide")
st.title("NoteWriter — Admission H&P Builder")

# ——— Sidebar: API key & models
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OpenAI API key", type="password", help="Starts with sk-…")
if api_key:
    # sanitize to avoid non‑ASCII breaks
    cleaned = api_key.strip().encode("ascii", "ignore").decode("ascii")
    os.environ["OPENAI_API_KEY"] = cleaned

model = st.sidebar.selectbox("Model", ["gpt-5", "gpt-5-mini", "gpt-4.1", "o4-mini"], index=0)
os.environ["NOTEWRITER_MODEL"] = model
fallback = st.sidebar.text_input("Fallback model", value=os.environ.get("NOTEWRITER_FALLBACK","gpt-4.1"))
os.environ["NOTEWRITER_FALLBACK"] = fallback

# ——— API connectivity test (Ping)
st.sidebar.write("")  # spacer
if st.sidebar.button("Ping API"):
    try:
        from openai import OpenAI
        c = OpenAI()
        test_model = os.environ.get("NOTEWRITER_MODEL", "gpt-4.1")
        r = c.responses.create(
            model=test_model,
            input="ping",
            max_output_tokens=16  # must be >= 16
        )
        st.sidebar.success(f"API OK with {test_model}: {r.output_text[:40]!r}")
    except Exception as e:
        st.sidebar.error(f"API error: {e}")

# ——— Prompt editor
st.subheader("Prompt")
if PROMPT_FILE.exists():
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
else:
    prompt_text = ""
prompt_text = st.text_area("Edit your H&P prompt", value=prompt_text, height=260)
if st.button("Save prompt"):
    PROMPT_FILE.write_text(prompt_text, encoding="utf-8")
    st.success("Prompt saved.")

# ——— Input tabs (Paste FIRST, Inbox SECOND) — no upload tab
tab_paste, tab_folder = st.tabs(["Paste text", "Process inbox folder"]) 

with tab_paste:
    st.markdown("**Paste raw text** (optionally start with a descriptor line like `NOTE: ED Provider note …`)")
    pasted = st.text_area("Pasted text", height=220)

with tab_folder:
    st.write(f"Inbox folder: `{INBOX}`")
    folder_files = sorted([p for p in INBOX.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg",".txt")], key=lambda p: p.name)
    st.write(f"Found {len(folder_files)} file(s)")
    if folder_files:
        st.code("\n".join(p.name for p in folder_files))

# ——— Process button (inbox + pasted text only)
if st.button("Process → Build Admission H&P"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("API key required in the sidebar.")
    else:
        t_total_start = time.perf_counter()
        t_ocr_start = time.perf_counter()

        sources = []
        # 1) Handle inbox folder files (images → OCR, .txt → read)
        for p in sorted(INBOX.iterdir()):
            if p.suffix.lower() not in (".png",".jpg",".jpeg",".txt"):
                continue
            name = p.name
            if p.suffix.lower() in (".png",".jpg",".jpeg"):
                txt = ocr_image(p)
                if not txt or txt == "[[NO_TEXT_FOUND]]":
                    st.warning(f"OCR yielded no readable text for {name}; skipping.")
                    continue
            else:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                if not txt.strip():
                    st.warning(f"Empty text file: {name}; skipping.")
                    continue
            (OUT/f"{p.stem}.txt").write_text(txt, encoding="utf-8")
            sources.append(f"===== SOURCE: {name} =====\n{txt}\n")

        # 2) Handle pasted text
        if pasted.strip():
            sources.append(f"===== SOURCE: pasted.txt =====\n{pasted.strip()}\n")

        t_ocr = time.perf_counter() - t_ocr_start

        # 3) Build H&P
        combined = "\n".join(sources)
        if not combined.strip():
            st.warning("No input text collected from Inbox or Paste.")
        else:
            st.info("Generating Admission H&P…")
            t_note_start = time.perf_counter()
            try:
                note = build_hnp(prompt_text, combined)
                t_note = time.perf_counter() - t_note_start

                out_note = OUT/"admission_hnp.txt"
                out_note.write_text(note, encoding="utf-8")
                t_total = time.perf_counter() - t_total_start

                st.success(f"Done: {out_note}")
                # Runtime metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("OCR time", f"{t_ocr:.1f}s")
                c2.metric("H&P generation", f"{t_note:.1f}s")
                c3.metric("Total runtime", f"{t_total:.1f}s")

                st.download_button("Download H&P", data=note, file_name="admission_hnp.txt")
            except Exception as e:
                st.error(str(e))

# ——— Output preview
st.subheader("Output preview")
np = OUT/"admission_hnp.txt"
if np.exists():
    st.code(np.read_text(encoding="utf-8"), language="markdown")