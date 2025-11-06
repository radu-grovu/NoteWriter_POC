import json
import os
import textwrap
import streamlit as st

# If using OpenAI >=1.0 SDK:
from openai import OpenAI

# -----------------------
# UI SETUP
# -----------------------
st.set_page_config(page_title="NoteWriter (Text-Only)", layout="wide")

st.title("NoteWriter – Text-Only (MyStyle)")
st.caption("Paste labeled clinical text → get HPI, A&P, and Med Review")

with st.sidebar:
    st.header("Settings")
    st.write("Add your API key in Streamlit **Secrets** as `OPENAI_API_KEY`.")
    default_model = "gpt-4.1"
    model = st.text_input("Model", value=default_model)
    max_tokens = st.slider("Max output tokens", 300, 4000, 1600, 50)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.divider()
    st.subheader("Per-Section Instructions (optional)")
    ed_instr = st.text_area("ED Note instructions", value="Extract chief complaint, initial HPI timeline, pertinent positives/negatives. Resolve obvious contradictions with labs/imaging.")
    disc_instr = st.text_area("Prior Discharge instructions", value="Carry forward only durable diagnoses, baseline functional status, and long-term meds. Ignore stale inpatient-only orders.")
    labs_instr = st.text_area("Labs instructions", value="Trend key labs (CBC, CMP, troponin, BNP, lactate). Flag critical values, deltas, and likely clinical implications.")
    img_instr = st.text_area("Imaging instructions", value="Summarize impressions only; avoid copying verbatim. Highlight new/worsening findings relevant to current HPI.")
    meds_instr = st.text_area("Med List instructions", value="Reconcile inpatient vs outpatient. Identify omissions, duplications, interactions, renally-dosed agents, QT-prolongers.")
    free_instr = st.text_area("Free Text instructions", value="Use as supplemental context; do not let it override objective data unless explicitly reasonable.")
    st.divider()
    ap_style = st.text_area(
        "Assessment & Plan style",
        value=(
            "# Problem 1 – short title\n"
            "- Assessment: pathophys reasoning, differential, supporting data\n"
            "- Workup: tests/consults\n"
            "- Treatment: meds with dose/route, monitoring, safety notes\n"
            "\n# Problem 2 – ...\n"
        ),
        height=160
    )

st.subheader("Paste Inputs")
c1, c2, c3 = st.columns(3)
with c1:
    ed_note = st.text_area("ED Note", height=220, placeholder="Paste ED provider note...")
    prior_discharge = st.text_area("Prior Discharge Note", height=220, placeholder="Paste last discharge summary...")

with c2:
    labs = st.text_area("Labs (structured or free text)", height=220, placeholder="e.g., CBC/CMP with dates, trends…")
    imaging = st.text_area("Imaging Impressions", height=220, placeholder="Final reads, key impressions…")

with c3:
    med_list = st.text_area("Medication List", height=220, placeholder="Home meds + current meds if available…")
    free_text = st.text_area("Free Text / Other", height=220, placeholder="Nursing notes, consult pearls, collateral…")

st.divider()

# -----------------------
# PROMPT CONSTRUCTION
# -----------------------
def build_instruction_block():
    return textwrap.dedent(f"""
    You are a clinical note assistant producing output in JSON only.
    You receive multiple labeled inputs and per-section instructions.
    Tasks:
      1) Compose an HPI (concise, chronological, pertinent positives/negatives).
      2) Compose a problem-oriented Assessment & Plan in this style:
         {ap_style}
      3) Create a Medication Review: reconciliation, discrepancies, safety flags, dosing adjustments, interactions/QT concerns, renal/hepatic notes.
      4) Add a brief Source Summary mapping key statements to the input(s) they came from.

    Rules:
      - Be specific, evidence-based, and internally consistent.
      - Prefer objective data over narrative when conflicting.
      - Quote nothing verbatim; summarize.
      - Keep HPI ≤ 180 words unless essential.
      - In A&P, include workup and treatment with dose/route when applicable.
      - Make medication safety notes explicit.
      - Output strictly the following JSON schema:

      {{
        "hpi": "string",
        "assessment_plan": "string",
        "medication_review": "string",
        "source_summary": "string"
      }}

    Per-section instructions the model must follow:
      - ED Note: {ed_instr}
      - Prior Discharge: {disc_instr}
      - Labs: {labs_instr}
      - Imaging: {img_instr}
      - Med List: {meds_instr}
      - Free Text: {free_instr}
    """)

def build_inputs_block():
    # Include only non-empty sections
    items = []
    if ed_note.strip():
        items.append({"label":"ED Note", "text": ed_note.strip()})
    if prior_discharge.strip():
        items.append({"label":"Prior Discharge", "text": prior_discharge.strip()})
    if labs.strip():
        items.append({"label":"Labs", "text": labs.strip()})
    if imaging.strip():
        items.append({"label":"Imaging", "text": imaging.strip()})
    if med_list.strip():
        items.append({"label":"Med List", "text": med_list.strip()})
    if free_text.strip():
        items.append({"label":"Free Text", "text": free_text.strip()})
    return items

def assemble_prompt():
    instruction = build_instruction_block()
    inputs = build_inputs_block()
    # Put inputs as a compact JSON array so the model can cite sources
    inputs_json = json.dumps(inputs, ensure_ascii=False)
    user_payload = f"{instruction}\n\nINPUTS_JSON:\n{inputs_json}\n\nReturn only the JSON object, no commentary."
    return user_payload

# -----------------------
# MODEL CALL
# -----------------------
def call_model(user_payload: str, model: str, max_tokens: int, temperature: float):
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No API key found. Add OPENAI_API_KEY to Streamlit Secrets.")
        return None

    client = OpenAI(api_key=api_key)
    # Prefer the /responses endpoint if available; otherwise fallback to chat.completions.
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content":[{"type":"text","text": user_payload}]}],
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        # Depending on SDK, extract text:
        text = resp.output_text  # OpenAI>=1.40 convenience
    except Exception:
        # Fallback to chat.completions for broader compatibility
        chat = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":user_payload}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = chat.choices[0].message.content

    return text

# -----------------------
# RUN
# -----------------------
if st.button("Generate Note", type="primary"):
    payload = assemble_prompt()

    with st.spinner("Generating…"):
        raw = call_model(payload, model=model, max_tokens=max_tokens, temperature=temperature)

    if not raw:
        st.error("No output received.")
    else:
        # Try parsing JSON; if it fails, show raw and a fix button
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # Some models may wrap code fences; try to strip them
            raw_stripped = raw.strip()
            if raw_stripped.startswith("```"):
                raw_stripped = raw_stripped.strip("`")
                # remove optional language hint
                if raw_stripped.startswith("json"):
                    raw_stripped = raw_stripped[len("json"):].strip()
            try:
                parsed = json.loads(raw_stripped)
            except Exception:
                parsed = None

        if parsed and all(k in parsed for k in ["hpi","assessment_plan","medication_review","source_summary"]):
            tab1, tab2, tab3, tab4 = st.tabs(["HPI", "Assessment & Plan", "Medication Review", "Source Summary"])
            with tab1: st.markdown(parsed["hpi"])
            with tab2: st.markdown(parsed["assessment_plan"])
            with tab3: st.markdown(parsed["medication_review"])
            with tab4: st.markdown(parsed["source_summary"])
            st.download_button("Download All (JSON)", data=json.dumps(parsed, indent=2), file_name="note_outputs.json")
        else:
            st.warning("Model returned non-JSON or malformed JSON. Showing raw output below.")
            st.code(raw, language="markdown")
