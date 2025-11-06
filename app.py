import json
import os
import textwrap
import streamlit as st
from datetime import date, timedelta
from openai import OpenAI

# -----------------------
# Utility
# -----------------------
def get_text_area(key: str, label: str, default: str, **kwargs):
    """Persistent text area stored in session_state."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.text_area(label, value=st.session_state[key], key=key, **kwargs)

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="NoteWriter – MyStyle", layout="wide")
st.title("NoteWriter – Text-Only (MyStyle)")
st.caption("Paste clinical text → generate structured hospitalist note output")

today = date.today()
cutoff = today - timedelta(days=183)

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Settings")

    st.write("Add your OpenAI API key in Streamlit **Secrets** as `OPENAI_API_KEY`.")

    model = st.selectbox("Model", ["gpt-4.1", "gpt-5", "gpt-5-pro", "gpt-5-mini"], index=0)
    fallback_model = st.selectbox("Fallback model", ["None", "gpt-4.1", "gpt-5"], index=1)
    if fallback_model == "None":
        fallback_model = None

    max_tokens = st.slider("Max output tokens", 300, 4000, 1800, 50)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.subheader("Per-Section Instructions")

    ed_instr = get_text_area(
        "ed_instr",
        "ED Note instructions",
        "Identify presenting complaint, acute timeline, and new findings differing from baseline.",
        height=120,
    )

    disc_instr = get_text_area(
        "disc_instr",
        "Prior Discharge instructions",
        "Summarize durable diagnoses, baseline functional status, long-term meds. Exclude resolved inpatient-only issues.",
        height=120,
    )

    labs_instr = get_text_area(
        "labs_instr",
        "Labs instructions",
        "List only significant or trending abnormal results, e.g., 'WBC 15.4 ↑, ESR 93 ↑, CRP 47 ↑'.",
        height=120,
    )

    img_instr = get_text_area(
        "img_instr",
        "Imaging instructions",
        "Highlight new or worsening findings relevant to current complaint.",
        height=120,
    )

    meds_instr = get_text_area(
        "meds_instr",
        "Med List instructions",
        "List home/inpatient meds with dose and status. Mark held meds with '– Holding' and reason.",
        height=120,
    )

    free_instr = get_text_area(
        "free_instr",
        "Free Text instructions",
        "Supplemental context (consult pearls, nursing notes). Do not override objective data.",
        height=120,
    )

    st.divider()
    st.subheader("Output Style Controls")

    hpi_style = get_text_area(
        "hpi_style",
        "HPI Style",
        "Start with age/sex/admission reason. Then PMH by organ system:\n"
        "- Cardiac:\n- Pulmonary:\n- Renal:\n- Endocrine:\n- Neuro/Psych:\n- Musculoskeletal:\n"
        "Then describe acute presentation, differences from baseline, functional status, and social context.",
        height=200,
    )

    hp_style = get_text_area(
        "hp_style",
        "Physical Exam (HP) Style",
        "Organize by system: General, HEENT, Cardiac, Pulmonary, Abdomen, Neuro, Skin, Extremities. "
        "Keep concise, emphasize abnormal findings.",
        height=120,
    )

    ap_style = get_text_area(
        "ap_style",
        "Assessment & Plan Style",
        "Each problem starts with a hashtag heading.\n"
        "# Problem Title – concise\n"
        "- Assessment: key data (vitals, labs, imaging) and PMH relevance.\n"
        "- Plan: diagnostics, consults, and meds (dose/route/frequency/status). "
        "State if continued, started, changed, or holding, with rationale.\n"
        "Finish with DVT prophylaxis, diet, activity, code status, disposition, med reconciliation.",
        height=250,
    )

# -----------------------
# Inputs
# -----------------------
st.subheader("Paste Inputs")

c1, c2, c3 = st.columns(3)
with c1:
    ed_note = st.text_area("ED Note", height=200)
    prior_discharge = st.text_area("Prior Discharge Note", height=200)
with c2:
    labs = st.text_area("Labs", height=200)
    imaging = st.text_area("Imaging Impressions", height=200)
with c3:
    med_list = st.text_area("Medication List", height=200)
    free_text = st.text_area("Free Text / Other", height=200)

# -----------------------
# Prompt construction
# -----------------------
def build_prompt():
    instruction = f"""
You are a clinical note assistant producing output in JSON only.

Follow these structural rules:
- Problems in A&P start with hashtag headings (# Problem ...).
- Include meds with dose/route/frequency/status (continued/started/changed/holding).
- Exclude meds prescribed before {cutoff} (6-month cutoff).
- Medication Review must contain:
  - included_medications
  - excluded_medications (older_than_6_months|duplicate|not_relevant)
  - redundancies
  - interactions
  - side_effects_relevant
  - summary

JSON schema:
{{
 "hpi": "string",
 "assessment_plan": "string",
 "physical_exam": "string",
 "medication_review": {{
   "included_medications": [{{"name": "string", "dose": "string", "route": "string", "frequency": "string",
     "status": "continued|started|changed|holding", "notes": "string"}}],
   "excluded_medications": [{{"name": "string", "reason": "string"}}],
   "redundancies": ["string"],
   "interactions": ["string"],
   "side_effects_relevant": ["string"],
   "summary": "string"
 }},
 "source_summary": "string"
}}

Styles:
- HPI: {hpi_style}
- Physical Exam: {hp_style}
- A&P: {ap_style}

Per-section instructions:
- ED Note: {ed_instr}
- Prior Discharge: {disc_instr}
- Labs: {labs_instr}
- Imaging: {img_instr}
- Med List: {meds_instr}
- Free Text: {free_instr}
"""
    inputs = [
        {"label": "ED Note", "text": ed_note},
        {"label": "Prior Discharge", "text": prior_discharge},
        {"label": "Labs", "text": labs},
        {"label": "Imaging", "text": imaging},
        {"label": "Med List", "text": med_list},
        {"label": "Free Text", "text": free_text},
    ]
    inputs_json = json.dumps([i for i in inputs if i["text"].strip()], ensure_ascii=False)
    return f"{instruction}\n\nINPUTS_JSON:\n{inputs_json}\n\nReturn only JSON."

# -----------------------
# Model call
# -----------------------
def call_model(prompt, model, fallback, max_tokens, temperature):
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No API key found.")
        return None
    client = OpenAI(api_key=api_key)

    # Try responses API
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
    except Exception:
        pass

    # Fallback chat
    if fallback:
        try:
            chat = client.chat.completions.create(
                model=fallback,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return chat.choices[0].message.content
        except Exception as e:
            st.error(f"Model error: {e}")
    return None

# -----------------------
# Run
# -----------------------
if st.button("Generate Note", type="primary"):
    with st.spinner(f"Generating with {model}..."):
        result = call_model(build_prompt(), model, fallback_model, max_tokens, temperature)
    if not result:
        st.error("No output received.")
    else:
        try:
            cleaned = result.strip().strip("`").replace("json", "")
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None

        if not parsed:
            st.warning("Output not valid JSON. Showing raw output.")
            st.code(result)
        else:
            tabs = st.tabs(["HPI", "A&P", "Physical Exam", "Medication Review", "Source Summary"])
            tab_names = ["hpi", "assessment_plan", "physical_exam", "medication_review", "source_summary"]
            for tab, key in zip(tabs, tab_names):
                with tab:
                    content = parsed[key]
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.markdown(content)
            st.download_button(
                "Download JSON",
                data=json.dumps(parsed, indent=2),
                file_name="note_output.json",
            )
