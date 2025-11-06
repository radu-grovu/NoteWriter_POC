import json
import os
import textwrap
import streamlit as st
from datetime import date, timedelta
from openai import OpenAI

# =========================
# Utilities (session-safe)
# =========================
def get_text_area(key: str, label: str, default: str, **kwargs):
    """Persistent text area stored in session_state (for sidebar settings)."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.text_area(label, value=st.session_state[key], key=key, **kwargs)

def get_input_area(key: str, label: str, **kwargs):
    """Persistent text area for main inputs (ED, Labs, etc.)."""
    if key not in st.session_state:
        st.session_state[key] = ""
    return st.text_area(label, value=st.session_state[key], key=key, **kwargs)

def set_input_from_testcase(tc: dict):
    """Populate all main inputs from a loaded test case dict."""
    def pull(d, *names):
        for n in names:
            if n in d and isinstance(d[n], str):
                return d[n]
        return ""
    st.session_state["ed_note_input"]         = pull(tc, "ED Note", "ed_note")
    st.session_state["prior_discharge_input"] = pull(tc, "Prior Discharge", "prior_discharge")
    st.session_state["labs_input"]            = pull(tc, "Labs", "labs")
    st.session_state["imaging_input"]         = pull(tc, "Imaging", "imaging", "imaging_impressions")
    st.session_state["med_list_input"]        = pull(tc, "Med List", "Medication List", "med_list", "meds")
    st.session_state["free_text_input"]       = pull(tc, "Free Text", "free_text", "other")

# =========================
# Page setup
# =========================
st.set_page_config(page_title="NoteWriter – MyStyle (Test Case + Prompt Save/Load)", layout="wide")
st.title("NoteWriter – Text-Only (MyStyle)")
st.caption("Upload a test case and/or prompts → tune sidebar → re-run instantly to evaluate output")

today = date.today()
cutoff = today - timedelta(days=183)  # ~6 months

# Keys included in prompt config save/load
PROMPT_KEYS_AND_DEFAULTS = {
    "ed_instr":   "Identify presenting complaint, acute timeline, and new findings differing from baseline.",
    "disc_instr": "Summarize durable diagnoses, baseline functional status, long-term meds. Exclude resolved inpatient-only issues.",
    "labs_instr": "List only significant or trending abnormal results, e.g., 'WBC 15.4 ↑, ESR 93 ↑, CRP 47 ↑'.",
    "img_instr":  "Highlight new or worsening findings relevant to current complaint.",
    "meds_instr": "List home/inpatient meds with dose and status. Mark held meds with '– Holding' and reason.",
    "free_instr": "Supplemental context (consult pearls, nursing notes). Do not override objective data.",
    "hpi_style": (
        "Start with age/sex/admission reason. Then PMH by organ system:\n"
        "- Cardiac:\n- Pulmonary:\n- Renal:\n- Endocrine:\n- Neuro/Psych:\n- Musculoskeletal:\n"
        "Then describe acute presentation, differences from baseline, functional status, and social context."
    ),
    "hp_style": (
        "Organize by system: General, HEENT, Cardiac, Pulmonary, Abdomen, Neuro, Skin, Extremities. "
        "Keep concise, emphasize abnormal findings."
    ),
    "ap_style": (
        "Organize each problem in order of clinical priority.\n"
        "For each, start with a hashtag heading:\n"
        "# Problem – concise title (e.g., Hyponatremia, CHF, COPD exacerbation)\n"
        "\n"
        "- **Impression:** Provide a clear, one-line interpretation of the problem's current state and likely etiology. "
        "Use terms like 'stable', 'improving', 'uncontrolled', or 'resolved' when appropriate. "
        "For example: 'Likely hypovolemic hyponatremia due to thiazide diuretic use'; or 'Stable chronic hypertension, well-controlled on home regimen.'\n"
        "\n"
        "- **Supporting Data:** List key objective findings—labs, imaging, vitals, and exam data—that support your impression. "
        "Include relevant values (Na 128 ↓, K 4.3, BUN 26, Cr 1.2 baseline, CT head neg).\n"
        "\n"
        "- **Treatment Plan:** Outline active management. "
        "List medications with full dose/route/frequency and indicate whether they are *continued, started, changed, or held*. "
        "Specify new interventions, adjustments, or monitoring needs. "
        "Example: 'Continue home Lisinopril 10 mg daily; start Lasix 40 mg IV BID; hold HCTZ given hyponatremia.'\n"
        "\n"
        "- **Follow-up & Consults:** State next steps in evaluation or care—tests, imaging, labs, consults, or monitoring intervals. "
        "Example: 'Repeat BMP q8h; nephrology consult if Na <125 or persistent after 24 h.'\n"
        "\n"
        "At the end of the section, summarize global orders:\n"
        "- DVT prophylaxis\n"
        "- Activity\n"
        "- Diet\n"
        "- Code status\n"
        "- Disposition / anticipated course\n"
    ),
}

def load_prompt_config_into_session(cfg: dict):
    """Load prompt settings from uploaded JSON into session_state BEFORE widgets render."""
    for k, default in PROMPT_KEYS_AND_DEFAULTS.items():
        if k in cfg and isinstance(cfg[k], str):
            st.session_state[k] = cfg[k]
        elif k not in st.session_state:
            st.session_state[k] = default  # ensure presence

def current_prompt_config_from_session() -> dict:
    return {k: st.session_state.get(k, v) for k, v in PROMPT_KEYS_AND_DEFAULTS.items()}

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Settings")
    st.write("Add your OpenAI API key in Streamlit **Secrets** as `OPENAI_API_KEY`.")

    # Model selection (safe default = gpt-4.1)
    model = st.selectbox("Model", ["gpt-4.1", "gpt-5", "gpt-5-pro", "gpt-5-mini"], index=0)
    fallback_model = st.selectbox("Fallback model", ["None", "gpt-4.1", "gpt-5"], index=1)
    if fallback_model == "None":
        fallback_model = None

    max_tokens = st.slider("Max output tokens", 500, 8000, 8000, 100)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.subheader("Prompt Config (Save / Load)")

    # Prompt config uploader (loads BEFORE prompt widgets render)
    up_prompts = st.file_uploader("Load prompts (.json)", type=["json"], key="prompt_uploader")
    if up_prompts is not None:
        try:
            cfg = json.loads(up_prompts.getvalue().decode("utf-8"))
            if not isinstance(cfg, dict):
                raise ValueError("Uploaded JSON must be an object with string fields.")
            load_prompt_config_into_session(cfg)
            st.success("Prompts loaded.")
        except Exception as e:
            st.error(f"Failed to load prompts: {e}")

    st.divider()
    st.subheader("Test Case")
    # Test case JSON upload (inputs)
    uploaded_tc = st.file_uploader("Load test case (.json)", type=["json"], key="testcase_uploader")
    if uploaded_tc is not None:
        try:
            tc_data = json.loads(uploaded_tc.getvalue().decode("utf-8"))
            set_input_from_testcase(tc_data)
            st.success("Test case loaded into inputs.")
        except Exception as e:
            st.error(f"Failed to load test case: {e}")

    # Download current inputs as test_case.json
    if st.button("📥 Prepare current inputs for download"):
        current_tc = {
            "ED Note":         st.session_state.get("ed_note_input", ""),
            "Prior Discharge": st.session_state.get("prior_discharge_input", ""),
            "Labs":            st.session_state.get("labs_input", ""),
            "Imaging":         st.session_state.get("imaging_input", ""),
            "Med List":        st.session_state.get("med_list_input", ""),
            "Free Text":       st.session_state.get("free_text_input", ""),
        }
        st.session_state["_prepared_testcase_json"] = json.dumps(current_tc, indent=2, ensure_ascii=False)
    if "_prepared_testcase_json" in st.session_state:
        st.download_button(
            "Save test_case.json",
            data=st.session_state["_prepared_testcase_json"],
            file_name="test_case.json",
            mime="application/json",
            use_container_width=True
        )

    # Quick reset of inputs
    if st.button("🧹 Clear all inputs"):
        for k in ["ed_note_input","prior_discharge_input","labs_input","imaging_input","med_list_input","free_text_input"]:
            st.session_state[k] = ""
        st.success("Inputs cleared.")

    st.divider()
    st.subheader("Per-Section Instructions")

    ed_instr = get_text_area(
        "ed_instr",
        "ED Note instructions",
        PROMPT_KEYS_AND_DEFAULTS["ed_instr"],
        height=120,
    )
    disc_instr = get_text_area(
        "disc_instr",
        "Prior Discharge instructions",
        PROMPT_KEYS_AND_DEFAULTS["disc_instr"],
        height=120,
    )
    labs_instr = get_text_area(
        "labs_instr",
        "Labs instructions",
        PROMPT_KEYS_AND_DEFAULTS["labs_instr"],
        height=120,
    )
    img_instr = get_text_area(
        "img_instr",
        "Imaging instructions",
        PROMPT_KEYS_AND_DEFAULTS["img_instr"],
        height=120,
    )
    meds_instr = get_text_area(
        "meds_instr",
        "Med List instructions",
        PROMPT_KEYS_AND_DEFAULTS["meds_instr"],
        height=120,
    )
    free_instr = get_text_area(
        "free_instr",
        "Free Text instructions",
        PROMPT_KEYS_AND_DEFAULTS["free_instr"],
        height=120,
    )

    st.divider()
    st.subheader("Output Style Controls")

    hpi_style = get_text_area(
        "hpi_style",
        "HPI Style",
        PROMPT_KEYS_AND_DEFAULTS["hpi_style"],
        height=200,
    )
    hp_style = get_text_area(
        "hp_style",
        "Physical Exam (HP) Style",
        PROMPT_KEYS_AND_DEFAULTS["hp_style"],
        height=120,
    )
    ap_style = get_text_area(
        "ap_style",
        "Assessment & Plan Style",
        PROMPT_KEYS_AND_DEFAULTS["ap_style"],
        height=250,
    )

    # Download current prompts as prompt_config.json
    st.divider()
    st.subheader("Export Current Prompts")
    prompt_cfg_json = json.dumps(current_prompt_config_from_session(), indent=2, ensure_ascii=False)
    st.download_button(
        "💾 Save prompt_config.json",
        data=prompt_cfg_json,
        file_name="prompt_config.json",
        mime="application/json",
        use_container_width=True
    )

# =========================
# Main Inputs (persisted)
# =========================
st.subheader("Paste Inputs (or load a Test Case in the sidebar)")

c1, c2, c3 = st.columns(3)
with c1:
    ed_note = get_input_area("ed_note_input", "ED Note", height=200)
    prior_discharge = get_input_area("prior_discharge_input", "Prior Discharge Note", height=200)
with c2:
    labs = get_input_area("labs_input", "Labs", height=200)
    imaging = get_input_area("imaging_input", "Imaging Impressions", height=200)
with c3:
    med_list = get_input_area("med_list_input", "Medication List", height=200)
    free_text = get_input_area("free_text_input", "Free Text / Other", height=200)

# =========================
# Prompt construction
# =========================
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
- HPI: {st.session_state.get("hpi_style")}
- Physical Exam: {st.session_state.get("hp_style")}
- A&P: {st.session_state.get("ap_style")}

Per-section instructions:
- ED Note: {st.session_state.get("ed_instr")}
- Prior Discharge: {st.session_state.get("disc_instr")}
- Labs: {st.session_state.get("labs_instr")}
- Imaging: {st.session_state.get("img_instr")}
- Med List: {st.session_state.get("meds_instr")}
- Free Text: {st.session_state.get("free_instr")}
"""
    inputs = [
        {"label": "ED Note", "text": ed_note},
        {"label": "Prior Discharge", "text": prior_discharge},
        {"label": "Labs", "text": labs},
        {"label": "Imaging", "text": imaging},
        {"label": "Med List", "text": med_list},
        {"label": "Free Text", "text": free_text},
    ]
    inputs_json = json.dumps([i for i in inputs if isinstance(i["text"], str) and i["text"].strip()], ensure_ascii=False)
    return f"{instruction}\n\nINPUTS_JSON:\n{inputs_json}\n\nReturn only JSON."

# Button to download the currently assembled prompt (for testing/debug)
assembled_prompt_str = build_prompt()
st.download_button(
    "📄 Download assembled_prompt.txt",
    data=assembled_prompt_str,
    file_name="assembled_prompt.txt",
    mime="text/plain",
    use_container_width=True
)

# =========================
# Model call
# =========================
def call_model(prompt, model, fallback, max_tokens, temperature):
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No API key found.")
        return None
    client = OpenAI(api_key=api_key)

    # Try Responses API first
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

    # Fallback to Chat Completions if configured
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

# =========================
# Run
# =========================
if st.button("Generate Note", type="primary"):
    with st.spinner(f"Generating with {model}..."):
        result = call_model(assembled_prompt_str, model, fallback_model, max_tokens, temperature)

    if not result:
        st.error("No output received.")
    else:
        # Attempt to parse JSON (and tolerate code fences)
        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[len("json"):].strip()
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None

        if not parsed:
            st.warning("Output not valid JSON. Showing raw output.")
            st.code(result)
        else:
            tabs = st.tabs(["HPI", "A&P", "Physical Exam", "Medication Review", "Source Summary"])
            keys = ["hpi", "assessment_plan", "physical_exam", "medication_review", "source_summary"]
            for tab, key in zip(tabs, keys):
                with tab:
                    content = parsed.get(key, "")
                    if key == "medication_review" and isinstance(content, dict):
                        # Render structured medication review
                        st.subheader("Included Medications")
                        inc = content.get("included_medications", [])
                        if inc:
                            for m in inc:
                                st.markdown(f"- {json.dumps(m, ensure_ascii=False)}")
                        else:
                            st.markdown("_None_")
                        st.subheader("Excluded Medications")
                        exc = content.get("excluded_medications", [])
                        if exc:
                            for m in exc:
                                st.markdown(f"- {json.dumps(m, ensure_ascii=False)}")
                        else:
                            st.markdown("_None_")
                        st.subheader("Redundancies")
                        for i in content.get("redundancies", []) or ["_None_"]:
                            st.markdown(f"- {i}")
                        st.subheader("Interactions")
                        for i in content.get("interactions", []) or ["_None_"]:
                            st.markdown(f"- {i}")
                        st.subheader("Side Effects Relevant")
                        for i in content.get("side_effects_relevant", []) or ["_None_"]:
                            st.markdown(f"- {i}")
                        st.subheader("Summary")
                        st.markdown(content.get("summary", "_None_"))
                    else:
                        st.markdown(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2))

            st.download_button(
                "Download JSON",
                data=json.dumps(parsed, indent=2, ensure_ascii=False),
                file_name="note_output.json",
                use_container_width=True
            )
