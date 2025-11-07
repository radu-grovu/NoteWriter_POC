import json
import os
import re
import textwrap
from datetime import date, timedelta
import streamlit as st
from openai import OpenAI

# =========================================
# Page setup
# =========================================
st.set_page_config(page_title="NoteWriter – MyStyle IDE", layout="wide")
st.title("NoteWriter – MyStyle IDE")
st.caption("Version, test, and optimize your medical note prompts directly on site (no local files).")

# Dates / constants
TODAY = date.today()
CUTOFF = TODAY - timedelta(days=183)  # ~6 months cutoff for meds

# Storage files (persist on the site)
PROMPT_STORE_FILE = "prompt_versions.json"
TESTCASE_STORE_FILE = "test_cases.json"

# Default prompt fields (Section + Style instructions)
PROMPT_DEFAULTS = {
    "ed_instr":   "Identify presenting complaint, acute timeline, and new findings differing from baseline.",
    "disc_instr": "Summarize durable diagnoses, baseline functional status, long-term meds. Exclude resolved inpatient-only issues.",
    "labs_instr": "List only significant or trending abnormal results, e.g., 'WBC 15.4 ↑, ESR 93 ↑, CRP 47 ↑'.",
    "img_instr":  "Highlight new or worsening findings relevant to current complaint.",
    "meds_instr": "List home/inpatient meds with dose and status. Mark held meds with '– Holding' and reason.",
    "free_instr": "Supplemental context (consult pearls, nursing notes). Do not override objective data.",

    # STYLE (no Physical Exam)
    "hpi_style": (
        "Start with age/sex/admission reason. Then PMH by organ system:\n"
        "- Cardiac:\n- Pulmonary:\n- Renal:\n- Endocrine:\n- Neuro/Psych:\n- Musculoskeletal:\n"
        "Then describe acute presentation, differences from baseline, functional status, and social context."
    ),

    # A&P style with priority, impression, supporting data, treatment, follow-up
    "ap_style": (
        "Organize each problem in order of clinical priority.\n"
        "For each, start with a hashtag heading:\n"
        "# Problem – concise title (e.g., Hyponatremia, CHF, COPD exacerbation)\n\n"
        "- **Impression:** One-line interpretation of current state and likely etiology. "
        "Use 'stable', 'improving', 'uncontrolled', or 'resolved' when appropriate.\n"
        "- **Supporting Data:** Key objective findings (labs, imaging, vitals, exam) supporting the impression.\n"
        "- **Treatment Plan:** Meds with dose/route/frequency and status (continued/started/changed/holding), "
        "plus interventions and monitoring.\n"
        "- **Follow-up & Consults:** Next steps (tests, imaging, labs), consults, and monitoring intervals.\n\n"
        "End with global orders: DVT prophylaxis, Activity, Diet, Code status, Disposition/anticipated course."
    ),
}

# =========================================
# Persistent on-site storage helpers
# =========================================
def _init_store_file(path: str, default_obj: dict):
    """Ensure a JSON store file exists with valid JSON object."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_obj, f, indent=2, ensure_ascii=False)

def _load_json(path: str, fallback: dict) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        _init_store_file(path, fallback)
        return fallback

def _save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# Initialize stores if missing
_init_store_file(PROMPT_STORE_FILE, {"versions": {"Default": PROMPT_DEFAULTS}, "selected": "Default"})
_init_store_file(TESTCASE_STORE_FILE, {"cases": {"Blank": {
    "ED Note": "", "Prior Discharge": "", "Labs": "", "Imaging": "", "Med List": "", "Free Text": ""
}}, "selected": "Blank"})

def load_prompt_store():
    return _load_json(PROMPT_STORE_FILE, {"versions": {"Default": PROMPT_DEFAULTS}, "selected": "Default"})

def save_prompt_store(store: dict):
    _save_json(PROMPT_STORE_FILE, store)

def load_testcase_store():
    return _load_json(TESTCASE_STORE_FILE, {"cases": {"Blank": {
        "ED Note": "", "Prior Discharge": "", "Labs": "", "Imaging": "", "Med List": "", "Free Text": ""
    }}, "selected": "Blank"})

def save_testcase_store(store: dict):
    _save_json(TESTCASE_STORE_FILE, store)

# =========================================
# Session-safe text areas (no double-assign)
# =========================================
def session_text_area(key: str, label: str, default: str, **kwargs):
    """Create a text_area that uses existing session value or default, without reassigning after creation."""
    value = st.session_state.get(key, default)
    return st.text_area(label, value=value, key=key, **kwargs)

def session_input_area(key: str, label: str, **kwargs):
    """Create a text_area for inputs (ED/Labs/etc.), persisted in session."""
    value = st.session_state.get(key, "")
    return st.text_area(label, value=value, key=key, **kwargs)

# =========================================
# Model call
# =========================================
def call_model(prompt: str, model: str, fallback: str | None, max_tokens: int, temperature: float):
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No API key found in Secrets or env.")
        return None

    client = OpenAI(api_key=api_key)

    # Try modern Responses API
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

    # Fallback to Chat Completions
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

# =========================================
# Assemble prompt (no Physical Exam)
# =========================================
def build_prompt():
    # Read current instructions/styles from session or defaults
    get = st.session_state.get
    instruction = f"""
You are a professional internal medicine note writer producing output in JSON only.
Be concise; audience are medical experts. Use standard acronyms.

Rules:
- Problems in A&P start with hashtag headings (# Problem ...).
- For each problem include: Impression (one line), Supporting Data (key objective values), Treatment Plan (meds with dose/route/frequency/status),
  and Follow-up & Consults (tests, imaging, labs, consults/monitoring).
- Exclude meds last prescribed before {CUTOFF.isoformat()} (6-month cutoff).
- Medication Review must contain:
  - included_medications
  - excluded_medications (older_than_6_months | duplicate | not_relevant; include reason and last known date if available)
  - redundancies
  - interactions
  - side_effects_relevant
  - summary

JSON schema (no extra fields, return ONLY JSON, fully closed braces/quotes):
{{
 "hpi": "string",
 "assessment_plan": "string",
 "medication_review": {{
   "included_medications": [{{"name": "string", "dose": "string", "route": "string", "frequency": "string",
     "status": "continued|started|changed|holding", "notes": "string"}}],
   "excluded_medications": [{{"name": "string", "reason": "older_than_6_months|duplicate|not_relevant", "last_prescribed_date": "YYYY-MM-DD or unknown"}}],
   "redundancies": ["string"],
   "interactions": ["string"],
   "side_effects_relevant": ["string"],
   "summary": "string"
 }},
 "source_summary": "string"
}}

Styles:
- HPI: {get("hpi_style", PROMPT_DEFAULTS["hpi_style"])}
- A&P:  {get("ap_style", PROMPT_DEFAULTS["ap_style"])}

Per-section instructions:
- ED Note:        {get("ed_instr", PROMPT_DEFAULTS["ed_instr"])}
- Prior Discharge:{get("disc_instr", PROMPT_DEFAULTS["disc_instr"])}
- Labs:           {get("labs_instr", PROMPT_DEFAULTS["labs_instr"])}
- Imaging:        {get("img_instr", PROMPT_DEFAULTS["img_instr"])}
- Med List:       {get("meds_instr", PROMPT_DEFAULTS["meds_instr"])}
- Free Text:      {get("free_instr", PROMPT_DEFAULTS["free_instr"])}
"""
    inputs = [
        {"label": "ED Note", "text": st.session_state.get("ed_note_input", "")},
        {"label": "Prior Discharge", "text": st.session_state.get("prior_discharge_input", "")},
        {"label": "Labs", "text": st.session_state.get("labs_input", "")},
        {"label": "Imaging", "text": st.session_state.get("imaging_input", "")},
        {"label": "Med List", "text": st.session_state.get("med_list_input", "")},
        {"label": "Free Text", "text": st.session_state.get("free_text_input", "")},
    ]
    inputs_json = json.dumps([i for i in inputs if isinstance(i["text"], str) and i["text"].strip()], ensure_ascii=False)
    return f"{instruction}\n\nINPUTS_JSON:\n{inputs_json}\n\nReturn only JSON. Ensure JSON is fully complete with matching brackets and quotes."

# =========================================
# UI – Model Settings
# =========================================
with st.expander("▸ Model Settings", expanded=True):
    cols = st.columns(3)
    with cols[0]:
        model = st.selectbox("Model", ["gpt-4.1", "gpt-5", "gpt-5-pro", "gpt-5-mini"], index=0)
    with cols[1]:
        fallback_model = st.selectbox("Fallback model", ["None", "gpt-4.1", "gpt-5"], index=1)
        if fallback_model == "None":
            fallback_model = None
    with cols[2]:
        temperature = st.slider("Temperature", 0.0, 0.6, 0.2, 0.05)

    max_tokens = st.slider("Max output tokens", 500, 8000, 4000, 100)
    st.write("Tip: Use 4k–8k for long multi-problem notes to avoid truncation.")

# =========================================
# UI – Inputs + Test Case manager
# =========================================
with st.expander("▸ Inputs", expanded=True):
    # Load stored test cases
    tc_store = load_testcase_store()
    tc_names = sorted(tc_store["cases"].keys())
    left, mid, right = st.columns([2,1,1])
    with left:
        selected_tc = st.selectbox("Saved Test Cases (on site)", options=tc_names, index=tc_names.index(tc_store.get("selected", tc_names[0])) if tc_names else 0)
    with mid:
        new_tc_name = st.text_input("New case name", placeholder="e.g., Cervical OM (VA transfer)")
    with right:
        colr1, colr2 = st.columns(2)
        with colr1:
            if st.button("Load Case"):
                case = tc_store["cases"].get(selected_tc, {})
                for k, v in {
                    "ed_note_input": case.get("ED Note", ""),
                    "prior_discharge_input": case.get("Prior Discharge", ""),
                    "labs_input": case.get("Labs", ""),
                    "imaging_input": case.get("Imaging", ""),
                    "med_list_input": case.get("Med List", ""),
                    "free_text_input": case.get("Free Text", ""),
                }.items():
                    st.session_state[k] = v
                tc_store["selected"] = selected_tc
                save_testcase_store(tc_store)
        with colr2:
            if st.button("Save as New"):
                name = new_tc_name.strip() or f"Case {len(tc_store['cases'])+1}"
                tc_store["cases"][name] = {
                    "ED Note": st.session_state.get("ed_note_input",""),
                    "Prior Discharge": st.session_state.get("prior_discharge_input",""),
                    "Labs": st.session_state.get("labs_input",""),
                    "Imaging": st.session_state.get("imaging_input",""),
                    "Med List": st.session_state.get("med_list_input",""),
                    "Free Text": st.session_state.get("free_text_input",""),
                }
                tc_store["selected"] = name
                save_testcase_store(tc_store)
                st.success(f"Saved test case: {name}")

    # Duplicate / Overwrite / Delete
    cdup, cover, cdel = st.columns(3)
    with cdup:
        if st.button("Duplicate Selected"):
            base = selected_tc
            dup_name = f"{base} (copy)"
            i = 2
            while dup_name in tc_store["cases"]:
                dup_name = f"{base} (copy {i})"
                i += 1
            tc_store["cases"][dup_name] = dict(tc_store["cases"][base])
            tc_store["selected"] = dup_name
            save_testcase_store(tc_store)
            st.success(f"Duplicated to: {dup_name}")
    with cover:
        if st.button("Overwrite Selected with Current Inputs"):
            tc_store["cases"][selected_tc] = {
                "ED Note": st.session_state.get("ed_note_input",""),
                "Prior Discharge": st.session_state.get("prior_discharge_input",""),
                "Labs": st.session_state.get("labs_input",""),
                "Imaging": st.session_state.get("imaging_input",""),
                "Med List": st.session_state.get("med_list_input",""),
                "Free Text": st.session_state.get("free_text_input",""),
            }
            save_testcase_store(tc_store)
            st.info(f"Overwrote: {selected_tc}")
    with cdel:
        if st.button("Delete Selected"):
            if selected_tc in tc_store["cases"] and selected_tc != "Blank":
                del tc_store["cases"][selected_tc]
                tc_store["selected"] = "Blank" if "Blank" in tc_store["cases"] else (sorted(tc_store["cases"].keys())[0] if tc_store["cases"] else "")
                save_testcase_store(tc_store)
                st.warning(f"Deleted case: {selected_tc}")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        ed_note = session_input_area("ed_note_input", "ED Note", height=200)
        prior_discharge = session_input_area("prior_discharge_input", "Prior Discharge Note", height=200)
    with c2:
        labs = session_input_area("labs_input", "Labs", height=200)
        imaging = session_input_area("imaging_input", "Imaging Impressions", height=200)
    with c3:
        med_list = session_input_area("med_list_input", "Medication List", height=200)
        free_text = session_input_area("free_text_input", "Free Text / Other", height=200)

# =========================================
# UI – Section Instructions (versioned)
# =========================================
with st.expander("▸ Section Instructions", expanded=False):
    store = load_prompt_store()
    version_names = sorted(store["versions"].keys())
    vcol1, vcol2, vcol3 = st.columns([2,1,1])
    with vcol1:
        sel_version = st.selectbox("Saved Prompt Versions (on site)", options=version_names, index=version_names.index(store.get("selected","Default")) if version_names else 0)
    with vcol2:
        new_version_name = st.text_input("New version name", placeholder="e.g., MyStyle v2 – concise")
    with vcol3:
        if st.button("Load Version"):
            data = store["versions"].get(sel_version, PROMPT_DEFAULTS)
            for k, v in PROMPT_DEFAULTS.items():
                st.session_state[k] = data.get(k, v)
            store["selected"] = sel_version
            save_prompt_store(store)
            st.success(f"Loaded version: {sel_version}")

    # Save / Duplicate / Delete / Overwrite
    vbtn1, vbtn2, vbtn3, vbtn4 = st.columns(4)
    if vbtn1.button("Save as New Version"):
        name = new_version_name.strip() or f"Version {len(store['versions'])+1}"
        store["versions"][name] = {k: st.session_state.get(k, v) for k, v in PROMPT_DEFAULTS.items()}
        store["selected"] = name
        save_prompt_store(store)
        st.success(f"Saved version: {name}")
    if vbtn2.button("Duplicate Selected"):
        base = sel_version
        dup = f"{base} (copy)"
        i = 2
        while dup in store["versions"]:
            dup = f"{base} (copy {i})"
            i += 1
        store["versions"][dup] = dict(store["versions"][base])
        store["selected"] = dup
        save_prompt_store(store)
        st.success(f"Duplicated to: {dup}")
    if vbtn3.button("Overwrite Selected"):
        store["versions"][sel_version] = {k: st.session_state.get(k, v) for k, v in PROMPT_DEFAULTS.items()}
        save_prompt_store(store)
        st.info(f"Overwrote version: {sel_version}")
    if vbtn4.button("Delete Selected"):
        if sel_version != "Default":
            del store["versions"][sel_version]
            store["selected"] = "Default" if "Default" in store["versions"] else (sorted(store["versions"].keys())[0] if store["versions"] else "")
            save_prompt_store(store)
            st.warning(f"Deleted version: {sel_version}")

    st.markdown("---")
    colsA = st.columns(3)
    with colsA[0]:
        ed_instr = session_text_area("ed_instr", "ED Note instructions", PROMPT_DEFAULTS["ed_instr"], height=140)
        disc_instr = session_text_area("disc_instr", "Prior Discharge instructions", PROMPT_DEFAULTS["disc_instr"], height=140)
    with colsA[1]:
        labs_instr = session_text_area("labs_instr", "Labs instructions", PROMPT_DEFAULTS["labs_instr"], height=140)
        img_instr = session_text_area("img_instr", "Imaging instructions", PROMPT_DEFAULTS["img_instr"], height=140)
    with colsA[2]:
        meds_instr = session_text_area("meds_instr", "Med List instructions", PROMPT_DEFAULTS["meds_instr"], height=140)
        free_instr = session_text_area("free_instr", "Free Text instructions", PROMPT_DEFAULTS["free_instr"], height=140)

# =========================================
# UI – Style Instructions (no Physical Exam)
# =========================================
with st.expander("▸ Style Instructions", expanded=False):
    colsS = st.columns(2)
    with colsS[0]:
        hpi_style = session_text_area("hpi_style", "HPI Style", PROMPT_DEFAULTS["hpi_style"], height=220)
    with colsS[1]:
        ap_style  = session_text_area("ap_style",  "Assessment & Plan Style", PROMPT_DEFAULTS["ap_style"], height=300)

# =========================================
# Assembled prompt (download) + Run
# =========================================
assembled_prompt_str = build_prompt()
st.download_button(
    "📄 Download assembled_prompt.txt",
    data=assembled_prompt_str,
    file_name="assembled_prompt.txt",
    mime="text/plain",
    use_container_width=True
)

if st.button("Generate Note", type="primary"):
    with st.spinner(f"Generating with {model}…"):
        result = call_model(assembled_prompt_str, model, fallback_model, max_tokens, temperature)

    if not result:
        st.error("No output received.")
    else:
        # Robust JSON parsing with salvage of truncated endings
        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.IGNORECASE).strip("` \n")
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            last_brace = cleaned.rfind("}")
            parsed = None
            if last_brace != -1:
                try:
                    parsed = json.loads(cleaned[:last_brace+1])
                except Exception:
                    parsed = None

        if not parsed:
            st.warning("Output not valid JSON. Showing raw output below.")
            st.code(result)
        else:
            tabs = st.tabs(["HPI", "Assessment & Plan", "Medication Review", "Source Summary"])
            keys = ["hpi", "assessment_plan", "medication_review", "source_summary"]
            for tab, key in zip(tabs, keys):
                with tab:
                    content = parsed.get(key, "")
                    if key == "medication_review" and isinstance(content, dict):
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
