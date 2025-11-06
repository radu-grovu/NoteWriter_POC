import json
import os
import textwrap
from datetime import datetime, timedelta, date
import streamlit as st

# OpenAI SDK v1.x
from openai import OpenAI

# -----------------------
# PAGE / LAYOUT
# -----------------------
st.set_page_config(page_title="NoteWriter – Text-Only (MyStyle)", layout="wide")
st.title("NoteWriter – Text-Only (MyStyle)")
st.caption("Paste labeled clinical text → get HPI, A&P (hashtag problems, med dosing/status), Physical Exam, and a structured Medication Review")

# Dates used for medication cutoff logic passed to the model
today = date.today()
six_months_ago = today - timedelta(days=183)  # ~6 months buffer
TODAY_STR = today.isoformat()
CUTOFF_STR = six_months_ago.isoformat()

# Predefined friendly model list (edit if needed)
FRIENDLY_MODELS = [
    ("ChatGPT-5 (recommended)", "gpt-5"),
    ("ChatGPT-5 Pro (bigger, pricier)", "gpt-5-pro"),
    ("ChatGPT-5 Mini (fast/cheap)", "gpt-5-mini"),
    ("GPT-4.1", "gpt-4.1"),
]
FRIENDLY_MODEL_NAMES = [name for name, _id in FRIENDLY_MODELS]
MODEL_NAME_TO_ID = {name: mid for name, mid in FRIENDLY_MODELS}

# -----------------------
# SIDEBAR: SETTINGS & FORMATS
# -----------------------
with st.sidebar:
    st.header("Settings")
    st.write("Add your API key in Streamlit **Secrets** as `OPENAI_API_KEY`.")

    # --- Model selection (dropdown) ---
    st.subheader("Model")
    chosen_friendly = st.selectbox(
        "Select a model",
        FRIENDLY_MODEL_NAMES,
        index=0,
        help="Pick a ChatGPT model for generation."
    )
    model = MODEL_NAME_TO_ID[chosen_friendly]

    # Optional: allow a custom override and fallback
    with st.expander("Advanced model options"):
        custom_model = st.text_input(
            "Custom model ID (optional)",
            value="",
            help="If provided, this overrides the dropdown choice."
        )
        use_fallback = st.checkbox("Enable fallback model", value=False)
        fallback_friendly = st.selectbox(
            "Fallback model",
            FRIENDLY_MODEL_NAMES,
            index=3,  # default to GPT-4.1
            disabled=not use_fallback
        )
        fallback_model = MODEL_NAME_TO_ID[fallback_friendly] if use_fallback else None

    if custom_model.strip():
        model = custom_model.strip()

    max_tokens = st.slider("Max output tokens", 300, 4000, 1800, 50)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.subheader("Per-Section Instructions (optional)")
    ed_instr = st.text_area(
        "ED Note instructions",
        value=(
            "Identify presenting complaint and acute timeline. "
            "Highlight new findings or symptoms differing from chronic baseline. "
            "Summarize consultations and ED management steps."
        ),
        height=120
    )
    disc_instr = st.text_area(
        "Prior Discharge instructions",
        value=(
            "Carry forward durable diagnoses, baseline status, and long-term meds. "
            "Exclude stale inpatient-only orders unless still relevant."
        ),
        height=120
    )
    labs_instr = st.text_area(
        "Labs instructions",
        value=(
            "Include only significant abnormal or trending results. "
            "Use compact notation (e.g., 'WBC 15.4 ↑, ESR 93 ↑, CRP 47 ↑')."
        ),
        height=120
    )
    img_instr = st.text_area(
        "Imaging instructions",
        value=(
            "Summarize impressions only; avoid verbatim copying. "
            "Highlight new or worsening findings relevant to the HPI."
        ),
        height=120
    )
    meds_instr = st.text_area(
        "Med List instructions",
        value=(
            "State home and inpatient meds with doses. "
            "Mark held meds with '– Holding' and include rationale if known."
        ),
        height=120
    )
    free_instr = st.text_area(
        "Free Text instructions",
        value="Use as supplemental context; do not override objective data unless justified.",
        height=120
    )

    st.divider()
    st.subheader("Output Style Controls")

    # HPI and Physical Exam (HP) style editors
    hpi_style = st.text_area(
        "HPI Style",
        value=(
            "Start with patient age, sex, and reason for admission.\n"
            "Then show PMH line by line, organized by organ system:\n"
            "- Cardiac: …\n"
            "- Pulmonary: …\n"
            "- Renal: …\n"
            "- Endocrine: …\n"
            "- Neuro/Psych: …\n"
            "- Musculoskeletal: …\n"
            "- Other chronic issues: …\n"
            "\n"
            "Next, summarize current presentation using short declarative sentences, "
            "emphasizing changes from baseline, functional status, and key limitations. "
            "Highlight when pain/symptoms differ from chronic pattern. Include notable social/functional details."
        ),
        height=220
    )
    hp_style = st.text_area(
        "Physical Exam (HP) Style",
        value=(
            "Organize by system (General, HEENT, Cardiac, Pulmonary, Abdomen, Neuro, Skin, Extremities). "
            "Use concise, professional phrasing. List normal if relevant; emphasize abnormal findings."
        ),
        height=140
    )

    # Assessment & Plan style editor — with hashtag problem headings and explicit med status/dosing
    ap_style = st.text_area(
        "Assessment & Plan Style",
        value=(
            "For each active problem (use hashtag heading):\n"
            "# Problem Title – brief\n"
            "- Assessment: significance, relevant PMH, and key findings (vitals, imaging, labs).\n"
            "- Plan: diagnostics, consults, management. "
            "List medications with name, dose, route, frequency, and status (continued/started/changed/holding) with reasoning. "
            "Include follow-up items (f/u) and monitoring.\n"
            "\n"
            "After system-based problems, include short lines for:\n"
            "- DVT prophylaxis\n"
            "- Activity\n"
            "- Diet\n"
            "- Code status\n"
            "- Disposition / Med reconciliation summary"
        ),
        height=260
    )

# -----------------------
# MAIN INPUT AREAS
# -----------------------
st.subheader("Paste Inputs (labeled sources)")

c1, c2, c3 = st.columns(3)
with c1:
    ed_note = st.text_area("ED Note", height=220, placeholder="Paste ED provider note...")
    prior_discharge = st.text_area("Prior Discharge Note", height=220, placeholder="Paste last discharge summary...")

with c2:
    labs = st.text_area("Labs (structured or free text)", height=220, placeholder="CBC/CMP with dates, trends…")
    imaging = st.text_area("Imaging Impressions", height=220, placeholder="Final reads, key impressions…")

with c3:
    med_list = st.text_area("Medication List", height=220, placeholder="Home meds + current meds if available…")
    free_text = st.text_area("Free Text / Other", height=220, placeholder="Nursing notes, consult pearls, collateral…")

st.divider()

# -----------------------
# PROMPT CONSTRUCTION
# -----------------------
def build_instruction_block() -> str:
    """
    Build the global instruction block including style controls, six-month med cutoff, and strict JSON schema.
    """
    formatting_guidance = (
        "Formatting guidance:\n"
        "- Always reproduce PMH as a labeled section before the narrative in HPI.\n"
        "- Maintain line breaks between items.\n"
        "- Use abbreviations such as PMH, CKD, COPD, HTN, DM2, CHF, AICD.\n"
        "- Avoid fluff; use short clinical sentences.\n"
        "- In A&P, prefix each problem with a hashtag heading ('# ').\n"
        "- Under each problem, show relevant objective data (labs/imaging/ECG) as bullets.\n"
        "- For each medication tied to a problem, include name, dose, route, frequency, and status: "
        "continued | started | changed | holding, with brief rationale.\n"
    )

    med_review_rules = (
        f"Medication Review rules (current date: {TODAY_STR}; 6-month cutoff: {CUTOFF_STR}):\n"
        f"- Exclude from 'included_medications' any medication last prescribed BEFORE {CUTOFF_STR}.\n"
        "- Present Medication Review as structured sections: included_medications, excluded_medications, "
        "redundancies, interactions, side_effects_relevant, and a concise summary.\n"
        "- 'excluded_medications' should include the last_prescribed_date (if known) and reason "
        "(older_than_6_months | not_relevant | duplicate).\n"
        "- Identify and list redundant or incorrect medications.\n"
        "- Identify key interactions and side effects relevant to this patient's presentation/problems.\n"
    )

    schema = (
        "Output strictly the following JSON schema:\n"
        "{\n"
        '  "hpi": "string",\n'
        '  "assessment_plan": "string",\n'
        '  "physical_exam": "string",\n'
        '  "medication_review": {\n'
        '    "included_medications": [\n'
        '      {"name": "string", "dose": "string", "route": "string", "frequency": "string", '
        '"indication": "string", "status": "continued|started|changed|holding", "notes": "string"}\n'
        '    ],\n'
        '    "excluded_medications": [\n'
        '      {"name": "string", "last_prescribed_date": "YYYY-MM-DD or unknown", '
        '"reason": "older_than_6_months|not_relevant|duplicate"}\n'
        '    ],\n'
        '    "redundancies": ["string"],\n'
        '    "interactions": ["string"],\n'
        '    "side_effects_relevant": ["string"],\n'
        '    "summary": "string"\n'
        '  },\n'
        '  "source_summary": "string"\n'
        "}\n"
        "- Return ONLY the JSON object. No surrounding text, no code fences.\n"
    )

    header = textwrap.dedent(f"""
    You are a clinical note assistant producing output in JSON only.
    You receive multiple labeled inputs and per-section instructions.

    Tasks:
      1) Compose an HPI following this style:
         {hpi_style}
      2) Compose a problem-oriented Assessment & Plan in this style (problems as '# Problem …'):
         {ap_style}
      3) Compose a Physical Exam (HP) section following this style:
         {hp_style}
      4) Create a structured Medication Review enforcing the 6-month cutoff and sections as defined below.
      5) Add a brief Source Summary mapping key statements to the input(s) they came from.

    {formatting_guidance}
    {med_review_rules}
    {schema}
    """)

    per_section = textwrap.dedent(f"""
    Per-section instructions the model must follow:
      - ED Note: {ed_instr}
      - Prior Discharge: {disc_instr}
      - Labs: {labs_instr}
      - Imaging: {img_instr}
      - Med List: {meds_instr}
      - Free Text: {free_instr}
    """)

    return header + "\n" + per_section

def build_inputs_block():
    """
    Create a compact JSON array of non-empty inputs so the model can cite sources.
    """
    items = []
    if ed_note.strip():
        items.append({"label": "ED Note", "text": ed_note.strip()})
    if prior_discharge.strip():
        items.append({"label": "Prior Discharge", "text": prior_discharge.strip()})
    if labs.strip():
        items.append({"label": "Labs", "text": labs.strip()})
    if imaging.strip():
        items.append({"label": "Imaging", "text": imaging.strip()})
    if med_list.strip():
        items.append({"label": "Med List", "text": med_list.strip()})
    if free_text.strip():
        items.append({"label": "Free Text", "text": free_text.strip()})
    return items

def assemble_prompt() -> str:
    instruction = build_instruction_block()
    inputs = build_inputs_block()
    inputs_json = json.dumps(inputs, ensure_ascii=False)
    user_payload = (
        f"{instruction}\n\n"
        f"INPUTS_JSON:\n{inputs_json}\n\n"
        f"Return only the JSON object."
    )
    return user_payload

# -----------------------
# MODEL CALL
# -----------------------
def _try_responses_api(client: OpenAI, user_payload: str, model_id: str, max_tokens: int, temperature: float):
    resp = client.responses.create(
        model=model_id,
        input=[{"role": "user", "content": [{"type": "text", "text": user_payload}]}],
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    text = getattr(resp, "output_text", None)
    if not text:
        if hasattr(resp, "output") and len(resp.output) > 0 and "content" in resp.output[0]:
            text = resp.output[0]["content"][0].get("text", "")
    return text

def _try_chat_completions(client: OpenAI, user_payload: str, model_id: str, max_tokens: int, temperature: float):
    chat = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": user_payload}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return chat.choices[0].message.content

def call_model_with_optional_fallback(user_payload: str, primary_model: str, fallback_model: str | None, max_tokens: int, temperature: float):
    """
    Try primary via Responses API -> Chat Completions; if any path fails and fallback is set, try fallback.
    """
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No API key found. Add OPENAI_API_KEY to Streamlit Secrets.")
        return None

    client = OpenAI(api_key=api_key)

    # Primary attempts
    try:
        text = _try_responses_api(client, user_payload, primary_model, max_tokens, temperature)
        if text:
            return text
    except Exception:
        pass
    try:
        text = _try_chat_completions(client, user_payload, primary_model, max_tokens, temperature)
        if text:
            return text
    except Exception:
        pass

    # Fallback attempts
    if fallback_model:
        try:
            text = _try_responses_api(client, user_payload, fallback_model, max_tokens, temperature)
            if text:
                return text
        except Exception:
            pass
        try:
            text = _try_chat_completions(client, user_payload, fallback_model, max_tokens, temperature)
            if text:
                return text
        except Exception:
            pass

    st.error("Model call failed for both primary and fallback (if set).")
    return None

# -----------------------
# RENDER HELPERS
# -----------------------
def render_med_review(med_review):
    """
    Render the structured medication review (dict) with sections.
    """
    if isinstance(med_review, str):
        st.markdown(med_review)
        return

    # Included medications
    st.markdown("### Included Medications")
    included = med_review.get("included_medications", [])
    if included:
        for m in included:
            name = m.get("name", "")
            dose = m.get("dose", "")
            route = m.get("route", "")
            freq = m.get("frequency", "")
            indication = m.get("indication", "")
            status = m.get("status", "")
            notes = m.get("notes", "")
            line = f"- **{name}** — {dose} {route} {freq}  • *{status}*"
            if indication:
                line += f"  • Indication: {indication}"
            if notes:
                line += f"  • Notes: {notes}"
            st.markdown(line)
    else:
        st.markdown("_None listed_")

    # Excluded medications
    st.markdown("### Excluded Medications")
    excluded = med_review.get("excluded_medications", [])
    if excluded:
        for m in excluded:
            name = m.get("name", "")
            lpd = m.get("last_prescribed_date", "unknown")
            reason = m.get("reason", "")
            st.markdown(f"- **{name}** — last prescribed: {lpd} • reason: {reason}")
    else:
        st.markdown("_None listed_")

    # Redundancies
    st.markdown("### Redundancies / Incorrect")
    redundancies = med_review.get("redundancies", [])
    if redundancies:
        for r in redundancies:
            st.markdown(f"- {r}")
    else:
        st.markdown("_None identified_")

    # Interactions
    st.markdown("### Interactions")
    interactions = med_review.get("interactions", [])
    if interactions:
        for it in interactions:
            st.markdown(f"- {it}")
    else:
        st.markdown("_None identified_")

    # Side effects relevant
    st.markdown("### Side Effects Relevant to Case")
    se = med_review.get("side_effects_relevant", [])
    if se:
        for sfx in se:
            st.markdown(f"- {sfx}")
    else:
        st.markdown("_None highlighted_")

    # Summary
    st.markdown("### Summary")
    st.markdown(med_review.get("summary", "_No summary provided_"))

# -----------------------
# RUN BUTTON
# -----------------------
if st.button("Generate Note", type="primary"):
    payload = assemble_prompt()
    with st.spinner(f"Generating with {model}…"):
        raw = call_model_with_optional_fallback(
            user_payload=payload,
            primary_model=model,
            fallback_model=fallback_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if not raw:
        st.error("No output received.")
    else:
        # Try strict JSON parse; handle common fence wrappers
        parsed = None
        candidate = raw.strip()

        # Remove code fences if present
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.lower().startswith("json"):
                candidate = candidate[len("json"):].strip()

        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None

        expected_keys = ["hpi", "assessment_plan", "physical_exam", "medication_review", "source_summary"]

        if parsed and all(k in parsed for k in expected_keys):
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["HPI", "Assessment & Plan", "Physical Exam", "Medication Review", "Source Summary"]
            )
            with tab1:
                st.markdown(parsed["hpi"])
            with tab2:
                # Ensure hashtag headings render as headers in Streamlit markdown
                st.markdown(parsed["assessment_plan"])
            with tab3:
                st.markdown(parsed["physical_exam"])
            with tab4:
                render_med_review(parsed["medication_review"])
            with tab5:
                st.markdown(parsed["source_summary"])

            st.download_button(
                "Download All (JSON)",
                data=json.dumps(parsed, indent=2),
                file_name="note_outputs.json"
            )
        else:
            st.warning("Model returned non-JSON or malformed JSON. Showing raw output below.")
            st.code(raw, language="markdown")
