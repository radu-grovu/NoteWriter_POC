import json
import os
import textwrap
import streamlit as st

# OpenAI SDK v1.x
from openai import OpenAI

# -----------------------
# PAGE / LAYOUT
# -----------------------
st.set_page_config(page_title="NoteWriter – Text-Only (MyStyle)", layout="wide")
st.title("NoteWriter – Text-Only (MyStyle)")
st.caption("Paste labeled clinical text → get HPI, A&P, Physical Exam, and Med Review")

# Predefined friendly model list (you can edit this)
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

    # Optional: allow a custom override (for advanced users)
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

    max_tokens = st.slider("Max output tokens", 300, 4000, 1600, 50)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.subheader("Per-Section Instructions (optional)")
    ed_instr = st.text_area(
        "ED Note instructions",
        value=(
            "Extract chief complaint, initial HPI timeline, and pertinent positives/negatives. "
            "Resolve obvious contradictions with labs/imaging when appropriate."
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
            "Trend key labs (CBC, CMP, troponin, BNP, lactate). "
            "Flag critical values, significant deltas, and likely implications."
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
            "Reconcile inpatient vs outpatient. Identify omissions, duplications, interactions, "
            "renally-dosed agents, and QT-prolongers."
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
            "Chronological narrative of symptom onset and evolution with context. "
            "Include pertinent positives/negatives and salient past history that informs the presentation. "
            "Avoid repeating exam or lab details unless essential to the timeline."
        ),
        height=140
    )
    hp_style = st.text_area(
        "Physical Exam (HP) Style",
        value=(
            "Organize by system (General, HEENT, Cardiac, Pulmonary, Abdomen, Neuro, Skin, Extremities). "
            "Use concise, professional phrasing. List normal if relevant; emphasize abnormal findings."
        ),
        height=140
    )

    # Assessment & Plan style editor
    ap_style = st.text_area(
        "Assessment & Plan Style",
        value=(
            "# Problem 1 – short title\n"
            "- Assessment: pathophysiology, differential, supporting data\n"
            "- Workup: tests/consults and monitoring\n"
            "- Treatment: meds with dose/route, supportive care, safety notes\n"
            "\n# Problem 2 – ...\n"
        ),
        height=180
    )

# -----------------------
# MAIN INPUT AREAS
# -----------------------
st.subheader("Paste Inputs (labelled sources)")

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
    Build the global instruction block including style controls and strict JSON schema.
    """
    return textwrap.dedent(f"""
    You are a clinical note assistant producing output in JSON only.
    You receive multiple labeled inputs and per-section instructions.

    Tasks:
      1) Compose an HPI following this style:
         {hpi_style}
      2) Compose a problem-oriented Assessment & Plan in this style:
         {ap_style}
      3) Compose a Physical Exam (HP) section following this style:
         {hp_style}
      4) Create a Medication Review: reconciliation, discrepancies, safety flags, dosing adjustments, interactions/QT concerns, renal/hepatic considerations.
      5) Add a brief Source Summary mapping key statements to the input(s) they came from.

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
        "physical_exam": "string",
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
        f"Return only the JSON object, no commentary."
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
    # Newer SDKs expose this convenience:
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
                st.markdown(parsed["assessment_plan"])
            with tab3:
                st.markdown(parsed["physical_exam"])
            with tab4:
                st.markdown(parsed["medication_review"])
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
