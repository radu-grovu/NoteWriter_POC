import json
import re
import textwrap
from datetime import date, timedelta
import streamlit as st
from openai import OpenAI
from st_supabase_connection import SupabaseConnection

# =========================================
# Page setup
# =========================================
st.set_page_config(page_title="NoteWriter – MyStyle IDE", layout="wide")
st.title("NoteWriter – MyStyle IDE (Supabase-Backed)")
st.caption("Prompts, styles, and test cases now persist permanently via Supabase.")

TODAY = date.today()
CUTOFF = TODAY - timedelta(days=183)

# =========================================
# Connect to Supabase
# =========================================
supabase = st.connection("supabase_conn", type=SupabaseConnection)

# Ensure tables exist
def init_supabase_tables():
    try:
        supabase.table("prompt_versions").select("*").limit(1).execute()
    except Exception:
        st.warning("Create table 'prompt_versions' in Supabase.")
    try:
        supabase.table("test_cases").select("*").limit(1).execute()
    except Exception:
        st.warning("Create table 'test_cases' in Supabase.")

init_supabase_tables()

# =========================================
# Defaults
# =========================================
PROMPT_DEFAULTS = {
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
    "ap_style": (
        "Organize each problem in order of clinical priority.\n"
        "# Problem – concise title (e.g., Hyponatremia, CHF, COPD exacerbation)\n\n"
        "- **Impression:** One-line interpretation of current state and likely etiology.\n"
        "- **Supporting Data:** Key objective findings (labs, imaging, vitals, exam).\n"
        "- **Treatment Plan:** Meds with dose/route/frequency/status (continued/started/changed/holding), "
        "plus interventions and monitoring.\n"
        "- **Follow-up & Consults:** Next steps (tests, imaging, labs, consults, monitoring).\n\n"
        "End with global orders: DVT prophylaxis, Activity, Diet, Code status, Disposition."
    ),
}

# =========================================
# Supabase helpers
# =========================================
def get_prompt_versions():
    data = supabase.table("prompt_versions").select("*").execute().data or []
    return {d["version_name"]: d["config"] for d in data}

def save_prompt_version(name, config):
    supabase.table("prompt_versions").upsert({"version_name": name, "config": config}).execute()

def delete_prompt_version(name):
    supabase.table("prompt_versions").delete().eq("version_name", name).execute()

def get_test_cases():
    data = supabase.table("test_cases").select("*").execute().data or []
    return {d["case_name"]: d["inputs"] for d in data}

def save_test_case(name, inputs):
    supabase.table("test_cases").upsert({"case_name": name, "inputs": inputs}).execute()

def delete_test_case(name):
    supabase.table("test_cases").delete().eq("case_name", name).execute()

# =========================================
# Widgets
# =========================================
def session_text_area(key, label, default, **kw):
    value = st.session_state.get(key, default)
    return st.text_area(label, value=value, key=key, **kw)

def session_input_area(key, label, **kw):
    value = st.session_state.get(key, "")
    return st.text_area(label, value=value, key=key, **kw)

# =========================================
# Build prompt
# =========================================
def build_prompt():
    get = st.session_state.get
    instruction = f"""
You are a professional internal medicine note writer producing output in JSON only.
Audience are medical experts. Be concise and structured.

Rules:
- Problems in A&P start with hashtag headings (# Problem ...).
- For each: Impression, Supporting Data, Treatment Plan, Follow-up/Consults.
- Exclude meds prescribed before {CUTOFF.isoformat()} (6-month cutoff).

JSON schema:
{{
 "hpi": "string",
 "assessment_plan": "string",
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
- HPI: {get("hpi_style")}
- A&P: {get("ap_style")}

Per-section instructions:
- ED Note: {get("ed_instr")}
- Prior Discharge: {get("disc_instr")}
- Labs: {get("labs_instr")}
- Imaging: {get("img_instr")}
- Med List: {get("meds_instr")}
- Free Text: {get("free_instr")}
"""
    inputs = [
        {"label": "ED Note", "text": get("ed_note_input", "")},
        {"label": "Prior Discharge", "text": get("prior_discharge_input", "")},
        {"label": "Labs", "text": get("labs_input", "")},
        {"label": "Imaging", "text": get("imaging_input", "")},
        {"label": "Med List", "text": get("med_list_input", "")},
        {"label": "Free Text", "text": get("free_text_input", "")},
    ]
    inputs_json = json.dumps([i for i in inputs if i["text"].strip()], ensure_ascii=False)
    return f"{instruction}\n\nINPUTS_JSON:\n{inputs_json}\n\nReturn only JSON."

# =========================================
# Model call
# =========================================
def call_model(prompt, model, fallback, max_tokens, temperature):
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No API key found.")
        return None
    client = OpenAI(api_key=api_key)
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role":"user","content":[{"type":"text","text":prompt}]}],
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        if hasattr(resp, "output_text"):
            return resp.output_text
    except Exception:
        pass
    if fallback:
        try:
            chat = client.chat.completions.create(
                model=fallback,
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return chat.choices[0].message.content
        except Exception as e:
            st.error(f"Model error: {e}")
    return None

# =========================================
# UI: Model Settings
# =========================================
with st.expander("▸ Model Settings", expanded=True):
    col = st.columns(3)
    model = col[0].selectbox("Model", ["gpt-4.1","gpt-5","gpt-5-pro","gpt-5-mini"], index=0)
    fallback_model = col[1].selectbox("Fallback", ["None","gpt-4.1","gpt-5"], index=1)
    if fallback_model == "None": fallback_model=None
    temperature = col[2].slider("Temperature",0.0,0.6,0.2,0.05)
    max_tokens = st.slider("Max output tokens",500,8000,4000,100)

# =========================================
# Test Case Manager
# =========================================
with st.expander("▸ Inputs / Test Cases", expanded=True):
    tc_dict = get_test_cases()
    tc_names = sorted(tc_dict.keys())
    c1,c2,c3 = st.columns([2,1,1])
    with c1:
        sel_case = st.selectbox("Saved Cases", tc_names, index=0)
    with c2:
        new_case = st.text_input("New Case Name", "")
    with c3:
        if st.button("Load", key="load_tc"):
            case = tc_dict.get(sel_case,{})
            for field,label in [("ed_note_input","ED Note"),("prior_discharge_input","Prior Discharge"),
                                ("labs_input","Labs"),("imaging_input","Imaging"),
                                ("med_list_input","Med List"),("free_text_input","Free Text")]:
                st.session_state[field] = case.get(label,"")
            st.success(f"Loaded case: {sel_case}")
    cdup,cover,cdel=st.columns(3)
    if cdup.button("Duplicate",key="dup_tc"):
        dup=f"{sel_case} (copy)"
        i=2
        while dup in tc_dict: dup=f"{sel_case} (copy {i})"; i+=1
        save_test_case(dup, tc_dict[sel_case])
        st.success(f"Duplicated: {dup}")
    if cover.button("Overwrite",key="overwrite_tc"):
        data={ "ED Note":st.session_state.get("ed_note_input",""),
               "Prior Discharge":st.session_state.get("prior_discharge_input",""),
               "Labs":st.session_state.get("labs_input",""),
               "Imaging":st.session_state.get("imaging_input",""),
               "Med List":st.session_state.get("med_list_input",""),
               "Free Text":st.session_state.get("free_text_input","") }
        save_test_case(sel_case,data)
        st.info(f"Overwrote: {sel_case}")
    if cdel.button("Delete",key="delete_tc"):
        delete_test_case(sel_case)
        st.warning(f"Deleted case: {sel_case}")

    st.markdown("---")
    colx=st.columns(3)
    with colx[0]:
        session_input_area("ed_note_input","ED Note",height=200)
        session_input_area("prior_discharge_input","Prior Discharge",height=200)
    with colx[1]:
        session_input_area("labs_input","Labs",height=200)
        session_input_area("imaging_input","Imaging Impressions",height=200)
    with colx[2]:
        session_input_area("med_list_input","Medication List",height=200)
        session_input_area("free_text_input","Free Text / Other",height=200)

# =========================================
# Prompt Versions Manager
# =========================================
with st.expander("▸ Section Instructions / Styles", expanded=False):
    versions = get_prompt_versions()
    vnames = sorted(versions.keys())
    col1,col2,col3=st.columns([2,1,1])
    with col1:
        sel_ver=st.selectbox("Saved Versions",vnames,index=0)
    with col2:
        new_ver=st.text_input("New Version Name","")
    with col3:
        if st.button("Load Version",key="load_ver"):
            data=versions.get(sel_ver,PROMPT_DEFAULTS)
            for k,v in PROMPT_DEFAULTS.items():
                st.session_state[k]=data.get(k,v)
            st.success(f"Loaded: {sel_ver}")

    b1,b2,b3,b4=st.columns(4)
    if b1.button("Save as New",key="save_ver"):
        nm=new_ver.strip() or f"Version {len(vnames)+1}"
        cfg={k:st.session_state.get(k,v) for k,v in PROMPT_DEFAULTS.items()}
        save_prompt_version(nm,cfg)
        st.success(f"Saved: {nm}")
    if b2.button("Duplicate",key="dup_ver"):
        dup=f"{sel_ver} (copy)"
        i=2
        while dup in versions: dup=f"{sel_ver} (copy {i})"; i+=1
        save_prompt_version(dup,versions[sel_ver])
        st.success(f"Duplicated: {dup}")
    if b3.button("Overwrite",key="overwrite_ver"):
        cfg={k:st.session_state.get(k,v) for k,v in PROMPT_DEFAULTS.items()}
        save_prompt_version(sel_ver,cfg)
        st.info(f"Overwrote: {sel_ver}")
    if b4.button("Delete",key="delete_ver"):
        delete_prompt_version(sel_ver)
        st.warning(f"Deleted: {sel_ver}")

    st.markdown("---")
    c1,c2,c3=st.columns(3)
    with c1:
        session_text_area("ed_instr","ED Note instructions",PROMPT_DEFAULTS["ed_instr"],height=140)
        session_text_area("disc_instr","Prior Discharge instructions",PROMPT_DEFAULTS["disc_instr"],height=140)
    with c2:
        session_text_area("labs_instr","Labs instructions",PROMPT_DEFAULTS["labs_instr"],height=140)
        session_text_area("img_instr","Imaging instructions",PROMPT_DEFAULTS["img_instr"],height=140)
    with c3:
        session_text_area("meds_instr","Med List instructions",PROMPT_DEFAULTS["meds_instr"],height=140)
        session_text_area("free_instr","Free Text instructions",PROMPT_DEFAULTS["free_instr"],height=140)

    st.markdown("---")
    colS1,colS2=st.columns(2)
    with colS1:
        session_text_area("hpi_style","HPI Style",PROMPT_DEFAULTS["hpi_style"],height=220)
    with colS2:
        session_text_area("ap_style","Assessment & Plan Style",PROMPT_DEFAULTS["ap_style"],height=300)

# =========================================
# Run
# =========================================
assembled_prompt=build_prompt()
st.download_button("📄 Download assembled_prompt.txt",data=assembled_prompt,file_name="assembled_prompt.txt",mime="text/plain")

if st.button("Generate Note",type="primary",key="gen_btn"):
    with st.spinner(f"Generating with {model}…"):
        result=call_model(assembled_prompt,model,fallback_model,max_tokens,temperature)
    if not result:
        st.error("No output received.")
    else:
        parsed=None
        cleaned=result.strip()
        try:
            if cleaned.startswith("```"):
                cleaned=re.sub(r"^```(json)?","",cleaned,flags=re.IGNORECASE).strip("` \n")
            parsed=json.loads(cleaned)
        except json.JSONDecodeError:
            last=cleaned.rfind("}")
            if last!=-1:
                try:
                    parsed=json.loads(cleaned[:last+1])
                except Exception: pass
        if not parsed:
            st.warning("Output not valid JSON. Showing raw output below.")
            st.code(result)
        else:
            tabs=st.tabs(["HPI","Assessment & Plan","Medication Review","Source Summary"])
            keys=["hpi","assessment_plan","medication_review","source_summary"]
            for tab,key in zip(tabs,keys):
                with tab:
                    content=parsed.get(key,"")
                    if key=="medication_review" and isinstance(content,dict):
                        st.subheader("Included Medications")
                        for m in content.get("included_medications",[]) or ["_None_"]:
                            st.markdown(f"- {json.dumps(m,ensure_ascii=False)}")
                        st.subheader("Excluded Medications")
                        for m in content.get("excluded_medications",[]) or ["_None_"]:
                            st.markdown(f"- {json.dumps(m,ensure_ascii=False)}")
                        st.subheader("Summary")
                        st.markdown(content.get("summary","_None_"))
                    else:
                        st.markdown(content if isinstance(content,str)
                                    else json.dumps(content,ensure_ascii=False,indent=2))
            st.download_button("Download JSON",
                               data=json.dumps(parsed,indent=2,ensure_ascii=False),
                               file_name="note_output.json",
                               use_container_width=True)
