# NoteWriter POC
Streamlit app that turns clinical screenshots + pasted text into an Admission H&P.

## Run locally
```bash
python -m venv .venv   # on Windows use 'py -3 -m venv .venv'
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app\app.py
