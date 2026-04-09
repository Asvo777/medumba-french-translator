# Medumba-French Dictionary (Streamlit)

A Streamlit app to search, add, and manage Medumba ↔ French dictionary entries. Includes a lightweight multilingual embedding model for fuzzy token-level translation suggestions.

## Features
- Search words or expressions in either language
- Add new translations with duplicate checks
- View full dictionaries for words and expressions
- Embedding-based token translation demo using `paraphrase-multilingual-MiniLM-L12-v2`

## Project layout
- `dictionary_manager.py` – Streamlit app
- `output/` – data folder with CSVs
  - `translations_words.csv`
  - `translations_expressions.csv`
- `requirements.txt` – Python dependencies

## Prerequisites
- Python 3.12 (tested)
- Git (to clone, optional)

## Setup (local)
```bash
# 1) Clone or unzip this repo, then inside the project folder:
python -m venv .venv

# 2) Activate the venv (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# or cmd.exe
.\.venv\Scripts\activate.bat

# 3) Install deps
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run dictionary_manager.py
```
Then open the printed local URL (default http://localhost).

## Data files
The app expects the CSVs in `output/`. Edits in the UI overwrite these files. If you deploy to Streamlit Cloud, commit the `output/` CSVs so they are available at startup.

## First-run download note
The SentenceTransformer model and PyTorch weights download on first run. Keep the terminal open and allow a minute. If you want a custom cache location, set `HF_HOME` to a folder with enough space before launching Streamlit.

## Deploy to Streamlit Cloud
1. Push this project to GitHub (include `output/` CSVs).
2. On share.streamlit.io, create a new app pointing to this repo.
3. Set the main file to `dictionary_manager.py` and Python version to 3.12.
4. Add `requirements.txt` as the dependency file; deployment will install them automatically.
5. Deploy. Subsequent edits to the repo will auto-trigger redeploys.

## Troubleshooting
- Missing data: ensure `output/translations_words.csv` and `output/translations_expressions.csv` exist and are readable.
- Package install issues: upgrade pip and retry. On Windows with long paths, enable long path support if needed.
- Slow startup: first-run model download is usually the cause; later runs will be faster due to caching.
