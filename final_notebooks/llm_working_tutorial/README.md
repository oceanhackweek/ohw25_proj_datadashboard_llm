# 🌊 LLM Working Tutorial

This folder contains a runnable tutorial that demonstrates how to:

- Load climate datasets from a simple **catalog** (`datasets.json`)
- Preview and explore them with **xarray/pandas**
- Prototype **LLM-assisted workflows** in Jupyter

If you want to reproduce the workflow locally, follow the steps below.

---

## 📂 Folder Contents

llm_working_tutorial/
├─ project_result_1.ipynb # Main notebook to run
├─ dataset.py # Loader functions used by the catalog
├─ datasets.json # Dataset catalog (paths + access functions)
└─ env/ # Local venv or secrets folder (ignored)


---

## ⚙️ Requirements

- Python **3.10+** (3.11 recommended)
- JupyterLab or Jupyter Notebook
- Packages:
  - `xarray`, `fsspec`, `s3fs`, `gcsfs`, `zarr`
  - `pandas`, `numpy`
  - `python-dotenv`
- (Optional for LLM features):  
  - `openai`, `huggingface_hub`, `requests`

---

## 🚀 Quickstart

### 1️⃣ Create and activate a virtual environment
```bash
# from repo root
cd final_notebooks/llm_working_tutorial

# (option A) venv
python -m venv env
source env/bin/activate           # macOS/Linux
# .\env\Scripts\activate          # Windows PowerShell
```
```bash
# (option B) conda
# conda create -n ohw-llm python=3.11 -y
# conda activate ohw-llm
```

### 2️⃣ Install dependencies
```bash
pip install -U pip
pip install jupyterlab xarray fsspec s3fs gcsfs zarr pandas numpy python-dotenv
# Optional (LLM features)
# pip install openai huggingface_hub requests
```

### 3️⃣ Create a local .env file 
In the same folder (llm_working_tutorial/), create a file named .env:

```bash
# .env  (do NOT commit this file)

# Hugging Face API key (if using HF inference models)
HF_TOKEN=hf_**************************************

# OpenAI API key (if using OpenAI models)
OPENAI_API_KEY=sk-********************************

# Add any other cloud storage creds here if needed
```

The notebook will load these automatically with:

```bash
from dotenv import load_dotenv
load_dotenv()
```

### 4️⃣ Launch Jupyter and run the notebook
```bash
jupyter lab
# or: jupyter notebook
```

## 📊 How the Catalog Works

*`datasets.json` defines datasets with:

name → human-readable key

path → location (e.g. S3/GCS URL)

access_function → loader function in dataset.py

*`dataset.py` provides loader functions such as:

load_climate_data(...) → generic fallback/dispatcher

## 🔒 Git Hygiene

Always ignore secrets and environments. Add this to .gitignore:

```bash
# environments & secrets
env/
.env
*.env
```

If a secret was committed accidentally: revoke it immediately, then remove with:
```bash
git rm --cached .env
git commit -m "Remove accidentally committed secrets"
git push
```

## 📜 License / Credit

This tutorial is part of OceanHackWeek 2025: Data Dashboard + LLM prototype.
Respect dataset provider licenses (e.g. GHRSST, ERA5).

## 🙋 Contact

For questions or issues, please open an issue in the repo or contact the maintainers.
