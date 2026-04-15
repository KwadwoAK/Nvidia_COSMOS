# Hugging Face, Cosmos 2B, and Ollama (local setup)

## What is Hugging Face?

**Hugging Face** ([huggingface.co](https://huggingface.co)) hosts **model weights and configs**. Your app does not run “on” Hugging Face: **Python downloads** files the first time you call `from_pretrained("nvidia/Cosmos-Reason2-2B")` (or load from a local folder).

- It is **not** where Streamlit runs.
- It **is** the registry used to **download** models when online (unless you point `COSMOS_MODEL` at a local directory).

## How this project selects the vision model

1. Environment variable **`COSMOS_MODEL`** in `.env` (default `nvidia/Cosmos-Reason2-2B`), or  
2. `DEFAULT_COSMOS_MODEL` in `model_handler.py`.

The first run with a good network caches weights under your user profile (e.g. `.cache/huggingface`). Later runs can work offline if the cache is complete.

### Useful commands

```powershell
.\.venv\Scripts\hf.exe auth whoami
.\.venv\Scripts\hf.exe download nvidia/Cosmos-Reason2-2B --local-dir .\models\Cosmos-Reason2-2B
```

Then in `.env`:

```env
MOCK_COSMOS=0
COSMOS_MODEL=C:/full/path/to/Nvidia_COSMOS/models/Cosmos-Reason2-2B
```

### If download fails

- Use **`MOCK_COSMOS=1`** to skip Cosmos (placeholder captions) for UI tests.
- Copy a full model folder from a teammate (USB / shared drive).

## Gated models and 403 errors

`Cosmos-Reason2-2B` is **gated**. You must:

1. Accept the license on the model page while logged in.  
2. Use a token that can read gated public repos (fine-grained: enable **Access to public gated repositories**, or use a **classic** read token).  
3. `.\.venv\Scripts\hf.exe auth login --token YOUR_TOKEN`

---

## Ollama (local LLM for summaries)

Used when you choose **Summary engine → Ollama** in the app.

### Install and run (Windows)

1. Install from [ollama.com/download](https://ollama.com/download).  
2. Start the app (tray icon) or run `ollama serve` — API defaults to `http://127.0.0.1:11434`.  
3. Pull a model: `ollama pull llama3.2` (match **Ollama model** in the sidebar / `OLLAMA_MODEL` in `.env`).  
4. Run Streamlit from the project folder:

```powershell
cd path\to\Nvidia_COSMOS
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

5. Select **Analysis type → Municipal report (detailed)** for long **English** incident-style narratives (police / municipal filing style). Prefer a capable model; increase generation limit if the answer truncates:

```env
OLLAMA_NUM_PREDICT_MUNICIPAL=8192
```

Optional global override:

```env
OLLAMA_NUM_PREDICT=4096
```

---

## Summary

| Piece | Role |
|-------|------|
| **Cosmos** | Per-frame **vision** captions (`COSMOS_MODEL`, or `MOCK_COSMOS=1`). |
| **Ollama** | **Text** summarization in the chosen format (including municipal report). |
| **Heuristic** | Rule-based formatting without an LLM (faster, less detailed). |
