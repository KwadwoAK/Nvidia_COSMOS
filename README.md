# Video Summarizer with Nvidia Cosmos AI

Streamlit app for authenticated video upload, frame-level analysis with `Cosmos-Reason2-8B`, summary generation with Gemma, optional Supabase Storage upload, and semantic retrieval over archived summaries.

## What It Does

- Uploads video files (`mp4`, `avi`, `mov`, `mkv`) and extracts representative frames.
- Runs visual analysis with Cosmos, then generates a structured summary with Gemma.
- Stores summary metadata and embedding in PostgreSQL/pgvector.
- Optionally uploads original videos to Supabase Storage and displays playable URLs in search results.
- Provides two pages:
  - `app.py`: upload + summarize + inline archive search
  - `pages/2_Semantic_search.py`: dedicated semantic search view

## Visual Flow

![Video processing and semantic retrieval flow](./diagrams/flow.png)



## Current Project Structure

```text
Nvidia_COSMOS/
├── app.py
├── auth.py
├── vision_search.py
├── video_processor.py
├── model_handler.py
├── smoke_check_pipeline.py
├── test_setup.py
├── pages/
│   └── 2_Semantic_search.py
├── ui/
│   ├── __init__.py
│   ├── components.py
│   ├── sidebar.py
│   └── theme.py
├── services/
│   ├── __init__.py
│   ├── archive_search.py
│   └── pipeline.py
├── state/
│   ├── __init__.py
│   └── session.py
├── db/
│   ├── connection.py
│   ├── search_video.py
│   ├── supabase_storage.py
│   └── video_store.py
├── embeddings/
│   ├── __init__.py
│   └── embedder.py
├── summarys/
│   ├── __init__.py
│   ├── gemma_summarizer.py
│   ├── ollama_summarizer.py
│   └── summary_templates.py
└── tests/
    ├── test_gemma_summarizer.py
    ├── test_summary_templates.py
    ├── test_video_processor.py
    ├── test_video_store.py
    └── test_vision_search.py
```

## Architecture Notes

- `app.py` is orchestration-focused; UI composition lives in `ui/`, workflows in `services/`, and session defaults in `state/`.
- Shared login/logout behavior is in `auth.py`.
- Shared light/dark theme is in `ui/theme.py` and is applied to both pages and the login UI.

## Setup

### 1) Clone and create environment

```bash
git clone <repo-url>
cd Nvidia_COSMOS
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment variables

Create `.env` in the project root:

```env
SUPABASE_DB_URL=postgresql://<user>:<password>@<host>:5432/<dbname>?sslmode=require
LOGIN_USERNAME=your_username
LOGIN_PASSWORD=your_password

# Optional storage upload support
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
SUPABASE_VIDEO_BUCKET=videos

# Optional model label override for stored metadata
COSMOS_MODEL_LABEL=Cosmos-Reason2-8B
```

### 3) Optional Streamlit secrets for multi-user login

Create `.streamlit/secrets.toml`:

```toml
[passwords]
alice = "password1"
bob = "password2"
```

If `[passwords]` is set, login checks this mapping first.

## Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Smoke Check

```bash
python smoke_check_pipeline.py
```

This validates:

- modular app orchestration wiring
- template metadata and search text behavior
- DB insert compatibility across old/new schemas

## Database Expectations

Requires PostgreSQL with `pgvector` enabled and table `video_summaries`.

Core columns:

- `id`, `filename`, `duration_sec`, `summary_style`, `summary_text`, `embedding`

Extended columns used when available:

- `summary_engine`, `vision_model`, `template_id`, `search_text`, `storage_object_path`

## Main User Flow

1. Log in.
2. Select theme and analysis settings in sidebar.
3. Upload video and click **Generate Summary**.
4. Pipeline runs:
  - frame extraction
  - Cosmos frame analysis
  - Gemma summary generation
  - optional Storage upload
  - summary + embedding insert into DB
5. Review summary, download text, preview sample frames.
6. Search archived summaries from sidebar or Semantic Search page.

## Troubleshooting

- Login fails:
  - set `LOGIN_USERNAME` / `LOGIN_PASSWORD`, or use `.streamlit/secrets.toml`.
- Search/storage errors:
  - verify `SUPABASE_DB_URL` and `pgvector` setup.
- Video URL not playable:
  - ensure `SUPABASE_URL` is set and `storage_object_path` exists for that row.
- Slow or OOM processing:
  - reduce max frames or increase frame interval.

## Acknowledgments

- Nvidia Cosmos model family
- Streamlit
- Hugging Face ecosystem

