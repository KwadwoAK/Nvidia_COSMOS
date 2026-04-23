# Video Summarizer with Nvidia Cosmos AI

A Streamlit application that generates AI-powered summaries of video content using Nvidia's Cosmos-reason2-8b vision-language model.

## Features

- 🎥 Upload and process video files (MP4, AVI, MOV, MKV)
- 🤖 AI-powered frame analysis using Cosmos-reason2-8b
- 📝 Generate summaries in multiple styles (Detailed, Concise, Bullet Points)
- ⚙️ Configurable frame sampling and analysis parameters
- 💾 Download summaries as text files
- 🖼️ View sample frames from the analyzed video

## Visual Flow

Video processing flow

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd Nvidia_COSMOS
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create required secret files (not included in repo)

These files are excluded from the repository via `.gitignore` and **must be created manually** after cloning.

#### `.env`

Create a `.env` file in the project root:

```
SUPABASE_DB_URL=postgresql://<user>:<password>@<host>:5432/<dbname>?sslmode=require
LOGIN_USERNAME=your_username
LOGIN_PASSWORD=your_password
```

- `SUPABASE_DB_URL` — your Supabase (or any PostgreSQL) connection string
- `LOGIN_USERNAME` / `LOGIN_PASSWORD` — credentials for the app login screen

#### `.streamlit/secrets.toml`

Create the `.streamlit/` directory and a `secrets.toml` file inside it:

```bash
mkdir .streamlit
```

Then create `.streamlit/secrets.toml` with the following content:

```toml
[passwords]
your_username = "your_password"
```

Add one line per user you want to allow. The username and password here must match what you set in `.env` (or you can use only one of the two approaches — both work).

> **Note:** If VSCode shows these files greyed out, that is expected — they are gitignored to keep secrets out of version control. The files still work normally.

### 5. Model Access

Make sure you have access to Nvidia's Cosmos-reason2-8b model. You may need to:

- Accept the model's terms on HuggingFace
- Log in to HuggingFace: `huggingface-cli login`
- Or download the model locally and update the model path in `model_handler.py`

## Usage

### Running the Application

```bash
streamlit run app.py
python3 -m streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Configure Settings** (in sidebar):
  - Adjust frame sampling interval (1-10 seconds)
  - Set maximum frames to analyze (5-50)
  - Choose summary style (Detailed/Concise/Bullet Points)
2. **Upload Video**:
  - Click "Browse files" or drag and drop
  - Supported formats: MP4, AVI, MOV, MKV
3. **Generate Summary**:
  - Click "🚀 Generate Summary"
  - Wait for processing (3 steps):
    - Frame extraction
    - AI analysis
    - Summary generation
4. **Review Results**:
  - Read the generated summary
  - Download summary as text file
  - View sample frames from the video

## Project Structure

Tracked files only (see `.gitignore` for excluded paths like `.env`, `.idea/`, `venv/`).

```
Nvidia_COSMOS/
├── app.py                  # Main page orchestration (Upload + summary + archive)
├── auth.py                 # Login/session guard and sidebar user controls
├── video_processor.py      # Frame extraction and video processing
├── model_handler.py        # Cosmos model interface
├── vision_search.py        # Build searchable text blob from summary + frame captions
├── pages/
│   └── 2_Semantic_search.py # Dedicated semantic search Streamlit page
├── ui/
│   ├── theme.py            # Shared light/dark theme CSS + cursor glow
│   ├── components.py       # Reusable UI widgets (metric cards, formatters)
│   ├── sidebar.py          # Sidebar rendering + settings config object
│   └── __init__.py
├── services/
│   ├── pipeline.py         # Generate-summary workflow (extract/analyze/summarize/store)
│   ├── archive_search.py   # Sidebar/archive query execution helpers
│   └── __init__.py
├── state/
│   ├── session.py          # Centralized st.session_state initialization/defaults
│   └── __init__.py
├── summarys/               # Gemma summarizer + templates
├── db/
│   ├── connection.py       # PostgreSQL connection (SUPABASE_DB_URL)
│   ├── video_store.py      # Insert video summaries with embeddings
│   ├── search_video.py     # Similarity search over summaries
│   └── supabase_storage.py # Optional video upload/public URL helpers
├── embeddings/
│   ├── embedder.py         # Text → 384-dim vector (sentence-transformers)
│   └── __init__.py
├── smoke_check_pipeline.py # Smoke checks for modular wiring + storage-compatible insert
├── test_setup.py           # Verify setup (imports, GPU, video processor)
├── requirements.txt
├── README.md
├── QUICKSTART.md
```

## Database schema

The app can store video summaries in PostgreSQL with the pgvector extension for similarity search.

In Supabase, make sure the `vector` extension is enabled and you have a `video_summaries` table with an `embedding` column of dimension `384` (matches `all-MiniLM-L6-v2`).

- **Extension:** `CREATE EXTENSION IF NOT EXISTS vector;`
- **Table:** `video_summaries`


| Column          | Type          | Description                                        |
| --------------- | ------------- | -------------------------------------------------- |
| `id`            | BIGSERIAL     | Primary key                                        |
| `created_at`    | TIMESTAMPTZ   | Default NOW()                                      |
| `filename`      | TEXT          | Original video filename (optional)                 |
| `duration_sec`  | NUMERIC(10,2) | Video duration in seconds (optional)               |
| `summary_style` | TEXT          | e.g. "detailed", "concise"                         |
| `summary_text`  | TEXT NOT NULL | Full summary text                                  |
| `embedding`     | vector(384)   | Embedding for similarity search (all-MiniLM-L6-v2) |


Optional index for faster search once you have many rows:

```sql
CREATE INDEX ON video_summaries
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

## Configuration

### Frame Extraction Methods

The `VideoProcessor` class provides two methods:

1. **Interval-based extraction** (default):
  - Extracts frames at regular time intervals
  - Good for consistent sampling
2. **Keyframe extraction**:
  - Detects scene changes
  - More intelligent sampling
  - To use, modify `app.py` to call `extract_keyframes()` instead of `extract_frames()`

### Model Configuration

In `model_handler.py`, you can adjust:

- `model_name`: HuggingFace model ID or local path
- `max_new_tokens`: Maximum length of generated descriptions
- `temperature`: Creativity of responses (0.0-1.0)
- `batch_size`: Number of frames to process together

## Customization

### Summary Styles

You can adjust summary prompts in `summarys/summary_templates.py` (summary user prompts) and `summarys/gemma_summarizer.py`.

### Prompts

Customize the prompts sent to the Cosmos model in `model_handler.py`:

- Edit the `prompt` parameter in `analyze_single_frame()`
- Modify context-aware prompts in `analyze_with_context()`

## Troubleshooting

### Common Issues

**Login fails / "no credentials found"**

- Ensure `.env` exists in the project root with `LOGIN_USERNAME` and `LOGIN_PASSWORD` set
- Or ensure `.streamlit/secrets.toml` exists with a `[passwords]` section
- Verify you are running `streamlit run app.py` from the project root — Streamlit looks for `.streamlit/secrets.toml` relative to the working directory
- On Windows, check the file was saved correctly: `Test-Path ".streamlit\secrets.toml"` should return `True`

**"Missing SUPABASE_DB_URL"**

- Ensure `.env` exists and contains `SUPABASE_DB_URL=...`
- The `python-dotenv` package must be installed (`pip install -r requirements.txt`)

**"Could not open video file"**

- Ensure the video file is not corrupted
- Check that the format is supported
- Try converting to MP4 if issues persist

**"CUDA out of memory"**

- Reduce `max_frames` in the sidebar
- Increase `frame_interval` to sample fewer frames
- Use CPU instead (slower but works without GPU)

**"Model not found"**

- Verify you have access to the Cosmos model
- Check your HuggingFace authentication
- Ensure the model name in `model_handler.py` is correct

**Slow processing**

- Enable GPU acceleration
- Reduce number of frames analyzed
- Use interval-based extraction instead of keyframe detection

## Performance Tips

1. **Start small**: Test with short videos first (30-60 seconds)
2. **Balance quality vs speed**: Fewer frames = faster processing
3. **GPU recommended**: CPU processing works but is significantly slower
4. **Resize frames**: Smaller frames process faster (default is 512px width)

## Future Enhancements

Potential improvements:

- Batch processing for multiple frames
- Support for live video streams
- Audio transcription integration
- Multi-language support
- Export summaries in multiple formats (PDF, Markdown)
- Fine-tuning prompts based on video category

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Nvidia for the Cosmos-reason2-8b model
- Streamlit for the web framework
- HuggingFace Transformers library

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the Cosmos model documentation
3. Check Streamlit documentation at [https://docs.streamlit.io](https://docs.streamlit.io)

