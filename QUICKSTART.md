# Quick Start Guide

## Setup (First Time Only)

1. **Install Python dependencies**
  ```bash
   pip install -r requirements.txt
  ```
2. **Verify setup**
  ```bash
   python test_setup.py
  ```
   This will check if all components are properly installed.
3. **Configure Cosmos Model Access**
  You need to ensure you have access to the Cosmos model. Update the model name in `model_handler.py` if needed:
   Replace with your actual model path or HuggingFace model ID.

## Running the Application

```bash
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501)

## First Test

1. Upload a short video (30-60 seconds recommended for first test)
2. Use default settings:
  - Frame Interval: 2 seconds
  - Max Frames: 20
  - Style: Detailed
3. Click "Generate Summary"
4. Wait for processing (may take a few minutes depending on video length and hardware)

## Important Notes

### Model Loading

- The first time you run the app and process a video, the Cosmos model will be downloaded
- This can take several minutes and requires a stable internet connection
- The model is ~15-20 GB, so ensure you have enough disk space
- Subsequent runs will use the cached model

### Performance

- **With GPU**: Processing is much faster (~30 seconds for a 1-minute video)
- **Without GPU**: Can take 5-10x longer
- Reduce `max_frames` if processing is too slow

### Supported Video Formats

- MP4 (recommended)
- AVI
- MOV  
- MKV

### Troubleshooting

**"Module not found" errors**

```bash
pip install -r requirements.txt --upgrade
```

**"CUDA out of memory"**

- Reduce max_frames to 10
- Increase frame_interval to 5
- Close other GPU-intensive applications

**"Cannot access model"**

- Check if you need to accept model terms on HuggingFace
- Log in: `huggingface-cli login`
- Verify model name is correct in `model_handler.py`

## Next Steps

Once your setup is working:

1. Try different summary styles
2. Experiment with frame intervals
3. Test with different types of videos
4. Customize prompts in `model_handler.py` for better results

## File Overview

- `app.py` - Main Streamlit interface
- `video_processor.py` - Extracts frames from videos
- `model_handler.py` - Interfaces with Cosmos model
- `summarys/ollama_summarizer.py` - Ollama summary generation
- `test_setup.py` - Verifies installation
- `requirements.txt` - Python dependencies
- `README.md` - Full documentation

## Getting Help

If you encounter issues:

1. Run `python test_setup.py` to diagnose problems
2. Check the console output for error messages
3. Verify your GPU drivers are up to date (if using GPU)
4. Make sure you have access to the Cosmos model

