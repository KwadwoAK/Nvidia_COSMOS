# Video Summarizer with Nvidia Cosmos AI

A Streamlit application that generates AI-powered summaries of video content using Nvidia's Cosmos-reason2-8b vision-language model.

## Features

- 🎥 Upload and process video files (MP4, AVI, MOV, MKV)
- 🤖 AI-powered frame analysis using Cosmos-reason2-8b
- 📝 Generate summaries in multiple styles (Detailed, Concise, Bullet Points)
- ⚙️ Configurable frame sampling and analysis parameters
- 💾 Download summaries as text files
- 🖼️ View sample frames from the analyzed video

## Installation

### 1. Clone or download this repository

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Model Access

Make sure you have access to Nvidia's Cosmos-reason2-8b model. You may need to:
- Accept the model's terms on HuggingFace
- Log in to HuggingFace: `huggingface-cli login`
- Or download the model locally and update the model path in `model_handler.py`

### 5. GPU Setup (Recommended)

For best performance, ensure you have:
- CUDA-compatible GPU
- Appropriate CUDA drivers installed
- PyTorch with CUDA support

To verify GPU availability:
```python
import torch
print(torch.cuda.is_available())
```

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

```
video-summarizer/
├── app.py                 # Main Streamlit application
├── video_processor.py     # Frame extraction and video processing
├── model_handler.py       # Cosmos model interface
├── summarizer.py          # Summary generation logic
├── requirements.txt       # Python dependencies
└── README.md             # This file
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

You can modify or add new summary styles in `summarizer.py`:
- `_generate_detailed_summary()`: Narrative format with scenes
- `_generate_concise_summary()`: Brief overview with key moments
- `_generate_bullet_summary()`: Point-by-point breakdown

### Prompts

Customize the prompts sent to the Cosmos model in `model_handler.py`:
- Edit the `prompt` parameter in `analyze_single_frame()`
- Modify context-aware prompts in `analyze_with_context()`

## Troubleshooting

### Common Issues

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
- [ ] Batch processing for multiple frames
- [ ] Support for live video streams
- [ ] Audio transcription integration
- [ ] Multi-language support
- [ ] Export summaries in multiple formats (PDF, Markdown)
- [ ] Fine-tuning prompts based on video category

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
3. Check Streamlit documentation at https://docs.streamlit.io
