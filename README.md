# Prediction Guard Vision Models Demo

Interactive demos for image and video analysis using [Prediction Guard](https://predictionguard.com)'s vision model (`Qwen3-VL-235B-A22B`), built with [Marimo](https://marimo.io).

| App | What it does |
|-----|-------------|
| `app.py` | Upload an image and ask questions about it |
| `video_app.py` | Upload a short video clip — frames are extracted and analyzed together |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (`pip install uv` or `brew install uv`)
- A [Prediction Guard API key](https://docs.predictionguard.com/getting-started/quick-start)

## Setup

```bash
git clone https://github.com/predictionguard/pg-vision-models
cd pg-vision-models

# Install dependencies
uv sync

# Configure credentials
cp .env_example .env
# Edit .env and add your PREDICTIONGUARD_API_KEY
```

## Running the Apps

Each app has two modes:

```bash
# Interactive notebook mode (edit & run cells reactively)
uv run marimo edit app.py
uv run marimo edit video_app.py

# Read-only app mode (cleaner UI for sharing/demos)
uv run marimo run app.py
uv run marimo run video_app.py
```

Both open in your browser automatically.

## Image Analysis (`app.py`)

1. Upload a JPEG, PNG, WebP, or GIF image
2. Enter a prompt (e.g. *"What objects are in this image?"*)
3. Click **Analyze Image**

## Video Analysis (`video_app.py`)

1. Upload a short video clip (MP4, WebM, MOV, AVI — keep it under ~50 MB)
2. Choose how many frames to extract (4–16, evenly spaced)
3. Enter a prompt (e.g. *"Describe the sequence of events in this video"*)
4. Click **Analyze Video**

The app uses OpenCV to extract frames, encodes them as base64 JPEG, and sends them all in a single multimodal request to the model.

## Project Structure

```
pg-vision-models/
├── app.py          # Image analysis Marimo app
├── video_app.py    # Video frame analysis Marimo app
├── pyproject.toml  # Dependencies (managed by uv)
└── .env_example    # Environment variable template
```

## License

MIT
