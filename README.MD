# Podcast Transcriber App

A cross-platform desktop tool for podcasters to prep an episode *after* the final MP3 is ready.

## Features
- **Local transcription** using `faster-whisper` (CPU/GPUц/A GPU when available)
- **Transcript editor** (VTT recommended) with playback + scrubbing
- **Chapter generation** with optional “better topic detection” via local embeddings
- **Chapter editor** (edit titles, times, URLs, images) + export **Podcasting 2.0 JSON Chapters**
- **ID3 tag editor** (read existing tags, write updated tags, embed cover art + thumbnail preview)
- **Show notes generator**:
  - Fast offline heuristic mode
  - Optional **local LLM** mode (GPT4All) for better titles + notes (downloads models locally)

## Local-first design
This app is designed to run locally:
- Models are downloaded on first use and cached in the app data folder
- No cloud APIs required for transcription, chapters, or show notes (unless you add them)

## Getting Started (Developer)
1. Install Python (recommended: **Python 3.11** on Windows for best wheel compatibility).
2. Create/activate a virtual environment (recommended):
   - Windows:
     ```bash
     py -3.11 -m venv .venv
     .venv\\Scripts\\activate
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -U pip
   pip install numpy pydub scikit-learn faster-whisper PySide6 mutagen sentence-transformers transformers
   # Optional local LLM:
   pip install gpt4all
