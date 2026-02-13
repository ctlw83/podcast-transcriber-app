# Podcast Transcriber App

A local-first desktop tool for podcasters to prepare episode assets from a finished audio file:

- **Transcription** (local Whisper via `faster-whisper`)
- **Transcript editing + export** (VTT/SRT)
- **Chapters** (Podcasting 2.0 JSON Chapters export + editor + waveform)
- **ID3 tags** (read existing tags + write updates + embedded cover art + thumbnail preview)
- **Show notes** (local generation; optional local LLM support)

## Local-first design goals

- No cloud APIs required
- Models and tools are downloaded/cached locally
- Works as a normal desktop app (Windows/macOS/Linux)

## Requirements (developer run)

- Python 3.10–3.11 recommended
- Internet connection on first run (to download models/tools)

> Note: Some optional packages (e.g., local LLM backends) may not provide wheels for every Python version.

## Run

```bash
python podcast_transcriber_app.py
