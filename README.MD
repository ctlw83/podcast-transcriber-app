# Podcast Transcriber App (Podcasting 2.0 Tools)

A cross-platform desktop app for podcasters to finish an episode after the MP3 is ready:
- **Transcribe locally** (Whisper via faster-whisper)
- **Edit transcript** and export **VTT / SRT**
- **Generate Podcasting 2.0 JSON Chapters** (with manual editing tools)
- **Edit/write ID3 tags** (Artist/Album/Title/Track/Year/Genre + cover art)
- **Generate local show notes** (optional local LLM)

Designed as an “indie” tool that runs on **Windows / macOS / Linux** with a simple UI and portable-friendly storage.

---

## Key Features

### 1) Audio tab
- Load common audio formats (WAV/MP3/FLAC/OGG/OPUS/M4A/AAC)
- Portable-first data storage:
  - If a `.portable` file exists next to the app/script, the app stores data in `./data/`
  - Otherwise uses a user home folder (`~/.podcast_chapter_app`)
- Dependency bootstrap:
  - FFmpeg auto-download if missing
  - Model catalog caching (to reduce 404s / filename changes)

### 2) Transcript tab
- Local transcription using faster-whisper
- Optional GPU acceleration (CUDA / Metal via MPS / CPU fallback)
- Editable transcript editor
- Export:
  - **WEBVTT** (`.vtt`)
  - **SRT** (`.srt`)
- Playback tools:
  - Play/Pause/Stop (Qt multimedia)
  - Optional external preview via `ffplay`
  - Scrub bar
  - “Follow playback in transcript” (scrolls transcript while playing)

### 3) Chapters tab
- Topic-based chapter recommendations
- Optional “Better topic detection” using local embeddings (downloads once)
- Waveform timeline + draggable markers
- Snap markers to silence points (when available)
- Full chapter editor:
  - Add / Update / Delete chapters
  - Edit start/end times
  - Set Start/End from current playback position
  - Optional URL + image fields
- Export Podcasting 2.0 chapters JSON (**version 1.2.0**)

### 4) ID3 Tags tab (MP3)
- Reads existing ID3 tags from the loaded MP3
- Writes updated tags back into the MP3 (in-place)
- Optional cover art:
  - Browse cover art image
  - Thumbnail preview in the UI

### 5) Show Notes tab
- Local show notes generator
- Optional local LLM for better titles + notes (downloads model on first use)
- Simple Markdown helper buttons (Bold/Italic/H1/H2/Link/List)
- Export notes as `.md` or `.txt`

---

## Podcasting 2.0 Chapters JSON Compatibility

The app exports **Podcasting 2.0 JSON Chapters** using:
- `version: "1.2.0"`
- `chapters: [...]` with `startTime` in seconds (and optional `title`, `img`, `url`, `toc`, `endTime`, etc.)

This structure is intended to match the Podcast Namespace JSON Chapters format.

---

## Privacy / Local-Only Design

- Transcription runs locally using Whisper (faster-whisper).
- “Better topic detection” uses a small local semantic model (downloaded once).
- Optional local LLM features run locally (no API keys required).

Note: If models are downloaded from public hubs/catalogs, your machine will fetch them over the network. No transcript text is uploaded by this app.

---

## Running From Source (Developer Mode)

1. Install Python 3.10+ (3.12 works)
2. Install dependencies (recommended):
   ```bash
   pip install pyside6 numpy pydub scikit-learn faster-whisper
