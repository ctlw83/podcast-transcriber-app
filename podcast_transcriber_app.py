# -*- coding: utf-8 -*-
# Podcasting 2.0 Transcript & Chapter Creator (Indie - Execution Hardened)
# Single-file cross-platform desktop app (Windows/macOS/Linux)
#
# Goals:
# - Runs as a normal desktop app (no Docker)
# - FFmpeg auto-download + extraction + PATH wiring
# - Optional GPU acceleration (CUDA/MPS/CPU fallback)
# - Robust error messages + fallbacks for audio preview
# - Zoomable waveform + scrubbing + silence snap markers
# - Local transcription via faster-whisper
# - Topic-based chapter recommendations
# - Batch processing
#
# NOTE: For a true consumer product, you package this with PyInstaller
# and include Python deps inside the installer. This script also supports
# a "developer run" mode where missing Python packages can be installed
# automatically via pip.

from __future__ import annotations

import os
import sys
import json
import math
import time
import shutil
import tarfile
import zipfile
import platform
import subprocess
import urllib.request
import warnings
import re
import hashlib
import threading
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"Couldn't find ffmpeg or avconv.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Enum value 'Qt::ApplicationAttribute\.AA_EnableHighDpiScaling'.*",
)
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple

# Detect restricted runtimes (e.g., Emscripten / Pyodide) where subprocesses aren't allowed.
IS_RESTRICTED_RUNTIME = (
    sys.platform == "emscripten" or
    "pyodide" in sys.modules or
    os.environ.get("PYODIDE", "") == "1"
)

# ------------------------------
# App paths (portable-first)
# ------------------------------

def _app_base_dir() -> str:
    """Folder where the app lives (exe folder for PyInstaller, script folder for source run)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _portable_root() -> Optional[str]:
    """If running as a portable app, store data next to the app."""
    try:
        base_dir = _app_base_dir()

        if os.environ.get("PODCAST_APP_PORTABLE", "").strip() == "1":
            return base_dir

        if os.path.exists(os.path.join(base_dir, ".portable")):
            return base_dir
    except Exception:
        pass
    return None


def app_data_dir() -> str:
    portable = _portable_root()
    if portable:
        base = os.path.join(portable, "data")
    else:
        base = os.path.join(os.path.expanduser("~"), ".podcast_chapter_app")
    os.makedirs(base, exist_ok=True)
    return base


def ffmpeg_dir() -> str:
    d = os.path.join(app_data_dir(), "ffmpeg")
    os.makedirs(d, exist_ok=True)
    return d


def models_dir() -> str:
    d = os.path.join(app_data_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d
# Keep Hugging Face / faster-whisper model cache inside our app data (portable-friendly)
MODEL_CACHE_ROOT = models_dir()
os.environ.setdefault("HF_HOME", MODEL_CACHE_ROOT)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(MODEL_CACHE_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(MODEL_CACHE_ROOT, "transformers"))

# ------------------------------
# Developer-friendly auto-install
# ------------------------------

def _can_pip_install() -> bool:
    # In packaged apps, pip may not exist; in dev mode it often does.
    # We try hard to make "auto-download deps" work in portable builds too.
    try:
        import pip  # noqa: F401
        return True
    except Exception:
        pass

    try:
        import ensurepip  # noqa: F401
        return True
    except Exception:
        pass

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test = os.path.join(path, f".__write_test__{os.getpid()}_{int(time.time())}")
        with open(test, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test)
        return True
    except Exception:
        return False


def pydeps_dir() -> str:
    """Directory used for auto-installed Python dependencies.

    IMPORTANT:
    - Even in "portable" mode, the app directory may live in protected or sync-managed locations
      (e.g., OneDrive Desktop) that can cause PermissionError/locks during pip installs.
    - To keep the UX reliable, we default pydeps to a per-user writable location, and only fall
      back to the portable data folder if needed.

    Users can force portable pydeps by setting PODCAST_APP_PORTABLE_PYDEPS=1.
    """
    force_portable = os.environ.get("PODCAST_APP_PORTABLE_PYDEPS", "").strip() == "1"

    # Preferred: per-user cache (more reliable than OneDrive Desktop/portable dirs)
    local_base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if local_base:
        preferred = os.path.join(local_base, "PodcastTranscriberApp", "pydeps")
    else:
        preferred = os.path.join(os.path.expanduser("~"), ".podcast_chapter_app", "pydeps")

    portable_candidate = os.path.join(app_data_dir(), "pydeps")

    base = portable_candidate if force_portable else preferred
    if not _is_writable_dir(base):
        base = portable_candidate

    os.makedirs(base, exist_ok=True)
    if base not in sys.path:
        sys.path.insert(0, base)
    return base


def _pydeps_active_site_dir() -> str:
    """A stable, writable target directory for pip installs.

    We avoid repeatedly overwriting the same files (which can trigger Windows file locks)
    by using a versioned site directory.
    """
    root = pydeps_dir()
    ver = f"py{sys.version_info.major}{sys.version_info.minor}"
    site = os.path.join(root, f"site-{ver}")
    os.makedirs(site, exist_ok=True)
    if site not in sys.path:
        sys.path.insert(0, site)
    return site


def _ensure_pip_available() -> None:
    """Best-effort: make pip available even in portable embedded Python builds."""
    try:
        import pip  # noqa: F401
        return
    except Exception:
        pass

    try:
        import ensurepip
        ensurepip.bootstrap(upgrade=True)
    except Exception:
        return


def ensure_import(module_name: str, pip_name: Optional[str] = None) -> None:
    """Try importing a module; if missing, attempt an auto-install into app-local pydeps.

    Hardening improvements:
    - Uses a per-user writable pydeps location by default (avoids OneDrive/portable permission issues)
    - Captures pip stdout/stderr so failures are actionable
    - Prefers binary wheels first (avoids requiring build tools), then retries without that constraint
    """
    try:
        __import__(module_name)
        return
    except Exception:
        pass

    if pip_name is None:
        pip_name = module_name

    if IS_RESTRICTED_RUNTIME:
        raise ImportError(
            f"Missing dependency: {module_name}.\n\n"
            "This environment does not support installing packages (subprocess disabled). "
            "Run the app locally (desktop Python) or use a packaged portable release."
        )


    if not _can_pip_install():
        raise ImportError(f"Missing dependency: {module_name} (pip/ensurepip unavailable)")

    _ensure_pip_available()

    def _run_pip(args: List[str]) -> None:
        p = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if p.returncode != 0:
            # Trim to keep dialogs readable
            out = (p.stdout or "")[-4000:]
            err = (p.stderr or "")[-4000:]
            raise RuntimeError(
                f"pip failed (exit {p.returncode}).\n\nstdout:\n{out}\n\nstderr:\n{err}"
            )


    target = _pydeps_active_site_dir()

    # 1) Prefer wheels to avoid build tool requirements
    base_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-warn-script-location",
        "--target",
        target,
        pip_name,
    ]

    # Some packages (esp. on Windows) fail if pip tries to build from source.
    wheel_first = base_cmd.copy()
    wheel_first.insert(6, "--only-binary=:all:")

    try:
        _run_pip(wheel_first)
    except Exception:
        # 2) Retry allowing sdists (best-effort)
        try:
            _run_pip(base_cmd)
        except Exception as e:
            # Last-mile: explain common causes clearly.
            hint = ""
            if pip_name.lower() == "gpt4all" and sys.version_info >= (3, 12):
                hint = (
                    "\n\nCommon cause on Windows: gpt4all may not provide a prebuilt wheel for your Python version. "
                    "If so, pip will fail unless build tools are installed.\n"
                    "Best fix for an indie release: bundle gpt4all inside the packaged app, or run dev with Python 3.11."
                )

            raise ImportError(f"Failed to auto-install dependency {pip_name}: {e}{hint}")


    # Make sure the newly-installed target is importable
    if target not in sys.path:
        sys.path.insert(0, target)

    __import__(module_name)


# Ensure core Python deps if running from source.

# In packaged builds, these are bundled and this is effectively a no-op.
# IMPORTANT: In restricted runtimes (like Emscripten/Pyodide), we cannot pip-install,
# so we skip these checks and let the app fail gracefully with clear messaging.
if (not IS_RESTRICTED_RUNTIME) and (not getattr(sys, "frozen", False)):
    for _mod, _pip in [
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("faster_whisper", "faster-whisper"),
    ]:
        ensure_import(_mod, _pip)


import numpy as np

# NOTE: pydub is imported lazily (inside workers/functions) to avoid import-time ffmpeg warnings
# and to reduce startup risk in packaged builds.
# from pydub import AudioSegment, silence

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from faster_whisper import WhisperModel

# ------------------------------
# Optional local LLM (GPT4All) for better chapter titles + show notes
# ------------------------------
# Why GPT4All?
# - Cross-platform binaries via pip
# - Auto-downloads GGUF models to a folder we control
# - Local inference (no API keys)
#
# Note: This is optional. If it can't load (no internet, install failure, etc.),
# the app falls back to the fast TF-IDF / embeddings heuristics.

LOCAL_LLM_DEFAULT_MODELS = [
    # Curated to run on an "average" computer (CPU-only is fine):
    # - Prefer <= ~4GB Q4 models
    # - Avoid giant 7B/13B defaults
    # NOTE: These filenames must match GPT4All's catalog exactly.
    "Llama-3.2-1B-Instruct-Q4_0.gguf",           # ~0.8GB, very light
    "Llama-3.2-3B-Instruct-Q4_0.gguf",           # ~1.9GB, solid
    "Phi-3-mini-4k-instruct.Q4_0.gguf",          # ~2.2GB, very fast
]

# Remote catalogs/manifests (override via env vars for your own hosted manifests)
DEFAULT_GPT4ALL_CATALOG_URL = os.environ.get(
    "PODCAST_APP_GPT4ALL_CATALOG_URL",
    "https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models3.json",
)
DEFAULT_APP_MANIFEST_URL = os.environ.get(
    "PODCAST_APP_MANIFEST_URL",
    "",
)
# ------------------------------
# Catalog caching (offline-friendly)
# ------------------------------
# Cache remote catalogs/manifests to disk so:
# - the app loads instantly on future runs
# - model dropdowns work offline
# - upstream filename changes are handled by refreshing catalogs
CATALOG_CACHE_TTL_S = int(os.environ.get("PODCAST_APP_CATALOG_TTL_S", "604800"))  # 7 days


def _catalog_cache_dir() -> str:
    d = os.path.join(app_data_dir(), "catalog_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path_for_url(url: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", url.strip())[:120]
    h = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return os.path.join(_catalog_cache_dir(), f"{safe}_{h}.json")


def _load_cached_json(url: str) -> Optional[object]:
    try:
        p = _cache_path_for_url(url)
        if not os.path.exists(p):
            return None
        age = time.time() - os.path.getmtime(p)
        if age > CATALOG_CACHE_TTL_S:
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cached_json(url: str, obj: object) -> None:
    try:
        p = _cache_path_for_url(url)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception:
        pass


def _download_json_cached(url: str, timeout: int = 15, *, force_refresh: bool = False) -> object:
    """Download JSON with disk cache + TTL.

    - If force_refresh=False, use cached response when valid.
    - If online fetch succeeds, cache the response.
    - If offline/fails, return cached response if available.
    """
    if not force_refresh:
        cached = _load_cached_json(url)
        if cached is not None:
            return cached

    try:
        raw = _download_bytes(url, timeout=timeout)
        obj = json.loads(raw.decode("utf-8", errors="replace"))
        _save_cached_json(url, obj)
        return obj
    except Exception:
        cached = _load_cached_json(url)
        if cached is not None:
            return cached
        raise


def gpt4all_models_dir() -> str:
    d = os.path.join(models_dir(), "gpt4all")
    os.makedirs(d, exist_ok=True)
    return d


def _llm_pick_device(prefer_gpu: bool = True) -> str | None:
    """Return a GPT4All device string.

    GPT4All uses different backends per platform. We keep this simple:
    - If CUDA is available: "cuda"
    - On Apple Silicon: "gpu" (metal)
    - Otherwise: None (let GPT4All choose kompute/cpu) or "cpu"

    The UI lets users force CPU/GPU; "Auto" returns None to let GPT4All decide.
    """
    if not prefer_gpu:
        return "cpu"

    # CUDA / MPS detection via torch if present (optional)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if sys.platform == "darwin" and platform.machine() == "arm64":
            # GPT4All expects device="gpu" for metal on arm64
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "gpu"
    except Exception:
        pass

    # Let GPT4All pick best available (kompute/cpu).
    return None


def _sanitize_llm_title(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'")
    # Remove common prefixes
    s = re.sub(r"^(chapter\s*[0-9]+\s*[:\-–-]\s*)", "", s, flags=re.I).strip()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Avoid empty / garbage
    if not s:
        return "Chapter"
    # Limit length
    if len(s) > 60:
        s = s[:60].rsplit(" ", 1)[0].strip() or s[:60]
    # Basic capitalization
    return s[:1].upper() + s[1:]


# Qt imports (can fail in some environments)
ensure_import("PySide6", "PySide6")
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QSlider,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QTabWidget,
    QSpinBox,
    QInputDialog,
)

# Qt Multimedia is optional; we will fall back to ffplay if unavailable.
try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    QT_MULTIMEDIA_OK = True
except Exception:
    QT_MULTIMEDIA_OK = False


# ------------------------------
# FFmpeg auto-download
# ------------------------------

# ------------------------------

def _download(url: str, dst: str, progress_cb=None) -> None:
    def _report(blocks, block_size, total_size):
        if not progress_cb or total_size <= 0:
            return
        downloaded = blocks * block_size
        progress_cb(min(100, int(downloaded * 100 / total_size)))

    urllib.request.urlretrieve(url, dst, reporthook=_report)


def _download_bytes(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "PodcastChapters/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download_json(url: str, timeout: int = 15) -> dict:
    raw = _download_bytes(url, timeout=timeout)
    return json.loads(raw.decode("utf-8", errors="replace"))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def _extract_archive(archive_path: str, out_dir: str) -> None:
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(out_dir)
        return

    # .tar.xz for Linux static builds
    if archive_path.endswith(".tar.xz"):
        with tarfile.open(archive_path, mode="r:xz") as t:
            t.extractall(out_dir)
        return

    raise ValueError(f"Unknown archive format: {archive_path}")


def _find_ffmpeg_binary(root: str) -> Optional[str]:
    exe = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    for r, _, files in os.walk(root):
        if exe in files:
            return os.path.join(r, exe)
    return None


def _load_app_manifest() -> Optional[dict]:
    """Optional: fetch a remote manifest that can update dependency URLs/filenames.

    For a real indie release, host a small JSON manifest on GitHub Pages (or similar)
    that you control, and update it when upstream filenames/structures change.

    If DEFAULT_APP_MANIFEST_URL is empty, this is disabled.
    """
    if not DEFAULT_APP_MANIFEST_URL:
        return None
    try:
        return _download_json(DEFAULT_APP_MANIFEST_URL, timeout=10)
    except Exception:
        return None


def ensure_ffmpeg(progress_cb=None) -> Optional[str]:
    """Return path to ffmpeg binary. Downloads/extracts if needed.

    Strategy:
    1) Use system ffmpeg if present.
    2) Use previously-downloaded ffmpeg in app data.
    3) Download from:
       - Your app manifest (if configured)
       - Else fallback community URLs.

    We don't rely on archive root folder names; we scan extracted content to find ffmpeg.
    """
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    base = ffmpeg_dir()

    # If we previously downloaded, reuse
    existing = _find_ffmpeg_binary(base)
    if existing:
        os.environ["PATH"] = os.pathsep.join([os.path.dirname(existing), os.environ.get("PATH", "")])
        return existing

    system = platform.system()

    # Optional manifest override
    manifest = _load_app_manifest()
    if manifest and isinstance(manifest, dict):
        try:
            ffm = manifest.get("ffmpeg", {})
            plat = ffm.get(system, {}) if isinstance(ffm, dict) else {}
            url = (plat.get("url") or "").strip()
            sha = (plat.get("sha256") or "").strip().lower()
            if url:
                fname = os.path.basename(url) or ("ffmpeg.zip" if system in ("Windows", "Darwin") else "ffmpeg.tar.xz")
                archive = os.path.join(base, fname)
                _download(url, archive, progress_cb=progress_cb)
                if sha:
                    got = _sha256_file(archive)
                    if got.lower() != sha:
                        try:
                            os.remove(archive)
                        except Exception:
                            pass
                        return None
                _extract_archive(archive, base)
                ff = _find_ffmpeg_binary(base)
                if not ff:
                    return None
                os.environ["PATH"] = os.pathsep.join([os.path.dirname(ff), os.environ.get("PATH", "")])
                return ff
        except Exception:
            # Fall through to default sources
            pass

    # Default community sources (fallback)
    urls = {
        "Windows": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "Darwin": "https://evermeet.cx/ffmpeg/getrelease/zip",
        "Linux": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
    }

    url = urls.get(system)
    if not url:
        return None

    fname = os.path.basename(url)
    if fname == "zip":
        # evermeet uses a non-file basename in URL
        fname = "ffmpeg-macos.zip"

    archive = os.path.join(base, fname)

    try:
        _download(url, archive, progress_cb=progress_cb)
        _extract_archive(archive, base)
        ff = _find_ffmpeg_binary(base)
        if not ff:
            return None

        # Add to PATH so pydub and subprocess can find it
        os.environ["PATH"] = os.pathsep.join([os.path.dirname(ff), os.environ.get("PATH", "")])

        # pydub is imported lazily; workers will set AudioSegment.converter when needed.
        return ff

    except Exception:
        return None


# ------------------------------
# Audio utilities
# ------------------------------

def seconds_to_hhmmss_mmm(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"  # HH:MM:SS.mmm


def seconds_to_timestamp(sec: float) -> str:
    # For UI display only.
    return seconds_to_hhmmss_mmm(sec)


def detect_silence_boundaries_fraction(audio_path: str) -> List[float]:
    """Return silence start times as fractions [0..1] of duration.

    NOTE: Optional feature (currently not used by default).
    Kept for future snapping, but intentionally not used during audio load.
    """
    ensure_import("pydub", "pydub")
    from pydub import AudioSegment, silence

    audio = AudioSegment.from_file(audio_path)
    dur_s = max(0.001, len(audio) / 1000.0)

    sil = silence.detect_silence(
        audio,
        min_silence_len=1200,
        silence_thresh=audio.dBFS - 16,
    )
    return [max(0.0, min(1.0, (s[0] / 1000.0) / dur_s)) for s in sil]


def _ffprobe_duration_seconds(audio_path: str, ffmpeg_path: Optional[str]) -> Optional[float]:
    """Fast duration lookup using ffprobe (preferred) or ffmpeg fallback."""
    try:
        exe = None
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            base = os.path.dirname(ffmpeg_path)
            cand = os.path.join(base, "ffprobe.exe" if platform.system() == "Windows" else "ffprobe")
            if os.path.exists(cand):
                exe = cand
        if exe is None:
            exe = shutil.which("ffprobe")
        if not exe:
            return None

        p = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if p.returncode != 0:
            return None
        val = (p.stdout or "").strip()
        if not val:
            return None
        return float(val)
    except Exception:
        return None


def _waveform_peaks_ffmpeg(
    audio_path: str,
    ffmpeg_path: Optional[str],
    *,
    target_points: int = 20000,
    sample_rate: int = 8000,
) -> Tuple[np.ndarray, float]:
    """Compute a lightweight waveform using ffmpeg piping.

    This avoids pydub decode overhead and is much faster for large MP3s.
    Returns (wave_peaks, duration_s).
    """
    ff = ffmpeg_path or shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("FFmpeg not found")

    dur = _ffprobe_duration_seconds(audio_path, ffmpeg_path)

    # Decode to raw 16-bit PCM mono at low sample rate
    cmd = [
        ff,
        "-hide_banner",
        "-nostdin",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]

    # Read PCM stream and aggregate into peaks
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    # If duration unknown, estimate from audio length later.
    # If duration known, compute block size so we yield ~target_points.
    if dur and dur > 0:
        total_samples = int(dur * sample_rate)
        block = max(1, total_samples // max(1, int(target_points)))
    else:
        block = 4096  # adaptive-ish default

    peaks: List[float] = []
    pending = np.empty(0, dtype=np.int16)

    bytes_per_sample = 2
    chunk_bytes = 1024 * 64

    read_samples = 0
    try:
        while True:
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype=np.int16)
            read_samples += int(arr.size)

            if pending.size:
                arr = np.concatenate([pending, arr])
                pending = np.empty(0, dtype=np.int16)

            n = arr.size
            if n < block:
                pending = arr
                continue

            usable = (n // block) * block
            cut = arr[:usable]
            pending = arr[usable:]

            # reshape and take max abs per block
            cut = cut.reshape((-1, block))
            p = np.max(np.abs(cut).astype(np.int32), axis=1).astype(np.float32)
            peaks.extend(p.tolist())

            # early stop if we already have plenty and duration was unknown
            if (dur is None) and len(peaks) > target_points * 2:
                # increase block size on the fly to reduce memory
                block *= 2

        # flush remainder
        if pending.size:
            peaks.append(float(np.max(np.abs(pending).astype(np.int32))))

        rc = proc.wait(timeout=20)
        if rc != 0:
            # Consume stderr for debug context
            err = (proc.stderr.read() if proc.stderr else b"")
            raise RuntimeError((err or b"ffmpeg failed").decode("utf-8", errors="replace")[-2000:])

    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass

    wave = np.asarray(peaks, dtype=np.float32)
    if wave.size == 0:
        wave = np.zeros(1, dtype=np.float32)

    # Derive duration if unknown
    if dur is None:
        dur = read_samples / float(sample_rate)

    return wave, float(dur or 0.0)


def load_waveform(audio_path: str, downsample: int = 1000) -> Tuple[np.ndarray, float]:
    """Legacy helper (not used by the UI). Kept for compatibility."""
    ensure_import("pydub", "pydub")
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    dur_s = len(audio) / 1000.0

    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)

    if len(samples) == 0:
        return np.zeros(1, dtype=np.float32), dur_s

    ds = max(1, int(downsample))
    wave = samples[::ds].astype(np.float32)
    return wave, dur_s



def load_waveform(audio_path: str, downsample: int = 1000) -> Tuple[np.ndarray, float]:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    dur_s = len(audio) / 1000.0

    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)

    if len(samples) == 0:
        return np.zeros(1, dtype=np.float32), dur_s

    ds = max(1, int(downsample))
    wave = samples[::ds].astype(np.float32)
    return wave, dur_s


# ------------------------------
# Chapters
# ------------------------------

@dataclass
class Chapter:
    title: str
    start: float
    end: float
    url: str = ""
    img: str = ""
    toc: bool = True
    location: Optional[dict] = None


FILLER_WORDS = {
    # Disfluencies / filler
    "um", "uh", "erm", "ah", "like", "you", "know", "just", "mean", "yeah", "okay", "right",
    "sort", "kind", "basically", "actually", "literally",
    # Colloquialisms / contraction fragments that make awful chapter titles
    "gonna", "wanna", "gotta", "kinda", "sorta", "wouldn", "shouldn", "couldn", "didn", "don", "isn",
    "aren", "wasn", "weren", "hasn", "haven", "hadn",
    # Generic verbs that dominate TF counts
    "get", "got", "make", "made", "take", "took", "say", "said", "think", "thought", "let", "try", "going",
}


def _normalize_text_for_topics(text: str) -> str:
    # Normalize apostrophes/dashes and strip non-word noise.
    t = (text or "").lower()
    t = t.replace("’", "'").replace("–", "-").replace("-", "-")
    # wouldn't -> wouldn (helps filter contraction fragments)
    t = re.sub(r"\b(\w+)'(t|re|ve|ll|d|m|s)\b", r"\1", t)
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _phrase_is_good(phrase: str) -> bool:
    p = phrase.strip()
    if not p:
        return False
    if len(p) < 3 or p.isdigit():
        return False

    toks = [x for x in re.split(r"\s+", p.lower()) if x]
    if not toks:
        return False

    if all((t in ENGLISH_STOP_WORDS) or (t in FILLER_WORDS) for t in toks):
        return False

    if len(toks) <= 2 and any(t in FILLER_WORDS for t in toks):
        return False

    if toks[0] in ENGLISH_STOP_WORDS or toks[-1] in ENGLISH_STOP_WORDS:
        return False

    return True


def _dedupe_phrases(phrases: List[str], max_terms: int) -> List[str]:
    # Prefer longer phrases; remove unigrams that are substrings of longer phrases.
    cleaned = [p.strip() for p in phrases if _phrase_is_good(p)]
    cleaned = [p.title() for p in cleaned]
    cleaned.sort(key=lambda s: (-len(s), s))

    kept: List[str] = []
    for p in cleaned:
        pl = p.lower()
        if any(pl in k.lower() for k in kept):
            continue
        kept.append(p)
        if len(kept) >= max_terms:
            break

    return kept


def _clean_keywords(words: List[str]) -> List[str]:
    out: List[str] = []
    for w in words:
        ww = w.strip().lower()
        if not ww:
            continue
        if ww in FILLER_WORDS:
            continue
        if ww in ENGLISH_STOP_WORDS:
            continue
        if len(ww) <= 2:
            continue
        out.append(w)

    seen = set()
    uniq: List[str] = []
    for w in out:
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        uniq.append(w)
    return uniq


_ST_MODEL = None
_ST_LOCK = threading.Lock()


def _maybe_load_sentence_transformer():
    """Optional enhanced topic model. Lazy-load so it doesn't slow startup.

    We also reduce HF/transformers logging noise so end users don't see scary warnings.
    """
    try:
        ensure_import("sentence_transformers", "sentence-transformers")
        from sentence_transformers import SentenceTransformer  # type: ignore

        # Quiet transformers/sentence-transformers logs (these messages are usually harmless)
        try:
            ensure_import("transformers", "transformers")
            from transformers.utils import logging as hf_logging  # type: ignore
            hf_logging.set_verbosity_error()
        except Exception:
            pass

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        cache = os.path.join(models_dir(), "sentence_transformers")
        os.makedirs(cache, exist_ok=True)
        global _ST_MODEL
        with _ST_LOCK:
            if _ST_MODEL is not None:
                return _ST_MODEL

            _ST_MODEL = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=cache,
                device="cpu",
            )
            return _ST_MODEL

    except Exception:
        return None


def _make_title_from_text(text: str, max_terms: int = 4, st_model=None) -> str:
    """Create a short, human-friendly chapter label.

    The previous version returned a *list* of top n-grams joined with slashes,
    which often looks like keyword soup in conversational transcripts.

    New behavior:
    - Prefer a single best phrase (or at most two) that matches the chapter semantics.
    - If an embedding model is available, do a KeyBERT-style rerank locally.
    - Apply strong filtering against filler words / fragments.
    """

    text = (text or "").strip()
    if not text:
        return "Chapter"

    norm = _normalize_text_for_topics(text)
    if not norm:
        return "Chapter"

    # Candidate n-grams with simple TF weighting (fast)
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=12000,
        use_idf=False,
        norm=None,
        sublinear_tf=True,
    )
    X = vec.fit_transform([norm])
    terms = vec.get_feature_names_out()
    weights = X.toarray()[0]

    order = np.argsort(weights)[::-1]
    ranked = [terms[i] for i in order if weights[i] > 0]
    ranked = _dedupe_phrases(ranked, max_terms=max(30, max_terms * 10))
    if not ranked:
        return "Chapter"

    # If we have a sentence-transformer model, rerank candidates by semantic match.
    if st_model is not None:
        try:
            cand = ranked[: min(len(ranked), 40)]
            doc_emb = st_model.encode([norm], normalize_embeddings=True, show_progress_bar=False)
            doc_emb = np.asarray(doc_emb, dtype=np.float32)[0]
            cand_emb = st_model.encode(cand, normalize_embeddings=True, show_progress_bar=False)
            cand_emb = np.asarray(cand_emb, dtype=np.float32)
            sims = cand_emb @ doc_emb

            # Blend TF weight rank with semantic similarity to avoid generic phrases.
            # Higher sims is better; slightly favor earlier (higher TF) candidates.
            tf_rank_bonus = np.linspace(1.0, 0.7, num=len(cand), dtype=np.float32)
            score = sims * tf_rank_bonus
            best = int(np.argmax(score))
            pick1 = cand[best]

            # Optional second phrase if it's very complementary.
            pick2 = None
            if max_terms >= 2:
                # pick the best phrase that isn't a substring of pick1 and isn't too redundant
                for j in np.argsort(score)[::-1]:
                    if j == best:
                        continue
                    p = cand[int(j)]
                    pl = p.lower()
                    if pl in pick1.lower() or pick1.lower() in pl:
                        continue
                    # avoid near-duplicates semantically
                    if float(cand_emb[int(j)] @ cand_emb[best]) > 0.85:
                        continue
                    pick2 = p
                    break

            if pick2 and max_terms >= 2:
                return " / ".join([pick1.title(), pick2.title()])
            return pick1.title()
        except Exception:
            pass

    # Fallback: just return the best filtered phrase (single)
    return ranked[0].title()


def topic_cluster_chapters(
    segments: List[dict],
    duration_s: Optional[float] = None,
    silence_points: Optional[List[float]] = None,
    target_chapters: int = 10,
    use_embeddings: bool = False,
) -> List[Chapter]:
    """Generate chapters using *block-level* topic shift detection.

    Previous version clustered per-segment, which caused rapid oscillation and junk titles.
    This version:
    - groups transcript into time blocks (60-90s typical)
    - computes cosine similarity between adjacent blocks
    - chooses boundaries at the largest topic shifts
    - enforces a minimum chapter duration
    - optionally snaps boundaries to nearby silence points
    """

    if not segments:
        return []

    if duration_s is None:
        duration_s = float(segments[-1]["end"]) if segments else 0.0
    duration_s = max(0.0, float(duration_s or 0.0))

    # Build blocks
    blocks: List[dict] = []
    block_text: List[str] = []
    b_start = float(segments[0]["start"])
    b_end = float(segments[0]["end"])

    # Choose block size based on episode length
    # Aim ~3 blocks per chapter so topic shifts are meaningful.
    if duration_s > 0:
        desired_block = max(60.0, min(180.0, duration_s / max(1, target_chapters * 3)))
    else:
        desired_block = 75.0

    for s in segments:
        st = float(s.get("start", 0.0))
        en = float(s.get("end", st))
        tx = (s.get("text") or "").strip()

        if not block_text:
            b_start = st
            b_end = en
        else:
            b_end = max(b_end, en)

        block_text.append(tx)

        if (b_end - b_start) >= desired_block and len(block_text) >= 3:
            blocks.append({"start": b_start, "end": b_end, "text": " ".join(block_text).strip()})
            block_text = []

    if block_text:
        blocks.append({"start": b_start, "end": b_end, "text": " ".join(block_text).strip()})

    if len(blocks) <= 1:
        return [Chapter(title="Episode", start=float(segments[0]["start"]), end=float(segments[-1]["end"]))]

    # Compute similarity between adjacent blocks.
    sims: List[float] = []

    st_model = _maybe_load_sentence_transformer() if use_embeddings else None

    # If embeddings are enabled but fail, fall back safely to TF-IDF.
    if st_model is not None:
        try:
            texts = [_normalize_text_for_topics(b["text"]) for b in blocks]
            emb = st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            emb = np.asarray(emb, dtype=np.float32)
            for i in range(len(blocks) - 1):
                sim = float(np.clip(np.dot(emb[i], emb[i + 1]), -1.0, 1.0))
                sims.append(sim)
        except Exception:
            st_model = None
            sims = []

    if st_model is None:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        X = vec.fit_transform([_normalize_text_for_topics(b["text"]) for b in blocks])
        for i in range(len(blocks) - 1):
            sim = float(cosine_similarity(X[i], X[i + 1])[0][0])
            sims.append(sim)


    # Candidate boundaries are the biggest topic shifts (lowest similarity)
    # Keep in time order.
    min_chapter = 180.0  # seconds
    max_chapter = 20 * 60.0

    # Decide how many boundaries to try.
    # target_chapters -> target_chapters-1 boundaries.
    want = max(1, min(len(blocks) - 1, int(target_chapters) - 1))

    # Sort indices by (similarity ascending), then take top `want*2` candidates to allow constraints.
    idx_sorted = sorted(range(len(sims)), key=lambda i: sims[i])
    candidates = idx_sorted[: min(len(idx_sorted), want * 2)]
    candidates = sorted(candidates)

    boundaries: List[float] = [float(segments[0]["start"])]
    last_t = boundaries[0]

    # Prepare silence snaps
    silence_s: List[float] = []
    if silence_points and duration_s > 0:
        silence_s = [max(0.0, min(duration_s, float(p) * duration_s)) for p in silence_points]

    def _snap(t: float) -> float:
        if not silence_s:
            return t
        # snap within 15s
        nearest = min(silence_s, key=lambda x: abs(x - t))
        if abs(nearest - t) <= 15.0:
            return nearest
        return t

    for i in candidates:
        t = float(blocks[i + 1]["start"])
        t = _snap(t)
        if t - last_t < min_chapter:
            continue
        if t - last_t > max_chapter:
            # If we have a gigantic chapter, accept the boundary anyway.
            pass
        boundaries.append(t)
        last_t = t
        if len(boundaries) - 1 >= want:
            break

    # Always end at the end of audio
    end_t = float(segments[-1]["end"])
    boundaries = sorted(set(boundaries))

    chapters: List[Chapter] = []
    for bi, bt in enumerate(boundaries):
        st = bt
        en = boundaries[bi + 1] if bi + 1 < len(boundaries) else end_t
        if en - st < 30.0:
            continue

        # Collect block texts in this range to label topic
        texts = [b["text"] for b in blocks if (float(b["start"]) >= st - 1e-6 and float(b["start"]) < en - 1e-6)]
        joined = " ".join(texts).strip()
        title = _make_title_from_text(joined, st_model=st_model) if joined else f"Chapter {len(chapters) + 1}"

        # A couple of friendly heuristics
        # Friendly labeling heuristics
        low = _normalize_text_for_topics(joined)
        if bi == 0 and st <= 5.0:
            title = "Introduction"
        # Common segment patterns
        if "value for value" in low or "support the show" in low or "patreon" in low or "boost" in low or "boostagram" in low:
            title = "Value for Value"
        if en >= (end_t - 120.0):
            # If this is the closing stretch, label it as Closing unless it's clearly another segment.
            if "thank" in low or "bye" in low or "see you" in low or "next time" in low:
                title = "Closing"

        chapters.append(Chapter(title=title, start=st, end=en))

    # Final merge of ultra-short chapters
    merged: List[Chapter] = []
    for ch in chapters:
        if merged and (ch.end - ch.start) < 45.0:
            merged[-1].end = ch.end
        else:
            merged.append(ch)

    return merged


def export_podcast20_json(
    chapters: List[Chapter],
    path: str,
    *,
    author: str = "",
    title: str = "",
    podcastName: str = "",
    description: str = "",
    fileName: str = "",
    waypoints: Optional[bool] = None,
) -> None:
    """Export Podcasting 2.0 JSON Chapters (v1.2.0).

    Required top-level fields:
      - version (string)
      - chapters (array)

    Required chapter fields:
      - startTime (float seconds)

    Common optional chapter fields we support:
      - title (string)
      - img (string URL)
      - url (string URL)
      - toc (boolean)
      - endTime (float seconds)
      - location (object)

    Optional top-level metadata we support:
      - author, title, podcastName, description, fileName, waypoints
    """

    out: dict = {
        "version": "1.2.0",
        "chapters": [],
    }

    # Optional top-level metadata (only include when non-empty)
    if author.strip():
        out["author"] = author.strip()
    if title.strip():
        out["title"] = title.strip()
    if podcastName.strip():
        out["podcastName"] = podcastName.strip()
    if description.strip():
        out["description"] = description.strip()
    if fileName.strip():
        out["fileName"] = fileName.strip()
    if waypoints is not None:
        out["waypoints"] = bool(waypoints)

    # Chapters must be in ascending startTime.
    cleaned: List[Chapter] = []
    for c in chapters:
        st = max(0.0, float(c.start))
        en = float(c.end) if c.end is not None else st
        if en < st:
            en = st
        cleaned.append(
            Chapter(
                title=(c.title or "").strip() or "Chapter",
                start=st,
                end=en,
                url=(c.url or "").strip(),
                img=(c.img or "").strip(),
                toc=bool(getattr(c, "toc", True)),
                location=getattr(c, "location", None),
            )
        )

    cleaned.sort(key=lambda x: float(x.start))

    for c in cleaned:
        obj: dict = {
            "startTime": round(float(c.start), 3),
        }
        # Optional per spec
        if (c.title or "").strip():
            obj["title"] = c.title
        if c.img:
            obj["img"] = c.img
        if c.url:
            obj["url"] = c.url
        if c.toc is False:
            obj["toc"] = False
        # endTime is optional, but helpful for editors/players that support it
        if float(c.end) > float(c.start):
            obj["endTime"] = round(float(c.end), 3)
        if c.location and isinstance(c.location, dict):
            obj["location"] = c.location

        out["chapters"].append(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")



# ------------------------------
# Waveform widget
# ------------------------------

class WaveformWidget(QWidget):
    scrubRequested = Signal(float)          # fraction [0..1]
    markerChanged = Signal(int, float)      # marker index, new fraction

    def __init__(self):
        super().__init__()
        self.setMinimumSize(QSize(800, 180))

        self.wave: Optional[np.ndarray] = None
        self.zoom: float = 1.0
        self.offset: float = 0.0  # [0..1] start of view in waveform fraction

        self.silence_points: List[float] = []
        self.markers: List[float] = []
        self._drag_idx: Optional[int] = None

    def set_data(self, wave: np.ndarray, silence_points: List[float]) -> None:
        self.wave = wave
        self.silence_points = silence_points
        self.zoom = 1.0
        self.offset = 0.0
        self.update()

    def set_markers(self, markers: List[float]) -> None:
        self.markers = [max(0.0, min(1.0, m)) for m in markers]
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 1.25
        else:
            self.zoom /= 1.25
        self.zoom = max(1.0, min(self.zoom, 20.0))
        self.update()

    def paintEvent(self, event):
        if self.wave is None or len(self.wave) < 2:
            return

        w = self.width()
        h = self.height()

        painter = QPainter(self)
        painter.fillRect(0, 0, w, h, QColor("#121212"))

        # Determine visible slice based on zoom and offset
        total = len(self.wave)
        visible = int(total / self.zoom)
        visible = max(2, min(total, visible))

        start = int(self.offset * (total - visible))
        start = max(0, min(total - visible, start))
        data = self.wave[start : start + visible]

        max_val = float(np.max(np.abs(data))) if len(data) else 1.0
        max_val = max(1.0, max_val)

        # Waveform
        pen = QPen(QColor("#3aa6ff"), 1)
        painter.setPen(pen)

        step = len(data) / max(1, w)
        mid = h / 2.0
        for x in range(w):
            idx = min(int(x * step), len(data) - 1)
            val = float(data[idx]) / max_val
            y = val * (h * 0.45)
            painter.drawLine(int(x), int(mid - y), int(x), int(mid + y))

        # Silence guides (grey)
        pen = QPen(QColor("#666666"), 1)
        painter.setPen(pen)
        for s in self.silence_points:
            # s in [0..1] global fraction; map to view
            x = self._fraction_to_x(s)
            if x is not None:
                painter.drawLine(int(x), 0, int(x), h)

        # Chapter markers (orange)
        pen = QPen(QColor("#ff7a3a"), 2)
        painter.setPen(pen)
        for m in self.markers:
            x = self._fraction_to_x(m)
            if x is not None:
                painter.drawLine(int(x), 0, int(x), h)

        # Labels
        painter.setPen(QColor("#e0e0e0"))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(10, 18, f"Zoom: {self.zoom:.1f}x")

    def _fraction_to_x(self, frac: float) -> Optional[float]:
        if self.wave is None or len(self.wave) < 2:
            return None

        # Convert global fraction to view x
        if self.zoom <= 0:
            return None

        view_span = 1.0 / self.zoom
        view_start = self.offset * (1.0 - view_span)
        view_end = view_start + view_span

        if frac < view_start or frac > view_end:
            return None

        local = (frac - view_start) / max(1e-9, view_span)
        return local * self.width()

    def _x_to_fraction(self, x: float) -> float:
        view_span = 1.0 / self.zoom
        view_start = self.offset * (1.0 - view_span)
        local = max(0.0, min(1.0, x / max(1, self.width())))
        return view_start + local * view_span

    def mousePressEvent(self, event):
        if not self.markers:
            return

        frac = self._x_to_fraction(event.position().x())
        # Select nearest marker
        d = [abs(m - frac) for m in self.markers]
        self._drag_idx = int(np.argmin(d))

    def mouseMoveEvent(self, event):
        if self._drag_idx is None:
            return

        frac = self._x_to_fraction(event.position().x())

        # Snap to nearest silence point within threshold
        if self.silence_points:
            nearest = min(self.silence_points, key=lambda s: abs(s - frac))
            if abs(nearest - frac) < 0.02:
                frac = nearest

        frac = max(0.0, min(1.0, frac))
        self.markers[self._drag_idx] = frac
        self.markerChanged.emit(self._drag_idx, frac)
        self.update()

    def mouseReleaseEvent(self, event):
        self._drag_idx = None

    def mouseDoubleClickEvent(self, event):
        frac = self._x_to_fraction(event.position().x())
        self.scrubRequested.emit(max(0.0, min(1.0, frac)))


# ------------------------------
# Transcript export + parsing
# ------------------------------

def _parse_timecode(tc: str) -> float:
    """Parse HH:MM:SS.mmm, HH:MM:SS,mmm, MM:SS.mmm, or seconds."""
    tc = tc.strip()
    if not tc:
        raise ValueError("empty timecode")

    # seconds
    try:
        if tc.replace(".", "", 1).isdigit():
            return float(tc)
    except Exception:
        pass

    tc = tc.replace(",", ".")
    parts = tc.split(":")
    if len(parts) == 2:
        m = int(parts[0])
        s = float(parts[1])
        return m * 60 + s
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s

    raise ValueError(f"bad timecode: {tc}")


def _format_srt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms >= 1000:
        ms -= 1000
        s += 1
    if s >= 60:
        s -= 60
        m += 1
    if m >= 60:
        m -= 60
        h += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms >= 1000:
        ms -= 1000
        s += 1
    if s >= 60:
        s -= 60
        m += 1
    if m >= 60:
        m -= 60
        h += 1
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def export_srt(segments: List[dict], path: str) -> None:
    lines: List[str] = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_srt_time(s['start'])} --> {_format_srt_time(s['end'])}")
        lines.append((s.get('text') or '').strip())
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def export_vtt(segments: List[dict], path: str) -> None:
    lines: List[str] = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{_format_vtt_time(s['start'])} --> {_format_vtt_time(s['end'])}")
        lines.append((s.get('text') or '').strip())
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def parse_transcript_editor(text: str) -> List[dict]:
    """Parse editor text into segments.

    Supports:
    - VTT (WEBVTT header)
    - SRT
    - Single-line cues:
        [HH:MM:SS.mmm --> HH:MM:SS.mmm] text
        HH:MM:SS.mmm --> HH:MM:SS.mmm | text
    """
    raw = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not raw:
        return []

    # VTT
    if raw.lstrip().startswith("WEBVTT"):
        lines = raw.split("\n")
        # drop header + blanks
        while lines and (not lines[0].strip() or lines[0].strip().startswith("WEBVTT")):
            lines.pop(0)

        segs: List[dict] = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "-->" in line:
                start_s, end_s = [t.strip() for t in line.split("-->")]
                i += 1
                text_lines: List[str] = []
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    text_lines.append(lines[i].strip())
                    i += 1
                # skip separators
                while i < len(lines) and not lines[i].strip():
                    i += 1
                segs.append({
                    "start": _parse_timecode(start_s),
                    "end": _parse_timecode(end_s),
                    "text": " ".join(text_lines).strip(),
                })
            else:
                i += 1
        return segs

    # SRT
    if "-->" in raw and "," in raw:
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        segs: List[dict] = []
        for b in blocks:
            ls = [l.strip() for l in b.split("\n") if l.strip()]
            if len(ls) < 2:
                continue
            if "-->" in ls[0]:
                time_line = ls[0]
                text_lines = ls[1:]
            else:
                time_line = ls[1]
                text_lines = ls[2:] if len(ls) > 2 else []
            if "-->" not in time_line:
                continue
            start_s, end_s = [t.strip() for t in time_line.split("-->")]
            segs.append({
                "start": _parse_timecode(start_s),
                "end": _parse_timecode(end_s),
                "text": " ".join(text_lines).strip(),
            })
        return segs

    # One-line cue formats
    segs: List[dict] = []
    for line in raw.split("\n"):
        l = line.strip()
        if not l:
            continue
        if l.startswith("[") and "]" in l and "-->" in l:
            inside, rest = l[1:].split("]", 1)
            start_s, end_s = [t.strip() for t in inside.split("-->")]
            segs.append({"start": _parse_timecode(start_s), "end": _parse_timecode(end_s), "text": rest.strip()})
            continue
        if "-->" in l and "|" in l:
            times, rest = l.split("|", 1)
            start_s, end_s = [t.strip() for t in times.split("-->")]
            segs.append({"start": _parse_timecode(start_s), "end": _parse_timecode(end_s), "text": rest.strip()})
            continue

    return segs


# ------------------------------
# Workers
# ------------------------------

class AudioLoadWorker(QThread):
    """Background audio analysis to avoid UI freezes.

    PERF CHANGE:
    - Waveform generation uses ffmpeg piping (fast) instead of pydub.
    - Silence detection is DISABLED by default because it can be slow.

    The UI treats a file as "loaded" immediately; this worker only produces
    the lightweight waveform data for drawing.
    """

    progress = Signal(int)
    status = Signal(str)
    finished = Signal(object, float, list)  # wave(np.ndarray), duration_s, silence_points(list[float])
    failed = Signal(str)

    def __init__(self, audio_path: str, ffmpeg_path: Optional[str]):
        super().__init__()
        self.audio_path = audio_path
        self.ffmpeg_path = ffmpeg_path

    def run(self) -> None:
        try:
            self.status.emit("Analyzing waveform (ffmpeg)...")
            self.progress.emit(5)

            wave, dur_s = _waveform_peaks_ffmpeg(
                self.audio_path,
                self.ffmpeg_path,
                target_points=20000,
                sample_rate=8000,
            )

            self.progress.emit(100)

            # Silence detection intentionally disabled (speed / UX).
            silence_points: List[float] = []

            self.finished.emit(wave, float(dur_s), silence_points)
        except Exception as e:
            self.failed.emit(str(e))


class TranscribeWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, audio_path: str, model_size: str, use_gpu: bool = True):
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.use_gpu = use_gpu

    def _detect_device(self) -> str:
        if not self.use_gpu:
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            # ROCm is exposed via CUDA APIs in many torch builds; if present, cuda.is_available() is True.
            return "cpu"
        except Exception:
            return "cpu"

    def run(self) -> None:
        try:
            device = self._detect_device()
            self.status.emit(f"Loading Whisper model ({self.model_size}) on {device}...")

            # Performance tuning:
            # - CPU: int8 quantization is much faster and lower memory
            # - GPU (CUDA/MPS): float16 is typically fastest
            compute_type = "float16" if device in ("cuda", "mps") else "int8"
            try:
                cpu_threads = max(1, (os.cpu_count() or 4) - 1)
            except Exception:
                cpu_threads = 4

            model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=2,
                download_root=models_dir(),
            )

            self.status.emit("Transcribing...")
            segments_iter, info = model.transcribe(
                self.audio_path,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                without_timestamps=False,
                word_timestamps=False,
            )

            segments: List[dict] = []
            dur = float(getattr(info, "duration", 0.0) or 0.0)
            dur = max(1.0, dur)

            for seg in segments_iter:
                segments.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
                pct = int(min(100, (float(seg.end) / dur) * 100))
                self.progress.emit(pct)

            self.progress.emit(100)
            self.finished.emit(segments)
        except Exception as e:
            self.failed.emit(str(e))
# ------------------------------
# Local LLM workers (optional)
# ------------------------------

class LLMChapterTitleWorker(QThread):
    """Generate human-friendly chapter titles using a local LLM (GPT4All)."""
    status = Signal(str)
    finished = Signal(list)  # list[str] titles
    failed = Signal(str)

    def __init__(self, model_name: str, device_mode: str, chapters: List[Chapter], segments: List[dict]):
        super().__init__()
        self.model_name = model_name
        self.device_mode = device_mode  # "Auto" | "CPU" | "GPU"
        self.chapters = chapters
        self.segments = segments

    def _device_string(self) -> str | None:
        m = (self.device_mode or "Auto").strip().lower()
        if m == "cpu":
            return "cpu"
        if m == "gpu":
            return _llm_pick_device(prefer_gpu=True) or "cpu"
        # Auto
        return None

    def _chapter_text(self, ch: Chapter, max_chars: int = 1800) -> str:
        parts: List[str] = []
        for s in self.segments:
            st = float(s.get("start", 0.0))
            if st < float(ch.start) - 1e-6:
                continue
            if st >= float(ch.end) - 1e-6:
                break
            tx = (s.get("text") or "").strip()
            if tx:
                parts.append(tx)
            if sum(len(p) for p in parts) >= max_chars:
                break
        return " ".join(parts).strip()[:max_chars]

    def run(self) -> None:
        try:
            ensure_import("gpt4all", "gpt4all")
            from gpt4all import GPT4All  # type: ignore

            dev = self._device_string()
            self.status.emit("Loading local LLM (first run may download a model)...")
            llm = _get_gpt4all_instance(self.model_name, dev)

            titles: List[str] = []
            for i, ch in enumerate(self.chapters, 1):
                self.status.emit(f"Generating chapter titles... ({i}/{len(self.chapters)})")
                excerpt = self._chapter_text(ch)
                if not excerpt:
                    titles.append(ch.title or "Chapter")
                    continue

                prompt = (
                    "You are generating a podcast chapter title. "
                    "Return ONLY one short title (2-6 words), no quotes, no trailing punctuation. "
                    "Avoid filler words like 'know', 'just', 'like'. Use natural phrasing.\n\n"
                    f"TRANSCRIPT EXCERPT:\n{excerpt}\n\nTITLE:"
                )


                try:
                    out = llm.generate(prompt, max_tokens=24, temp=0.2, top_p=0.9)
                except TypeError:
                    out = llm.generate(prompt)

                title = _sanitize_llm_title(str(out).splitlines()[0] if out else "")
                titles.append(title)

            self.finished.emit(titles)

        except Exception as e:
            self.failed.emit(str(e))


_GPT4ALL_LOCK = threading.Lock()
_GPT4ALL_CACHE = {}  # key=(model_name, device_str) -> GPT4All


def _get_gpt4all_instance(model_name: str, device: str | None):
    """Cache GPT4All instances so we don't trigger multiple downloads/loads."""
    ensure_import("gpt4all", "gpt4all")
    from gpt4all import GPT4All  # type: ignore

    key = (model_name, device or "auto")
    with _GPT4ALL_LOCK:
        inst = _GPT4ALL_CACHE.get(key)
        if inst is not None:
            return inst
        inst = GPT4All(
            model_name,
            model_path=gpt4all_models_dir(),
            allow_download=True,
            device=device,
        )
        _GPT4ALL_CACHE[key] = inst
        return inst


class LLMShowNotesWorker(QThread):
    """Generate show title + summary + bullets with a local LLM (GPT4All)."""
    status = Signal(str)
    finished = Signal(str, str, str)  # title, summary, bullets(markdown)
    failed = Signal(str)

    def __init__(self, model_name: str, device_mode: str, segments: List[dict], chapters: List[Chapter]):
        super().__init__()
        self.model_name = model_name
        self.device_mode = device_mode
        self.segments = segments
        self.chapters = chapters

    def _device_string(self) -> str | None:
        m = (self.device_mode or "Auto").strip().lower()
        if m == "cpu":
            return "cpu"
        if m == "gpu":
            return _llm_pick_device(prefer_gpu=True) or "cpu"
        return None

    def _build_context(self, max_chars: int = 9000) -> str:
        chunks: List[str] = []
        if self.chapters:
            for ch in self.chapters[:12]:
                chunks.append(f"[{seconds_to_timestamp(ch.start)} - {seconds_to_timestamp(ch.end)}] {ch.title}")
                excerpt_parts: List[str] = []
                for s in self.segments:
                    st = float(s.get("start", 0.0))
                    if st < float(ch.start) - 1e-6:
                        continue
                    if st >= float(ch.end) - 1e-6:
                        break
                    tx = (s.get("text") or "").strip()
                    if tx:
                        excerpt_parts.append(tx)
                    if sum(len(p) for p in excerpt_parts) >= 500:
                        break
                if excerpt_parts:
                    chunks.append("Excerpt: " + " ".join(excerpt_parts)[:520])
                if sum(len(x) for x in chunks) >= max_chars:
                    break
        else:
            for s in self.segments:
                if float(s.get("start", 0.0)) > 240.0:
                    break
                tx = (s.get("text") or "").strip()
                if tx:
                    chunks.append(tx)
                if sum(len(x) for x in chunks) >= max_chars:
                    break
        return "\n".join(chunks)[:max_chars]

    def run(self) -> None:
        try:
            ensure_import("gpt4all", "gpt4all")
            from gpt4all import GPT4All  # type: ignore

            dev = self._device_string()
            self.status.emit("Loading local LLM (first run may download a model)...")
            llm = _get_gpt4all_instance(self.model_name, dev)

            ctx = self._build_context()
            self.status.emit("Generating show notes...")

            prompt = (
                "You help podcasters write episode metadata. "
                "Write in a friendly, fun tone, but keep it accurate. "
                "Return EXACTLY in this format (no extra text):\n"
                "TITLE: <one catchy title>\n"
                "SUMMARY: <2-4 sentences>\n"
                "BULLETS:\n- <bullet 1>\n- <bullet 2>\n- <bullet 3>\n"
                "(5-10 bullets total, Markdown)\n\n"
                f"CONTEXT:\n{ctx}\n"
            )

            try:
                out = llm.generate(prompt, max_tokens=450, temp=0.3, top_p=0.9)
            except TypeError:
                out = llm.generate(prompt)

            text = str(out or "").strip()

            title = ""
            summary = ""
            bullets = ""

            m_title = re.search(r"(?im)^TITLE:\s*(.+)$", text)
            if m_title:
                title = m_title.group(1).strip()
            m_sum = re.search(r"(?is)SUMMARY:\s*(.+?)(?:\n\s*BULLETS:|\Z)", text)
            if m_sum:
                summary = m_sum.group(1).strip()
            m_bul = re.search(r"(?is)BULLETS:\s*(.+)$", text)
            if m_bul:
                bullets = m_bul.group(1).strip()

            title = _sanitize_llm_title(title)
            if not summary:
                summary = text[:800].strip()
            if not bullets:
                bullets = "- " + "\n- ".join([x.strip() for x in text.splitlines() if x.strip()][:8])

            self.finished.emit(title, summary, bullets)

        except Exception as e:
            self.failed.emit(str(e))


# ------------------------------
# Main App
# ------------------------------

class ChapterGenerationWorker(QThread):
    """Generate chapters (and optionally LLM titles) off the UI thread.

    This prevents the UI from freezing during:
    - sentence-transformers model download/load ("Better topic detection")
    - GPT4All model download/load (LLM titling)
    """

    status = Signal(str)
    progress = Signal(int)
    finished = Signal(list)  # List[Chapter]
    failed = Signal(str)

    def __init__(
        self,
        segments: List[dict],
        duration_s: float,
        silence_points: List[float],
        target_chapters: int,
        use_embeddings: bool,
        do_llm_titles: bool,
        llm_model_name: str,
        llm_device_mode: str,
    ):
        super().__init__()
        self.segments = segments
        self.duration_s = duration_s
        self.silence_points = silence_points
        self.target_chapters = target_chapters
        self.use_embeddings = use_embeddings
        self.do_llm_titles = do_llm_titles
        self.llm_model_name = llm_model_name
        self.llm_device_mode = llm_device_mode

    def _device_string(self) -> str | None:
        m = (self.llm_device_mode or "Auto").strip().lower()
        if m == "cpu":
            return "cpu"
        if m == "gpu":
            return _llm_pick_device(prefer_gpu=True) or "cpu"
        return None

    def _chapter_text(self, ch: Chapter, max_chars: int = 1800) -> str:
        parts: List[str] = []
        for s in self.segments:
            st = float(s.get("start", 0.0))
            if st < float(ch.start) - 1e-6:
                continue
            if st >= float(ch.end) - 1e-6:
                break
            tx = (s.get("text") or "").strip()
            if tx:
                parts.append(tx)
            if sum(len(p) for p in parts) >= max_chars:
                break
        return " ".join(parts).strip()[:max_chars]

    def run(self) -> None:
        try:
            self.status.emit("Analyzing transcript for chapter boundaries...")
            self.progress.emit(5)

            # Heavy work: topic clustering (may download embeddings model)
            try:
                chs = topic_cluster_chapters(
                    self.segments,
                    duration_s=self.duration_s,
                    silence_points=self.silence_points,
                    target_chapters=int(self.target_chapters),
                    use_embeddings=bool(self.use_embeddings),
                )
            except Exception:
                self.status.emit("Better topic detection failed; using fast fallback...")
                chs = topic_cluster_chapters(
                    self.segments,
                    duration_s=self.duration_s,
                    silence_points=self.silence_points,
                    target_chapters=int(self.target_chapters),
                    use_embeddings=False,
                )

            self.progress.emit(60)

            # Optional: local LLM titles (may download model)
            if self.do_llm_titles and chs:
                self.status.emit("Generating titles with local LLM (first run may download)...")
                dev = self._device_string()
                llm = _get_gpt4all_instance(self.llm_model_name, dev)

                for i, ch in enumerate(chs, 1):
                    excerpt = self._chapter_text(ch)
                    if excerpt:
                        prompt = (
                    "You are generating a podcast chapter title. "
                    "Return ONLY one short title (2-6 words), no quotes, no trailing punctuation. "
                    "Avoid filler words like 'know', 'just', 'like'. Use natural phrasing.\n\n"
                    f"TRANSCRIPT EXCERPT:\n{excerpt}\n\nTITLE:"
                )
                        try:
                            out = llm.generate(prompt, max_tokens=24, temp=0.2, top_p=0.9)
                        except TypeError:
                            out = llm.generate(prompt)
                        ch.title = _sanitize_llm_title(str(out).splitlines()[0] if out else ch.title)

                    # Progress from 60..95
                    self.progress.emit(60 + int((i / max(1, len(chs))) * 35))

            self.progress.emit(100)
            self.finished.emit(chs)
        except Exception as e:
            self.failed.emit(str(e))


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Podcasting 2.0 Chapter Creator")
        self.resize(1100, 820)

        self.audio_path: Optional[str] = None
        self._pending_audio_path: Optional[str] = None
        self.wave: Optional[np.ndarray] = None
        self.duration_s: float = 0.0

        self.segments: List[dict] = []
        self.chapters: List[Chapter] = []
        self.silence_points: List[float] = []
        self.llm_model_name = LOCAL_LLM_DEFAULT_MODELS[0]
        self.llm_device_mode = "Auto"   # Auto | CPU | GPU

        # Transcript navigation helpers (used for playback-follow + seek)
        self._cue_starts: List[float] = []
        self._cue_offsets: List[int] = []
        self._scrub_is_dragging: bool = False

        self._ffmpeg_path: Optional[str] = None
        self._ffplay_path: Optional[str] = None

        # UI
        root = QVBoxLayout()

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # ------------------------------
        # Tab 1 - Audio
        # ------------------------------
        tab_audio = QWidget()
        t1 = QVBoxLayout()

        # Status
        self.status = QLabel("Ready")
        t1.addWidget(self.status)

        # Dependency bootstrap
        dep_row = QHBoxLayout()
        self.btn_deps = QPushButton("Check/Install Dependencies")
        self.btn_deps.clicked.connect(self.bootstrap_dependencies)
        dep_row.addWidget(self.btn_deps)

        self.dep_progress = QProgressBar()
        self.dep_progress.setValue(0)
        dep_row.addWidget(self.dep_progress)
        t1.addLayout(dep_row)

        # Dependency status (quick-glance UX)
        dep_status = QGroupBox("Dependency Status")
        dep_status_layout = QVBoxLayout()

        self.lbl_dep_ffmpeg = QLabel("FFmpeg: unknown")
        self.lbl_dep_whisper = QLabel("Whisper models: unknown")
        self.lbl_dep_embeddings = QLabel("Embeddings model: unknown")
        self.lbl_dep_llm = QLabel("Local LLM: unknown")

        for lab in [self.lbl_dep_ffmpeg, self.lbl_dep_whisper, self.lbl_dep_embeddings, self.lbl_dep_llm]:
            dep_status_layout.addWidget(lab)

        dep_btn_row = QHBoxLayout()
        self.btn_refresh_deps = QPushButton("Refresh Status")
        self.btn_refresh_deps.clicked.connect(self.refresh_dependency_status)
        dep_btn_row.addWidget(self.btn_refresh_deps)

        self.btn_update_catalogs = QPushButton("Update Catalogs")
        self.btn_update_catalogs.setToolTip("Refresh remote catalogs and update cached copies (LLM model list, manifests).")
        self.btn_update_catalogs.clicked.connect(lambda: self.refresh_dependency_status(force_refresh=True))
        dep_btn_row.addWidget(self.btn_update_catalogs)

        dep_btn_row.addStretch(1)
        dep_status_layout.addLayout(dep_btn_row)

        dep_status.setLayout(dep_status_layout)
        t1.addWidget(dep_status)


        llm_box = QGroupBox("Local LLM (Optional - improves chapter titles & show notes)")
        llm_layout = QHBoxLayout()

        self.llm_model_select = QComboBox()
        self.llm_model_select.addItems(LOCAL_LLM_DEFAULT_MODELS)
        self.llm_model_select.currentTextChanged.connect(lambda t: setattr(self, "llm_model_name", t))
        llm_layout.addWidget(QLabel("Model:"))
        llm_layout.addWidget(self.llm_model_select, 1)

        self.btn_llm_refresh = QPushButton("Refresh Model List")
        self.btn_llm_refresh.clicked.connect(self.refresh_llm_model_list)
        llm_layout.addWidget(self.btn_llm_refresh)

        self.llm_device_select = QComboBox()
        self.llm_device_select.addItems(["Auto", "CPU", "GPU"])    
        self.llm_device_select.currentTextChanged.connect(lambda t: setattr(self, "llm_device_mode", t))
        llm_layout.addWidget(QLabel("Device:"))
        llm_layout.addWidget(self.llm_device_select)

        self.chk_llm_titles = QCheckBox("Use LLM titles")
        self.chk_llm_titles.setChecked(True)
        llm_layout.addWidget(self.chk_llm_titles)

        self.chk_llm_notes = QCheckBox("Use LLM notes")
        self.chk_llm_notes.setChecked(True)
        llm_layout.addWidget(self.chk_llm_notes)

        llm_box.setLayout(llm_layout)
        t1.addWidget(llm_box)


        # Load audio
        row = QHBoxLayout()
        self.btn_load = QPushButton("Load Audio")
        self.btn_load.clicked.connect(self.load_audio_file)
        row.addWidget(self.btn_load)

        self.lbl_file = QLabel("No file loaded")
        row.addWidget(self.lbl_file)
        t1.addLayout(row)
        export_all_row = QHBoxLayout()
        self.btn_export_all = QPushButton("Export All (Transcript + Chapters + Tags)")
        self.btn_export_all.setEnabled(False)
        self.btn_export_all.clicked.connect(self.export_all)
        export_all_row.addWidget(self.btn_export_all)
        t1.addLayout(export_all_row)


        # Audio load progress
        self.load_progress = QProgressBar()
        self.load_progress.setRange(0, 100)
        self.load_progress.setValue(0)
        self.load_progress.setVisible(False)
        t1.addWidget(self.load_progress)

        # Basic info
        self.lbl_duration = QLabel("Duration: -")
        t1.addWidget(self.lbl_duration)

        tab_audio.setLayout(t1)
        self.tabs.addTab(tab_audio, "1) Audio")

        # ------------------------------
        # Tab 2 - Transcript
        # ------------------------------
        tab_tx = QWidget()
        t2 = QVBoxLayout()

        # Waveform + playback (for transcript review)
        self.waveform = WaveformWidget()
        self.waveform.scrubRequested.connect(self.scrub)
        self.waveform.markerChanged.connect(self.on_marker_changed)
        t2.addWidget(self.waveform)

        play_row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop)
        play_row.addWidget(self.btn_play)
        play_row.addWidget(self.btn_pause)
        play_row.addWidget(self.btn_stop)

        self.chk_preview_external = QCheckBox("Use external preview (ffplay)")
        self.chk_preview_external.setChecked(False)
        play_row.addWidget(self.chk_preview_external)
        t2.addLayout(play_row)

        scrub_row = QHBoxLayout()
        self.lbl_time = QLabel("00:00:00.000 / 00:00:00.000")
        scrub_row.addWidget(self.lbl_time)

        self.slider_pos = QSlider(Qt.Horizontal)
        self.slider_pos.setRange(0, 1000)
        self.slider_pos.setValue(0)
        self.slider_pos.sliderPressed.connect(lambda: setattr(self, "_scrub_is_dragging", True))
        self.slider_pos.sliderReleased.connect(self._seek_from_slider)
        scrub_row.addWidget(self.slider_pos, 1)

        self.chk_follow = QCheckBox("Follow playback in transcript")
        self.chk_follow.setChecked(False)
        scrub_row.addWidget(self.chk_follow)
        t2.addLayout(scrub_row)

        # Transcription options
        opt_box = QGroupBox("Transcription Options")
        opt_layout = QFormLayout()

        self.model_select = QComboBox()
        self.model_select.addItems(["small", "medium", "large-v3"])
        opt_layout.addRow("Whisper model", self.model_select)

        self.chk_use_gpu = QCheckBox("Use GPU if available")
        self.chk_use_gpu.setChecked(True)
        opt_layout.addRow("Acceleration", self.chk_use_gpu)

        self.chk_diarization = QCheckBox("Enable speaker diarization (optional)")
        self.chk_diarization.setChecked(False)
        opt_layout.addRow("Speakers", self.chk_diarization)

        opt_box.setLayout(opt_layout)
        t2.addWidget(opt_box)

        trans_row = QHBoxLayout()
        self.btn_transcribe = QPushButton("Transcribe")
        self.btn_transcribe.clicked.connect(self.transcribe)
        trans_row.addWidget(self.btn_transcribe)

        self.trans_progress = QProgressBar()
        self.trans_progress.setValue(0)
        trans_row.addWidget(self.trans_progress)
        t2.addLayout(trans_row)

        # Transcript editor
        editor_row = QHBoxLayout()

        self.btn_apply_edits = QPushButton("Apply Transcript Edits")
        self.btn_apply_edits.clicked.connect(self.apply_transcript_edits)
        editor_row.addWidget(self.btn_apply_edits)

        self.btn_seek_cursor = QPushButton("Seek to Cursor")
        self.btn_seek_cursor.clicked.connect(self.seek_to_cursor_cue)
        self.btn_seek_cursor.setEnabled(False)
        editor_row.addWidget(self.btn_seek_cursor)

        self.btn_export_srt = QPushButton("Export SRT")
        self.btn_export_srt.clicked.connect(self.export_srt_clicked)
        self.btn_export_srt.setEnabled(False)
        editor_row.addWidget(self.btn_export_srt)

        self.btn_export_vtt = QPushButton("Export VTT")
        self.btn_export_vtt.clicked.connect(self.export_vtt_clicked)
        self.btn_export_vtt.setEnabled(False)
        editor_row.addWidget(self.btn_export_vtt)

        t2.addLayout(editor_row)

        self.txt_transcript = QTextEdit()
        self.txt_transcript.setPlaceholderText(
            "Transcript will appear here... (editable)\n\n"
            "Prefer WEBVTT cues (recommended). You can paste/edit SRT or VTT here, or use lines like:\n"
            "  [00:00:10.000 --> 00:00:15.000] Hello world\n"
            "  00:00:10.000 --> 00:00:15.000 | Hello world"
        )
        t2.addWidget(self.txt_transcript, 1)


        tab_tx.setLayout(t2)
        self.tabs.addTab(tab_tx, "2) Transcript")

        # ------------------------------
        # Tab 3 - Chapters
        # ------------------------------
        tab_ch = QWidget()
        t3 = QVBoxLayout()

        # Playback bar (chapters tab)
        play_row_ch = QHBoxLayout()
        self.btn_play_ch = QPushButton("Play")
        self.btn_pause_ch = QPushButton("Pause")
        self.btn_stop_ch = QPushButton("Stop")
        self.btn_play_ch.clicked.connect(self.play)
        self.btn_pause_ch.clicked.connect(self.pause)
        self.btn_stop_ch.clicked.connect(self.stop)
        play_row_ch.addWidget(self.btn_play_ch)
        play_row_ch.addWidget(self.btn_pause_ch)
        play_row_ch.addWidget(self.btn_stop_ch)

        self.lbl_time_ch = QLabel("00:00:00.000 / 00:00:00.000")
        play_row_ch.addWidget(self.lbl_time_ch)

        self.slider_pos_ch = QSlider(Qt.Horizontal)
        self.slider_pos_ch.setRange(0, 1000)
        self.slider_pos_ch.setValue(0)
        self._scrub_is_dragging_ch = False
        self.slider_pos_ch.sliderPressed.connect(lambda: setattr(self, "_scrub_is_dragging_ch", True))
        self.slider_pos_ch.sliderReleased.connect(self._seek_from_slider_ch)
        play_row_ch.addWidget(self.slider_pos_ch, 1)

        t3.addLayout(play_row_ch)

        chap_row = QHBoxLayout()
        self.btn_chapters = QPushButton("Generate Chapters")
        self.btn_chapters.clicked.connect(self.generate_chapters)
        chap_row.addWidget(self.btn_chapters)

        self.chk_topic_embeddings = QCheckBox("Better topic detection (embeddings)")
        self.chk_topic_embeddings.setChecked(True)
        self.chk_topic_embeddings.setToolTip(
            "Uses a small local semantic model (auto-downloads once) to improve chapter boundaries and titles.\n"
            "Note: This does NOT require silence detection."
        )

        chap_row.addWidget(self.chk_topic_embeddings)

        self.spn_target_chapters = QSpinBox()
        self.spn_target_chapters.setRange(3, 30)
        self.spn_target_chapters.setValue(8)
        self.spn_target_chapters.setToolTip("How many chapters to aim for (coarse vs detailed)")
        chap_row.addWidget(QLabel("Target:"))
        chap_row.addWidget(self.spn_target_chapters)

        self.btn_export = QPushButton("Export Podcasting 2.0 Chapters JSON")
        self.btn_export.clicked.connect(self.export_chapters)
        chap_row.addWidget(self.btn_export)

        t3.addLayout(chap_row)

        # Chapter generation progress (prevents "Not Responding" UX)
        self.chap_progress = QProgressBar()
        self.chap_progress.setRange(0, 100)
        self.chap_progress.setValue(0)
        self.chap_progress.setVisible(False)
        t3.addWidget(self.chap_progress)

        self.lbl_ch_status = QLabel("")
        t3.addWidget(self.lbl_ch_status)

        self.list_chapters = QListWidget()
        self.list_chapters.currentRowChanged.connect(self.on_chapter_selected)

        editor = QGroupBox("Chapter Editor")
        fl = QFormLayout()

        self.ed_ch_title = QLineEdit()
        self.ed_ch_start = QLineEdit()
        self.ed_ch_end = QLineEdit()
        self.ed_ch_url = QLineEdit()
        self.ed_ch_img = QLineEdit()

        fl.addRow("Title", self.ed_ch_title)
        fl.addRow("Start (HH:MM:SS.mmm)", self.ed_ch_start)
        fl.addRow("End (HH:MM:SS.mmm)", self.ed_ch_end)
        fl.addRow("URL (optional)", self.ed_ch_url)
        fl.addRow("Image (optional)", self.ed_ch_img)

        btns = QHBoxLayout()
        self.btn_ch_add = QPushButton("Add")
        self.btn_ch_update = QPushButton("Update")
        self.btn_ch_delete = QPushButton("Delete")
        self.btn_ch_add.clicked.connect(self.add_chapter)
        self.btn_ch_update.clicked.connect(self.update_chapter)
        self.btn_ch_delete.clicked.connect(self.delete_chapter)
        btns.addWidget(self.btn_ch_add)
        btns.addWidget(self.btn_ch_update)
        btns.addWidget(self.btn_ch_delete)

        self.btn_ch_set_start = QPushButton("Use Playback as Start")
        self.btn_ch_set_end = QPushButton("Use Playback as End")
        self.btn_ch_set_start.clicked.connect(self.set_chapter_start_from_playback)
        self.btn_ch_set_end.clicked.connect(self.set_chapter_end_from_playback)
        btns.addWidget(self.btn_ch_set_start)
        btns.addWidget(self.btn_ch_set_end)

        fl.addRow(btns)

        editor.setLayout(fl)
        t3.addWidget(editor)
        t3.addWidget(self.list_chapters, 1)

        tab_ch.setLayout(t3)
        self.tabs.addTab(tab_ch, "3) Chapters")

        # ------------------------------
        # Tab 4 - ID3 Tags
        # ------------------------------
        tab_id3 = QWidget()
        t4 = QVBoxLayout()

        id3_box = QGroupBox("ID3 Tags (MP3)")
        id3_form = QFormLayout()

        self.id3_artist = QLineEdit()
        self.id3_album = QLineEdit()
        self.id3_title = QLineEdit()
        self.id3_track = QSpinBox()
        self.id3_track.setRange(0, 9999)
        self.id3_year = QSpinBox()
        self.id3_year.setRange(0, 9999)
        self.id3_genre = QLineEdit()

        self.id3_cover_path = QLineEdit()
        self.id3_cover_path.setPlaceholderText("Optional cover art (JPG/PNG)")
        self.btn_cover_browse = QPushButton("Browse")
        self.btn_cover_browse.clicked.connect(self.browse_cover_art)

        cover_row = QHBoxLayout()
        cover_row.addWidget(self.id3_cover_path, 1)
        cover_row.addWidget(self.btn_cover_browse)
        cover_wrap = QWidget()
        cover_wrap.setLayout(cover_row)

        id3_form.addRow("Artist", self.id3_artist)
        id3_form.addRow("Album", self.id3_album)
        id3_form.addRow("Title", self.id3_title)
        id3_form.addRow("Track #", self.id3_track)
        id3_form.addRow("Year", self.id3_year)
        id3_form.addRow("Genre", self.id3_genre)
        id3_form.addRow("Cover Art", cover_wrap)

        id3_box.setLayout(id3_form)
        t4.addWidget(id3_box)

        id3_btn_row = QHBoxLayout()
        self.btn_write_id3 = QPushButton("Write ID3 Tags to Audio")
        self.btn_write_id3.clicked.connect(self.write_id3_tags)
        id3_btn_row.addWidget(self.btn_write_id3)
        t4.addLayout(id3_btn_row)

        self.lbl_id3_hint = QLabel("Tip: This modifies the audio file in-place. Keep a backup if needed.")
        t4.addWidget(self.lbl_id3_hint)

        self.lbl_id3_existing = QLabel("")
        t4.addWidget(self.lbl_id3_existing)
      
        # Cover thumbnail preview
        self.cover_thumb = QLabel("No cover")
        self.cover_thumb.setFixedSize(160, 160)
        self.cover_thumb.setAlignment(Qt.AlignCenter)
        self.cover_thumb.setStyleSheet("border: 1px solid #444; background: #111; color: #ddd;")
        t4.addWidget(self.cover_thumb)

        # Update preview when cover path changes
        self.id3_cover_path.textChanged.connect(self.update_cover_thumbnail)


        tab_id3.setLayout(t4)
        self.tabs.addTab(tab_id3, "4) ID3 Tags")

        # ------------------------------
        # Tab 5 - Show Notes
        # ------------------------------
        tab_notes = QWidget()
        t5 = QVBoxLayout()

        notes_row = QHBoxLayout()
        self.btn_gen_notes = QPushButton("Generate Show Notes (local)")
        self.btn_gen_notes.clicked.connect(self.generate_show_notes)
        notes_row.addWidget(self.btn_gen_notes)

        self.btn_export_notes = QPushButton("Export Notes")
        self.btn_export_notes.clicked.connect(self.export_show_notes)
        notes_row.addWidget(self.btn_export_notes)
        t5.addLayout(notes_row)

        # Markdown helper toolbar (applies to whichever notes editor is focused)
        md_row = QHBoxLayout()
        self.btn_md_bold = QPushButton("Bold")
        self.btn_md_italic = QPushButton("Italic")
        self.btn_md_h1 = QPushButton("H1")
        self.btn_md_h2 = QPushButton("H2")
        self.btn_md_link = QPushButton("Link")
        self.btn_md_bullet = QPushButton("• List")

        self.btn_md_bold.clicked.connect(lambda: self._md_wrap("**", "**"))
        self.btn_md_italic.clicked.connect(lambda: self._md_wrap("*", "*"))
        self.btn_md_h1.clicked.connect(lambda: self._md_prefix_lines("# "))
        self.btn_md_h2.clicked.connect(lambda: self._md_prefix_lines("## "))
        self.btn_md_link.clicked.connect(self._md_link)
        self.btn_md_bullet.clicked.connect(lambda: self._md_prefix_lines("- "))

        for b in [self.btn_md_bold, self.btn_md_italic, self.btn_md_h1, self.btn_md_h2, self.btn_md_link, self.btn_md_bullet]:
            md_row.addWidget(b)
        md_row.addStretch(1)
        t5.addLayout(md_row)

        self.notes_title = QLineEdit()
        self.notes_title.setPlaceholderText("Suggested episode title (editable)")
        t5.addWidget(self.notes_title)

        self.notes_summary = QTextEdit()
        self.notes_summary.setPlaceholderText("Fun summary (editable)")
        t5.addWidget(self.notes_summary, 1)

        self.notes_bullets = QTextEdit()
        self.notes_bullets.setPlaceholderText("Key points / show notes bullets (editable)")
        t5.addWidget(self.notes_bullets, 2)

        tab_notes.setLayout(t5)
        self.tabs.addTab(tab_notes, "5) Show Notes")

        self.setLayout(root)

        # Multimedia
        self.player = None
        self.audio_out = None
        self._external_preview_proc = None

        if QT_MULTIMEDIA_OK:
            try:
                self.player = QMediaPlayer()
                self.audio_out = QAudioOutput()
                self.player.setAudioOutput(self.audio_out)
                self.player.durationChanged.connect(self._on_player_duration_changed)
                self.player.positionChanged.connect(self._on_player_position_changed)

            except Exception:
                self.player = None

        # Bootstrap dependencies on first launch (non-blocking expectations)
        self.bootstrap_dependencies(silent=True)
        try:
            self.refresh_dependency_status(force_refresh=False)
        except Exception:
            pass


    # ------------------------------
    # Dependency bootstrap
    # ------------------------------
    def bootstrap_dependencies(self, silent: bool = False) -> None:
        """Ensure critical runtime dependencies exist (FFmpeg, catalogs).

        Runs quickly and safely:
        - FFmpeg download/extract if missing
        - Optional refresh of small remote catalogs (non-fatal)
        - Updates dependency status panel
        """
        self.dep_progress.setValue(0)

        def setp(p: int) -> None:
            try:
                self.dep_progress.setValue(int(p))
            except Exception:
                pass

        # FFmpeg is required for pydub decoding and optional external preview
        self.status.setText("Checking FFmpeg...")
        ff = ensure_ffmpeg(progress_cb=setp)
        self._ffmpeg_path = ff
        self._ffplay_path = None

        if ff:
            # Try to locate ffplay next to ffmpeg if available
            base = os.path.dirname(ff)
            cand = os.path.join(base, "ffplay.exe" if platform.system() == "Windows" else "ffplay")
            if os.path.exists(cand):
                self._ffplay_path = cand
            setp(100)
            self.status.setText("Dependencies ready")
        else:
            msg = "FFmpeg could not be installed automatically. Audio decoding may fail."
            self.status.setText(msg)
            if not silent:
                QMessageBox.warning(self, "Dependency Issue", msg)

        # Best-effort: refresh GPT4All catalog so filenames stay valid (non-fatal).
        try:
            self.refresh_llm_model_list(silent=True, force_refresh=False)
        except Exception:
            pass

        # Update dependency status labels
        try:
            self.refresh_dependency_status(force_refresh=False)
        except Exception:
            pass


    # ------------------------------
    # LLM model catalog
    # ------------------------------

    def refresh_llm_model_list(self, silent: bool = False, force_refresh: bool = False) -> None:
        """Fetch GPT4All's model catalog and populate the dropdown with *small* models.

        UX goal: only show options that can run on an average computer.
        - Prefer <= ~4GB Q4 models
        - Prefer 1B-4B families
        - Keep list short and friendly

        If offline or catalog parsing fails, we keep the built-in defaults.
        """
        try:
            self.status.setText("Refreshing LLM model list...")
            cat = _download_json_cached(DEFAULT_GPT4ALL_CATALOG_URL, timeout=12, force_refresh=force_refresh)

            # Collect candidate models from catalog
            items: List[str] = []
            meta: dict[str, dict] = {}

            if isinstance(cat, list):
                for it in cat:
                    if not isinstance(it, dict):
                        continue
                    fn = (it.get("filename") or it.get("file") or "").strip()
                    if not fn:
                        u = (it.get("url") or "").strip()
                        if u:
                            fn = os.path.basename(u)
                    if not (fn and fn.lower().endswith(".gguf")):
                        continue

                    items.append(fn)
                    meta[fn] = it

            # De-dupe preserve order
            seen = set()
            clean: List[str] = []
            for x in items:
                if x in seen:
                    continue
                seen.add(x)
                clean.append(x)

            def _looks_small_enough(name: str, it: dict) -> bool:
                n = (name or "").lower()
                # Heuristic: allow curated small families
                allow_families = (
                    "llama-3.2-1b" in n or
                    "llama-3.2-3b" in n or
                    "phi-3-mini" in n or
                    "phi-3" in n and "mini" in n or
                    "gemma-2-2b" in n or
                    "qwen2.5-1.5b" in n or
                    "qwen2.5-3b" in n
                )
                if not allow_families:
                    return False
                # Prefer Q4-ish quantizations for CPU
                if "q4" not in n and "q4_0" not in n and "q4_k" not in n:
                    return False
                # If size is present in metadata, enforce <= 4.5GB
                sz = it.get("filesize") or it.get("size") or it.get("file_size")
                try:
                    # Some catalogs store bytes
                    if isinstance(sz, (int, float)):
                        if float(sz) > 4.5 * 1024 * 1024 * 1024:
                            return False
                except Exception:
                    pass
                return True

            # Filter to small-friendly list
            filtered: List[str] = []
            for fn in clean:
                it = meta.get(fn, {}) if isinstance(meta.get(fn, {}), dict) else {}
                if _looks_small_enough(fn, it):
                    filtered.append(fn)

            # Ensure defaults always appear (even if catalog changes/offline)
            for m in LOCAL_LLM_DEFAULT_MODELS:
                if m not in filtered:
                    filtered.insert(0, m)

            # Final short list: keep it compact
            # Order: defaults first, then a few additional small models
            final: List[str] = []
            for m in filtered:
                if m not in final:
                    final.append(m)
                if len(final) >= 12:
                    break

            cur = self.llm_model_select.currentText().strip() if hasattr(self, "llm_model_select") else ""
            self.llm_model_select.blockSignals(True)
            self.llm_model_select.clear()
            self.llm_model_select.addItems(final)
            self.llm_model_select.blockSignals(False)

            # Restore selection if possible
            if cur and cur in final:
                self.llm_model_select.setCurrentText(cur)
            else:
                self.llm_model_select.setCurrentIndex(0)
            self.llm_model_name = self.llm_model_select.currentText()

            self.status.setText("LLM model list updated")

        except Exception as e:
            if not silent:
                QMessageBox.information(
                    self,
                    "LLM Models",
                    "Could not refresh model list (offline?). Keeping defaults.\n\nDetails: " + str(e)

                )
            self.status.setText("LLM model list unchanged")
        
    # ------------------------------
    # Dependency Status UX
    # ------------------------------

    def refresh_dependency_status(self, force_refresh: bool = False) -> None:
        """Update dependency status labels.

        Fast and safe:
        - never blocks on large downloads
        - uses cached catalogs unless force_refresh=True
        """
        # Optionally refresh catalogs (small network request)
        if force_refresh:
            try:
                self.refresh_llm_model_list(silent=True, force_refresh=True)
            except Exception:
                pass

        # FFmpeg
        ff = self._ffmpeg_path or shutil.which("ffmpeg")
        if ff and os.path.exists(ff):
            self.lbl_dep_ffmpeg.setText(f"FFmpeg: ready ({os.path.basename(ff)})")
        else:
            self.lbl_dep_ffmpeg.setText("FFmpeg: missing (click Check/Install)")

        # Whisper: show whether the selected model looks cached
        try:
            sel = self.model_select.currentText() if hasattr(self, "model_select") else "small"
            found = False
            root = models_dir()
            for r, ds, _ in os.walk(root):
                for d in ds:
                    dl = d.lower()
                    if "whisper" in dl and sel.lower() in dl:
                        found = True
                        break
                if found:
                    break
            if found:
                self.lbl_dep_whisper.setText(f"Whisper models: cached ({sel})")
            else:
                self.lbl_dep_whisper.setText(f"Whisper models: will download on first use ({sel})")
        except Exception:
            self.lbl_dep_whisper.setText("Whisper models: unknown")

        # Embeddings model cache
        try:
            st_cache = os.path.join(models_dir(), "sentence_transformers")
            ok = os.path.exists(st_cache) and any(os.scandir(st_cache))
            self.lbl_dep_embeddings.setText(
                "Embeddings model: cached" if ok else "Embeddings model: will download on first use"
            )
        except Exception:
            self.lbl_dep_embeddings.setText("Embeddings model: unknown")

        # Local LLM file presence
        try:
            mname = self.llm_model_select.currentText().strip() if hasattr(self, "llm_model_select") else ""
            if mname:
                mp = os.path.join(gpt4all_models_dir(), mname)
                self.lbl_dep_llm.setText(
                    f"Local LLM: installed ({mname})" if os.path.exists(mp)
                    else f"Local LLM: will download on first use ({mname})"
                )
            else:
                self.lbl_dep_llm.setText("Local LLM: optional")
        except Exception:
            self.lbl_dep_llm.setText("Local LLM: unknown")

    # ------------------------------
    # File loading
    # ------------------------------

    def load_audio_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.opus *.m4a *.aac)"
        )
        if not path:
            return

        if not self._ffmpeg_path and not shutil.which("ffmpeg"):
            self.bootstrap_dependencies(silent=False)

        # Treat file selection as "loaded" immediately (like normal players).
        # Waveform/silence analysis continues in the background.
        self.audio_path = path
        self._pending_audio_path = path
        self.lbl_file.setText(os.path.basename(path))
        self.status.setText("Loaded audio (analyzing waveform in background...)")

        # Populate ID3 tab (fast; non-fatal if it fails)
        self.load_id3_tags_from_file()

        # Enable core actions immediately
        self.btn_transcribe.setEnabled(True)
        self.btn_chapters.setEnabled(True)
        self.btn_export.setEnabled(True)
        if hasattr(self, "btn_export_all"):
            self.btn_export_all.setEnabled(True)

        # Load into Qt player immediately for instant playback
        if self.player is not None:
            try:
                self.player.setSource(QUrl.fromLocalFile(self.audio_path))
            except Exception:
                pass

        # Start background analysis (non-blocking)
        self.load_progress.setVisible(True)
        self.load_progress.setValue(0)

        self.audio_worker = AudioLoadWorker(path, self._ffmpeg_path)
        self.audio_worker.progress.connect(self.load_progress.setValue)
        self.audio_worker.status.connect(self.status.setText)
        self.audio_worker.finished.connect(self.on_audio_loaded)
        self.audio_worker.failed.connect(self.on_audio_load_failed)
        self.audio_worker.start()

    def on_audio_loaded(self, wave, dur_s: float, silence_points: List[float]) -> None:
        try:
            # Keep whatever audio_path was set at selection time.
            self.wave = wave
            self.duration_s = float(dur_s)
            self.silence_points = list(silence_points)

            self.waveform.set_data(wave, self.silence_points)
            self.waveform.set_markers([])

            self.lbl_duration.setText(f"Duration: {seconds_to_timestamp(self.duration_s)}")
            # Set a sensible default target chapters based on duration (~1 per 6 min)
            try:
                rec = int(round(self.duration_s / 360.0))
                rec = max(5, min(12, rec))
                if hasattr(self, "spn_target_chapters"):
                    self.spn_target_chapters.setValue(rec)
            except Exception:
                pass
            self.status.setText("Loaded")
        finally:
            self.load_progress.setVisible(False)

    def on_audio_load_failed(self, err: str) -> None:
        self.load_progress.setVisible(False)
        self.btn_load.setEnabled(True)
        self.btn_transcribe.setEnabled(True)
        self.btn_chapters.setEnabled(True)
        self.btn_export.setEnabled(True)
        if hasattr(self, "btn_export_all"):
            self.btn_export_all.setEnabled(True)

        QMessageBox.critical(self, "Load Error", err)
        self.status.setText("Load failed")

    # ------------------------------
    # Playback
    # ------------------------------

    def _ensure_external_preview(self) -> Optional[str]:
        # Prefer bundled/system ffplay if Qt multimedia isn't working.
        if self._ffplay_path and os.path.exists(self._ffplay_path):
            return self._ffplay_path
        sys_ffplay = shutil.which("ffplay")
        if sys_ffplay:
            return sys_ffplay
        return None

    def play(self) -> None:
        if not self.audio_path:
            return

        if self.chk_preview_external.isChecked() or self.player is None:
            ffplay = self._ensure_external_preview()
            if not ffplay:
                QMessageBox.warning(self, "Preview", "Audio preview unavailable (ffplay not found).")
                return

            # Launch external preview
            try:
                self._external_preview_proc = subprocess.Popen(
                    [ffplay, "-nodisp", "-autoexit", self.audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                QMessageBox.warning(self, "Preview", f"Failed to start external preview: {e}")
            return

        try:
            self.player.play()
        except Exception as e:
            QMessageBox.warning(self, "Preview", f"Qt preview failed; enable external preview. ({e})")

    def pause(self) -> None:
        if self.player is not None:
            try:
                self.player.pause()
            except Exception:
                pass

    def stop(self) -> None:
        if self.player is not None:
            try:
                self.player.stop()
            except Exception:
                pass

        if self._external_preview_proc is not None:
            try:
                self._external_preview_proc.terminate()
            except Exception:
                pass
            self._external_preview_proc = None

    def scrub(self, frac: float) -> None:
        if self.player is not None and self.duration_s > 0:
            try:
                self.player.setPosition(int(frac * self.duration_s * 1000))
            except Exception:
                pass

    def _on_player_duration_changed(self, ms: int) -> None:
        if ms <= 0:
            return
        self.duration_s = max(self.duration_s, ms / 1000.0)
        self._update_time_label(self.player.position() if self.player else 0, ms)

    def _on_player_position_changed(self, ms: int) -> None:
        if self.player is None:
            return
        dur_ms = max(1, self.player.duration())

        if not self._scrub_is_dragging:
            self.slider_pos.setValue(int((ms / dur_ms) * 1000))
        if hasattr(self, "slider_pos_ch") and (not getattr(self, "_scrub_is_dragging_ch", False)):
            try:
                self.slider_pos_ch.setValue(int((ms / dur_ms) * 1000))
            except Exception:
                pass

        self._update_time_label(ms, dur_ms)

        # Update chapters-tab time label too
        if hasattr(self, "lbl_time_ch"):
            try:
                pos_s = max(0.0, ms / 1000.0)
                dur_s = max(0.0, dur_ms / 1000.0)
                self.lbl_time_ch.setText(f"{seconds_to_timestamp(pos_s)} / {seconds_to_timestamp(dur_s)}")
            except Exception:
                pass

        if self.chk_follow.isChecked() and self._cue_starts:
            self.scroll_transcript_to_time(ms / 1000.0)

    def _seek_from_slider(self) -> None:
        if self.player is None:
            return
        self._scrub_is_dragging = False
        dur_ms = max(1, self.player.duration())
        frac = self.slider_pos.value() / 1000.0
        self.player.setPosition(int(frac * dur_ms))

    def _seek_from_slider_ch(self) -> None:
        if self.player is None:
            return
        self._scrub_is_dragging_ch = False
        dur_ms = max(1, self.player.duration())
        frac = self.slider_pos_ch.value() / 1000.0
        self.player.setPosition(int(frac * dur_ms))

    def _update_time_label(self, pos_ms: int, dur_ms: int) -> None:
        pos_s = max(0.0, pos_ms / 1000.0)
        dur_s = max(0.0, dur_ms / 1000.0)
        self.lbl_time.setText(f"{seconds_to_timestamp(pos_s)} / {seconds_to_timestamp(dur_s)}")

    def scroll_transcript_to_time(self, t_s: float) -> None:
        if not self._cue_starts or not self._cue_offsets:
            return
        import bisect
        idx = bisect.bisect_right(self._cue_starts, float(t_s)) - 1
        idx = max(0, min(idx, len(self._cue_offsets) - 1))

        cur = self.txt_transcript.textCursor()
        cur.setPosition(int(self._cue_offsets[idx]))
        self.txt_transcript.setTextCursor(cur)
        self.txt_transcript.centerCursor()

    # ------------------------------
    # Transcription
    # ------------------------------

    def transcribe(self) -> None:
        if not self.audio_path:
            QMessageBox.information(self, "Transcribe", "Load an audio file first.")
            return

        self.txt_transcript.clear()
        self.trans_progress.setValue(0)

        model_size = self.model_select.currentText()
        use_gpu = self.chk_use_gpu.isChecked()

        self.worker = TranscribeWorker(self.audio_path, model_size, use_gpu=use_gpu)
        self.worker.progress.connect(self.trans_progress.setValue)
        self.worker.status.connect(self.status.setText)
        self.worker.finished.connect(self.on_transcribed)
        self.worker.failed.connect(self.on_transcribe_failed)
        self.worker.start()

    def on_transcribed(self, segments: List[dict]) -> None:
        self.segments = segments

        # Enable exports now that we have timestamps.
        self.btn_export_srt.setEnabled(True)
        self.btn_export_vtt.setEnabled(True)
        self.btn_seek_cursor.setEnabled(True)

        # Build VTT text and remember cue offsets so we can scroll to the right cue during playback.
        parts: List[str] = []
        self._cue_starts = []
        self._cue_offsets = []

        header = "WEBVTT\n\n"
        parts.append(header)
        pos = len(header)

        for s in segments:
            self._cue_starts.append(float(s["start"]))
            self._cue_offsets.append(pos)

            cue = (
                f"{_format_vtt_time(s['start'])} --> {_format_vtt_time(s['end'])}\n"
                f"{(s.get('text') or '').strip()}\n\n"
            )
            parts.append(cue)
            pos += len(cue)

        self.txt_transcript.setPlainText("".join(parts))
        self.status.setText("Transcription complete")

    def on_transcribe_failed(self, err: str) -> None:
        QMessageBox.critical(self, "Transcription Error", err)
        self.status.setText("Transcription failed")

    # ------------------------------
    # Transcript editing + export
    # ------------------------------

    def apply_transcript_edits(self) -> None:
        """Update internal segments from editor contents (best-effort)."""
        raw = self.txt_transcript.toPlainText()
        try:
            parsed = parse_transcript_editor(raw)
            if not parsed:
                QMessageBox.information(
                    self,
                    "Transcript",
                    "No parseable cues found.\n\n"
                    "Tip: Paste SRT/VTT, or use: [00:00:10.000 --> 00:00:15.000] Text",
                )
                return

            parsed.sort(key=lambda s: s["start"])

            for i in range(len(parsed) - 1):
                parsed[i]["end"] = max(parsed[i]["end"], parsed[i + 1]["start"])

            self.segments = parsed
            self.status.setText(f"Applied transcript edits ({len(self.segments)} cues)")
            self.btn_export_srt.setEnabled(True)
            self.btn_export_vtt.setEnabled(True)
        except Exception as e:
            QMessageBox.warning(self, "Transcript", f"Could not apply edits: {e}")

    def seek_to_cursor_cue(self) -> None:
        if self.player is None:
            return

        cursor = self.txt_transcript.textCursor()
        pos = cursor.position()
        text_up_to = self.txt_transcript.toPlainText()[:pos]

        last = text_up_to.rfind("-->")
        if last == -1:
            return

        line_start = text_up_to.rfind("\n", 0, last)
        if line_start == -1:
            line_start = 0
        line = text_up_to[line_start:last].strip()

        try:
            start_tc = line.split("-->")[0].strip()
            t = _parse_timecode(start_tc)
            self.player.setPosition(int(t * 1000))
        except Exception:
            return


    def export_srt_clicked(self) -> None:
        if not self.segments:
            QMessageBox.information(self, "Export", "No transcript available yet.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save SRT", "", "SubRip (*.srt)")
        if not path:
            return

        try:
            export_srt(self.segments, path)
            QMessageBox.information(self, "Export", "SRT saved.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_vtt_clicked(self) -> None:

        if not self.segments:
            QMessageBox.information(self, "Export", "No transcript available yet.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save VTT", "", "WebVTT (*.vtt)")
        if not path:
            return

        try:
            export_vtt(self.segments, path)
            QMessageBox.information(self, "Export", "VTT saved.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
    def export_all(self) -> None:
        """Export transcript + chapters JSON + optionally write ID3 tags."""
        if not self.audio_path:
            QMessageBox.information(self, "Export All", "Load an audio file first.")
            return

        # Sync segments from editor if possible (silent)
        try:
            raw = self.txt_transcript.toPlainText()
            parsed = parse_transcript_editor(raw)
            if parsed:
                parsed.sort(key=lambda s: s["start"])
                self.segments = parsed
        except Exception:
            pass

        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not out_dir:
            return

        stem = os.path.splitext(os.path.basename(self.audio_path))[0]

        fmt, ok = QInputDialog.getItem(
            self,
            "Transcript Export",
            "Export transcript as:",
            ["VTT", "SRT", "Both"],
            0,
            False,
        )
        if not ok:
            return

        errors: List[str] = []

        # Transcript export
        try:
            if self.segments:
                if fmt in ("VTT", "Both"):
                    export_vtt(self.segments, os.path.join(out_dir, f"{stem}.vtt"))
                if fmt in ("SRT", "Both"):
                    export_srt(self.segments, os.path.join(out_dir, f"{stem}.srt"))
            else:
                errors.append("Transcript: no segments available (transcribe first).")
        except Exception as e:
            errors.append(f"Transcript export failed: {e}")

        # Chapters export
        try:
            if self.chapters:
                export_podcast20_json(
                    self.chapters,
                    os.path.join(out_dir, f"{stem}.chapters.json"),
                    author=(self.id3_artist.text().strip() if hasattr(self, "id3_artist") else ""),
                    title=(self.notes_title.text().strip() if hasattr(self, "notes_title") else "")
                    or (self.id3_title.text().strip() if hasattr(self, "id3_title") else ""),
                    podcastName=(self.id3_album.text().strip() if hasattr(self, "id3_album") else ""),
                    description=(self.notes_summary.toPlainText().strip() if hasattr(self, "notes_summary") else ""),
                    fileName=os.path.basename(self.audio_path) if self.audio_path else "",
                )
            else:
                errors.append("Chapters: none available (generate chapters first).")
        except Exception as e:
            errors.append(f"Chapters export failed: {e}")

        # ID3 write (optional)
        try:
            if self.audio_path.lower().endswith(".mp3"):
                resp = QMessageBox.question(
                    self,
                    "Write ID3 Tags?",
                    "Also write the current ID3 tag fields into the MP3 file?\n\n"
                    "This modifies the audio file in-place.",
                    QMessageBox.Yes | QMessageBox.No,
                )

                if resp == QMessageBox.Yes:
                    self._write_id3_tags_silent()
        except Exception as e:
            errors.append(f"ID3 write failed: {e}")

        if errors:
            QMessageBox.warning(self, "Export All", "Completed with issues:\n\n- " + "\n- ".join(errors))

        else:
            QMessageBox.information(self, "Export All", "Export complete.")



    # ------------------------------
    # Chapters
    # ------------------------------

    def generate_chapters(self) -> None:
        if not self.segments:
            QMessageBox.information(self, "Chapters", "Transcribe first.")
            return

        # Prevent re-entry / repeated downloads
        if hasattr(self, "chap_worker") and getattr(self, "chap_worker", None) is not None:
            try:
                if self.chap_worker.isRunning():
                    QMessageBox.information(self, "Chapters", "Chapter generation is already running.")
                    return
            except Exception:
                pass

        use_emb = self.chk_topic_embeddings.isChecked()
        target = int(self.spn_target_chapters.value()) if hasattr(self, "spn_target_chapters") else 10
        do_llm = bool(getattr(self, "chk_llm_titles", None) and self.chk_llm_titles.isChecked())

        self.btn_chapters.setEnabled(False)
        self.chap_progress.setVisible(True)
        self.chap_progress.setValue(0)
        self.lbl_ch_status.setText("Starting...")
        self.status.setText("Generating chapters...")

        self.chap_worker = ChapterGenerationWorker(
            segments=list(self.segments),
            duration_s=float(self.duration_s),
            silence_points=list(self.silence_points),
            target_chapters=target,
            use_embeddings=use_emb,
            do_llm_titles=do_llm,
            llm_model_name=str(self.llm_model_name),
            llm_device_mode=str(self.llm_device_mode),
        )

        self.chap_worker.status.connect(self.lbl_ch_status.setText)
        self.chap_worker.progress.connect(self.chap_progress.setValue)

        def _done(chs: list):
            self.chapters = chs
            self._refresh_chapter_view()
            self.status.setText("Chapters generated")
            self.lbl_ch_status.setText("Done")
            self.chap_progress.setVisible(False)
            self.btn_chapters.setEnabled(True)

        def _fail(err: str):
            self.chap_progress.setVisible(False)
            self.btn_chapters.setEnabled(True)
            self.status.setText("Chapter generation failed")
            QMessageBox.critical(self, "Chapters", "Chapter generation failed:\n\n" + str(err))

        self.chap_worker.finished.connect(_done)
        self.chap_worker.failed.connect(_fail)
        self.chap_worker.start()

    def on_marker_changed(self, idx: int, frac: float) -> None:
        # Update chapter start times based on marker dragging
        if not self.chapters or self.duration_s <= 0:
            return

        if idx < 0 or idx >= len(self.chapters):
            return

        new_start = frac * self.duration_s
        self.chapters[idx].start = new_start

        # Maintain ordering; also adjust end times
        self.chapters.sort(key=lambda c: c.start)
        for i in range(len(self.chapters) - 1):
            self.chapters[i].end = max(self.chapters[i].start, self.chapters[i + 1].start)
        self.chapters[-1].end = max(self.chapters[-1].start, float(self.segments[-1]["end"]))

        self._refresh_chapter_view()


    def _refresh_chapter_view(self) -> None:
        self.list_chapters.clear()
        markers: List[float] = []

        for ch in self.chapters:
            self.list_chapters.addItem(
                QListWidgetItem(
                    f"{seconds_to_timestamp(ch.start)} - {seconds_to_timestamp(ch.end)}  |  {ch.title}"
                )
            )
            if self.duration_s > 0:
                markers.append(max(0.0, min(1.0, ch.start / self.duration_s)))

        self.waveform.set_markers(markers)

    def export_chapters(self) -> None:
        if not self.chapters:
            QMessageBox.information(self, "Export", "Generate chapters first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Chapters JSON", "", "JSON (*.json)")
        if not path:
            return

        try:
            export_podcast20_json(
                self.chapters,
                path,
                author=(self.id3_artist.text().strip() if hasattr(self, "id3_artist") else ""),
                title=(self.notes_title.text().strip() if hasattr(self, "notes_title") else "")
                or (self.id3_title.text().strip() if hasattr(self, "id3_title") else ""),
                podcastName=(self.id3_album.text().strip() if hasattr(self, "id3_album") else ""),
                description=(self.notes_summary.toPlainText().strip() if hasattr(self, "notes_summary") else ""),
                fileName=os.path.basename(self.audio_path) if self.audio_path else "",
            )
            QMessageBox.information(self, "Export", "Saved.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def on_chapter_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.chapters):
            return
        ch = self.chapters[row]
        self.ed_ch_title.setText(ch.title)
        self.ed_ch_start.setText(seconds_to_timestamp(ch.start))
        self.ed_ch_end.setText(seconds_to_timestamp(ch.end))
        self.ed_ch_url.setText(ch.url or "")
        self.ed_ch_img.setText(ch.img or "")

    def add_chapter(self) -> None:
        if self.duration_s <= 0:
            QMessageBox.information(self, "Chapters", "Load audio first.")
            return
        try:
            st = _parse_timecode(self.ed_ch_start.text())
            en = _parse_timecode(self.ed_ch_end.text()) if self.ed_ch_end.text().strip() else st + 60.0
            if en <= st:
                en = st + 30.0
            ch = Chapter(
                title=self.ed_ch_title.text().strip() or f"Chapter {len(self.chapters)+1}",
                start=max(0.0, st),
                end=min(self.duration_s, en),
                url=self.ed_ch_url.text().strip(),
                img=self.ed_ch_img.text().strip(),
            )
            self.chapters.append(ch)
            self.chapters.sort(key=lambda c: c.start)
            # re-derive end times from start order if needed
            for i in range(len(self.chapters) - 1):
                self.chapters[i].end = max(self.chapters[i].start, self.chapters[i + 1].start)
            self.chapters[-1].end = max(self.chapters[-1].start, self.duration_s)
            self._refresh_chapter_view()
        except Exception as e:
            QMessageBox.warning(self, "Chapters", f"Could not add chapter: {e}")

    def update_chapter(self) -> None:
        row = self.list_chapters.currentRow()
        if row < 0 or row >= len(self.chapters):
            return
        try:
            st = _parse_timecode(self.ed_ch_start.text())
            en = _parse_timecode(self.ed_ch_end.text()) if self.ed_ch_end.text().strip() else st + 60.0
            if en <= st:
                en = st + 30.0
            ch = self.chapters[row]
            ch.title = self.ed_ch_title.text().strip() or ch.title
            ch.start = max(0.0, st)
            ch.end = min(self.duration_s, en)
            ch.url = self.ed_ch_url.text().strip()
            ch.img = self.ed_ch_img.text().strip()

            self.chapters.sort(key=lambda c: c.start)
            for i in range(len(self.chapters) - 1):
                self.chapters[i].end = max(self.chapters[i].start, self.chapters[i + 1].start)
            self.chapters[-1].end = max(self.chapters[-1].start, self.duration_s)

            self._refresh_chapter_view()
        except Exception as e:
            QMessageBox.warning(self, "Chapters", f"Could not update chapter: {e}")

    def delete_chapter(self) -> None:
        row = self.list_chapters.currentRow()
        if row < 0 or row >= len(self.chapters):
            return
        del self.chapters[row]
        if self.chapters:
            for i in range(len(self.chapters) - 1):
                self.chapters[i].end = max(self.chapters[i].start, self.chapters[i + 1].start)
            self.chapters[-1].end = max(self.chapters[-1].start, self.duration_s)
        self._refresh_chapter_view()

    def _current_playback_time_s(self) -> Optional[float]:
        if self.player is None:
            return None
        try:
            return max(0.0, float(self.player.position()) / 1000.0)
        except Exception:
            return None

    def _snap_time_to_silence(self, t: float, window_s: float = 1.5) -> float:
        # Snap to nearest detected silence boundary if close enough.
        # Silence detection is currently disabled for performance.
        # This remains as a safe no-op when no silence points exist.
        if not self.silence_points or self.duration_s <= 0:
            return t
        silence_s = [max(0.0, min(self.duration_s, float(p) * self.duration_s)) for p in self.silence_points]
        nearest = min(silence_s, key=lambda x: abs(x - t))
        if abs(nearest - t) <= window_s:
            return nearest
        return t

    def set_chapter_start_from_playback(self) -> None:
        t = self._current_playback_time_s()
        if t is None:
            QMessageBox.information(self, "Chapters", "Playback position is unavailable in external preview mode.")
            return
        t = self._snap_time_to_silence(t)
        self.ed_ch_start.setText(seconds_to_timestamp(t))

    def set_chapter_end_from_playback(self) -> None:
        t = self._current_playback_time_s()
        if t is None:
            QMessageBox.information(self, "Chapters", "Playback position is unavailable in external preview mode.")
            return
        t = self._snap_time_to_silence(t)
        self.ed_ch_end.setText(seconds_to_timestamp(t))


    # ------------------------------
    # ID3 Tags
    # ------------------------------

    def browse_cover_art(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Cover Art",
            "",
            "Images (*.jpg *.jpeg *.png *.webp)"
        )
        if path:
            self.id3_cover_path.setText(path)
            self.update_cover_thumbnail(path)


    def update_cover_thumbnail(self, path: str = "") -> None:
        """Render a small cover-art preview in the ID3 tab."""
        if not hasattr(self, "cover_thumb"):
            return

        p = (path or self.id3_cover_path.text() or "").strip()
        if not p or not os.path.exists(p):
            self.cover_thumb.setPixmap(QPixmap())
            self.cover_thumb.setText("No cover")
            return

        pm = QPixmap(p)
        if pm.isNull():
            self.cover_thumb.setPixmap(QPixmap())
            self.cover_thumb.setText("Unsupported image")
            return

        scaled = pm.scaled(self.cover_thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cover_thumb.setText("")
        self.cover_thumb.setPixmap(scaled)

    def load_id3_tags_from_file(self) -> None:
        """If the loaded file is an MP3, read existing ID3 tags and display them in the UI."""
        # Clear fields first
        try:
            self.id3_artist.setText("")
            self.id3_album.setText("")
            self.id3_title.setText("")
            self.id3_track.setValue(0)
            self.id3_year.setValue(0)
            self.id3_genre.setText("")
            self.id3_cover_path.setText("")
            if hasattr(self, "lbl_id3_existing"):
                self.lbl_id3_existing.setText("")
        except Exception:
            pass

        if not self.audio_path or not self.audio_path.lower().endswith(".mp3"):
            self.update_cover_thumbnail("")
            return

        try:
            ensure_import("mutagen", "mutagen")
            from mutagen.id3 import ID3

            tags = ID3(self.audio_path)

            def _get_text(frame_id: str) -> str:
                try:
                    fr = tags.get(frame_id)
                    if not fr:
                        return ""
                    tx = getattr(fr, "text", None)
                    if isinstance(tx, (list, tuple)) and tx:
                        return str(tx[0])
                    return str(fr)
                except Exception:
                    return ""

            self.id3_artist.setText(_get_text("TPE1"))
            self.id3_album.setText(_get_text("TALB"))
            self.id3_title.setText(_get_text("TIT2"))

            tr = _get_text("TRCK")
            try:
                self.id3_track.setValue(int(str(tr).split("/")[0]))
            except Exception:
                self.id3_track.setValue(0)

            yr = _get_text("TDRC")
            try:
                m = re.findall(r"[0-9]{4}", str(yr))
                self.id3_year.setValue(int(m[0]) if m else 0)
            except Exception:
                self.id3_year.setValue(0)

            self.id3_genre.setText(_get_text("TCON"))

            # Cover art: if embedded, extract to app data so user can replace/keep it.
            apic = tags.getall("APIC")
            if apic:
                try:
                    covers_dir = os.path.join(app_data_dir(), "covers")
                    os.makedirs(covers_dir, exist_ok=True)
                    ext = ".png" if "png" in (getattr(apic[0], "mime", "") or "").lower() else ".jpg"
                    outp = os.path.join(
                        covers_dir,
                        os.path.splitext(os.path.basename(self.audio_path))[0] + "_cover" + ext,
                    )
                    with open(outp, "wb") as f:
                        f.write(apic[0].data)
                    self.id3_cover_path.setText(outp)
                    self.update_cover_thumbnail(outp)

                    if hasattr(self, "lbl_id3_existing"):
                        self.lbl_id3_existing.setText("Existing ID3 tags loaded (including embedded cover art).")
                except Exception:
                    if hasattr(self, "lbl_id3_existing"):
                        self.lbl_id3_existing.setText("Existing ID3 tags loaded.")
            else:
                if hasattr(self, "lbl_id3_existing"):
                    self.lbl_id3_existing.setText("Existing ID3 tags loaded.")

        except Exception as e:
            if hasattr(self, "lbl_id3_existing"):
                self.lbl_id3_existing.setText(f"Could not read existing ID3 tags: {e}")


    def write_id3_tags(self) -> None:
        if not self.audio_path:
            QMessageBox.information(self, "ID3", "Load an audio file first.")
            return
        if not self.audio_path.lower().endswith(".mp3"):
            QMessageBox.information(self, "ID3", "ID3 tagging is currently implemented for MP3 files only.")
            return

        try:
            ensure_import("mutagen", "mutagen")
            from mutagen.id3 import ID3, TALB, TIT2, TPE1, TRCK, TCON, TDRC, APIC

            try:
                tags = ID3(self.audio_path)
            except Exception:
                tags = ID3()

            artist = self.id3_artist.text().strip()
            album = self.id3_album.text().strip()
            title = self.id3_title.text().strip()
            track = int(self.id3_track.value())
            year = int(self.id3_year.value())
            genre = self.id3_genre.text().strip()

            if artist:
                tags.setall("TPE1", [TPE1(encoding=3, text=[artist])])
            if album:
                tags.setall("TALB", [TALB(encoding=3, text=[album])])
            if title:
                tags.setall("TIT2", [TIT2(encoding=3, text=[title])])
            if track > 0:
                tags.setall("TRCK", [TRCK(encoding=3, text=[str(track)])])
            if year > 0:
                tags.setall("TDRC", [TDRC(encoding=3, text=[str(year)])])
            if genre:
                tags.setall("TCON", [TCON(encoding=3, text=[genre])])

            cover = self.id3_cover_path.text().strip()
            if cover and os.path.exists(cover):
                with open(cover, "rb") as f:
                    data = f.read()
                mime = "image/jpeg" if cover.lower().endswith((".jpg", ".jpeg")) else "image/png"
                tags.setall(
                    "APIC",
                    [APIC(encoding=3, mime=mime, type=3, desc="Cover", data=data)],
                )

            tags.save(self.audio_path)
            QMessageBox.information(self, "ID3", "Tags written successfully.")
        except Exception as e:
            QMessageBox.critical(self, "ID3", f"Failed to write ID3 tags: {e}")

    def _write_id3_tags_silent(self) -> None:
        ensure_import("mutagen", "mutagen")
        from mutagen.id3 import ID3, TALB, TIT2, TPE1, TRCK, TCON, TDRC, APIC

        try:
            tags = ID3(self.audio_path)
        except Exception:
            tags = ID3()

        artist = self.id3_artist.text().strip()
        album = self.id3_album.text().strip()
        title = self.id3_title.text().strip()
        track = int(self.id3_track.value())
        year = int(self.id3_year.value())
        genre = self.id3_genre.text().strip()

        if artist:
            tags.setall("TPE1", [TPE1(encoding=3, text=[artist])])
        if album:
            tags.setall("TALB", [TALB(encoding=3, text=[album])])
        if title:
            tags.setall("TIT2", [TIT2(encoding=3, text=[title])])
        if track > 0:
            tags.setall("TRCK", [TRCK(encoding=3, text=[str(track)])])
        if year > 0:
            tags.setall("TDRC", [TDRC(encoding=3, text=[str(year)])])
        if genre:
            tags.setall("TCON", [TCON(encoding=3, text=[genre])])

        cover = self.id3_cover_path.text().strip()
        if cover and os.path.exists(cover):
            with open(cover, "rb") as f:
                data = f.read()
            mime = "image/jpeg" if cover.lower().endswith((".jpg", ".jpeg")) else "image/png"
            tags.setall("APIC", [APIC(encoding=3, mime=mime, type=3, desc="Cover", data=data)])

        tags.save(self.audio_path)

    # ------------------------------
    # Show Notes
    # ------------------------------

    def _active_notes_editor(self) -> QTextEdit:
        # Prefer the widget with focus; default to summary.
        if hasattr(self, "notes_bullets") and self.notes_bullets.hasFocus():
            return self.notes_bullets
        return self.notes_summary

    def _md_wrap(self, left: str, right: str) -> None:
        ed = self._active_notes_editor()
        cur = ed.textCursor()
        sel = cur.selectedText()
        if not sel:
            cur.insertText(left + "text" + right)
            # select 'text'
            cur.movePosition(cur.Left, cur.MoveAnchor, len(right))
            cur.movePosition(cur.Left, cur.KeepAnchor, len("text"))
            ed.setTextCursor(cur)
            return
        cur.insertText(left + sel + right)

    def _md_prefix_lines(self, prefix: str) -> None:
        ed = self._active_notes_editor()
        cur = ed.textCursor()
        if not cur.hasSelection():
            cur.select(cur.LineUnderCursor)

        # QTextEdit.selectedText uses U+2029 for line breaks
        text = cur.selectedText().replace("\u2029", "\n")
        lines = text.split("\n")

        new_lines: List[str] = []
        for ln in lines:
            new_lines.append(prefix + ln if ln.strip() else ln)

        cur.insertText("\n".join(new_lines))

    def _md_link(self) -> None:
        ed = self._active_notes_editor()
        cur = ed.textCursor()
        sel = cur.selectedText()
        if not sel:
            sel = "link text"
        url, ok = QInputDialog.getText(self, "Insert Link", "URL:")
        if not ok:
            return
        url = (url or "").strip() or "https://example.com"
        cur.insertText(f"[{sel}]({url})")

    def generate_show_notes(self) -> None:
        if not self.segments:
            QMessageBox.information(self, "Show Notes", "Transcribe first so we have text to work with.")
            return

        full = " ".join([(s.get("text") or "").strip() for s in self.segments]).strip()
        if not full:
            QMessageBox.information(self, "Show Notes", "Transcript is empty.")
            return
        if hasattr(self, "chk_llm_notes") and self.chk_llm_notes.isChecked():
            self.llm_notes_worker = LLMShowNotesWorker(
                model_name=self.llm_model_name,
                device_mode=self.llm_device_mode,
                segments=self.segments,
                chapters=self.chapters,
            )
            self.llm_notes_worker.status.connect(self.status.setText)
            self.llm_notes_worker.failed.connect(
                lambda e: QMessageBox.warning(
                    self,
                    "LLM Notes",
                    "Local LLM show-notes failed; using fast local fallback.\n\nDetails: " + str(e),
                )
            )

            def _apply_notes(title: str, summary: str, bullets: str):
                self.notes_title.setText(title)
                self.notes_summary.setPlainText(summary)
                self.notes_bullets.setPlainText(bullets)
                self.status.setText("Show notes generated (LLM)")

            self.llm_notes_worker.finished.connect(_apply_notes)
            self.llm_notes_worker.start()
            return

        # Sentence split
        sents = [x.strip() for x in re.split(r"(?<=[.!?]) +", full) if x.strip()]
        if len(sents) < 6:
            sents = [full]

        st_model = _maybe_load_sentence_transformer()

        def _tfidf_pick(sentences: List[str], k: int) -> List[str]:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
            X = vec.fit_transform([_normalize_text_for_topics(s) for s in sentences])
            centroid = np.asarray(X.mean(axis=0)).ravel()
            # X is sparse; X @ centroid returns a dense ndarray. Use np.asarray for safety.
            scores = np.asarray(X.dot(centroid)).ravel()
            idx = np.argsort(scores)[::-1][:k]
            return [sentences[i] for i in sorted(idx)]

        head = " ".join(
            [(s.get("text") or "").strip() for s in self.segments if float(s.get("start", 0.0)) <= 180.0]
        ).strip() or full

        suggested_title = _make_title_from_text(head, max_terms=3)
        self.notes_title.setText(suggested_title)

        k_summary = 4
        k_bullets = 8

        # TF-IDF fallback (fast, local)
        summary = "\n\n".join(_tfidf_pick(sents, k_summary))
        bullets = "\n".join([f"- {s}" for s in _tfidf_pick(sents, k_bullets)])

        # Optional embeddings refinement
        if st_model is not None and len(sents) > 1:
            try:
                emb = st_model.encode(
                    [_normalize_text_for_topics(s) for s in sents],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                emb = np.asarray(emb, dtype=np.float32)
                centroid = emb.mean(axis=0)
                centroid = centroid / max(1e-9, np.linalg.norm(centroid))
                sim_to_cent = emb @ centroid

                def pick(k: int) -> List[int]:
                    picked: List[int] = []
                    cand = list(range(len(sents)))
                    for _ in range(min(k, len(cand))):
                        best_i = None
                        best_score = -1e9
                        for i in cand:
                            red = 0.0
                            if picked:
                                red = float(max(emb[i] @ emb[j] for j in picked))
                            score = float(sim_to_cent[i]) - 0.4 * red
                            if score > best_score:
                                best_score = score
                                best_i = i
                        if best_i is None:
                            break
                        picked.append(best_i)
                        cand.remove(best_i)
                    return sorted(picked)

                sum_idx = pick(k_summary)
                bul_idx = pick(k_bullets)
                summary = "\n\n".join([sents[i] for i in sum_idx])
                bullets = "\n".join([f"- {sents[i]}" for i in bul_idx])
            except Exception:
                pass

        fun = f"In this episode: {suggested_title}.\n\n{summary}"

        self.notes_summary.setPlainText(fun)
        self.notes_bullets.setPlainText(bullets)
        self.status.setText("Show notes generated")

    def export_show_notes(self) -> None:
        title = (self.notes_title.text() or "").strip()
        summary = self.notes_summary.toPlainText().strip()
        bullets = self.notes_bullets.toPlainText().strip()

        if not (title or summary or bullets):
            QMessageBox.information(self, "Show Notes", "Nothing to export yet.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Notes", "", "Markdown (*.md);;Text (*.txt)")
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            if ext not in (".md", ".txt"):
                path += ".md"

            out: List[str] = []
            if title:
                out.append(f"# {title}\n")
            if summary:
                out.append(summary + "\n")
            if bullets:
                out.append("\n## Show Notes\n" + bullets + "\n")

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(out).strip() + "\n")

            QMessageBox.information(self, "Show Notes", "Exported.")
        except Exception as e:
            QMessageBox.critical(self, "Show Notes", f"Export failed: {e}")
   

# ------------------------------
# Packaging Notes (Execution Hardening)
# ------------------------------
# For a real consumer product, bundle everything:
#
# PyInstaller (single-folder is more reliable than one-file for Qt):
#   pyinstaller --noconfirm --windowed --name PodcastChapters app.py \
#     --collect-all PySide6 --collect-all sklearn --collect-all pydub \
#     --collect-all faster_whisper --collect-all numpy
#
# Then build platform installers:
# - Windows: use WiX Toolset or Inno Setup to wrap the dist folder to MSI/EXE
# - macOS: create-dmg to wrap .app to DMG
# - Linux: appimagetool to wrap dist folder to AppImage
#
# FFmpeg:
# - Recommended: ship known-good ffmpeg/ffplay binaries inside the installer
# - This code also supports first-run download if missing


def main() -> int:
    # Qt 6 enables high-DPI behavior by default; the old AA_EnableHighDpiScaling flag is deprecated.
    # We keep the console quiet by not setting deprecated attributes.

    app = QApplication(sys.argv)
    w = App()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
