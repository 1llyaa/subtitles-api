from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pathlib import Path
import tempfile
import os
import textwrap
import whisper

app = FastAPI(title="Whisper Subtitle API")

# --- Helper: lazy‑load and cache models so že se znovu nenačítají -----------------
_models: dict[str, whisper.Whisper] = {}

def get_model(model_size: str = "small") -> whisper.Whisper:
    """Return already‑loaded Whisper model or load it once and cache."""
    if model_size not in _models:
        _models[model_size] = whisper.load_model(model_size)
    return _models[model_size]

# --- Timestamp & wrapping helpers ------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Convert seconds → SRT timestamp (HH:MM:SS,mmm)."""
    millis = int(round(seconds * 1000))
    hrs, millis = divmod(millis, 3_600_000)
    mins, millis = divmod(millis,   60_000)
    secs, millis = divmod(millis,    1_000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def wrap_caption(text: str, width: int = 42) -> str | None:
    """Wrap to ≤2 lines of max `width` chars (returns None if too long)."""
    lines = textwrap.wrap(text, width=width, break_long_words=False)
    return None if len(lines) > 2 else "\n".join(lines)

# --- Builders for SRT / VTT ------------------------------------------------------

def build_srt(segments: list[dict], max_chars: int) -> str:
    lines, idx = [], 1
    for seg in segments:
        wrapped = wrap_caption(seg["text"].strip(), max_chars) or seg["text"].strip()
        lines.append(
            f"{idx}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{wrapped}\n"
        )
        idx += 1
    return "\n".join(lines)

def build_vtt(segments: list[dict], max_chars: int) -> str:
    out = ["WEBVTT\n"]
    for seg in segments:
        wrapped = wrap_caption(seg["text"].strip(), max_chars) or seg["text"].strip()
        out.append(
            f"{format_timestamp(seg['start']).replace(',', '.')} --> {format_timestamp(seg['end']).replace(',', '.')}\n{wrapped}\n"
        )
    return "\n".join(out)

# --- Main endpoint ---------------------------------------------------------------

@app.post("/subtitles", summary="Generate SRT/VTT subtitles from audio")
async def subtitles(
    file: UploadFile = File(..., description="Audio or video file"),
    model_size: str = Query("small", regex="^(tiny|base|small|medium|large)$"),
    language: str | None = Query(None, description="Force language ISO (e.g. 'en', 'cs')"),
    task: str = Query("transcribe", regex="^(transcribe|translate)$"),
    max_chars: int = Query(42, ge=20, le=80, description="Max characters per line"),
    response_format: str = Query("srt", regex="^(srt|vtt)$"),
):
    """Return subtitles file (SRT or VTT) built by Whisper.

    Parameters are passed as query params, file via multipart/form‑data.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File missing")

    # Save upload to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="whisper_")
    audio_path = Path(tmp_dir) / file.filename
    audio_path.write_bytes(await file.read())

    # Whisper transcription
    try:
        model = get_model(model_size)
        result = model.transcribe(
            str(audio_path), task=task, language=language, verbose=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper error: {e}")

    # Build chosen subtitle format
    if response_format == "srt":
        out_text = build_srt(result["segments"], max_chars)
        ext = ".srt"
        media_type = "text/plain"
    else:
        out_text = build_vtt(result["segments"], max_chars)
        ext = ".vtt"
        media_type = "text/vtt"

    sub_path = audio_path.with_suffix(ext)
    sub_path.write_text(out_text, encoding="utf-8")

    # Clean‑up after response is sent
    def cleanup():
        try:
            for p in Path(tmp_dir).iterdir():
                p.unlink()
            os.rmdir(tmp_dir)
        except Exception:
            pass

    return FileResponse(
        path=sub_path,
        media_type=media_type,
        filename=(Path(file.filename).stem + ext),
        background=BackgroundTask(cleanup),
    )

