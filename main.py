import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from audio_analysis import analyze_audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _warm_analysis_pipeline() -> None:
    """Trigger librosa/numba once at startup so the first real upload is not stalled."""
    import numpy as np
    import soundfile as sf

    sr = 22050
    t = np.arange(sr * 2, dtype=np.float64) / sr
    y = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = os.path.join(UPLOAD_DIR, ".warmup.wav")
    sf.write(path, y, sr)
    try:
        analyze_audio(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _warm_analysis_pipeline)
    except Exception:
        pass
    yield


app = FastAPI(title="Audio-Decoded", lifespan=lifespan)

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".webm"
}

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def _safe_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return ext if ext in ALLOWED_EXTENSIONS else ".wav"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = _safe_extension(file.filename)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # librosa is CPU-heavy; run off the event loop so the server stays responsive.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, analyze_audio, file_path)
        return JSONResponse(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not analyze audio. {str(exc)}"
        ) from exc

    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass
