import asyncio
import logging
import os
import shutil
import uuid

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from audio_analysis import analyze_audio
from database import SessionLocal, initialize_db
from models import AnalysisResult, AudioUpload, User

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)


app = FastAPI(title="Audio-Decoded")

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".webm"
}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

INITIALIZED_DB = initialize_db()
logger = logging.getLogger("uvicorn.error")


def _safe_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return ext if ext in ALLOWED_EXTENSIONS else ".wav"


def _persist_analysis(filename: str, bpm: str, key: str, note: str, size_bytes: int) -> None:
    if not INITIALIZED_DB:
        return

    db = SessionLocal()
    try:
        user = db.query(User).first()
        if user is None:
            user = User(email="guest@audiodecoded.local")
            db.add(user)
            db.flush()

        upload = AudioUpload(
            user_id=user.id,
            filename=filename,
            file_path=None,  # Temporary audio is deleted after analysis.
            content_type="audio",
            size_bytes=size_bytes,
        )
        db.add(upload)
        db.flush()

        analysis = AnalysisResult(
            upload_id=upload.id,
            bpm=bpm,
            key=key,
            note=note,
        )
        db.add(analysis)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Could not save analysis history.")
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = _safe_extension(file.filename)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, analyze_audio, file_path)

        bpm = result.get("bpm", "Unknown")
        key = result.get("key", "Unknown")
        note = "Analysis complete."
        # Return the result before optional database I/O. Capture the size now,
        # because the temporary file is removed before this task runs.
        background_tasks.add_task(
            _persist_analysis, file.filename, bpm, key, note,
            os.path.getsize(file_path),
        )

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
