# Audio-Decoded

**Audio-Decoded** is a full-stack web application that analyzes audio files or live microphone input to detect a track’s **tempo (BPM)** and **musical key**. The platform combines Python-based digital signal processing with a minimalist, responsive interface designed for quick and intuitive audio analysis.

The application allows users to upload music files through drag-and-drop or record audio directly through their microphone. Once analyzed, the system displays the detected BPM and key. If the harmonic profile suggests more than one likely key, the application will display both possibilities.

The app now also includes a lightweight PostgreSQL persistence layer to store user data and analysis history for each upload.

---

## Features

### BPM Detection
- Detects the tempo of an audio excerpt using onset-envelope analysis.
- Normalizes tempo estimates to common musical ranges.

### Musical Key Detection
- Uses chroma features and tonal profile correlation to estimate key.
- Automatically detects ambiguous keys and displays both candidates when appropriate.

### Drag and Drop Upload
- Upload MP3, WAV, FLAC, OGG, M4A, AAC, or WEBM files directly into the interface.

### Live Microphone Analysis
- Record audio through the browser microphone and analyze it once recording stops.

### Persistent Analysis History
- Saves upload metadata and analysis results in PostgreSQL.
- Stores users, uploaded audio records, and analysis outcomes.
- Keeps the app fast by only writing to the database when an analysis is triggered.

### Minimalist Interface
- Dark themed UI with orange accents.
- Smooth animated transitions for text and results.
- Fully responsive layout using the entire viewport.

---

## Tech Stack

### Core Stack
- **Python**
- **FastAPI**
- **JavaScript**
- **HTML/CSS**
- **PostgreSQL**
- **Librosa**
- **NumPy**
- **SciPy**

### Database Layer
- **SQLAlchemy**
- **psycopg**
- **PostgreSQL** for user, upload, and analysis tracking

### Audio Processing
- Librosa DSP pipeline
- Onset strength analysis
- Chroma feature extraction

### Frontend
- HTML
- CSS
- JavaScript

---

## How It Works

### BPM Detection
1. Decode at most the first 30 seconds at 22,050 Hz, in mono.
2. Trim silence and compute onset strength directly from the mix.
3. Estimate global tempo from the onset envelope and normalize the BPM range.

### Key Detection
1. Use the same bounded audio excerpt without a separate source-separation pass.
2. Chroma features measure energy distribution across pitch classes.
3. The chroma profile is compared against known major and minor key profiles.
4. The highest scoring key is selected.
5. If two keys have very similar scores, both are returned.

### Data Persistence
1. A user upload triggers the BPM/key analysis.
2. The result is generated in the FastAPI route.
3. After responding, a background task saves metadata and the outcome to PostgreSQL when available.
4. If the database is offline, the app continues to function normally without slowing the page load.

---

## Installation

Analysis logs include decoding, BPM, key, and total processing times. The first
request can take longer while audio libraries initialize. Upload time is separate
from processing time, and the browser stops waiting after two minutes. This browser
timeout does not cancel server processing. Estimates reflect the first 30 seconds;
long intros or later tempo/key changes may need a different excerpt. Background
history saving is best-effort and is not durable across server restarts.

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/audio-decoded.git
cd audio-decoded
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment (Windows PowerShell):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local PostgreSQL database named `audio_decoded`.

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/audio_decoded
```

Run the application:

```bash
python -m uvicorn main:app --reload
```

Open in your browser:

```text
http://127.0.0.1:8000
```

---

## Supported Audio Formats

The application supports:

- MP3
- WAV
- FLAC
- OGG
- M4A
- AAC
- WEBM

For best compatibility with compressed formats, installing **FFmpeg** is recommended.

---

## Usage

1. Open the application in your browser.
2. Drag and drop an audio file or click **Browse Files**.
3. Alternatively, record audio using **Live Microphone Scan**.
4. After analysis, the application displays:
   - **BPM**
   - **Musical Key**
5. The upload and analysis metadata can be saved to PostgreSQL when the database is available.

If the algorithm detects multiple likely keys, they are displayed together (for example: `C Maj / A Min`).

---

## Database Schema

The app includes a simple schema for:

- `users`
- `audio_uploads`
- `analysis_results`

This enables basic history tracking and future expansion into saved playlists, favorites, or searchable track data.

---

## Future Improvements

Potential enhancements include:

- waveform visualization
- confidence scoring for predictions
- batch analysis for multiple tracks
- real-time BPM detection
- improved key detection using machine learning models
- cloud deployment for public access
- searchable analysis history from PostgreSQL

---

## Author

Built by **Aaliyan Muhammad**
