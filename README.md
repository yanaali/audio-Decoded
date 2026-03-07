# Audio-Decoded

**Audio-Decoded** is a full-stack web application that analyzes audio files or live microphone input to detect a track’s **tempo (BPM)** and **musical key**. The platform combines Python-based digital signal processing with a minimalist, responsive interface designed for quick and intuitive audio analysis.

The application allows users to upload music files through drag-and-drop or record audio directly through their microphone. Once analyzed, the system displays the detected BPM and key. If the harmonic profile suggests more than one likely key, the application will display both possibilities.

---

## Features

### BPM Detection
- Detects the tempo of an audio track using onset-envelope analysis and beat tracking.
- Normalizes tempo estimates to common musical ranges.

### Musical Key Detection
- Uses chroma features and tonal profile correlation to estimate key.
- Automatically detects ambiguous keys and displays both candidates when appropriate.

### Drag and Drop Upload
- Upload MP3, WAV, FLAC, OGG, M4A, AAC, or WEBM files directly into the interface.

### Live Microphone Analysis
- Record audio through the browser microphone and analyze it once recording stops.

### Minimalist Interface
- Dark themed UI with orange accents.
- Smooth animated transitions for text and results.
- Fully responsive layout using the entire viewport.

---

## Tech Stack

### Backend
- **Python**
- **FastAPI**
- **Librosa**
- **NumPy**
- **SciPy**

### Frontend
- **HTML**
- **CSS**
- **JavaScript**

### Audio Processing
- Librosa DSP pipeline
- Harmonic/percussive source separation
- Onset strength analysis
- Chroma feature extraction

---

## How It Works

### BPM Detection
1. The audio signal is trimmed to remove silence.
2. Harmonic-Percussive Source Separation isolates rhythmic elements.
3. Onset strength is computed to detect rhythmic peaks.
4. Tempo candidates are calculated using both beat tracking and global tempo estimation.
5. The most reliable BPM estimate is selected and normalized.

### Key Detection
1. Harmonic components of the audio are extracted.
2. Chroma features measure energy distribution across pitch classes.
3. The chroma profile is compared against known major and minor key profiles.
4. The highest scoring key is selected.
5. If two keys have very similar scores, both are returned.

---

## Installation

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

Run the application:

```bash
python -m uvicorn main:app --reload
```

Open in your browser:

```
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

If the algorithm detects multiple likely keys, they are displayed together (for example: `C Maj / A Min`).

---

## Future Improvements

Potential enhancements include:

- waveform visualization
- confidence scoring for predictions
- batch analysis for multiple tracks
- real-time BPM detection
- improved key detection using machine learning models
- cloud deployment for public access

---

## Author

Built by **Aaliyan Muhammad**