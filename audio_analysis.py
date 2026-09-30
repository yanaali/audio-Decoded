from typing import Dict, List, Tuple
import logging
from time import perf_counter

import librosa
import numpy as np

# Bound decoding and feature extraction for predictable CPU and memory use.
# Estimate directly from the mix instead of separating and reconstructing audio.
MAX_ANALYSIS_SECONDS = 30.0
KEY_AUDIO_SECONDS = 30.0
logger = logging.getLogger("uvicorn.error")

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88
], dtype=float)

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17
], dtype=float)


def _normalize_tempo(bpm: float) -> float:
    if not np.isfinite(bpm) or bpm <= 0:
        return 0.0

    while bpm < 70:
        bpm *= 2
    while bpm > 200:
        bpm /= 2

    return bpm


def detect_bpm(y: np.ndarray, sr: int) -> int:
    if y.size == 0:
        return 0

    y, _ = librosa.effects.trim(y, top_db=30)

    if y.size == 0:
        return 0

    # Spectral flux works directly on the mix, without two inverse STFTs
    # and the median filters required by harmonic/percussive separation.
    hop_length = 128

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        aggregate=np.mean,
        hop_length=hop_length,
        n_mels=64,
    )

    if onset_env.size == 0 or np.allclose(onset_env, 0):
        return 0

    global_tempo = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
    )

    if isinstance(global_tempo, np.ndarray):
        global_tempo = float(global_tempo.flat[0])

    global_tempo = _normalize_tempo(float(global_tempo))
    if global_tempo <= 0:
        return 0

    return int(round(global_tempo))


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def _score_keys(chroma_avg: np.ndarray) -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []

    for i in range(12):
        major_score = _corrcoef_safe(chroma_avg, np.roll(MAJOR_PROFILE, i))
        minor_score = _corrcoef_safe(chroma_avg, np.roll(MINOR_PROFILE, i))

        scores.append((f"{KEY_NAMES[i]} Maj", major_score))
        scores.append((f"{KEY_NAMES[i]} Min", minor_score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def detect_key(y: np.ndarray, sr: int) -> str:
    if y.size == 0:
        return "Unknown"

    y, _ = librosa.effects.trim(y, top_db=30)

    if y.size == 0:
        return "Unknown"

    # chroma_stft is far cheaper than chroma_cqt on long clips; sufficient for key guess.
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=4096,
        hop_length=1024,
    )
    chroma_avg = np.mean(chroma, axis=1)

    if np.allclose(chroma_avg, 0):
        return "Unknown"

    chroma_avg = chroma_avg / np.sum(chroma_avg)
    scores = _score_keys(chroma_avg)

    best_key, best_score = scores[0]
    second_key, second_score = scores[1]

    if np.isfinite(best_score) and np.isfinite(second_score) and abs(best_score - second_score) < 0.035:
        return f"{best_key}/{second_key}"

    return best_key


def analyze_audio(file_path: str) -> Dict[str, str]:
    started = perf_counter()
    y, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True,
        duration=MAX_ANALYSIS_SECONDS,
    )
    decoded = perf_counter()

    if y.size == 0 or np.max(np.abs(y)) < 1e-6:
        return {"bpm": "Unknown", "key": "Unknown"}

    key_len = int(sr * KEY_AUDIO_SECONDS)
    y_key = y[:key_len] if y.size > key_len else y

    bpm = detect_bpm(y, sr)
    tempo_done = perf_counter()
    key = detect_key(y_key, sr)
    logger.info(
        "Audio analysis: clip=%.1fs decode=%.2fs bpm=%.2fs key=%.2fs total=%.2fs",
        y.size / sr, decoded - started, tempo_done - decoded,
        perf_counter() - tempo_done, perf_counter() - started,
    )

    return {
        "bpm": str(bpm) if bpm > 0 else "Unknown",
        "key": key
    }
