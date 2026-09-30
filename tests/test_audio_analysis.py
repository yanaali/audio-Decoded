import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_analysis import analyze_audio, detect_bpm, detect_key


class AudioAnalysisTests(unittest.TestCase):
    def test_known_tempos(self):
        sr = 22050
        for bpm in (90, 120, 150):
            with self.subTest(bpm=bpm):
                y = np.zeros(sr * 15, dtype=np.float32)
                pulse = np.random.default_rng(0).normal(size=441)
                pulse *= np.exp(-np.arange(441) / 80)
                for start in np.arange(sr // 2, len(y) - 441, sr * 60 / bpm):
                    y[int(start):int(start) + 441] += pulse
                self.assertLessEqual(abs(detect_bpm(y, sr) - bpm), 3)

    def test_c_major_chord(self):
        sr = 22050
        t = np.arange(sr * 5) / sr
        y = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0))
        self.assertIn("C Maj", detect_key(y.astype(np.float32), sr))

    def test_silence_and_decode_limit(self):
        # Audio after the first 30 seconds must not affect the bounded analysis.
        sr = 22050
        y = np.zeros(sr * 35, dtype=np.float32)
        y[sr * 31:] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silence.wav"
            sf.write(path, y, sr)
            self.assertEqual(analyze_audio(str(path)), {"bpm": "Unknown", "key": "Unknown"})


if __name__ == "__main__":
    unittest.main()
