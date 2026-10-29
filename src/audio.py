"""
Audio Signal Processing Module (Librosa)
Acoustic Feature Extraction, MFCC Profiling, Spectral Analysis,
and Voice Biometric Verification.
"""

import io
import math
import struct
import wave
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

try:
    import librosa
    import librosa.display
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .config import AUDIO_SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, SILENCE_TOP_DB


@dataclass
class AudioFeatures:
    mfcc: np.ndarray             # (n_mfcc, time_frames)
    mfcc_mean: np.ndarray        # (n_mfcc,)
    mfcc_std: np.ndarray         # (n_mfcc,)
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    chroma: np.ndarray
    duration: float
    rms_energy: float


class AudioEngine:
    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE, n_mfcc: int = N_MFCC):
        self.sr = sample_rate
        self.n_mfcc = n_mfcc

    def load_audio(self, audio_source) -> Tuple[np.ndarray, int]:
        """
        Load audio from file path, bytes, or file-like buffer into 1D float32 array at target sample rate.
        """
        if not LIBROSA_AVAILABLE:
            # Fallback if librosa is not installed
            if isinstance(audio_source, (str, bytes, io.BytesIO)):
                return self._fallback_wav_reader(audio_source)
            return np.zeros(self.sr * 2, dtype=np.float32), self.sr

        if isinstance(audio_source, bytes):
            audio_buf = io.BytesIO(audio_source)
            y, sr = librosa.load(audio_buf, sr=self.sr, mono=True)
        elif hasattr(audio_source, "read"):
            # File-like object (e.g., Streamlit UploadedFile)
            audio_source.seek(0)
            y, sr = librosa.load(audio_source, sr=self.sr, mono=True)
        elif isinstance(audio_source, str):
            y, sr = librosa.load(audio_source, sr=self.sr, mono=True)
        elif isinstance(audio_source, np.ndarray):
            y = audio_source.astype(np.float32)
            sr = self.sr
        else:
            raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

        # Trim leading and trailing silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)
        if len(y_trimmed) < 100:  # If silence trimming removed everything
            y_trimmed = y

        return y_trimmed, sr

    def _fallback_wav_reader(self, source) -> Tuple[np.ndarray, int]:
        """Built-in standard library WAV reader fallback."""
        try:
            if isinstance(source, bytes):
                f = io.BytesIO(source)
            elif hasattr(source, "read"):
                source.seek(0)
                f = source
            else:
                f = open(source, "rb")

            with wave.open(f, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)

                if sampwidth == 2:
                    data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 1:
                    data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128) / 128.0
                else:
                    data = np.zeros(framerate, dtype=np.float32)

                if n_channels > 1:
                    data = data.reshape(-1, n_channels).mean(axis=1)

                return data, framerate
        except Exception:
            return np.zeros(self.sr * 2, dtype=np.float32), self.sr

    def extract_features(self, y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> AudioFeatures:
        """
        Extract detailed acoustic features using Librosa:
        - MFCC (Mel-Frequency Cepstral Coefficients)
        - Spectral Centroid
        - Spectral Rolloff
        - Zero Crossing Rate
        - Chroma STFT
        - RMS Energy
        """
        if not LIBROSA_AVAILABLE:
            mfcc_mock = np.random.randn(self.n_mfcc, 50).astype(np.float32)
            return AudioFeatures(
                mfcc=mfcc_mock,
                mfcc_mean=np.mean(mfcc_mock, axis=1),
                mfcc_std=np.std(mfcc_mock, axis=1),
                spectral_centroid=np.array([1500.0]),
                spectral_rolloff=np.array([3000.0]),
                zero_crossing_rate=np.array([0.05]),
                chroma=np.zeros((12, 50)),
                duration=len(y) / sr if sr > 0 else 1.0,
                rms_energy=0.08
            )

        duration = librosa.get_duration(y=y, sr=sr)

        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # 2. Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)

        # 3. Spectral Rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)

        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)

        # 5. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

        # 6. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        rms_val = float(np.mean(rms)) if rms.size > 0 else 0.0

        return AudioFeatures(
            mfcc=mfcc,
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            spectral_centroid=spec_cent,
            spectral_rolloff=spec_roll,
            zero_crossing_rate=zcr,
            chroma=chroma,
            duration=duration,
            rms_energy=rms_val
        )

    def extract_voice_embedding(self, audio_source) -> Optional[np.ndarray]:
        """
        Generate a compact normalized voice biometric feature vector (length = n_mfcc).
        """
        y, sr = self.load_audio(audio_source)
        if len(y) < sr * 0.2:  # Less than 200ms of sound
            return None

        features = self.extract_features(y, sr)
        # We use the mean MFCC across time as the speaker vocal tract fingerprint
        emb = features.mfcc_mean.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb

    @staticmethod
    def match_voice(target_embedding: np.ndarray, enrolled_users: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Compare voice embedding against enrolled users using Cosine Similarity.
        Returns (best_match_user_dict, similarity_score [0..1]).
        """
        if target_embedding is None or not enrolled_users:
            return None, 0.0

        best_user = None
        highest_sim = -1.0

        target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-8)

        for user in enrolled_users:
            user_emb = user.get("voice_embedding")
            if user_emb is None:
                continue

            user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-8)
            dot = np.dot(target_norm, user_norm)
            similarity = (float(dot) + 1.0) / 2.0

            if similarity > highest_sim:
                highest_sim = similarity
                best_user = user

        return best_user, max(0.0, min(1.0, highest_sim))

    @staticmethod
    def generate_synthetic_voice_wav(frequency: float = 220.0, duration: float = 2.0, sample_rate: int = 16000) -> bytes:
        """
        Synthesize clean speech-like harmonic vocal audio in pure Python (WAV format).
        Used for offline testing, demos, and instant enrollments without a microphone.
        """
        num_samples = int(sample_rate * duration)
        buf = io.BytesIO()

        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)       # Mono
            wf.setsampwidth(2)       # 16-bit
            wf.setframerate(sample_rate)

            # Generate fundamental frequency + vocal formants (harmonics)
            samples = []
            for i in range(num_samples):
                t = i / sample_rate
                # Fundamental + 1st formant + 2nd formant + envelope
                env = math.sin(math.pi * t / duration)  # smooth envelope
                val = (
                    0.50 * math.sin(2.0 * math.pi * frequency * t) +
                    0.30 * math.sin(2.0 * math.pi * (frequency * 2.1) * t) +
                    0.15 * math.sin(2.0 * math.pi * (frequency * 3.4) * t) +
                    0.05 * math.sin(2.0 * math.pi * (frequency * 5.0) * t)
                ) * env

                clamped = max(-1.0, min(1.0, val))
                int_val = int(clamped * 32767)
                samples.append(struct.pack("<h", int_val))

            wf.writeframes(b"".join(samples))

        buf.seek(0)
        return buf.getvalue()
      
