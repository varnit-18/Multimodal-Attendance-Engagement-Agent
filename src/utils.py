"""
Utilities and Visualizations Module
Plotting charts with Plotly/Matplotlib, Audio spectrogram rendering,
and synthetic asset generation for offline demonstrations.
"""

import cv2
import io
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, List

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .config import FACE_SAMPLES_DIR, AUDIO_SAMPLES_DIR
from .audio import AudioEngine


def plot_waveform(y: np.ndarray, sr: int):
    """Generate interactive Plotly audio waveform."""
    # Downsample for snappy browser rendering
    step = max(1, len(y) // 2000)
    y_sub = y[::step]
    time_axis = np.linspace(0, len(y) / sr, len(y_sub))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=y_sub,
        mode="lines",
        line=dict(color="#6366F1", width=1.5),
        name="Audio Signal"
    ))
    fig.update_layout(
        title="Time-Domain Acoustic Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=260
    )
    return fig


def plot_spectrogram(y: np.ndarray, sr: int):
    """Generate Mel-Spectrogram matplotlib figure for Librosa Lab."""
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0F172A")
    ax.set_facecolor("#1E293B")

    if LIBROSA_AVAILABLE and len(y) > 0:
        s_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        s_db = librosa.power_to_db(s_mel, ref=np.max)
        img = librosa.display.specshow(s_db, x_axis="time", y_axis="mel", sr=sr, ax=ax, cmap="magma")
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.ax.yaxis.set_tick_params(color="#94A3B8")
        plt.setp(cbar.ax.get_yticklabels(), color="#94A3B8")
    else:
        # Fallback representation
        mock_spec = np.random.randn(64, 100)
        ax.imshow(mock_spec, aspect="auto", origin="lower", cmap="magma")

    ax.set_title("Librosa Mel-Spectrogram (Acoustic Frequency Distribution)", color="#F8FAFC", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", color="#94A3B8")
    ax.set_ylabel("Hz", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    return fig


def plot_mfcc_heatmap(mfcc: np.ndarray):
    """Plot MFCC Heatmap with Plotly."""
    fig = px.imshow(
        mfcc,
        labels=dict(x="Time Frames", y="MFCC Coefficients", color="Energy"),
        x=list(range(mfcc.shape[1])),
        y=[f"C{i+1}" for i in range(mfcc.shape[0])],
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig.update_layout(
        title="Mel-Frequency Cepstral Coefficients (Vocal Timbre Matrix)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=280
    )
    return fig


def plot_biometric_gauge(face_score: float, voice_score: float, liveness_score: float, total_score: float):
    """Generate Radar Chart comparing component biometric scores."""
    categories = ["Face Match", "Voice Match", "Anti-Spoof Liveness", "Combined Score"]
    values = [face_score * 100, voice_score * 100, liveness_score * 100, total_score * 100]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(79, 70, 229, 0.35)",
        line=dict(color="#6366F1", width=2),
        name="Biometric Confidence"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#94A3B8")),
            bgcolor="#1E293B"
        ),
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        margin=dict(l=40, r=40, t=30, b=30),
        height=280,
        showlegend=False
    )
    return fig


def generate_synthetic_assets_if_missing():
    """Create sample face portrait images and audio samples for instant offline testing."""
    # 1. Create sample face images
    sample_faces = {
        "aarav_sharma.jpg": (200, 150, 120),
        "priya_nair.jpg": (180, 130, 160),
        "rahul_verma.jpg": (140, 160, 200),
        "sample_target.jpg": (210, 160, 130)
    }

    for fname, color in sample_faces.items():
        fpath = FACE_SAMPLES_DIR / fname
        if not fpath.exists():
            img = np.full((320, 320, 3), (25, 30, 42), dtype=np.uint8)
            # Draw synthetic stylized head
            cv2.circle(img, (160, 140), 75, color, -1)
            # Neck and shoulders
            cv2.ellipse(img, (160, 310), (100, 80), 0, 0, 180, color, -1)
            # Eyes
            cv2.circle(img, (135, 125), 8, (255, 255, 255), -1)
            cv2.circle(img, (185, 125), 8, (255, 255, 255), -1)
            cv2.circle(img, (135, 125), 4, (40, 40, 40), -1)
            cv2.circle(img, (185, 125), 4, (40, 40, 40), -1)
            # Mouth smile
            cv2.ellipse(img, (160, 170), (25, 12), 0, 0, 180, (50, 50, 180), 3)
            # Name tag
            cv2.putText(img, fname.split(".")[0].upper(), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(str(fpath), img)

    # 2. Create sample audio voice files
    sample_voices = {
        "aarav_voice.wav": 210.0,
        "priya_voice.wav": 290.0,
        "rahul_voice.wav": 175.0,
        "ananya_voice.wav": 310.0,
        "test_sample.wav": 212.0
    }

    for fname, freq in sample_voices.items():
        fpath = AUDIO_SAMPLES_DIR / fname
        if not fpath.exists():
            wav_bytes = AudioEngine.generate_synthetic_voice_wav(frequency=freq, duration=2.5)
            with open(fpath, "wb") as f:
                f.write(wav_bytes)
