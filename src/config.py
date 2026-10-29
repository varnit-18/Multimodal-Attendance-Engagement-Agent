"""
Configuration Module for Multimodal AI Smart Attendance System
FutureSkills Prime Capstone Project
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "attendance_system.db"
SAMPLES_DIR = DATA_DIR / "samples"
FACE_SAMPLES_DIR = SAMPLES_DIR / "faces"
AUDIO_SAMPLES_DIR = SAMPLES_DIR / "audio"
ENROLLED_DIR = DATA_DIR / "enrolled"

# Ensure all directories exist
for folder in [DATA_DIR, SAMPLES_DIR, FACE_SAMPLES_DIR, AUDIO_SAMPLES_DIR, ENROLLED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Vision Processing Constants
FACE_DETECTION_CONFIDENCE = 0.6
FACE_MESH_CONFIDENCE = 0.6
EYE_ASPECT_RATIO_THRESHOLD = 0.22  # Below this is considered a blink/eye closed
HEAD_POSE_THRESHOLD = 18.0         # Head yaw/pitch threshold in degrees for liveness check
FACE_EMBEDDING_DIM = 128           # Normalized geometric landmark feature vector length

# Audio Processing Constants (Librosa)
AUDIO_SAMPLE_RATE = 16000          # 16 kHz standard for speech processing
AUDIO_DURATION = 3.0               # Standard clip duration in seconds
N_MFCC = 20                        # Number of Mel-Frequency Cepstral Coefficients
HOP_LENGTH = 512                   # Librosa hop length
N_FFT = 2048                       # FFT window size
SILENCE_TOP_DB = 25                # Top dB threshold for trimming silence

# Multimodal Verification Fusion Weights (Must sum to 1.0)
DEFAULT_WEIGHT_FACE = 0.50
DEFAULT_WEIGHT_VOICE = 0.35
DEFAULT_WEIGHT_LIVENESS = 0.15

# Verification Thresholds
OVERALL_PASS_THRESHOLD = 0.65      # Multimodal combined score to mark attendance
FACE_PASS_THRESHOLD = 0.60         # Minimum face match score
VOICE_PASS_THRESHOLD = 0.55        # Minimum voice match score
LIVENESS_PASS_THRESHOLD = 0.60     # Minimum liveness score

# Attendance Policy Configuration
OFFICE_START_TIME = "09:30:00"
LATE_GRACE_MINUTES = 15
REPEAT_PUNCH_COOLDOWN_MINUTES = 5  # Prevent duplicate punches within this window

# GenAI Settings
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
GENAI_API_KEY_ENV = "GEMINI_API_KEY"
