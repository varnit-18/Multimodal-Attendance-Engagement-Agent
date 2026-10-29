#!/bin/bash
# Multimodal AI Smart Attendance System - FutureSkills Prime
echo "=============================================================================="
echo "Multimodal AI Smart Attendance & Verification System"
echo "FutureSkills Prime Capstone Project"
echo "=============================================================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.9+."
    exit 1
fi

echo "Installing / Verifying dependencies..."
pip install -r requirements.txt

echo ""
echo "=============================================================================="
echo "Launching Streamlit Dashboard on http://localhost:8501..."
echo "=============================================================================="
streamlit run app.py
