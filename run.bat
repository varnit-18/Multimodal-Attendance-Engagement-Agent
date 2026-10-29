@echo off
TITLE Multimodal AI Smart Attendance System - FutureSkills Prime
echo ==============================================================================
echo Multimodal AI Smart Attendance & Verification System
echo FutureSkills Prime Capstone Project
echo ==============================================================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.9+ from https://python.org or Microsoft Store.
    pause
    exit /b 1
)

echo.
echo Installing / Verifying dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ==============================================================================
echo Launching Streamlit Dashboard...
echo Local URL: http://localhost:8501
echo ==============================================================================
streamlit run app.py

pause
