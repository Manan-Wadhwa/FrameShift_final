@echo off
REM F1 Visual Difference Engine - Quick Launcher for Windows

echo.
echo ============================================================
echo    F1 VISUAL DIFFERENCE ENGINE - STREAMLIT LAUNCHER
echo ============================================================
echo.
echo Starting Streamlit Dashboard...
echo This will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
python -m streamlit run demo/streamlit_app.py

pause
