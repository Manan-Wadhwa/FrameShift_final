@echo off
REM F1 Visual Difference Engine - Installation Test

echo.
echo ============================================================
echo    F1 VISUAL DIFFERENCE ENGINE - INSTALLATION TEST
echo ============================================================
echo.

cd /d "%~dp0"
python test_installation.py

echo.
pause
