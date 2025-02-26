@echo off
REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python and try again.
    exit /b 1
)

REM Create virtual environment named .env
python -m venv .env

REM Activate the virtual environment
call .env\Scripts\activate

REM Install the required packages
pip install -r requirements.txt

REM Deactivate the virtual environment
deactivate

echo Virtual environment setup complete.