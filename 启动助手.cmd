@echo off
cd /d "%~dp0"
if exist "%CD%\.hf-cache" (
    set "HF_HOME=%CD%\.hf-cache"
    set "HF_HUB_OFFLINE=1"
)
if exist "%CD%\models" (
    set "OLLAMA_MODELS=%CD%\models"
    set "OLLAMA_NO_CLOUD=1"
)
set "PYTHONIOENCODING=utf-8"
powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null } catch { $exe = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\ollama.exe'; if (-not (Test-Path $exe)) { $cmd = Get-Command ollama -ErrorAction SilentlyContinue; if ($cmd) { $exe = $cmd.Source } else { $exe = $null } }; if ($exe) { Start-Process -FilePath $exe -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 3 } }"
set "PY_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"
"%PY_EXE%" -m streamlit run research_assistant\presentation\streamlit_app.py --server.address 127.0.0.1
pause
