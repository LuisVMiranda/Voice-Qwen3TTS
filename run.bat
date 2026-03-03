@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: -----------------------------------------------------------------------------
:: Qwen3-TTS Windows launcher bootstrapper
:: -----------------------------------------------------------------------------

echo [Qwen3-TTS] Configuring UTF-8 console...
chcp 65001 >nul
set PYTHONUTF8=1

if /I not "%OS%"=="Windows_NT" (
    echo [Qwen3-TTS] This launcher is intended for Windows only.
    echo [Qwen3-TTS] Please use a shell script on Linux/macOS.
    exit /b 0
)

echo [Qwen3-TTS] Checking PowerShell availability...
where powershell >nul 2>nul
if errorlevel 1 (
    echo [Qwen3-TTS] PowerShell not found. Please install/enable PowerShell and retry.
    goto :die
)

echo [Qwen3-TTS] Checking uv...
set "UV_BIN="
for /f "delims=" %%I in ('where uv 2^>nul') do (
    if not defined UV_BIN set "UV_BIN=%%~fI"
)

if not defined UV_BIN (
    echo [Qwen3-TTS] Installing uv via PowerShell...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { irm https://astral.sh/uv/install.ps1 | iex; exit 0 } catch { exit 1 }"

    for /f "delims=" %%I in ('where uv 2^>nul') do (
        if not defined UV_BIN set "UV_BIN=%%~fI"
    )

    if not defined UV_BIN if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe"
    )
)

if not defined UV_BIN (
    echo [Qwen3-TTS] PowerShell install did not expose uv. Downloading local uv.exe...
    if not exist ".tools\uv" mkdir ".tools\uv"

    set "UV_ZIP=.tools\uv\uv.zip"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '.tools\\uv\\uv.zip'; Expand-Archive -Path '.tools\\uv\\uv.zip' -DestinationPath '.tools\\uv' -Force; exit 0 } catch { exit 1 }"
    if errorlevel 1 goto :die

    if exist ".tools\uv\uv-x86_64-pc-windows-msvc\uv.exe" (
        copy /Y ".tools\uv\uv-x86_64-pc-windows-msvc\uv.exe" ".tools\uv\uv.exe" >nul
    )

    if exist ".tools\uv\uv.exe" (
        set "UV_BIN=%CD%\.tools\uv\uv.exe"
    )
)

if not defined UV_BIN (
    echo [Qwen3-TTS] Failed to install or locate uv.
    goto :die
)

echo [Qwen3-TTS] Using uv at: %UV_BIN%

echo [Qwen3-TTS] Installing/provisioning Python 3.12 (uv-managed)...
"%UV_BIN%" python install 3.12
if errorlevel 1 goto :die

echo [Qwen3-TTS] Creating virtual environment (.venv)...
"%UV_BIN%" venv .venv --python 3.12
if errorlevel 1 goto :die

echo [Qwen3-TTS] Installing package in editable mode...
"%UV_BIN%" pip install --python .venv\Scripts\python.exe -e .
if errorlevel 1 goto :die

echo [Qwen3-TTS] Configuring local Hugging Face cache paths...
set "HF_HOME=%CD%\.cache\huggingface"
set "TRANSFORMERS_CACHE=%CD%\.cache\huggingface\transformers"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TRANSFORMERS_CACHE%" mkdir "%TRANSFORMERS_CACHE%"

echo [Qwen3-TTS] Checking optional dependency: SoX...
where sox >nul 2>nul
if errorlevel 1 (
    where winget >nul 2>nul
    if not errorlevel 1 (
        echo [Qwen3-TTS] SoX not found. Attempting install via winget...
        winget install -e --id ChrisBagwell.SoX --accept-source-agreements --accept-package-agreements
    )

    where sox >nul 2>nul
    if errorlevel 1 (
        echo [Qwen3-TTS] SoX missing (optional). Voice generation still works.
    ) else (
        echo [Qwen3-TTS] SoX installed successfully.
    )
) else (
    echo [Qwen3-TTS] SoX already available.
)

echo [Qwen3-TTS] Launching local UI...
.venv\Scripts\python.exe -m qwen_tts.cli.demo
if errorlevel 1 (
    echo [Qwen3-TTS] UI exited with an error.
    goto :die
)

exit /b 0

:die
echo [Qwen3-TTS] Setup failed. Press any key to keep this window open and inspect logs.
pause >nul
exit /b 1
