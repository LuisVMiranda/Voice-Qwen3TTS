# Local Run Guide

## Quick start
Double-click `run.bat`.

The launcher starts the local Gradio UI with CPU-safe defaults and attempts to auto-open your browser.

## First-run expectations
On the first run, the launcher will set up everything automatically, which can take a while:

- Downloads Python via `uv` if a suitable local Python is not already present.
- Installs project dependencies.
- Downloads large model weights (this is usually the slowest step).

Subsequent runs are much faster after these assets are cached.

When the UI starts, the launcher prints a line like:

- `Ready at http://127.0.0.1:8000`

If port `8000` is already in use, it automatically picks the next free localhost port and prints the exact URL.

## Logs

Each `run.bat` session writes a timestamped log file to:

- `logs/run_YYYYMMDD_HHMMSS.log`

If the UI exits or the window closes after a failure, reopen `run.bat` from a terminal and inspect the latest file in `logs/`.

## Output location
Generated outputs are written to:

- `outputs/`

## Troubleshooting

### No internet connection
**Typical symptoms/messages:**

- Dependency download failures (e.g., `uv`/package install errors).
- Model download failures (Hugging Face/network timeouts).

**Remedy:**

- Connect to the internet and run `run.bat` again.
- If your network uses a proxy or firewall, allow Python/`uv` and Hugging Face downloads.

### PowerShell execution policy or script blocking
**Typical messages:**

- `... cannot be loaded because running scripts is disabled on this system`
- `PSSecurityException`

**Remedy:**

- Open PowerShell as your user and run:
  - `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Then re-run `run.bat`.
- If policy is managed by your organization, use cmd.exe launch paths or contact IT.

### SoX missing (optional)
**Typical message:**

- `sox` not found (or similar warning).

**Remedy:**

- This is optional for core generation in common setups.
- You can ignore this warning unless you specifically need SoX-based audio post-processing utilities.

### Hugging Face symlink warnings
**Typical message:**

- Hugging Face cache warning about symlinks/hardlinks on Windows.

**Remedy:**

- Safe to ignore in normal usage.
- Downloads still work; caching may be less efficient without symlink support.

## Tested defaults
The default local run flow has been tested for:

- CPU-only execution.
- AMD RX 580 / no GPU acceleration scenario (falls back to CPU path).
