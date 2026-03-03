#!/usr/bin/env python3
"""Launch a safe local Gradio UI for Qwen3-TTS."""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import webbrowser
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import urlopen

import torch

from qwen_tts import Qwen3TTSModel
from qwen_tts.cli.demo import build_demo

DEFAULT_CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
LOCAL_HOST = "127.0.0.1"
START_PORT = 8000


def _detect_ram_gb() -> Optional[float]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        return (page_size * phys_pages) / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        return None


def _detect_runtime_capabilities() -> Dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    ram_gb = _detect_ram_gb()
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def _set_thread_guardrails() -> int:
    threads = min(8, os.cpu_count() or 1)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    return threads


def _find_free_port(start_port: int = START_PORT, host: str = LOCAL_HOST) -> int:
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((host, port)) != 0:
                return port
        port += 1
    raise RuntimeError("Unable to find a free TCP port.")


def _checkpoint_is_cached(checkpoint: str) -> bool:
    if os.path.isdir(checkpoint):
        return True
    if os.path.isfile(checkpoint):
        return True

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=checkpoint, local_files_only=True)
        return True
    except Exception:
        return False


def _start_browser_poll(url: str, timeout_s: int = 60) -> threading.Thread:
    def _poll_open() -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with urlopen(url, timeout=2):
                    webbrowser.open(url)
                    return
            except (URLError, OSError):
                time.sleep(0.5)

    t = threading.Thread(target=_poll_open, daemon=True)
    t.start()
    return t


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a safe local Qwen3-TTS Gradio UI.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF checkpoint id or local path.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Execution device preference.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None, help="Optional dtype override.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Default generation max_new_tokens.")
    parser.add_argument("--port", type=int, default=START_PORT, help="Preferred start port (first free port will be used).")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    print("Setting up…")
    capabilities = _detect_runtime_capabilities()
    ram_display = f"{capabilities['ram_gb']:.1f} GB" if capabilities["ram_gb"] is not None else "unknown"
    print(
        "Runtime capabilities: "
        f"CUDA={capabilities['cuda_available']}"
        + (f" ({capabilities['gpu_name']})" if capabilities["gpu_name"] else "")
        + f", CPU cores={capabilities['cpu_count']}, RAM={ram_display}"
    )

    threads = _set_thread_guardrails()
    print(f"Thread guardrails set: OMP_NUM_THREADS={threads}, MKL_NUM_THREADS={threads}, torch={threads}")

    use_cuda = args.device == "cuda" and capabilities["cuda_available"]
    if args.device == "cuda" and not capabilities["cuda_available"]:
        print("CUDA requested but unavailable; falling back to CPU.")

    device = "cuda:0" if use_cuda else "cpu"

    if args.dtype is not None:
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        dtype = dtype_map[args.dtype]
    else:
        dtype = torch.bfloat16 if use_cuda else torch.float32

    if not _checkpoint_is_cached(args.checkpoint):
        print("Downloading weights…")

    print("Loading model…")
    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )

    port = _find_free_port(start_port=max(1, args.port), host=LOCAL_HOST)
    url = f"http://{LOCAL_HOST}:{port}"

    demo = build_demo(tts, args.checkpoint, {"max_new_tokens": args.max_new_tokens})

    print(f"Ready at <{url}>")

    launch_kwargs: Dict[str, Any] = {
        "server_name": LOCAL_HOST,
        "server_port": port,
        "share": False,
        "inbrowser": True,
    }

    try:
        demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
    except TypeError:
        launch_kwargs.pop("inbrowser", None)
        _start_browser_poll(url)
        demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
