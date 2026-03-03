# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A gradio demo for Qwen3 TTS models.
"""

import argparse
import os
import re
import tempfile
import wave
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from .. import Qwen3TTSModel, VoiceClonePromptItem


LONG_TEXT_CHAR_THRESHOLD = 450
CHUNK_JOIN_SILENCE_S = 0.12


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _ordered_language_choices(items: Optional[List[str]]) -> List[str]:
    preferred = ["auto", "english", "portuguese"]
    ordered = []
    seen = set()
    values = [x for x in (items or []) if x]

    for key in preferred:
        for item in values:
            if item.lower() == key and item not in seen:
                ordered.append(item)
                seen.add(item)

    for item in values:
        if item not in seen:
            ordered.append(item)

    return ordered


def _pick_default_speaker(items: Optional[List[str]]) -> Optional[str]:
    if not items:
        return None

    preferred = ["ryan", "aiden"]
    for key in preferred:
        for item in items:
            if item.lower() == key:
                return item

    for item in items:
        lowered = item.lower()
        if "en" in lowered or "english" in lowered:
            return item

    return items[0]


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _maybe(v):
    return v if v is not None else gr.update()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-demo",
        description=(
            "Launch a Gradio demo for Qwen3 TTS models (CustomVoice / VoiceDesign / Base).\n\n"
            "Examples:\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --port 8000 --ip 127.0.0.01\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --device cuda:0\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --dtype bfloat16 --no-flash-attn\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    # Positional checkpoint (also supports -c/--checkpoint)
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (positional).",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (optional if positional is provided).",
    )

    # Model loading / from_pretrained args
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for device_map, e.g. cpu, cuda, cuda:0 (default: cpu).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help=(
            "Torch dtype for loading the model "
            "(default: float32 on cpu devices, bfloat16 otherwise)."
        ),
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: disabled).",
    )

    # Gradio server args
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="Server bind IP for Gradio (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for Gradio (default: 8000).",
    )
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link (default: disabled).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Gradio queue concurrency (default: 1).",
    )

    # HTTPS args
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to SSL certificate file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to SSL key file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificate (default: enabled).",
    )

    # Optional generation args
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for generation (optional).")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (optional).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (optional).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (optional).")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (optional).")
    parser.add_argument("--subtalker-top-k", type=int, default=None, help="Subtalker top-k (optional, only for tokenizer v2).")
    parser.add_argument("--subtalker-top-p", type=float, default=None, help="Subtalker top-p (optional, only for tokenizer v2).")
    parser.add_argument(
        "--subtalker-temperature", type=float, default=None, help="Subtalker temperature (optional, only for tokenizer v2)."
    )

    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> str:
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt:
        raise SystemExit(0)  # main() prints help
    return ckpt


def _collect_gen_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)

        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid

    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0

        if m <= 1.0 + 1e-6:
            pass
        else:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)
    
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav


def _save_wav_file(path: str, wav: np.ndarray, sr: int) -> None:
    audio = np.asarray(wav, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sr))
        f.writeframes(pcm.tobytes())


def _detect_model_kind(ckpt: str, tts: Qwen3TTSModel) -> str:
    mt = getattr(tts.model, "tts_model_type", None)
    if mt in ("custom_voice", "voice_design", "base"):
        return mt
    else:
        raise ValueError(f"Unknown Qwen-TTS model type: {mt}")


def _split_long_text(text: str, char_threshold: int = LONG_TEXT_CHAR_THRESHOLD) -> List[str]:
    normalized = (text or "").strip()
    if not normalized:
        return []

    if len(normalized) <= char_threshold:
        return [normalized]

    chunks: List[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", normalized) if p and p.strip()]
    for paragraph in paragraphs or [normalized]:
        if len(paragraph) <= char_threshold:
            chunks.append(paragraph)
            continue

        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", paragraph) if s and s.strip()]
        if not sentences:
            sentences = [paragraph]

        current = ""
        for sentence in sentences:
            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= char_threshold:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(sentence) <= char_threshold:
                    current = sentence
                else:
                    parts = [sentence[i : i + char_threshold] for i in range(0, len(sentence), char_threshold)]
                    chunks.extend(parts[:-1])
                    current = parts[-1]

        if current:
            chunks.append(current)

    return chunks or [normalized]


def _concat_wavs(wavs: List[np.ndarray], sr: int) -> np.ndarray:
    if not wavs:
        return np.zeros((1,), dtype=np.float32)
    if len(wavs) == 1:
        return np.asarray(wavs[0], dtype=np.float32)

    silence = np.zeros((max(1, int(sr * CHUNK_JOIN_SILENCE_S)),), dtype=np.float32)
    stitched: List[np.ndarray] = []
    for i, wav in enumerate(wavs):
        stitched.append(np.asarray(wav, dtype=np.float32))
        if i < len(wavs) - 1:
            stitched.append(silence)
    return np.concatenate(stitched)


def build_demo(
    tts: Qwen3TTSModel,
    ckpt: str,
    gen_kwargs_default: Dict[str, Any],
    runtime_warnings: Optional[List[str]] = None,
) -> gr.Blocks:
    model_kind = _detect_model_kind(ckpt, tts)

    supported_langs_raw = None
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported_langs_raw = tts.model.get_supported_languages()

    supported_spks_raw = None
    if callable(getattr(tts.model, "get_supported_speakers", None)):
        supported_spks_raw = tts.model.get_supported_speakers()

    ordered_langs = _ordered_language_choices([x for x in (supported_langs_raw or [])])
    lang_choices_disp, lang_map = _build_choices_and_map(ordered_langs)
    spk_choices_disp, spk_map = _build_choices_and_map([x for x in (supported_spks_raw or [])])
    default_speaker_raw = _pick_default_speaker([x for x in (supported_spks_raw or [])])
    default_speaker_disp = _title_case_display(default_speaker_raw) if default_speaker_raw else None

    def _gen_common_kwargs() -> Dict[str, Any]:
        return dict(gen_kwargs_default)

    theme = gr.themes.Base(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0b1220",
        body_text_color="#e2e8f0",
        background_fill_primary="#111827",
        background_fill_secondary="#1f2937",
        block_background_fill="#111827",
        block_border_color="#334155",
        block_label_text_color="#e2e8f0",
        input_background_fill="#1f2937",
        input_border_color="#475569",
        input_placeholder_color="#94a3b8",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="#f8fafc",
    )

    css = """
    .gradio-container {
        max-width: 960px !important;
        margin: 0 auto !important;
        background: #0b1220 !important;
        color: #e2e8f0 !important;
    }
    .gradio-container .wrap,
    .gradio-container .contain,
    .gradio-container .block,
    .gradio-container .panel,
    .gradio-container .form,
    .gradio-container .prose,
    .gradio-container .gr-form,
    .gradio-container .gr-box,
    .gradio-container .gr-group,
    .gradio-container .gr-panel,
    .gradio-container .gradio-markdown,
    .gradio-container .gradio-audio,
    .gradio-container .gradio-textbox,
    .gradio-container .gradio-dropdown,
    .gradio-container .gradio-file,
    .gradio-container .gradio-checkbox {
        background: #111827 !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    .gradio-container label,
    .gradio-container p,
    .gradio-container span,
    .gradio-container h1,
    .gradio-container h2,
    .gradio-container h3,
    .gradio-container h4,
    .gradio-container h5,
    .gradio-container h6,
    .gradio-container .prose * {
        color: #e2e8f0 !important;
    }
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select,
    .gradio-container option {
        background: #1f2937 !important;
        color: #e2e8f0 !important;
        border-color: #475569 !important;
    }
    .gradio-container button {
        background: #2563eb !important;
        color: #f8fafc !important;
        border-color: #1d4ed8 !important;
    }
    .gradio-container button:hover {
        background: #1d4ed8 !important;
    }
    .gradio-container [role="alert"],
    .gradio-container .warning,
    .gradio-container .gradio-warning,
    .gradio-container .gr-alert {
        background: #7c2d12 !important;
        color: #ffedd5 !important;
        border-color: #ea580c !important;
    }
    .gradio-container .theme-toggle,
    .gradio-container button[aria-label*="theme" i],
    .gradio-container button[title*="theme" i] {
        display: none !important;
    }
    """

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(
            f"""
# Qwen3 TTS Demo
**Checkpoint:** `{ckpt}`  
**Model Type:** `{model_kind}`  
"""
        )
        if runtime_warnings:
            gr.Markdown("\n".join([f"> ⚠️ {w}" for w in runtime_warnings]))

        if model_kind == "custom_voice":
            gr.Markdown(
                "> ⚠️ **Local-only. Do not use voice cloning for impersonation.**"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label="Text (待合成文本)",
                        lines=4,
                        placeholder="Enter text to synthesize (输入要合成的文本).",
                        info="Short text works best. PT-BR and English supported.",
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="Language (语种)",
                            choices=lang_choices_disp,
                            value=("Auto" if "Auto" in lang_choices_disp else (lang_choices_disp[0] if lang_choices_disp else None)),
                            interactive=True,
                            info="Use Auto unless language is mixed.",
                        )
                        spk_in = gr.Dropdown(
                            label="Speaker (说话人)",
                            choices=spk_choices_disp,
                            value=(default_speaker_disp if default_speaker_disp in spk_choices_disp else (spk_choices_disp[0] if spk_choices_disp else None)),
                            interactive=True,
                            info="Ryan/Aiden are good defaults for English.",
                        )
                    instruct_in = gr.Textbox(
                        label="Instruction (Optional) (控制指令，可不输入)",
                        lines=2,
                        placeholder="e.g. Say it in a very angry tone (例如：用特别伤心的语气说).",
                        info="Optional style cue. Keep it short.",
                    )
                    auto_split_in = gr.Checkbox(
                        label="Auto-split long text (自动分段长文本)",
                        value=True,
                        info=f"Recommended for text longer than {LONG_TEXT_CHAR_THRESHOLD} chars.",
                    )
                    save_out_in = gr.Checkbox(
                        label="Save to outputs/ (salvar em outputs/)",
                        value=False,
                        info="Writes a WAV copy after generation.",
                    )
                    prefix_in = gr.Textbox(
                        label="Filename prefix (optional)",
                        lines=1,
                        placeholder="ex: en_ptbr_demo",
                        info="Used as outputs/<prefix>_<speaker>.wav",
                    )
                    btn = gr.Button("Generate (生成)", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                    err = gr.Textbox(label="Status (状态)", lines=2)

            def run_instruct(text: str, lang_disp: str, spk_disp: str, instruct: str, auto_split: bool, save_out: bool, filename_prefix: str):
                try:
                    if not text or not text.strip():
                        return None, "Text is required (必须填写文本)."
                    if not spk_disp:
                        return None, "Speaker is required (必须选择说话人)."
                    language = lang_map.get(lang_disp, "Auto")
                    speaker = spk_map.get(spk_disp, spk_disp)
                    kwargs = _gen_common_kwargs()
                    text_clean = text.strip()
                    chunks = _split_long_text(text_clean)
                    if len(chunks) > 1 and not auto_split:
                        return None, (
                            f"Text is long ({len(text_clean)} chars). Enable 'Auto-split long text' or split into smaller paragraphs/sentences first."
                        )

                    wav_chunks: List[np.ndarray] = []
                    sr: Optional[int] = None
                    to_generate = chunks if auto_split else [text_clean]
                    for chunk in to_generate:
                        wavs, chunk_sr = tts.generate_custom_voice(
                            text=chunk,
                            language=language,
                            speaker=speaker,
                            instruct=(instruct or "").strip() or None,
                            **kwargs,
                        )
                        sr = chunk_sr if sr is None else sr
                        wav_chunks.append(wavs[0])

                    assert sr is not None
                    merged_wav = _concat_wavs(wav_chunks, sr)
                    status = "Finished. (生成完成)"
                    if auto_split and len(to_generate) > 1:
                        status += f" Auto-split into {len(to_generate)} chunks."
                    if save_out:
                        out_dir = "outputs"
                        os.makedirs(out_dir, exist_ok=True)
                        safe_prefix = (filename_prefix or "").strip().replace(" ", "_")
                        safe_speaker = (speaker or "speaker").strip().replace(" ", "_")
                        if safe_prefix:
                            name = f"{safe_prefix}_{safe_speaker}.wav"
                        else:
                            name = f"{safe_speaker}.wav"
                        out_path = os.path.join(out_dir, name)
                        _save_wav_file(out_path, merged_wav, sr)
                        status = f"Finished. Saved WAV to {out_path}"
                    return _wav_to_gradio_audio(merged_wav, sr), status
                except Exception as e:
                    return None, f"{type(e).__name__}: {e}"

            btn.click(
                run_instruct,
                inputs=[text_in, lang_in, spk_in, instruct_in, auto_split_in, save_out_in, prefix_in],
                outputs=[audio_out, err],
            )

        elif model_kind == "voice_design":
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label="Text (待合成文本)",
                        lines=4,
                        value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="Language (语种)",
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True,
                        )
                    design_in = gr.Textbox(
                        label="Voice Design Instruction (音色描述)",
                        lines=3,
                        value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
                    )
                    auto_split_in = gr.Checkbox(
                        label="Auto-split long text (自动分段长文本)",
                        value=True,
                        info=f"Recommended for text longer than {LONG_TEXT_CHAR_THRESHOLD} chars.",
                    )
                    btn = gr.Button("Generate (生成)", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                    err = gr.Textbox(label="Status (状态)", lines=2)

            def run_voice_design(text: str, lang_disp: str, design: str, auto_split: bool):
                try:
                    if not text or not text.strip():
                        return None, "Text is required (必须填写文本)."
                    if not design or not design.strip():
                        return None, "Voice design instruction is required (必须填写音色描述)."
                    language = lang_map.get(lang_disp, "Auto")
                    kwargs = _gen_common_kwargs()
                    text_clean = text.strip()
                    chunks = _split_long_text(text_clean)
                    if len(chunks) > 1 and not auto_split:
                        return None, (
                            f"Text is long ({len(text_clean)} chars). Enable 'Auto-split long text' or split into smaller paragraphs/sentences first."
                        )
                    wav_chunks: List[np.ndarray] = []
                    sr: Optional[int] = None
                    to_generate = chunks if auto_split else [text_clean]
                    for chunk in to_generate:
                        wavs, chunk_sr = tts.generate_voice_design(
                            text=chunk,
                            language=language,
                            instruct=design.strip(),
                            **kwargs,
                        )
                        sr = chunk_sr if sr is None else sr
                        wav_chunks.append(wavs[0])

                    assert sr is not None
                    merged_wav = _concat_wavs(wav_chunks, sr)
                    status = "Finished. (生成完成)"
                    if auto_split and len(to_generate) > 1:
                        status += f" Auto-split into {len(to_generate)} chunks."
                    return _wav_to_gradio_audio(merged_wav, sr), status
                except Exception as e:
                    return None, f"{type(e).__name__}: {e}"

            btn.click(run_voice_design, inputs=[text_in, lang_in, design_in, auto_split_in], outputs=[audio_out, err])

        else:  # voice_clone for base
            with gr.Tabs():
                with gr.Tab("Clone & Generate (克隆并合成)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            ref_audio = gr.Audio(
                                label="Reference Audio (参考音频)",
                            )
                            ref_text = gr.Textbox(
                                label="Reference Text (参考音频文本)",
                                lines=2,
                                placeholder="Required if not set use x-vector only (不勾选use x-vector only时必填).",
                            )
                            xvec_only = gr.Checkbox(
                                label="Use x-vector only (仅用说话人向量，效果有限，但不用传入参考音频文本)",
                                value=False,
                            )

                        with gr.Column(scale=2):
                            text_in = gr.Textbox(
                                label="Target Text (待合成文本)",
                                lines=4,
                                placeholder="Enter text to synthesize (输入要合成的文本).",
                            )
                            lang_in = gr.Dropdown(
                                label="Language (语种)",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            auto_split_in = gr.Checkbox(
                                label="Auto-split long text (自动分段长文本)",
                                value=True,
                                info=f"Recommended for text longer than {LONG_TEXT_CHAR_THRESHOLD} chars.",
                            )
                            btn = gr.Button("Generate (生成)", variant="primary")

                        with gr.Column(scale=3):
                            audio_out = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                            err = gr.Textbox(label="Status (状态)", lines=2)

                    def run_voice_clone(ref_aud, ref_txt: str, use_xvec: bool, text: str, lang_disp: str, auto_split: bool):
                        try:
                            if not text or not text.strip():
                                return None, "Target text is required (必须填写待合成文本)."
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return None, "Reference audio is required (必须上传参考音频)."
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, (
                                    "Reference text is required when use x-vector only is NOT enabled.\n"
                                    "(未勾选 use x-vector only 时，必须提供参考音频文本；否则请勾选 use x-vector only，但效果会变差.)"
                                )
                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            text_clean = text.strip()
                            chunks = _split_long_text(text_clean)
                            if len(chunks) > 1 and not auto_split:
                                return None, (
                                    f"Text is long ({len(text_clean)} chars). Enable 'Auto-split long text' or split into smaller paragraphs/sentences first."
                                )
                            wav_chunks: List[np.ndarray] = []
                            sr: Optional[int] = None
                            to_generate = chunks if auto_split else [text_clean]
                            for chunk in to_generate:
                                wavs, chunk_sr = tts.generate_voice_clone(
                                    text=chunk,
                                    language=language,
                                    ref_audio=at,
                                    ref_text=(ref_txt.strip() if ref_txt else None),
                                    x_vector_only_mode=bool(use_xvec),
                                    **kwargs,
                                )
                                sr = chunk_sr if sr is None else sr
                                wav_chunks.append(wavs[0])

                            assert sr is not None
                            merged_wav = _concat_wavs(wav_chunks, sr)
                            status = "Finished. (生成完成)"
                            if auto_split and len(to_generate) > 1:
                                status += f" Auto-split into {len(to_generate)} chunks."
                            return _wav_to_gradio_audio(merged_wav, sr), status
                        except Exception as e:
                            return None, f"{type(e).__name__}: {e}"

                    btn.click(
                        run_voice_clone,
                        inputs=[ref_audio, ref_text, xvec_only, text_in, lang_in, auto_split_in],
                        outputs=[audio_out, err],
                    )

                with gr.Tab("Save / Load Voice (保存/加载克隆音色)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### Save Voice (保存音色)
Upload reference audio and text, choose use x-vector only or not, then save a reusable voice prompt file.  
(上传参考音频和参考文本，选择是否使用 use x-vector only 模式后保存为可复用的音色文件)
"""
                            )
                            ref_audio_s = gr.Audio(label="Reference Audio (参考音频)", type="numpy")
                            ref_text_s = gr.Textbox(
                                label="Reference Text (参考音频文本)",
                                lines=2,
                                placeholder="Required if not set use x-vector only (不勾选use x-vector only时必填).",
                            )
                            xvec_only_s = gr.Checkbox(
                                label="Use x-vector only (仅用说话人向量，效果有限，但不用传入参考音频文本)",
                                value=False,
                            )
                            save_btn = gr.Button("Save Voice File (保存音色文件)", variant="primary")
                            prompt_file_out = gr.File(label="Voice File (音色文件)")

                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### Load Voice & Generate (加载音色并合成)
Upload a previously saved voice file, then synthesize new text.  
(上传已保存提示文件后，输入新文本进行合成)
"""
                            )
                            prompt_file_in = gr.File(label="Upload Prompt File (上传提示文件)")
                            text_in2 = gr.Textbox(
                                label="Target Text (待合成文本)",
                                lines=4,
                                placeholder="Enter text to synthesize (输入要合成的文本).",
                            )
                            lang_in2 = gr.Dropdown(
                                label="Language (语种)",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            auto_split_in2 = gr.Checkbox(
                                label="Auto-split long text (自动分段长文本)",
                                value=True,
                                info=f"Recommended for text longer than {LONG_TEXT_CHAR_THRESHOLD} chars.",
                            )
                            gen_btn2 = gr.Button("Generate (生成)", variant="primary")

                        with gr.Column(scale=3):
                            audio_out2 = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                            err2 = gr.Textbox(label="Status (状态)", lines=2)

                    def save_prompt(ref_aud, ref_txt: str, use_xvec: bool):
                        try:
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return None, "Reference audio is required (必须上传参考音频)."
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, (
                                    "Reference text is required when use x-vector only is NOT enabled.\n"
                                    "(未勾选 use x-vector only 时，必须提供参考音频文本；否则请勾选 use x-vector only，但效果会变差.)"
                                )
                            items = tts.create_voice_clone_prompt(
                                ref_audio=at,
                                ref_text=(ref_txt.strip() if ref_txt else None),
                                x_vector_only_mode=bool(use_xvec),
                            )
                            payload = {
                                "items": [asdict(it) for it in items],
                            }
                            fd, out_path = tempfile.mkstemp(prefix="voice_clone_prompt_", suffix=".pt")
                            os.close(fd)
                            torch.save(payload, out_path)
                            return out_path, "Finished. (生成完成)"
                        except Exception as e:
                            return None, f"{type(e).__name__}: {e}"

                    def load_prompt_and_gen(file_obj, text: str, lang_disp: str, auto_split: bool):
                        try:
                            if file_obj is None:
                                return None, "Voice file is required (必须上传音色文件)."
                            if not text or not text.strip():
                                return None, "Target text is required (必须填写待合成文本)."

                            path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
                            payload = torch.load(path, map_location="cpu", weights_only=True)
                            if not isinstance(payload, dict) or "items" not in payload:
                                return None, "Invalid file format (文件格式不正确)."

                            items_raw = payload["items"]
                            if not isinstance(items_raw, list) or len(items_raw) == 0:
                                return None, "Empty voice items (音色为空)."

                            items: List[VoiceClonePromptItem] = []
                            for d in items_raw:
                                if not isinstance(d, dict):
                                    return None, "Invalid item format in file (文件内部格式错误)."
                                ref_code = d.get("ref_code", None)
                                if ref_code is not None and not torch.is_tensor(ref_code):
                                    ref_code = torch.tensor(ref_code)
                                ref_spk = d.get("ref_spk_embedding", None)
                                if ref_spk is None:
                                    return None, "Missing ref_spk_embedding (缺少说话人向量)."
                                if not torch.is_tensor(ref_spk):
                                    ref_spk = torch.tensor(ref_spk)

                                items.append(
                                    VoiceClonePromptItem(
                                        ref_code=ref_code,
                                        ref_spk_embedding=ref_spk,
                                        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                                        icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                                        ref_text=d.get("ref_text", None),
                                    )
                                )

                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            text_clean = text.strip()
                            chunks = _split_long_text(text_clean)
                            if len(chunks) > 1 and not auto_split:
                                return None, (
                                    f"Text is long ({len(text_clean)} chars). Enable 'Auto-split long text' or split into smaller paragraphs/sentences first."
                                )
                            wav_chunks: List[np.ndarray] = []
                            sr: Optional[int] = None
                            to_generate = chunks if auto_split else [text_clean]
                            for chunk in to_generate:
                                wavs, chunk_sr = tts.generate_voice_clone(
                                    text=chunk,
                                    language=language,
                                    voice_clone_prompt=items,
                                    **kwargs,
                                )
                                sr = chunk_sr if sr is None else sr
                                wav_chunks.append(wavs[0])

                            assert sr is not None
                            merged_wav = _concat_wavs(wav_chunks, sr)
                            status = "Finished. (生成完成)"
                            if auto_split and len(to_generate) > 1:
                                status += f" Auto-split into {len(to_generate)} chunks."
                            return _wav_to_gradio_audio(merged_wav, sr), status
                        except Exception as e:
                            return None, (
                                f"Failed to read or use voice file. Check file format/content.\n"
                                f"(读取或使用音色文件失败，请检查文件格式或内容)\n"
                                f"{type(e).__name__}: {e}"
                            )

                    save_btn.click(save_prompt, inputs=[ref_audio_s, ref_text_s, xvec_only_s], outputs=[prompt_file_out, err2])
                    gen_btn2.click(load_prompt_and_gen, inputs=[prompt_file_in, text_in2, lang_in2, auto_split_in2], outputs=[audio_out2, err2])

        gr.Markdown(
            """
**Disclaimer (免责声明)**  
- The audio is automatically generated/synthesized by an AI model solely to demonstrate the model’s capabilities; it may be inaccurate or inappropriate, does not represent the views of the developer/operator, and does not constitute professional advice. You are solely responsible for evaluating, using, distributing, or relying on this audio; to the maximum extent permitted by applicable law, the developer/operator disclaims liability for any direct, indirect, incidental, or consequential damages arising from the use of or inability to use the audio, except where liability cannot be excluded by law. Do not use this service to intentionally generate or replicate unlawful, harmful, defamatory, fraudulent, deepfake, or privacy/publicity/copyright/trademark‑infringing content; if a user prompts, supplies materials, or otherwise facilitates any illegal or infringing conduct, the user bears all legal consequences and the developer/operator is not responsible.
- 音频由人工智能模型自动生成/合成，仅用于体验与展示模型效果，可能存在不准确或不当之处；其内容不代表开发者/运营方立场，亦不构成任何专业建议。用户应自行评估并承担使用、传播或依赖该音频所产生的一切风险与责任；在适用法律允许的最大范围内，开发者/运营方不对因使用或无法使用本音频造成的任何直接、间接、附带或后果性损失承担责任（法律另有强制规定的除外）。严禁利用本服务故意引导生成或复制违法、有害、诽谤、欺诈、深度伪造、侵犯隐私/肖像/著作权/商标等内容；如用户通过提示词、素材或其他方式实施或促成任何违法或侵权行为，相关法律后果由用户自行承担，与开发者/运营方无关。
"""
        )

    return demo


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.checkpoint and not args.checkpoint_pos:
        parser.print_help()
        return 0

    ckpt = _resolve_checkpoint(args)

    if args.dtype:
        dtype = _dtype_from_str(args.dtype)
    else:
        is_cpu_device = str(args.device).strip().lower().startswith("cpu")
        dtype = torch.float32 if is_cpu_device else torch.bfloat16
    attn_impl = "flash_attention_2" if args.flash_attn else None

    tts = Qwen3TTSModel.from_pretrained(
        ckpt,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    gen_kwargs_default = _collect_gen_kwargs(args)
    demo = build_demo(tts, ckpt, gen_kwargs_default)

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        ssl_verify=True if args.ssl_verify else False,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
