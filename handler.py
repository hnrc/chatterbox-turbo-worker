"""
RunPod Serverless Handler for Chatterbox Turbo
Text-to-Speech with voice cloning and paralinguistic tags

Based on: https://github.com/geronimi73/runpod_chatterbox
"""

import runpod
import base64
import io
import os
import tempfile
import urllib.request
from typing import Optional

import torch
import soundfile as sf
import numpy as np

# Global model instance (loaded once per worker)
tts_model = None
_orig_prepare = None


def _patch_prepare_conditionals():
    """Monkey-patch prepare_conditionals to cast all outputs to float32.

    librosa.load returns float64 numpy → torch.from_numpy preserves float64
    → dtype mismatch with float32 model weights during inference.
    This patch catches every call (built-in conds AND per-request voice cloning).
    """
    global _orig_prepare
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if _orig_prepare is not None:
        return  # already patched

    _orig_prepare = ChatterboxTurboTTS.prepare_conditionals

    def _patched(self, *args, **kwargs):
        conds = _orig_prepare(self, *args, **kwargs)
        if hasattr(conds, 't3'):
            for field in vars(conds.t3):
                v = getattr(conds.t3, field, None)
                if torch.is_tensor(v) and v.is_floating_point():
                    setattr(conds.t3, field, v.float())
        if hasattr(conds, 'gen') and isinstance(conds.gen, dict):
            for k, v in conds.gen.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    conds.gen[k] = v.float()
        return conds

    ChatterboxTurboTTS.prepare_conditionals = _patched
    print("[Handler] Patched prepare_conditionals for float32 dtype consistency")


def load_model():
    """Load Chatterbox Turbo model."""
    global tts_model

    if tts_model is not None:
        return tts_model

    print("[Handler] Loading Chatterbox Turbo model...")

    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from huggingface_hub import snapshot_download

    _patch_prepare_conditionals()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Handler] Using device: {device}")

    # Download with token=False (model is public, avoids auth requirement)
    local_path = snapshot_download(
        repo_id="ResembleAI/chatterbox-turbo",
        token=False,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    )
    tts_model = ChatterboxTurboTTS.from_local(local_path, device=device)

    print("[Handler] Model loaded successfully")
    return tts_model


def download_reference_audio(url: str) -> str:
    """Download reference audio from URL to temp file."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    try:
        urllib.request.urlretrieve(url, temp_file.name)
        return temp_file.name
    except Exception as e:
        os.unlink(temp_file.name)
        raise Exception(f"Failed to download reference audio: {e}")


def base64_to_audio_file(b64_data: str) -> str:
    """Convert base64 audio to temp file."""
    # Remove data URL prefix if present
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]

    audio_bytes = base64.b64decode(b64_data)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


def audio_to_base64(audio_array: np.ndarray, sample_rate: int = 24000) -> str:
    """Convert audio array to base64 encoded WAV."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def parse_emotion_tags(text: str) -> tuple[str, Optional[str]]:
    """
    Parse emotion tags from text.
    Format: [happy] Hello there! or <emotion:sad> How are you?
    Returns: (clean_text, emotion)
    """
    import re

    # Check for [emotion] format
    bracket_match = re.match(r"^\[(\w+)\]\s*(.+)$", text, re.DOTALL)
    if bracket_match:
        return bracket_match.group(2).strip(), bracket_match.group(1).lower()

    # Check for <emotion:name> format
    tag_match = re.match(r"^<emotion:(\w+)>\s*(.+)$", text, re.DOTALL)
    if tag_match:
        return tag_match.group(2).strip(), tag_match.group(1).lower()

    return text, None


def handler(job: dict) -> dict:
    """Main RunPod handler function."""

    job_input = job.get("input", {})

    # Health check - respond immediately without generating audio
    if job_input.get("health_check"):
        return {
            "status": "healthy",
            "message": "Chatterbox Turbo handler ready",
            "model_loaded": tts_model is not None
        }

    # Required: text to synthesize
    text = job_input.get("text", "")
    if not text:
        return {"error": "No text provided"}

    # Optional: reference audio for voice cloning
    reference_audio_url = job_input.get("reference_audio_url")
    reference_audio_base64 = job_input.get("reference_audio_base64")

    # Optional: emotion override (if not in text tags)
    emotion = job_input.get("emotion")

    # Optional: generation parameters
    temperature = job_input.get("temperature", 0.7)
    speed = job_input.get("speed", 1.0)
    min_p = job_input.get("min_p", 0.05)
    top_p = job_input.get("top_p", 0.8)
    top_k = job_input.get("top_k", 50)
    repetition_penalty = job_input.get("repetition_penalty", 1.1)

    # Backward compat: accept but ignore old params
    exaggeration = job_input.get("exaggeration")
    cfg_weight = job_input.get("cfg_weight")

    # Parse emotion from text if not provided
    clean_text, text_emotion = parse_emotion_tags(text)
    if text_emotion and not emotion:
        emotion = text_emotion
        text = clean_text

    try:
        # Load model
        model = load_model()

        # Handle reference audio
        ref_audio_path = None

        if reference_audio_url:
            print(f"[Handler] Downloading reference audio from URL...")
            ref_audio_path = download_reference_audio(reference_audio_url)
        elif reference_audio_base64:
            print(f"[Handler] Decoding reference audio from base64...")
            ref_audio_path = base64_to_audio_file(reference_audio_base64)

        # Generate speech
        print(f"[Handler] Generating speech for: {text[:50]}...")
        print(f"[Handler] Emotion: {emotion}, Temp: {temperature}, Speed: {speed}")

        turbo_params = dict(
            temperature=temperature,
            min_p=min_p,
            top_p=top_p,
            top_k=int(top_k),
            repetition_penalty=repetition_penalty,
        )

        if ref_audio_path:
            # Voice cloning mode
            audio = model.generate(
                text=text,
                audio_prompt_path=ref_audio_path,
                **turbo_params,
            )
            # Clean up temp file
            os.unlink(ref_audio_path)
        else:
            # Default voice mode
            audio = model.generate(
                text=text,
                **turbo_params,
            )

        # Apply speed adjustment if not 1.0
        if speed != 1.0:
            from scipy import signal
            # Resample to adjust speed
            original_length = len(audio)
            new_length = int(original_length / speed)
            audio = signal.resample(audio, new_length)

        # Convert to numpy if tensor
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()

        # Ensure 1D
        if len(audio.shape) > 1:
            audio = audio.squeeze()

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.95

        # Convert to base64
        audio_b64 = audio_to_base64(audio, sample_rate=24000)

        return {
            "audio_base64": audio_b64,
            "sample_rate": 24000,
            "duration_seconds": len(audio) / 24000,
            "text": text,
            "emotion": emotion,
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Pre-load model on worker start
print("[Handler] Pre-loading model...")
try:
    load_model()
except Exception as e:
    print(f"[Handler] Warning: Could not pre-load model: {e}")

# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
