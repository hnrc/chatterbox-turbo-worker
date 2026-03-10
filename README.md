# Chatterbox Turbo TTS Worker

RunPod Serverless worker for Chatterbox Turbo Text-to-Speech with voice cloning.

## Capabilities

- **Text-to-Speech** - Fast, natural speech synthesis (up to 6x real-time)
- **Voice Cloning** - Clone any voice from a reference audio sample
- **Paralinguistic Tags** - Native support for `[laugh]`, `[cough]`, `[chuckle]` etc.
- **Speed Control** - Adjust playback speed

## Deployment on RunPod

### 1. Fork this repo to your GitHub

### 2. Create Serverless Endpoint

1. Go to Serverless → New Endpoint
2. Source: GitHub repo URL
3. GPU: RTX 4090 or similar (8GB+ VRAM sufficient)
4. Max Workers: 1-2
5. Idle Timeout: 5 seconds
6. No network volume needed (model is ~3GB, included in Docker image)

### 3. Environment Variables

None required - model is bundled.

## API Usage

### Basic TTS (Default Voice)

```json
{
  "input": {
    "text": "Welcome to your daily wellness moment. Take a deep breath."
  }
}
```

### Voice Cloning

```json
{
  "input": {
    "text": "Hello, this is my cloned voice speaking.",
    "reference_audio_url": "https://example.com/my-voice-sample.wav"
  }
}
```

Or with base64 audio:

```json
{
  "input": {
    "text": "Hello, this is my cloned voice speaking.",
    "reference_audio_base64": "UklGRi..."
  }
}
```

### With Paralinguistic Tags

```json
{
  "input": {
    "text": "I can't believe it [laugh] that's amazing!"
  }
}
```

Supported tags: `[laugh]`, `[chuckle]`, `[cough]`, and more — placed inline in text.

### With Parameters

```json
{
  "input": {
    "text": "This is a calm meditation guide.",
    "temperature": 0.5,
    "speed": 0.9,
    "top_p": 0.8,
    "repetition_penalty": 1.1
  }
}
```

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `temperature` | 0.7 | 0.05-2.0 | Creativity/variability |
| `speed` | 1.0 | 0.5-2.0 | Playback speed |
| `min_p` | 0.05 | 0.0-1.0 | Minimum probability threshold |
| `top_p` | 0.8 | 0.0-1.0 | Top-p (nucleus) sampling |
| `top_k` | 50 | 0-1000 | Top-k sampling |
| `repetition_penalty` | 1.1 | 1.0-2.0 | Penalizes repetition |

## Response Format

```json
{
  "audio_base64": "UklGRi...",
  "sample_rate": 24000,
  "duration_seconds": 3.5,
  "text": "The synthesized text"
}
```

## Reference Audio Requirements

For best voice cloning results:
- WAV format (16-bit, mono)
- 5-15 seconds of clear speech
- Minimal background noise
- Single speaker only
- Natural speaking pace

## Local Development

```bash
# Build
docker build -t chatterbox-turbo-worker .

# Run (requires NVIDIA GPU)
docker run --gpus all chatterbox-turbo-worker
```

## License

MIT
