# Day 18: Speech to Text Transcription

## Overview
Convert audio speech to text using OpenAI's Whisper model or Google Speech Recognition API.

## Features
- Multiple model sizes (tiny, base, small, medium, large)
- Multi-language support (auto-detection)
- Timestamp generation
- SRT subtitle export
- Audio file analysis

## Requirements
```bash
# Option 1: Whisper (recommended, offline)
pip install openai-whisper

# Option 2: SpeechRecognition (requires internet)
pip install SpeechRecognition

# For audio processing
pip install numpy
```

## Usage
```bash
python speech_to_text.py
```

## Whisper Models
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| tiny | 39M | Fastest | Basic |
| base | 74M | Fast | Good |
| small | 244M | Medium | Better |
| medium | 769M | Slow | Great |
| large | 1.5GB | Slowest | Best |

## Transcribe Your Audio
```python
import whisper
model = whisper.load_model("base")
result = model.transcribe("your_audio.mp3")
print(result["text"])
```

## Supported Formats
- WAV, MP3, MP4, M4A, FLAC, OGG, and more

## Output Files
- `sample_audio.wav` - Generated test audio
- `transcription_result.json` - Transcription with metadata
