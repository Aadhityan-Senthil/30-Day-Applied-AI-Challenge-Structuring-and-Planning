"""
Day 18: Speech to Text Transcription
30-Day AI Challenge

Convert speech audio to text using OpenAI's Whisper model.
Supports multiple audio formats and languages.
"""

import numpy as np
import wave
import json
import os

# Check for whisper availability
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Note: Install openai-whisper for full functionality")

# Alternative: Use speech_recognition library
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

def generate_sample_audio():
    """Generate a sample audio file with a simple tone for testing."""
    filename = "sample_audio.wav"
    
    # Audio parameters
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # Hz (A4 note)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate sine wave
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some variation (simulate speech-like amplitude modulation)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    audio = audio * modulation
    
    # Normalize and convert to 16-bit PCM
    audio = (audio * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    print(f"Generated sample audio: {filename}")
    return filename

def transcribe_with_whisper(audio_path, model_size="base"):
    """Transcribe audio using OpenAI Whisper."""
    if not WHISPER_AVAILABLE:
        return None, "Whisper not installed. Run: pip install openai-whisper"
    
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    
    return result, None

def transcribe_with_speech_recognition(audio_path):
    """Transcribe using Google Speech Recognition (free, requires internet)."""
    if not SR_AVAILABLE:
        return None, "speech_recognition not installed. Run: pip install SpeechRecognition"
    
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
        return {"text": text}, None
    except sr.UnknownValueError:
        return None, "Could not understand audio"
    except sr.RequestError as e:
        return None, f"API error: {e}"

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

def create_transcript_with_timestamps(result):
    """Create a formatted transcript with timestamps."""
    if 'segments' not in result:
        return result.get('text', '')
    
    transcript_lines = []
    for segment in result['segments']:
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip()
        transcript_lines.append(f"[{start} --> {end}] {text}")
    
    return '\n'.join(transcript_lines)

def create_srt_subtitle(result, output_file="subtitles.srt"):
    """Create SRT subtitle file from transcription."""
    if 'segments' not in result:
        print("No segments available for SRT generation")
        return
    
    srt_content = []
    for i, segment in enumerate(result['segments'], 1):
        start = format_timestamp(segment['start']).replace('.', ',')
        end = format_timestamp(segment['end']).replace('.', ',')
        text = segment['text'].strip()
        
        srt_content.append(f"{i}")
        srt_content.append(f"{start} --> {end}")
        srt_content.append(text)
        srt_content.append("")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(srt_content))
    
    print(f"Subtitles saved to '{output_file}'")

def analyze_audio_stats(audio_path):
    """Analyze basic audio file statistics."""
    with wave.open(audio_path, 'r') as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        n_frames = wav.getnframes()
        duration = n_frames / frame_rate
    
    return {
        'filename': audio_path,
        'channels': channels,
        'sample_width_bytes': sample_width,
        'sample_rate': frame_rate,
        'duration_seconds': round(duration, 2),
        'total_frames': n_frames
    }

class SpeechToText:
    """Speech to Text transcription class."""
    
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
        
    def load_model(self):
        """Load the Whisper model."""
        if WHISPER_AVAILABLE:
            print(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
            print("Model loaded!")
        else:
            print("Whisper not available, using speech_recognition fallback")
    
    def transcribe(self, audio_path, language=None):
        """Transcribe audio file."""
        if not os.path.exists(audio_path):
            return {"error": f"File not found: {audio_path}"}
        
        if self.model:
            options = {}
            if language:
                options['language'] = language
            result = self.model.transcribe(audio_path, **options)
            return result
        elif SR_AVAILABLE:
            result, error = transcribe_with_speech_recognition(audio_path)
            if error:
                return {"error": error}
            return result
        else:
            return {"error": "No transcription backend available"}
    
    def transcribe_to_file(self, audio_path, output_path="transcript.txt"):
        """Transcribe and save to file."""
        result = self.transcribe(audio_path)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        with open(output_path, 'w') as f:
            f.write(result.get('text', ''))
        
        print(f"Transcript saved to '{output_path}'")
        return result

def main():
    print("=" * 50)
    print("Day 18: Speech to Text Transcription")
    print("=" * 50)
    
    # Check available backends
    print("\n[1] Checking available backends...")
    print(f"  Whisper: {'✓ Available' if WHISPER_AVAILABLE else '✗ Not installed'}")
    print(f"  SpeechRecognition: {'✓ Available' if SR_AVAILABLE else '✗ Not installed'}")
    
    # Generate sample audio
    print("\n[2] Generating sample audio file...")
    audio_file = generate_sample_audio()
    
    # Analyze audio
    print("\n[3] Analyzing audio file...")
    stats = analyze_audio_stats(audio_file)
    print(f"  Duration: {stats['duration_seconds']} seconds")
    print(f"  Sample rate: {stats['sample_rate']} Hz")
    print(f"  Channels: {stats['channels']}")
    
    # Transcribe
    print("\n[4] Transcribing audio...")
    
    if WHISPER_AVAILABLE:
        result, error = transcribe_with_whisper(audio_file, "base")
        if error:
            print(f"  Error: {error}")
        else:
            print(f"  Detected language: {result.get('language', 'unknown')}")
            print(f"  Transcription: {result.get('text', 'No text')[:200]}")
            
            # Save results
            with open('transcription_result.json', 'w') as f:
                # Convert numpy types for JSON serialization
                json_result = {
                    'text': result.get('text', ''),
                    'language': result.get('language', ''),
                }
                json.dump(json_result, f, indent=2)
            print("\n  Results saved to 'transcription_result.json'")
    
    elif SR_AVAILABLE:
        result, error = transcribe_with_speech_recognition(audio_file)
        if error:
            print(f"  Error: {error}")
        else:
            print(f"  Transcription: {result.get('text', 'No text')}")
    
    else:
        print("  No transcription backend available!")
        print("  Install one of:")
        print("    pip install openai-whisper")
        print("    pip install SpeechRecognition")
    
    # Demo usage instructions
    print("\n[5] Usage with your own audio:")
    print("""
    # Using Whisper (recommended)
    import whisper
    model = whisper.load_model("base")  # or "small", "medium", "large"
    result = model.transcribe("your_audio.mp3")
    print(result["text"])
    
    # Using SpeechRecognition
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile("audio.wav") as source:
        audio = r.record(source)
    text = r.recognize_google(audio)
    """)
    
    print("\n" + "=" * 50)
    print("Day 18 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
