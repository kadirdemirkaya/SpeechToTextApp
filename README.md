# Speech to Text Application

This is a real-time speech-to-text transcription application with AI capabilities. The application uses the Whisper model for transcription and integrates with Gemini AI for advanced text analysis.

## Features

- 🎙️ Real-time speech-to-text transcription
- 📁 Audio file transcription support
- 🔊 Multiple audio input device support
- 🤖 AI-powered text analysis using Gemini AI
- 📝 Text embedding and semantic search capabilities
- 🔍 Question-answering about transcribed content
- 🌐 Support for both English and Turkish languages

## Requirements

- Python 3.8+
- PyAudio
- faster-whisper
- webrtcvad
- numpy
- scikit-learn
- tkinter
- requests

## Configuration

The application's configuration can be found in `config.py`:

```python
- API_KEY: Your Gemini AI API key
- RATE: Audio sampling rate (default: 16000)
- CHANNELS: Audio channels (default: 1)
- CHUNK: Audio chunk size
- DURATION: Recording duration in seconds
- MODEL_SIZE: Whisper model size ("tiny" by default)
```

## Usage

1. Launch the application:
```bash
python main.py
```

2. Select your input source:
   - Microphone: Choose your input device from the dropdown
   - File: Select an audio file to transcribe

3. Use the interface buttons:
   - 🎙️ Start: Begin recording/transcribing
   - 🛑 Stop: Stop the current session
   - 📜 Show All Text: View complete transcript
   - ❓ Ask Questions: Query the transcribed content

## Project Structure

```
speechToTextApp/
├── config.py         # Configuration settings
├── main.py          # Main application code
├── arrays.py        # Shared data structures
└── image/
    └── ss.png       # Application screenshots
```

## Technical Details

- Uses WebRTC VAD (Voice Activity Detection) for better speech detection
- Implements threading for non-blocking audio processing
- Employs queue system for audio data management
- Utilizes Whisper model for accurate speech recognition
- Integrates with Gemini AI for advanced text processing
- Implements text embedding for semantic search capabilities
