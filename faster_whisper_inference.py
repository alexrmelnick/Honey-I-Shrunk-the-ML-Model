from faster_whisper import WhisperModel

# Load the Faster-Whisper model
model_path = "tiny.en"  # Choose 'tiny', 'base', 'small', 'medium', or 'large'
model = WhisperModel(model_path, device="cpu", compute_type ="int8")  # Use CPU since you're on a Raspberry Pi

# Path to the WAV file
audio_file = "Whisper_Test.wav"

# Run transcription
segments, info = model.transcribe(audio_file)

# Display transcription results
print("Transcription:")
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")

