# Written by Alex Melnick with the aid of GPT-4o

import torch
import torchaudio
from transformers import WhisperProcessor

# Step 1: Load the quantized model and processor
quantized_model = torch.load("Models/quantized_whisper_tiny_en/quantized_model.pth")

# Load the processor for pre-processing the audio input
processor = WhisperProcessor.from_pretrained("Models/quantized_whisper_tiny_en")

# Set the quantized model to evaluation mode
quantized_model.eval()

# Step 2: Load the test audio file "Whisper_Test.mp3"
audio_file = "Whisper_Test.mp3"
waveform, sample_rate = torchaudio.load(audio_file)

# Step 3: Resample the audio to 16000 Hz (Whisper models expect 16kHz audio input)
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Step 4: Preprocess the audio into input features that the Whisper model can understand
input_features = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features

# Step 5: Run inference with the quantized model to generate transcription
with torch.no_grad():
    generated_ids = quantized_model.generate(input_features)

# Step 6: Decode the generated IDs into text using the processor
decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)

# Print the transcription result
print(f"Decoded output: {decoded_output}")



