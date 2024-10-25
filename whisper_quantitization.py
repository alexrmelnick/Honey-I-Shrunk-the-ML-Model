# Written by Alex Melnick with the aid of GPT-4o and GitHub Copilot

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

# Load the Whisper model and processor
model_name = "openai/whisper-tiny.en"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# Set the model to evaluation mode before quantization
model.eval()

# Print the size of the original model
print(f"Original model size: {model.state_dict().__sizeof__() / 1e6:.2f} MB")

# Apply dynamic range quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,  # The model to quantize
    {torch.nn.Linear},  # Specify the types of layers to quantize (here, nn.Linear layers)
    dtype=torch.qint8  # Use int8 for dynamic quantization
)

# Print the size of the quantized model
print(f"Quantized model size: {quantized_model.state_dict().__sizeof__() / 1e6:.2f} MB")

# Load your test audio file
audio_file = "Whisper_Test.mp3"
waveform, sample_rate = torchaudio.load(audio_file)

# Resample the audio to 16000 Hz (the required input rate for Whisper models)
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Preprocess the audio input
input_features = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features

# Test the quantized model with the actual audio input
with torch.no_grad():
    generated_ids = quantized_model.generate(input_features)

# Decode the output to text
decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(f"Decoded output: {decoded_output}")

# Save the entire quantized model instead of using save_pretrained
save_dir = "Models/quantized_whisper_tiny_en"
os.makedirs(save_dir, exist_ok=True)

# Save the entire quantized model using torch.save()
torch.save(quantized_model, os.path.join(save_dir, "quantized_model.pth"))

# Save the processor (tokenizer and configuration)
processor.save_pretrained(save_dir)

print(f"Quantized model and processor saved to {save_dir}")
