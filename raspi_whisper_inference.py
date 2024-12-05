import os
import torch
import torchaudio
from transformers import WhisperProcessor

# Paths to the model, processor, and test audio file
MODEL_PATH = "Models/quantized_whisper_tiny_en/quantized_model.pth"
PROCESSOR_PATH = "Models/quantized_whisper_tiny_en"
AUDIO_FILE = "Whisper_Test.mp3"

# Suppress warnings for deprecated features
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

# Check if the required files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Quantized model not found at {MODEL_PATH}")
if not os.path.exists(PROCESSOR_PATH):
    raise FileNotFoundError(f"Processor directory not found at {PROCESSOR_PATH}")
if not os.path.exists(AUDIO_FILE):
    raise FileNotFoundError(f"Audio file not found at {AUDIO_FILE}")

# Load the processor for preprocessing the audio input
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(PROCESSOR_PATH)

# Load the quantized model and move it to CPU
print("Loading quantized model...")
quantized_model = torch.load(MODEL_PATH, map_location="cpu")  # Use weights_only=False (default)
quantized_model.eval()  # Ensure the model is in evaluation mode

# Fix generation_config if necessary
if hasattr(quantized_model, "generation_config"):
    # Safely remove or recreate generation_config
    if hasattr(quantized_model.generation_config, "is_assistant"):
        delattr(quantized_model.generation_config, "is_assistant")
    else:
        from transformers import GenerationConfig
        quantized_model.generation_config = GenerationConfig.from_model_config(quantized_model.config)

# Load the test audio file
print(f"Loading audio file: {AUDIO_FILE}")
waveform, sample_rate = torchaudio.load(AUDIO_FILE)

# Resample the audio to 16000 Hz (required by Whisper models)
print("Resampling audio to 16 kHz...")
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Preprocess the audio into input features
print("Preprocessing audio...")
input_features = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features

# Perform inference with the quantized model
print("Running inference...")
with torch.no_grad():
    generated_ids = quantized_model.generate(input_features)

# Decode the generated IDs into text
print("Decoding transcription...")
decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)

# Display the transcription result
if decoded_output:
    print("\nDecoded Transcription:")
    print(decoded_output[0])
else:
    print("No output generated.")
