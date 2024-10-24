# Written by Alex Melnick with the aid of GPT-4o and GitHub Copilot

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load the saved quantized model state dict
quantized_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
quantized_model.load_state_dict(torch.load("quantized_whisper_tiny_en/pytorch_model_quantized.bin"))

# Load the processor
processor = WhisperProcessor.from_pretrained("quantized_whisper_tiny_en")

# Set to evaluation mode
quantized_model.eval()

# The model is now ready for inference
