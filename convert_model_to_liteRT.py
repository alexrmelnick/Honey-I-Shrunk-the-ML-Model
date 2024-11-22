import ai_edge_torch
import numpy
import torch
import torchvision

# Load the model
model = torch.load('Models/quantized_whisper_tiny_en.pth')

# Convert the model
edge_model = ai_edge_torch.convert()

# Save the model
edge_model.export('Models/quantized_whisper_tiny_en_liteRT.tflite')