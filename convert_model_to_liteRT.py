import ai_edge_torch
import torch
import numpy
import torchvision

resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()

sample_input = (torch.randn(1, 3, 224, 224),)
edge_model = ai_edge_torch.convert(resnet18.eval(), sample_input)

output = edge_model(*sample_inputs)


# import ai_edge_torch
# import torch

# # Load the model
# model = torch.load('Models/quantized_whisper_tiny_en/quantized_model.pth')

# # Convert the model
# print("Starting conversion...")
# edge_model = ai_edge_torch.convert()

# # Save the model
# print("Saving the converted model...")
# edge_model.export('Models/quantized_whisper_tiny_en_liteRT.tflite')
# print("Model saved successfully.")