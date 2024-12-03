import torch
import os

class WhisperWrapper(torch.nn.Module):
    """
    A wrapper to modify the forward method of the Whisper model
    for ONNX export compatibility.
    """
    def __init__(self, model):
        super(WhisperWrapper, self).__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        # Call the original forward method
        outputs = self.model(input_features=input_features, decoder_input_ids=decoder_input_ids)
        return outputs.logits  # Extract only the logits for ONNX export

def convert_to_onnx(model_path, output_path, dummy_input_shape, decoder_input_shape):
    """
    Convert a PyTorch model to ONNX format.

    Args:
        model_path (str): Path to the .pth file of the PyTorch model.
        output_path (str): Path to save the ONNX model.
        dummy_input_shape (tuple): Shape of the dummy encoder input tensor.
        decoder_input_shape (tuple): Shape of the dummy decoder input tensor.
    """
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # Wrap the model for ONNX export compatibility
    wrapped_model = WhisperWrapper(model)

    # Create dummy inputs for the encoder and decoder
    dummy_input_features = torch.randn(*dummy_input_shape)
    dummy_decoder_input_ids = torch.zeros(*decoder_input_shape, dtype=torch.long)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting model to {output_path}")
    torch.onnx.export(
        wrapped_model,
        (dummy_input_features, dummy_decoder_input_ids),
        output_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input_features", "decoder_input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_features": {0: "batch_size"}, "decoder_input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print(f"Model successfully converted to ONNX and saved as {output_path}")

# Define models and their dummy input shapes
models = [
    {
        "model_path": "Models/quantized_whisper_tiny_en/quantized_model.pth",
        "output_path": "Models/quantized_whisper_tiny_en/quantized_model.onnx",
        "dummy_input_shape": (1, 80, 3000),  # Encoder input shape: (batch_size, 80, 3000)
        "decoder_input_shape": (1, 1)       # Decoder input shape: (batch_size, sequence_length)
    },
    {
        "model_path": "Models/quantized_whisper_base/quantized_model_base.pth",
        "output_path": "Models/quantized_whisper_base/quantized_model_base.onnx",
        "dummy_input_shape": (1, 80, 3000),
        "decoder_input_shape": (1, 1)
    }
]

for model_info in models:
    convert_to_onnx(
        model_path=model_info["model_path"],
        output_path=model_info["output_path"],
        dummy_input_shape=model_info["dummy_input_shape"],
        decoder_input_shape=model_info["decoder_input_shape"]
    )
