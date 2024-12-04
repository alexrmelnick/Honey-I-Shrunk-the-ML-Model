import torch
from torch.quantization import convert


class WrapperModel(torch.nn.Module):
    def __init__(self, model, decoder_start_token_id):
        super(WrapperModel, self).__init__()
        self.model = model
        self.decoder_start_token_id = decoder_start_token_id

    def forward(self, input_features):
        # Create a dummy decoder input
        batch_size = input_features.size(0)
        decoder_input_ids = torch.full(
            (batch_size, 1), self.decoder_start_token_id, dtype=torch.long
        )
        # Call the model with both encoder and decoder inputs
        outputs = self.model(input_features=input_features, decoder_input_ids=decoder_input_ids)
        return outputs.logits  # Return only the logits (primary output)


def convert_to_onnx(model_path, output_path, dummy_input_shape, decoder_start_token_id):
    """
    Converts a PyTorch model to ONNX format.

    Args:
        model_path (str): Path to the PyTorch model (.pth file).
        output_path (str): Path to save the ONNX model.
        dummy_input_shape (tuple): Shape of the dummy input tensor.
        decoder_start_token_id (int): ID of the start-of-sequence token for the decoder.
    """
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location="cpu")

    # Dequantize the model
    print("Dequantizing the model...")
    model = convert(model, inplace=False)
    model.eval()

    # Wrap the model
    model = WrapperModel(model, decoder_start_token_id)

    # Create a dummy input tensor
    dummy_input = torch.randn(*dummy_input_shape)

    # Export to ONNX
    print(f"Exporting model to {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=15,
        input_names=["input_features"],
        output_names=["logits"],
        dynamic_axes={"input_features": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print("Export complete.")


# Define paths and input shapes
model_paths = [
    ("Models/quantized_whisper_tiny_en/quantized_model.pth", "Models/quantized_whisper_tiny_en/quantized_model.onnx"),
    (
    "Models/quantized_whisper_base/quantized_model_base.pth", "Models/quantized_whisper_base/quantized_model_base.onnx")
]
dummy_input_shape = (1, 80, 3000)  # Batch size, features, sequence length
decoder_start_token_id = 50256  # Adjust this based on your model's config

for model_path, output_path in model_paths:
    convert_to_onnx(model_path, output_path, dummy_input_shape, decoder_start_token_id)
