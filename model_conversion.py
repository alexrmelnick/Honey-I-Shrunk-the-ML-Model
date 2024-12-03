import torch

def convert_to_onnx(model_path, output_path, dummy_input_shape):
    """
    Convert a PyTorch model to ONNX format.

    Args:
        model_path (str): Path to the .pth file of the PyTorch model.
        output_path (str): Path to save the ONNX model.
        dummy_input_shape (tuple): Shape of the dummy input tensor.
    """
    # Load the PyTorch model
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode

    # Create a dummy input
    dummy_input = torch.randn(*dummy_input_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,  # Store trained parameter weights inside the ONNX model
        opset_version=11,  # ONNX opset version
        do_constant_folding=True,  # Optimize constant folding for ONNX
        input_names=["input"],  # Name of the input tensor
        output_names=["output"],  # Name of the output tensor
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Support dynamic batch size
    )
    print(f"Model converted to ONNX and saved as: {output_path}")


# Paths to the model files and output ONNX files
models = [
    {
        "model_path": "Models/quantized_whisper_tiny_en/quantized_model.pth",
        "output_path": "Models/quantized_whisper_tiny_en/quantized_model.onnx",
        "dummy_input_shape": (1, 80, 2000)  # For 20 seconds of audio
    },
    {
        "model_path": "Models/quantized_whisper_base/quantized_model_base.pth",
        "output_path": "Models/quantized_whisper_base/quantized_model_base.onnx",
        "dummy_input_shape": (1, 80, 2000)  # For 20 seconds of audio
    }
]

# Convert each model to ONNX
for model_info in models:
    convert_to_onnx(
        model_path=model_info["model_path"],
        output_path=model_info["output_path"],
        dummy_input_shape=model_info["dummy_input_shape"]
    )
