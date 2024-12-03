import onnx
import onnxruntime
import numpy as np

def validate_onnx_model(onnx_path, dummy_input_shape):
    # Load the ONNX model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)  # Check for any structural issues
    print(f"ONNX model {onnx_path} is valid.")

    # Test inference with ONNX Runtime
    session = onnxruntime.InferenceSession(onnx_path)
    dummy_input = np.random.randn(*dummy_input_shape).astype(np.float32)
    inputs = {"input": dummy_input}
    outputs = session.run(None, inputs)

    print(f"ONNX model {onnx_path} inference successful with output shape: {outputs[0].shape}")


# Validate the converted models
for model_info in models:
    validate_onnx_model(
        onnx_path=model_info["output_path"],
        dummy_input_shape=model_info["dummy_input_shape"]
    )
