# Script for executing Post-training dynamic range quantization on Whisper Tiny.en

# Setup
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("/home/$USER/.cache/whisper/") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    tf.write(tflite_model)

