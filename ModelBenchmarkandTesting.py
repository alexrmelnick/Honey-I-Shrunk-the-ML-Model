# Written by Sergio Rodriguez with the aid of GPT-4o

import torch
import torchaudio
import soundfile as sf
from transformers import WhisperProcessor
from torch.utils.data import DataLoader
import psutil
import time
import os
import numpy as np

# Ensure the data directory exists
os.makedirs("./data", exist_ok=True)

# Limit PyTorch to use a single CPU core to simulate Raspberry Pi's performance
torch.set_num_threads(1)

# Allocate a significant portion of memory to simulate a low-memory environment
dummy_memory = np.zeros((500 * 1024 * 1024), dtype='uint8')  # Allocate around 500MB

# Load the quantized model and processor
quantized_model = torch.load("Models/quantized_whisper_tiny_en/quantized_model.pth")
processor = WhisperProcessor.from_pretrained("Models/quantized_whisper_tiny_en")
quantized_model.eval()

# Load LibriSpeech dataset (test-clean subset for accuracy evaluation)
dataset = torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# Function to measure memory and CPU usage
def measure_memory_and_cpu_usage(func, *args, **kwargs):
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    start_time = time.time()

    # Measure CPU usage during execution
    with process.oneshot():
        start_cpu_times = process.cpu_times()
    result = func(*args, **kwargs)
    with process.oneshot():
        end_cpu_times = process.cpu_times()

    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB

    memory_used = end_memory - start_memory
    time_taken = end_time - start_time

    cpu_user_time = end_cpu_times.user - start_cpu_times.user
    cpu_system_time = end_cpu_times.system - start_cpu_times.system

    return result, memory_used, time_taken, cpu_user_time, cpu_system_time
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)  # Start memory in MB
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 * 1024)  # End memory in MB

    memory_used = end_memory - start_memory
    time_taken = end_time - start_time
    return result, memory_used, time_taken


# Iterate over dataset and perform inference to measure memory usage and accuracy
total_correct = 0
total_samples = 0


def trim_waveform(waveform, sample_rate, duration_sec):
    num_samples = int(sample_rate * duration_sec)
    return waveform[:, :num_samples]


for i, data in enumerate(dataloader):
    waveform, sample_rate = data[0], data[1]
    # Limit the number of samples to process
    if i >= 100:
        break

    # Trim waveform to desired length (10, 15, or 20 seconds)
    duration_sec = [10, 15, 20][i % 3]  # Cycle through 10, 15, and 20 seconds
    waveform = trim_waveform(waveform, sample_rate, duration_sec)

    # Resample to 16kHz (Whisper model requirement)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

    # Preprocess audio input
    input_features = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features

    # Measure memory usage and time taken during inference
    with torch.no_grad():
        from transformers import GenerationConfig

        generation_config = GenerationConfig()
        generated_ids, memory_used, time_taken, cpu_user_time, cpu_system_time = measure_memory_and_cpu_usage(
            quantized_model.generate, input_features, generation_config=generation_config)

        # Decode generated IDs into text
        decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"Decoded output: {decoded_output}")
        print(f"Memory used: {memory_used:.2f} MB, Time taken: {time_taken:.2f} seconds")
        print(f"CPU user time: {cpu_user_time:.2f} seconds, CPU system time: {cpu_system_time:.2f} seconds")

        # Accuracy calculation would require ground truth comparison (not included here for brevity)
        total_samples += 1

# Print summary of results
print(f"Total samples processed: {total_samples}")
