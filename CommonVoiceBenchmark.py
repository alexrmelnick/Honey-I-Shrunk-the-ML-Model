import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import psutil
import time
import gc
import numpy as np
import jiwer
from tabulate import tabulate

# Load the quantized model and processor
quantized_model = torch.load("Models/quantized_whisper_base/quantized_model_base.pth")
processor = WhisperProcessor.from_pretrained("Models/quantized_whisper_base")
quantized_model.eval()

# Load Common Voice dataset (test split)
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test[:300]")

# Variables to track progress
total_samples_10s = 0
total_samples_15s = 0
total_samples_20s = 0

wer_sum_10s, wer_sum_15s, wer_sum_20s = 0.0, 0.0, 0.0
wer_10s, wer_15s, wer_20s = [], [], []
memory_usage_10s, memory_usage_15s, memory_usage_20s = [], [], []


# Preprocessing function for the dataset
def preprocess_common_voice(sample):
    global total_samples_10s, total_samples_15s, total_samples_20s
    # Load waveform and resample to 16kHz
    waveform, sample_rate = torchaudio.load(sample["path"])
    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    
    # Trim the waveform to match durations of 10, 15, or 20 seconds
    if total_samples_10s < 100:
        duration_sec = 10
        total_samples_10s += 1
    elif total_samples_15s < 100:
        duration_sec = 15
        total_samples_15s += 1
    elif total_samples_20s < 100:
        duration_sec = 20
        total_samples_20s += 1
    else:
        return None  # Stop processing once all categories are full
    
    waveform = waveform[:, :16000 * duration_sec]  # Trim the waveform to desired length
    return {
        "waveform": waveform,
        "transcript": sample["sentence"],
        "sample_rate": 16000
    }


# Apply preprocessing
dataset = dataset.map(preprocess_common_voice, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda x: x is not None)  # Remove None values after preprocessing


# Convert to PyTorch DataLoader
def data_collate_fn(batch):
    waveforms = [item["waveform"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    return waveforms, transcripts


dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=data_collate_fn)


# Function to measure memory and CPU usage
def measure_memory_and_cpu_usage(func, *args, **kwargs):
    process = psutil.Process()
    gc.collect()
    start_time = time.time()

    result = func(*args, **kwargs)
    end_time = time.time()

    memory_info = process.memory_info()
    memory_used = memory_info.rss / (1024 * 1024)  # Convert memory usage to MB
    time_taken = end_time - start_time
    return result, memory_used, time_taken


# Inference loop
for i, (waveforms, transcripts) in enumerate(dataloader):
    waveform = waveforms[0]  # Only one sample per batch
    transcript = transcripts[0]

    # Preprocess audio input
    input_features = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features

    # Measure memory usage and time taken during inference
    with torch.no_grad():
        generation_config = GenerationConfig()
        generated_ids, memory_used, time_taken = measure_memory_and_cpu_usage(
            quantized_model.generate, input_features, generation_config=generation_config
        )

        # Decode generated IDs into text
        decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Calculate Word Error Rate (WER)
        ground_truth_normalized = jiwer.transforms.RemovePunctuation()(transcript.lower().strip())
        predicted_text_normalized = jiwer.transforms.RemovePunctuation()(decoded_output.lower().strip())
        wer = jiwer.wer(ground_truth_normalized, predicted_text_normalized)

        # Log memory and WER based on duration
        if total_samples_10s <= 100:
            memory_usage_10s.append(memory_used)
            wer_10s.append(wer)
            wer_sum_10s += wer
        elif total_samples_15s <= 100:
            memory_usage_15s.append(memory_used)
            wer_15s.append(wer)
            wer_sum_15s += wer
        elif total_samples_20s <= 100:
            memory_usage_20s.append(memory_used)
            wer_20s.append(wer)
            wer_sum_20s += wer


# Summary tables
memory_summary = [
    ["10-second recordings", np.mean(memory_usage_10s), np.median(memory_usage_10s),
     min(memory_usage_10s), max(memory_usage_10s)],
    ["15-second recordings", np.mean(memory_usage_15s), np.median(memory_usage_15s),
     min(memory_usage_15s), max(memory_usage_15s)],
    ["20-second recordings", np.mean(memory_usage_20s), np.median(memory_usage_20s),
     min(memory_usage_20s), max(memory_usage_20s)],
]

wer_summary = [
    ["10-second recordings", wer_sum_10s / total_samples_10s, np.median(wer_10s),
     min(wer_10s), max(wer_10s)],
    ["15-second recordings", wer_sum_15s / total_samples_15s, np.median(wer_15s),
     min(wer_15s), max(wer_15s)],
    ["20-second recordings", wer_sum_20s / total_samples_20s, np.median(wer_20s),
     min(wer_20s), max(wer_20s)],
]

print("Memory Usage Summary:")
print(tabulate(memory_summary, headers=["Recording Duration", "Average (MB)", "Median (MB)", "Min (MB)", "Max (MB)"], tablefmt="pretty"))

print("WER Summary:")
print(tabulate(wer_summary, headers=["Recording Duration", "Average WER", "Median WER", "Min WER", "Max WER"], tablefmt="pretty"))
