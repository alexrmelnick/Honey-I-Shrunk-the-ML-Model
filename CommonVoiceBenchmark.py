import os
import torch
import numpy as np
from datasets import load_dataset
from torchaudio.transforms import Resample
from transformers import WhisperProcessor
from torch.utils.data import DataLoader
import psutil
import time
import gc
import tracemalloc
from tabulate import tabulate
import jiwer

# Ensure the data directory exists
os.makedirs("./data", exist_ok=True)

# Limit PyTorch to use a single CPU core to simulate Raspberry Pi's performance
torch.set_num_threads(1)

# Allocate a significant portion of memory to simulate a low-memory environment
dummy_memory = np.zeros((500 * 1024 * 1024), dtype='uint8')  # Allocate around 500MB

# Load the quantized model and processor
quantized_model = torch.load("Models/quantized_whisper_base/quantized_model_base.pth")
processor = WhisperProcessor.from_pretrained("Models/quantized_whisper_base")
quantized_model.eval()

# Load the Common Voice dataset (test set)
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test[:300]")

# Preprocessing function
def preprocess_common_voice(sample):
    waveform = torch.tensor(sample["audio"]["array"])  # Convert waveform to tensor
    sample_rate = sample["audio"]["sampling_rate"]
    transcript = sample["sentence"]
    return {"waveform": waveform, "sample_rate": sample_rate, "transcript": transcript}

# Preprocess the dataset
dataset = dataset.map(preprocess_common_voice, remove_columns=dataset.column_names)

# Convert to DataLoader
def data_collate_fn(batch):
    waveforms = [torch.tensor(item["waveform"]) for item in batch]  # Ensure tensors
    sample_rates = [item["sample_rate"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    return waveforms, sample_rates, transcripts

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=data_collate_fn)

# Inference loop
total_samples_10s = total_samples_15s = total_samples_20s = 0
wer_sum_10s = wer_sum_15s = wer_sum_20s = 0.0
memory_usage_10s, memory_usage_15s, memory_usage_20s = [], [], []

for i, (waveforms, sample_rates, transcripts) in enumerate(dataloader):
    if i >= 300:  # Limit to 300 samples
        break

    waveform = waveforms[0]
    sample_rate = sample_rates[0]
    transcript = transcripts[0]

    # Assign duration based on sample count
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
        break

    # Ensure waveform is padded or trimmed to required duration
    required_samples = 16000 * duration_sec
    if waveform.numel() < required_samples:
        padding = required_samples - waveform.numel()
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:required_samples]

    # Resample to 16 kHz
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    # Preprocess audio input
    input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features

    # Perform inference
    with torch.no_grad():
        generated_ids = quantized_model.generate(input_features)
        decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Calculate Word Error Rate (WER)
        ground_truth_normalized = jiwer.transforms.RemovePunctuation()(transcript.lower().strip())
        predicted_text_normalized = jiwer.transforms.RemovePunctuation()(decoded_output.lower().strip())
        wer = jiwer.wer(ground_truth_normalized, predicted_text_normalized)

        # Log memory and WER for the current duration
        if duration_sec == 10:
            memory_usage_10s.append(wer)
        elif duration_sec == 15:
            memory_usage_15s.append(wer)
        elif duration_sec == 20:
            memory_usage_20s.append(wer)


# Prepare summary tables for memory usage and WER
memory_summary = [
    ["10-second recordings", np.mean(memory_usage_10s), np.median(memory_usage_10s), min(memory_usage_10s), max(memory_usage_10s)],
    ["15-second recordings", np.mean(memory_usage_15s), np.median(memory_usage_15s), min(memory_usage_15s), max(memory_usage_15s)],
    ["20-second recordings", np.mean(memory_usage_20s), np.median(memory_usage_20s), min(memory_usage_20s), max(memory_usage_20s)],
]

wer_summary = [
    ["10-second recordings", wer_sum_10s / total_samples_10s, np.median(wer_10s), min(wer_10s), max(wer_10s)],
    ["15-second recordings", wer_sum_15s / total_samples_15s, np.median(wer_15s), min(wer_15s), max(wer_15s)],
    ["20-second recordings", wer_sum_20s / total_samples_20s, np.median(wer_20s), min(wer_20s), max(wer_20s)],
]

# Print tables
print("Memory Usage Summary:")
print(tabulate(memory_summary, headers=["Recording Duration", "Average (MB)", "Median (MB)", "Lowest (MB)", "Highest (MB)"], tablefmt="pretty"))

print("Word Error Rate (WER) Summary:")
print(tabulate(wer_summary, headers=["Recording Duration", "Average WER", "Median WER", "Lowest WER", "Highest WER"], tablefmt="pretty"))
