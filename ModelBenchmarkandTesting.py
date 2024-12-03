# Written by Sergio Rodriguez with the aid of GPT-4o

import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset
from torchaudio.transforms import Resample
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from torch.utils.data import DataLoader
import psutil
import time
import os
import numpy as np
import jiwer
import gc
import tracemalloc
from tabulate import tabulate


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

# model_name = "openai/whisper-base"
# original_model = WhisperForConditionalGeneration.from_pretrained(model_name)
# processor = WhisperProcessor.from_pretrained(model_name)
# original_model.eval()

# Load LibriSpeech dataset (test-clean subset for accuracy evaluation)
dataset = torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# Function to measure memory and CPU usage
def measure_memory_and_cpu_usage(func, *args, **kwargs):
    process = psutil.Process()
    gc.collect()
    tracemalloc.start()
    start_time = time.time()

    # Measure CPU usage during execution
    with process.oneshot():
        start_cpu_times = process.cpu_times()
    result = func(*args, **kwargs)
    with process.oneshot():
        end_cpu_times = process.cpu_times()

    end_time = time.time()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_used = peak_memory / (1024 * 1024)  # Convert peak memory usage to MB
    time_taken = end_time - start_time

    cpu_user_time = end_cpu_times.user - start_cpu_times.user
    cpu_system_time = end_cpu_times.system - start_cpu_times.system

    return result, abs(memory_used), time_taken, cpu_user_time, cpu_system_time


# Iterate over dataset and perform inference to measure memory usage and accuracy
total_correct = 0
total_samples_10s = 0
total_samples = 0
total_samples_15s = 0
total_samples_20s = 0

wer_sum_10s = 0.0
wer_sum_15s = 0.0
wer_sum_20s = 0.0

wer_10s = []
wer_15s = []
wer_20s = []

memory_usage_10s = []
memory_usage_15s = []
memory_usage_20s = []


def trim_waveform(waveform, sample_rate, duration_sec):
    num_samples = int(sample_rate * duration_sec)
    return waveform[:, :num_samples]


for i, data in enumerate(dataloader):
    if i >= 300:  # Limit to 300 samples in total, 100 for each duration
        break
    waveform, sample_rate, transcript = data[0], data[1], data[2]

    # Trim waveform to desired length (10, 15, or 20 seconds)
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
        break  # Cycle through 10, 15, and 20 seconds
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

        # Collect memory usage for different durations
        if duration_sec == 10:
            memory_usage_10s.append(memory_used)
        elif duration_sec == 15:
            memory_usage_15s.append(memory_used)
        elif duration_sec == 20:
            memory_usage_20s.append(memory_used)
        print(f"CPU user time: {cpu_user_time:.2f} seconds, CPU system time: {cpu_system_time:.2f} seconds")

        # Calculate accuracy using Word Error Rate (WER)
        ground_truth = transcript[0]
        predicted_text = decoded_output[0]

        # Normalize texts for comparison
        ground_truth_normalized = jiwer.transforms.RemovePunctuation()(ground_truth.lower().strip())
        predicted_text_normalized = jiwer.transforms.RemovePunctuation()(predicted_text.lower().strip())

        # Calculate WER
        wer = jiwer.wer(ground_truth_normalized, predicted_text_normalized)
        print(f"Word Error Rate (WER): {wer:.2f}")

        # Accumulate WER for each duration
        if duration_sec == 10:
            wer_sum_10s += wer
            wer_10s.append(wer)
        elif duration_sec == 15:
            wer_sum_15s += wer
            wer_15s.append(wer)
        elif duration_sec == 20:
            wer_sum_20s += wer
            wer_20s.append(wer)

        total_samples += 1



# Prepare summary tables for memory usage and WER
memory_summary = [
    ["10-second recordings", np.mean(memory_usage_10s) if memory_usage_10s else 0,
     np.median(memory_usage_10s) if memory_usage_10s else 0, min(memory_usage_10s) if memory_usage_10s else 0,
     max(memory_usage_10s) if memory_usage_10s else 0],
    ["15-second recordings", np.mean(memory_usage_15s) if memory_usage_15s else 0,
     np.median(memory_usage_15s) if memory_usage_15s else 0, min(memory_usage_15s) if memory_usage_15s else 0,
     max(memory_usage_15s) if memory_usage_15s else 0],
    ["20-second recordings", np.mean(memory_usage_20s) if memory_usage_20s else 0,
     np.median(memory_usage_20s) if memory_usage_20s else 0, min(memory_usage_20s) if memory_usage_20s else 0,
     max(memory_usage_20s) if memory_usage_20s else 0]
]

wer_summary = [
    ["10-second recordings", wer_sum_10s / total_samples_10s if total_samples_10s > 0 else 0,
     np.median(wer_10s) if wer_10s else 0, min(wer_10s) if wer_10s else 0, max(wer_10s) if wer_10s else 0],
    ["15-second recordings", wer_sum_15s / total_samples_15s if total_samples_15s > 0 else 0,
     np.median(wer_15s) if wer_15s else 0, min(wer_15s) if wer_15s else 0, max(wer_15s) if wer_15s else 0],
    ["20-second recordings", wer_sum_20s / total_samples_20s if total_samples_20s > 0 else 0,
     np.median(wer_20s) if wer_20s else 0, min(wer_20s) if wer_20s else 0, max(wer_20s) if wer_20s else 0]
]

# Print tables
print("Memory Usage Summary: ")
print(tabulate(memory_summary,
               headers=["Recording Duration", "Average (MB)", "Median (MB)", "Lowest (MB)", "Highest (MB)"],
               tablefmt="pretty"))

print("Word Error Rate (WER) Summary:")
print(tabulate(wer_summary, headers=["Recording Duration", "Average WER", "Median WER", "Lowest WER", "Highest WER"],
               tablefmt="pretty"))
print(f"Total samples processed for 10-second recordings: {total_samples_10s}")
print(f"Average WER for 10-second recordings: {wer_sum_10s / total_samples_10s if total_samples_10s > 0 else 0:.2f}")

print(f"Total samples processed for 15-second recordings: {total_samples_15s}")
print(f"Average WER for 15-second recordings: {wer_sum_15s / total_samples_15s if total_samples_15s > 0 else 0:.2f}")

print(f"Total samples processed for 20-second recordings: {total_samples_20s}")
print(f"Average WER for 20-second recordings: {wer_sum_20s / total_samples_20s if total_samples_20s > 0 else 0:.2f}")



"""

Quantized Tiny_en:
Memory Usage Summary: 
+----------------------+---------------------+---------------------+----------------------+---------------------+
|  Recording Duration  |    Average (MB)     |     Median (MB)     |     Lowest (MB)      |    Highest (MB)     |
+----------------------+---------------------+---------------------+----------------------+---------------------+
| 10-second recordings | 0.06741405487060546 | 0.06396675109863281 | 0.05375194549560547  | 0.5165681838989258  |
| 15-second recordings | 0.0626408576965332  | 0.06392478942871094 | 0.054566383361816406 | 0.0659322738647461  |
| 20-second recordings | 0.06368687629699707 | 0.06396675109863281 | 0.05584526062011719  | 0.06632137298583984 |
+----------------------+---------------------+---------------------+----------------------+---------------------+
Word Error Rate (WER) Summary:
+----------------------+---------------------+--------------------+------------+--------------------+
|  Recording Duration  |     Average WER     |     Median WER     | Lowest WER |    Highest WER     |
+----------------------+---------------------+--------------------+------------+--------------------+
| 10-second recordings | 0.3193642061563498  | 0.3181818181818182 |    0.0     | 0.8070175438596491 |
| 15-second recordings | 0.3201951244329157  | 0.2426470588235294 |    0.0     | 0.8472222222222222 |
| 20-second recordings | 0.35201279216441017 | 0.3137254901960784 |    0.0     | 0.8666666666666667 |
+----------------------+---------------------+--------------------+------------+--------------------+
Total samples processed for 10-second recordings: 100
Average WER for 10-second recordings: 0.32
Total samples processed for 15-second recordings: 100
Average WER for 15-second recordings: 0.32
Total samples processed for 20-second recordings: 100
Average WER for 20-second recordings: 0.35


Tiny_en:
Memory Usage Summary: 
+----------------------+---------------------+----------------------+----------------------+---------------------+
|  Recording Duration  |    Average (MB)     |     Median (MB)      |     Lowest (MB)      |    Highest (MB)     |
+----------------------+---------------------+----------------------+----------------------+---------------------+
| 10-second recordings | 0.06062358856201172 | 0.057627201080322266 | 0.04269695281982422  | 0.5117874145507812  |
| 15-second recordings | 0.05559160232543945 | 0.057435035705566406 | 0.043735504150390625 | 0.06049919128417969 |
| 20-second recordings | 0.05714850425720215 | 0.05777311325073242  | 0.045706748962402344 | 0.06101512908935547 |
+----------------------+---------------------+----------------------+----------------------+---------------------+
Word Error Rate (WER) Summary:
+----------------------+---------------------+--------------------+------------+--------------------+
|  Recording Duration  |     Average WER     |     Median WER     | Lowest WER |    Highest WER     |
+----------------------+---------------------+--------------------+------------+--------------------+
| 10-second recordings | 0.31539303125666607 | 0.3330039525691699 |    0.0     | 0.8070175438596491 |
| 15-second recordings | 0.3237399251359314  | 0.2620137299771167 |    0.0     | 0.8301886792452831 |
| 20-second recordings | 0.34300472079627176 | 0.3021739130434783 |    0.0     |        0.85        |
+----------------------+---------------------+--------------------+------------+--------------------+
Total samples processed for 10-second recordings: 100
Average WER for 10-second recordings: 0.32
Total samples processed for 15-second recordings: 100
Average WER for 15-second recordings: 0.32
Total samples processed for 20-second recordings: 100
Average WER for 20-second recordings: 0.34

Whisper Base:
Memory Usage Summary: 
+----------------------+----------------------+----------------------+----------------------+---------------------+
|  Recording Duration  |     Average (MB)     |     Median (MB)      |     Lowest (MB)      |    Highest (MB)     |
+----------------------+----------------------+----------------------+----------------------+---------------------+
| 10-second recordings | 0.06762524604797364  | 0.06028032302856445  | 0.043845176696777344 | 0.5093307495117188  |
| 15-second recordings | 0.058406105041503904 | 0.05909252166748047  | 0.046527862548828125 | 0.06310081481933594 |
| 20-second recordings | 0.059731788635253906 | 0.059627532958984375 | 0.048150062561035156 | 0.06374645233154297 |
+----------------------+----------------------+----------------------+----------------------+---------------------+
Word Error Rate (WER) Summary:
+----------------------+---------------------+---------------------+------------+--------------------+
|  Recording Duration  |     Average WER     |     Median WER      | Lowest WER |    Highest WER     |
+----------------------+---------------------+---------------------+------------+--------------------+
| 10-second recordings | 0.36843071678282896 | 0.4105263157894737  |    0.0     |        1.5         |
| 15-second recordings | 0.3629425866573451  | 0.3514705882352941  |    0.0     | 0.8666666666666667 |
| 20-second recordings | 0.39287492694984216 | 0.36363636363636365 |    0.0     |        0.9         |
+----------------------+---------------------+---------------------+------------+--------------------+
Total samples processed for 10-second recordings: 100
Average WER for 10-second recordings: 0.37
Total samples processed for 15-second recordings: 100
Average WER for 15-second recordings: 0.36
Total samples processed for 20-second recordings: 100
Average WER for 20-second recordings: 0.39

Quantized Whisper Base:
Memory Usage Summary: 
+----------------------+---------------------+---------------------+----------------------+---------------------+
|  Recording Duration  |    Average (MB)     |     Median (MB)     |     Lowest (MB)      |    Highest (MB)     |
+----------------------+---------------------+---------------------+----------------------+---------------------+
| 10-second recordings | 0.06808197975158692 |  0.064361572265625  |  0.0542449951171875  | 0.5165262222290039  |
| 15-second recordings | 0.06307750701904297 | 0.0639801025390625  | 0.054996490478515625 | 0.06687641143798828 |
| 20-second recordings | 0.06411605834960937 | 0.06436920166015625 | 0.056580543518066406 | 0.06723880767822266 |
+----------------------+---------------------+---------------------+----------------------+---------------------+
Word Error Rate (WER) Summary:
+----------------------+---------------------+---------------------+------------+-------------+
|  Recording Duration  |     Average WER     |     Median WER      | Lowest WER | Highest WER |
+----------------------+---------------------+---------------------+------------+-------------+
| 10-second recordings | 0.37057226506812063 | 0.4373913043478261  |    0.0     |     1.0     |
| 15-second recordings | 0.35165972465553397 | 0.3117408906882591  |    0.0     |    0.85     |
| 20-second recordings | 0.4192931386001764  | 0.41883116883116883 |    0.0     |     0.9     |
+----------------------+---------------------+---------------------+------------+-------------+
Total samples processed for 10-second recordings: 100
Average WER for 10-second recordings: 0.37
Total samples processed for 15-second recordings: 100
Average WER for 15-second recordings: 0.35
Total samples processed for 20-second recordings: 100
Average WER for 20-second recordings: 0.42
"""