import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import DataLoader
import jiwer
import numpy as np
import psutil
import time
import gc
import tracemalloc
from tabulate import tabulate
import os

# Ensure the data directory exists
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

# Limit PyTorch to use a single CPU core to simulate Raspberry Pi's performance
torch.set_num_threads(1)

# Allocate a significant portion of memory to simulate a low-memory environment
dummy_memory = np.zeros((500 * 1024 * 1024), dtype='uint8')  # Allocate around 500MB

# Load the quantized model and processor
# model = torch.load("Models/quantized_whisper_base/quantized_model_base.pth")
# processor = WhisperProcessor.from_pretrained("Models/quantized_whisper_base")
model = torch.load("Models/quantized_whisper_tiny_en/quantized_model.pth")
processor = WhisperProcessor.from_pretrained("Models/quantized_whisper_tiny_en")
model.eval()

# model_name = "openai/whisper-tiny.en"
# # model_name = "openai/whisper-base"
# model = WhisperForConditionalGeneration.from_pretrained(model_name)
# processor = WhisperProcessor.from_pretrained(model_name)
# model.eval()



# Load TED-LIUM 3 dataset
dataset = torchaudio.datasets.TEDLIUM(data_dir, release="release3", subset="test", download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Function to trim waveform to a specific duration
def trim_waveform(waveform, sample_rate, duration_sec):
    num_samples = int(sample_rate * duration_sec)
    return waveform[:, :num_samples]

# Initialize tracking variables
total_samples = 0
wer_10s, wer_15s, wer_20s = [], [], []
memory_usage_10s, memory_usage_15s, memory_usage_20s = [], [], []

# Benchmark and process the dataset
for i, data in enumerate(dataloader):
    if total_samples >= 300:  # Process up to 300 samples
        break

    # Extract waveform, sample_rate, and transcript
    waveform, sample_rate, transcript = data[0], data[1], data[2]

    # Unpack transcript from tuple (TED-LIUM format)
    if isinstance(transcript, tuple):
        transcript = transcript[0]  # Extract the actual transcript string

    # Determine duration category
    if len(wer_10s) < 100:
        duration_sec = 10
    elif len(wer_15s) < 100:
        duration_sec = 15
    elif len(wer_20s) < 100:
        duration_sec = 20
    else:
        break

    # Trim and resample the waveform
    waveform = trim_waveform(waveform, sample_rate, duration_sec)
    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

    # Preprocess audio input
    input_features = processor(
        waveform.squeeze(),
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True
    ).input_features

    # Measure memory and time during inference
    tracemalloc.start()
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(input_features)
    end_time = time.time()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_used = peak_memory / (1024 * 1024)  # Convert to MB
    inference_time = end_time - start_time

    # Decode generated text
    decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Calculate Word Error Rate (WER)
    ground_truth_normalized = jiwer.transforms.RemovePunctuation()(transcript.lower().strip())
    predicted_text_normalized = jiwer.transforms.RemovePunctuation()(decoded_output.lower().strip())
    wer = jiwer.wer(ground_truth_normalized, predicted_text_normalized)

    # Log results
    if duration_sec == 10:
        wer_10s.append(wer)
        memory_usage_10s.append(memory_used)
    elif duration_sec == 15:
        wer_15s.append(wer)
        memory_usage_15s.append(memory_used)
    elif duration_sec == 20:
        wer_20s.append(wer)
        memory_usage_20s.append(memory_used)

    total_samples += 1


# Prepare summary tables for memory usage and WER
memory_summary = [
    ["10-second recordings", np.mean(memory_usage_10s), np.median(memory_usage_10s), min(memory_usage_10s), max(memory_usage_10s)],
    ["15-second recordings", np.mean(memory_usage_15s), np.median(memory_usage_15s), min(memory_usage_15s), max(memory_usage_15s)],
    ["20-second recordings", np.mean(memory_usage_20s), np.median(memory_usage_20s), min(memory_usage_20s), max(memory_usage_20s)],
]

wer_summary = [
    ["10-second recordings", np.mean(wer_10s), np.median(wer_10s), min(wer_10s), max(wer_10s)],
    ["15-second recordings", np.mean(wer_15s), np.median(wer_15s), min(wer_15s), max(wer_15s)],
    ["20-second recordings", np.mean(wer_20s), np.median(wer_20s), min(wer_20s), max(wer_20s)],
]

# Print tables
print("Memory Usage Summary:")
print(tabulate(memory_summary, headers=["Recording Duration", "Average (MB)", "Median (MB)", "Lowest (MB)", "Highest (MB)"], tablefmt="pretty"))

print("Word Error Rate (WER) Summary:")
print(tabulate(wer_summary, headers=["Recording Duration", "Average WER", "Median WER", "Lowest WER", "Highest WER"], tablefmt="pretty"))
"""
Whisper Base 
Memory Usage Summary:
+----------------------+---------------------+---------------------+----------------------+----------------------+
|  Recording Duration  |    Average (MB)     |     Median (MB)     |     Lowest (MB)      |     Highest (MB)     |
+----------------------+---------------------+---------------------+----------------------+----------------------+
| 10-second recordings | 0.04801161766052246 | 0.04738664627075195 | 0.037873268127441406 | 0.06840991973876953  |
| 15-second recordings | 0.04136317253112793 | 0.04086780548095703 |   0.03924560546875   | 0.045607566833496094 |
| 20-second recordings | 0.04131581306457519 | 0.04114866256713867 | 0.03799724578857422  | 0.043229103088378906 |
+----------------------+---------------------+---------------------+----------------------+----------------------+
Word Error Rate (WER) Summary:
+----------------------+---------------------+---------------------+------------+-------------+
|  Recording Duration  |     Average WER     |     Median WER      | Lowest WER | Highest WER |
+----------------------+---------------------+---------------------+------------+-------------+
| 10-second recordings | 0.4375554255373098  | 0.11145510835913312 |    0.0     |     3.0     |
| 15-second recordings | 0.35416831517235414 | 0.13397129186602869 |    0.0     |     3.0     |
| 20-second recordings | 0.33295083698587014 | 0.17391304347826086 |    0.0     |     7.5     |
+----------------------+---------------------+---------------------+------------+-------------+
Quantized Whisper Base:
Memory Usage Summary:
+----------------------+---------------------+----------------------+----------------------+---------------------+
|  Recording Duration  |    Average (MB)     |     Median (MB)      |     Lowest (MB)      |    Highest (MB)     |
+----------------------+---------------------+----------------------+----------------------+---------------------+
| 10-second recordings | 0.05113487243652344 | 0.048012733459472656 | 0.042450904846191406 | 0.2147665023803711  |
| 15-second recordings | 0.04409256935119629 | 0.04412651062011719  | 0.041541099548339844 | 0.04553031921386719 |
| 20-second recordings | 0.04470427513122559 | 0.044440269470214844 | 0.042014122009277344 | 0.04935932159423828 |
+----------------------+---------------------+----------------------+----------------------+---------------------+
Word Error Rate (WER) Summary:
+----------------------+--------------------+---------------------+------------+-------------------+
|  Recording Duration  |    Average WER     |     Median WER      | Lowest WER |    Highest WER    |
+----------------------+--------------------+---------------------+------------+-------------------+
| 10-second recordings | 3.0554066182341075 | 0.12310606060606061 |    0.0     |       264.0       |
| 15-second recordings | 1.801206009023633  | 0.15384615384615385 |    0.0     |       148.0       |
| 20-second recordings | 0.5069215245492741 | 0.18518518518518517 |    0.0     | 9.833333333333334 |
+----------------------+--------------------+---------------------+------------+-------------------+

Tiny En
Memory Usage Summary:
+----------------------+---------------------+----------------------+----------------------+---------------------+
|  Recording Duration  |    Average (MB)     |     Median (MB)      |     Lowest (MB)      |    Highest (MB)     |
+----------------------+---------------------+----------------------+----------------------+---------------------+
| 10-second recordings | 0.03992022514343262 | 0.040816307067871094 | 0.030005455017089844 | 0.06135845184326172 |
| 15-second recordings | 0.03347084999084473 | 0.033080101013183594 | 0.030054092407226562 |  0.038909912109375  |
| 20-second recordings | 0.03309560775756836 | 0.03278827667236328  | 0.03006744384765625  |  0.035797119140625  |
+----------------------+---------------------+----------------------+----------------------+---------------------+
Word Error Rate (WER) Summary:
+----------------------+---------------------+---------------------+------------+-------------+
|  Recording Duration  |     Average WER     |     Median WER      | Lowest WER | Highest WER |
+----------------------+---------------------+---------------------+------------+-------------+
| 10-second recordings | 0.2362809366113932  | 0.11145510835913312 |    0.0     |     1.0     |
| 15-second recordings | 0.29817153078024444 | 0.15268065268065267 |    0.0     |     3.0     |
| 20-second recordings | 0.34524023222457273 | 0.17902930402930403 |    0.0     |    10.0     |
+----------------------+---------------------+---------------------+------------+-------------+

Quantized Tiny En
Memory Usage Summary:
+----------------------+---------------------+---------------------+----------------------+---------------------+
|  Recording Duration  |    Average (MB)     |     Median (MB)     |     Lowest (MB)      |    Highest (MB)     |
+----------------------+---------------------+---------------------+----------------------+---------------------+
| 10-second recordings | 0.04274847984313965 | 0.04105949401855469 | 0.03296947479248047  | 0.20885562896728516 |
| 15-second recordings | 0.03588282585144043 | 0.03580331802368164 | 0.033217430114746094 | 0.0386199951171875  |
| 20-second recordings | 0.03563480377197266 | 0.03568840026855469 | 0.03309345245361328  | 0.03779315948486328 |
+----------------------+---------------------+---------------------+----------------------+---------------------+
Word Error Rate (WER) Summary:
+----------------------+---------------------+---------------------+------------+-------------+
|  Recording Duration  |     Average WER     |     Median WER      | Lowest WER | Highest WER |
+----------------------+---------------------+---------------------+------------+-------------+
| 10-second recordings | 0.24419047457161372 | 0.12310606060606061 |    0.0     |     1.0     |
| 15-second recordings | 0.30779991224245945 | 0.1620405101275319  |    0.0     |     3.0     |
| 20-second recordings | 0.2581943369709067  | 0.18181818181818182 |    0.0     |     2.0     |
+----------------------+---------------------+---------------------+------------+-------------+

"""