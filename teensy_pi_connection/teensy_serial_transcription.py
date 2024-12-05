import serial
import os
from faster_whisper import WhisperModel

# Configurable parameters
SERIAL_PORT = "/dev/ttyACM0"  # Replace with the Teensy's serial port
BAUD_RATE = 115200            # Ensure this matches the Teensy's settings
WAV_FILE = "received_audio.wav"

# Initialize serial communication
def setup_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        return ser
    except Exception as e:
        print(f"Error opening serial port: {e}")
        exit(1)

# Receive a .wav file over serial
def receive_audio_file(ser, file_name):
    with open(file_name, 'wb') as f:
        print("Receiving audio file...")
        while True:
            data = ser.read(1024)  # Read 1KB chunks
            if data == b"END":     # Signal from Teensy indicating file transfer is complete
                print("File transfer complete.")
                break
            if data:
                f.write(data)
    print(f"Audio file saved as {file_name}")

# Transcribe audio using Faster-Whisper
def transcribe_audio(file_name):
    print("Loading Faster-Whisper model...")
    model = WhisperModel("tiny.en", device="cpu",compute_type = "int8")  # Use "tiny" for faster performance on Raspberry Pi
    print(f"Transcribing {file_name}...")
    segments, _ = model.transcribe(file_name)
    transcription = "".join(segment.text for segment in segments)
    print("Transcription complete.")
    return transcription

# Send transcription back to Teensy
def send_transcription(ser, transcription):
    print("Sending transcription back to Teensy...")
    ser.write(transcription.encode() + b"\n")
    print("Transcription sent.")

# Main workflow
def main():
    ser = setup_serial()

    while True:
        print("Waiting for Teensy to send audio...")
        if ser.readline().strip() == b"START":  # Signal from Teensy to start receiving
            receive_audio_file(ser, WAV_FILE)

            # Run transcription
            transcription = transcribe_audio(WAV_FILE)

            # Send transcription back
            send_transcription(ser, transcription)

if __name__ == "__main__":
    main()
