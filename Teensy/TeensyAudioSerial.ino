#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>

// Audio objects
AudioInputI2S mic;              // Microphone input
AudioRecordQueue queue;         // Queue for recording audio
AudioConnection patchCord1(mic, queue);

// Constants
const int RECORDING_TIME_MS = 5000;  // Duration to record (5 seconds)
const int SAMPLE_RATE = 44100;      // Standard audio sample rate

// SD card setup
const int SD_CS_PIN = BUILTIN_SDCARD;

// File name for recording
const char *filename = "recording.wav";

void setup() {
  // Serial connection
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for Serial connection
  }
  Serial.println("Teensy ready!");

  // Audio system setup
  AudioMemory(12);          // Allocate memory for the audio library
  queue.begin();            // Start the recording queue

  // SD card setup
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("Unable to access SD card.");
    while (1) ;
  }
}

void loop() {
  Serial.println("Recording audio...");
  recordAudioToSD(RECORDING_TIME_MS);

  Serial.println("Sending audio to Raspberry Pi...");
  sendFileToRaspberryPi(filename);

  Serial.println("Waiting for transcription...");
  String transcription = receiveTranscription();
  Serial.println("Transcription received:");
  Serial.println(transcription);

  // Delay before next recording
  delay(10000); // 10 seconds
}

// Function to record audio to SD card
void recordAudioToSD(int duration_ms) {
  File audioFile = SD.open(filename, FILE_WRITE);
  if (!audioFile) {
    Serial.println("Error opening file for writing.");
    return;
  }

  // Write WAV file header
  writeWavHeader(audioFile);

  unsigned long startMillis = millis();
  while (millis() - startMillis < duration_ms) {
    if (queue.available() > 0) {
      int16_t *buffer = (int16_t *)queue.readBuffer();
      audioFile.write((byte *)buffer, 256);
      queue.freeBuffer();
    }
  }

  // Update WAV file size in header
  updateWavHeader(audioFile);
  audioFile.close();

  Serial.println("Audio recorded successfully.");
}

// Function to send the WAV file to the Raspberry Pi
void sendFileToRaspberryPi(const char *filename) {
  File audioFile = SD.open(filename);
  if (!audioFile) {
    Serial.println("Error opening file for reading.");
    return;
  }

  // Signal the Raspberry Pi to start receiving
  Serial.println("START");

  // Send the file contents
  while (audioFile.available()) {
    byte buffer[1024];
    int bytesRead = audioFile.read(buffer, sizeof(buffer));
    Serial.write(buffer, bytesRead);
  }

  // Signal the Raspberry Pi that the transfer is complete
  Serial.println("END");
  audioFile.close();
}

// Function to receive transcription from Raspberry Pi
String receiveTranscription() {
  String transcription = "";
  while (Serial.available() > 0) {
    transcription += Serial.readStringUntil('\n');
  }
  return transcription;
}

// Helper function to write a basic WAV header
void writeWavHeader(File &file) {
  file.write("RIFF");
  file.write(0); file.write(0); file.write(0); file.write(0); // Placeholder for file size
  file.write("WAVE");
  file.write("fmt ");
  file.write(16, 1, 1, 0);            // PCM format
  file.write(2, SAMPLE_RATE);         // Channels and sample rate
  file.write(4 * SAMPLE_RATE);        // Byte rate
  file.write(4, 16);                  // Block align and bits per sample
  file.write("data");
  file.write(0); file.write(0); file.write(0); file.write(0); // Placeholder for data size
}

// Helper function to update the WAV header with actual file size
void updateWavHeader(File &file) {
  int fileSize = file.size();
  file.seek(4);
  file.write((fileSize - 8) & 0xFF);
  file.write(((fileSize - 8) >> 8) & 0xFF);
  file.write(((fileSize - 8) >> 16) & 0xFF);
  file.write(((fileSize - 8) >> 24) & 0xFF);
  file.seek(40);
  file.write((fileSize - 44) & 0xFF);
  file.write(((fileSize - 44) >> 8) & 0xFF);
  file.write(((fileSize - 44) >> 16) & 0xFF);
  file.write(((fileSize - 44) >> 24) & 0xFF);
}
