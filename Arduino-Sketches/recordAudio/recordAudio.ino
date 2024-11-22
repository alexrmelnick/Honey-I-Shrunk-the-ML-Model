// Adapted from the Teensy Recorder.ino tutorial
// This file records audio from the onboard mic for 5 seconds, then plays the recording back for 5 seconds, then loops. 

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SerialFlash.h>

// GUItool: begin automatically generated code
AudioInputI2S            i2s2;
AudioRecordQueue         queue1;
AudioPlaySerialflashRaw  playRaw1;
AudioOutputI2S           i2s1;
AudioConnection          patchCord1(i2s2, 0, queue1, 0);
AudioConnection          patchCord2(playRaw1, 0, i2s1, 0);
AudioConnection          patchCord3(playRaw1, 0, i2s1, 1);
AudioControlSGTL5000     sgtl5000_1;
// GUItool: end automatically generated code

SerialFlashFile frec;
const int myInput = AUDIO_INPUT_MIC;
const uint32_t estimatedFileSize = 44100 * 2 * 5; // 5 seconds of mono audio
const int recordDuration = 5000;  // 5 seconds in milliseconds

void setup() {
  Serial.begin(9600);
  AudioMemory(60);
  sgtl5000_1.enable();
  sgtl5000_1.inputSelect(myInput);
  sgtl5000_1.volume(0.5);

  if (!SerialFlash.begin()) {
    while (1) {
      Serial.println("Unable to access SerialFlash");
      delay(1000);
    }
  }
}

void loop() {
  recordAudio();
  playAudio();
}

void recordAudio() {
  Serial.println("Recording...");

  if (SerialFlash.exists("RECORD.RAW")) {
    SerialFlash.remove("RECORD.RAW");
  }

  if (SerialFlash.create("RECORD.RAW", estimatedFileSize)) {
    frec = SerialFlash.open("RECORD.RAW");
    if (!frec) {
      Serial.println("Failed to open file for recording");
      return;
    }

    queue1.begin();
    unsigned long startMillis = millis();

    while (millis() - startMillis < recordDuration) {
      if (queue1.available() >= 2) {
        byte buffer[512];
        memcpy(buffer, queue1.readBuffer(), 256);
        queue1.freeBuffer();
        memcpy(buffer + 256, queue1.readBuffer(), 256);
        queue1.freeBuffer();

        if (frec) {
          frec.write(buffer, 512);
        } else {
          Serial.println("File write error!");
        }
      }
    }

    queue1.end();
    while (queue1.available() > 0) {
      frec.write((byte*)queue1.readBuffer(), 256);
      queue1.freeBuffer();
    }
    frec.close();

    Serial.println("Recording complete");
  } else {
    Serial.println("Failed to create file on SerialFlash");
  }
}

void playAudio() {
  Serial.println("Playing...");

  if (SerialFlash.exists("RECORD.RAW")) {
    playRaw1.play("RECORD.RAW");
    unsigned long startMillis = millis();

    while (millis() - startMillis < recordDuration) {
      if (!playRaw1.isPlaying()) {
        break;
      }
    }

    playRaw1.stop();
    Serial.println("Playback complete");
  } else {
    Serial.println("File not found");
  }
}
