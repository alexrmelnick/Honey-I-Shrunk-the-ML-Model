// Adapted from the Teensy Recorder.ino tutorial
// This file records audio from the onboard mic for 5 seconds, then plays the recording back for 5 seconds, then loops. 

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>

// GUItool: begin automatically generated code
AudioInputI2S            i2s2;
AudioRecordQueue         queue1;
AudioPlaySdRaw  playRaw1;
AudioOutputI2S           i2s1;
AudioConnection          patchCord1(i2s2, 0, queue1, 0);
AudioConnection          patchCord2(playRaw1, 0, i2s1, 0);
AudioConnection          patchCord3(playRaw1, 0, i2s1, 1);
AudioControlSGTL5000     sgtl5000_1;
// GUItool: end automatically generated code

const int chipSelect = BUILTIN_SDCARD; // Use Teensy 4.1's built-in SD card
const uint32_t estimatedFileSize = 44100 * 2 * 5; // 5 seconds of mono audio
const int recordDuration = 5000;  // 5 seconds in milliseconds
File frec; // SD file object

// Only use one of these
const int myInput = AUDIO_INPUT_MIC;
// const int myInput = AUDIO_INPUT_LINEIN;

void setup() {
  Serial.begin(9600);
  AudioMemory(60);
  sgtl5000_1.enable();
  sgtl5000_1.inputSelect(myInput);
  sgtl5000_1.volume(0.5);

  
  if (!SD.begin(chipSelect)) {
    while (1) {
      Serial.println("!!!Main SD card initialization failed!!!");
      delay(1000);
    }
  }
  Serial.println("Main SD card successfully initialized");
}

void loop() {
  recordAudio();
  playAudio();
}

void recordAudio() {
  Serial.println("Recording...");

  if (SD.exists("RECORD.RAW")) {
    SD.remove("RECORD.RAW");
  }

  frec = SD.open("RECORD.RAW", FILE_WRITE);
  if (frec) {
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
          delay(1000);
        }
      }
    }

    queue1.end();
    while (queue1.available() > 0) {
      frec.write((byte*)queue1.readBuffer(), 256);
      queue1.freeBuffer();
    }
    frec.close();

  }
  else { 
    Serial.println("!!!Failed to open file for recording!!!");
    Serial.println(frec);
    delay(1000);
    return;
  }
  
  Serial.println("Recording complete");
}

void playAudio() {
  Serial.println("Playing...");

  if (SD.exists("RECORD.RAW")) {
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
    Serial.println("!!!File not found for playback!!!");
    delay(1000);
  }
}