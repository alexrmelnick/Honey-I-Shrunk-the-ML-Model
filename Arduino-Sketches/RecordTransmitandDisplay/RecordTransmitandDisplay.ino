#include <Bounce2.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h> 
#include <hd44780.h> // main hd44780 header
#include <hd44780ioClass/hd44780_I2Cexp.h> // i2c expander i/o class header
#include <string.h>


const int buttonPinNo = 32;
const int myInput = AUDIO_INPUT_MIC;

Bounce button = Bounce();



// GUItool: begin automatically generated code
AudioInputI2S            i2s2;
AudioRecordQueue         queue1;
AudioPlaySdRaw           playRaw1;
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
const char* filename = "RECORD.wav";

// Only use one of these
// const int myInput = AUDIO_INPUT_LINEIN;

// WAVE FILE HEADER INFORMATION
// The first 4 byte of a wav file should be the characters "RIFF" */
char chunkID[4] = {'R', 'I', 'F', 'F'};
/// 36 + SubChunk2Size
uint32_t chunkSize = 36; // We don't know this until we write our data but at a minimum it is 36 for an empty file
/// "should be characters "WAVE"
char format[4] = {'W', 'A', 'V', 'E'};
/// " This should be the letters "fmt ", note the space character
char subChunk1ID[4] = {'f', 'm', 't', ' '};
///: For PCM == 16, since audioFormat == uint16_t
uint32_t subChunk1Size = 16;
///: For PCM this is 1, other values indicate compression
uint16_t audioFormat = 1;
///: Mono = 1, Stereo = 2, etc.
uint16_t numChannels = 1;
///: Sample Rate of file
uint32_t sampleRate = 44100;
///: SampleRate * NumChannels * BitsPerSample/8
uint32_t byteRate = 44100 * 2;
///: The number of byte for one frame NumChannels * BitsPerSample/8
uint16_t blockAlign = 2;
///: 8 bits = 8, 16 bits = 16
uint16_t bitsPerSample = 16;
///: Contains the letters "data"
char subChunk2ID[4] = {'d', 'a', 't', 'a'};
///: == NumSamples * NumChannels * BitsPerSample/8  i.e. number of byte in the data.
uint32_t subChunk2Size = 0; // We don't know this until we write our data

const int MIN_DATA_VALUE = 0;
const int MAX_DATA_VALUE = 1023; // For 16-bit ADC (Arduino analog input)



hd44780_I2Cexp LCD(0x27, 20, 4); // Create an LCD struct with I2C addr 0x27 that is 20 chars wide and 4 lines tall


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
  // Serial.println("Main SD card successfully initialized");

  LCD.begin(20, 4);
  LCD.backlight();

  button.attach(buttonPinNo, INPUT_PULLUP);
  button.interval(10);

  // Serial.println("Setup complete");
}

void loop() {
  button.update();

  if(button.fell()) {
    // Record audio to transmit and save it as a WAVE file on SD card
    recordAudio();

    // Transmit file to Pi
    sendWAVE();

    waitForMessage();

    // Receive string
    String transcribedText = recieveString();

    printMessage(transcribedText);
  }
}

void sendWAVE() {
  if (SD.exists(filename)) {
    frec = SD.open(filename, FILE_READ);

    // Serial.println("Sending file");
    Serial.print("START");
    
    // Read and send the file in chunks
    const size_t bufferSize = 64; // Adjust buffer size as needed
    uint8_t buffer[bufferSize];
    size_t bytesRead;

    while ((bytesRead = frec.read(buffer, bufferSize)) > 0) {
        Serial.write(buffer, bytesRead); // Send chunk over serial
    }

    Serial.print("END");

    frec.close();
  }
  else {
    Serial.println("Cannot find file to send");
  }
}

String recieveString() {
  // Serial.println("Recieving string");
  String receivedString = Serial.readStringUntil('\n'); // Read until newline character
  // Serial.println("Recieved string:");
  // Serial.println(receivedString);
  return receivedString;
}

void recordAudio() {
  // Serial.println("Recording...");

  if (SD.exists(filename)) {
    SD.remove(filename);
  }

  frec = SD.open(filename, FILE_WRITE);
  if (frec) {
    writeWavHeader(frec);

    queue1.begin();
    unsigned long startMillis = millis();
    unsigned long totalSamples = 0; // Declare and initialize totalSamples

    while (millis() - startMillis < recordDuration) {
      if (queue1.available() >= 2) {
        // byte buffer[512];
        // int16_t processedBuffer[256];
        int16_t buffer[256];

        memcpy(buffer, queue1.readBuffer(), 256);
        queue1.freeBuffer();
        memcpy(buffer + 128, queue1.readBuffer(), 256);
        queue1.freeBuffer();

        // Print the raw data for debugging:
        // Serial.print("Raw data: ");
        // for (int i = 0; i < 10; i++) {
        //   Serial.println(buffer[i]);
        // }

        // Map and process raw data
        // Serial.println("Mapped data: ");
        // for (int i = 0; i < 256; i++) {
        //   int rawData = buffer[i]; // Assuming raw data is in the range MIN_DATA_VALUE to MAX_DATA_VALUE
        //   processedBuffer[i] = map(rawData, MIN_DATA_VALUE, MAX_DATA_VALUE, -32767, 32767);
        //   // if (i < 10) { Serial.println(processedBuffer[i]); }
        // }

        if (frec) {
          // frec.write((byte*)processedBuffer, sizeof(processedBuffer));
          frec.write((byte*)buffer, sizeof(buffer));
          totalSamples += 256;
        } 
        else {
          Serial.println("File write error!");
          delay(1000);
        }
      }
    }

    queue1.end();
    // while (queue1.available() > 0) {
    //   frec.write((byte*)queue1.readBuffer(), 256);
    //   queue1.freeBuffer();
    // }

    // Update the WAVE header
    subChunk2Size = totalSamples * numChannels * bitsPerSample / 8;
    chunkSize = 36 + subChunk2Size;

    // Serial.print("Final chunkSize: ");
    // Serial.println(chunkSize);
    // Serial.print("Final subChunk2Size: ");
    // Serial.println(subChunk2Size);

    frec.seek(4);  // Update chunkSize
    frec.write((byte*)&chunkSize, 4);

    frec.seek(40); // Update subChunk2Size
    frec.write((byte*)&subChunk2Size, 4);

       // Verify file size
        // Serial.p?rint("File size: ");
        // Serial.println(frec.size());

    frec.close();
  }
  else { 
    Serial.println("!!!Failed to open file for recording!!!");
    Serial.println(frec);
    delay(1000);
    return;
  }
  
  

  // Serial.println("Recording complete");
}

// void playAudio() {
//   Serial.println("Playing...");

//   if (SD.exists(filename)) {
//     playRaw1.play(filename);
//     unsigned long startMillis = millis();

//     while (millis() - startMillis < recordDuration) {
//       if (!playRaw1.isPlaying()) {
//         break;
//       }
//     }

//     playRaw1.stop();
//     Serial.println("Playback complete");
//   } else {
//     Serial.println("!!!File not found for playback!!!");
//     delay(1000);
//   }
// }

void writeWavHeader(File wavFile)
{
   wavFile.seek(0);
   wavFile.write(chunkID,4);
   wavFile.write((byte*)&chunkSize,4);
   wavFile.write(format,4);
   wavFile.write(subChunk1ID,4);
   wavFile.write((byte*)&subChunk1Size,4);
   wavFile.write((byte*)&audioFormat,2);
   wavFile.write((byte*)&numChannels,2);
   wavFile.write((byte*)&sampleRate,4);
   wavFile.write((byte*)&byteRate,4);
   wavFile.write((byte*)&blockAlign,2);
   wavFile.write((byte*)&bitsPerSample,2);
   wavFile.write(subChunk2ID,4);
   wavFile.write((byte*)&subChunk2Size,4);
}

void printMessage(String message) {
  // Serial.println("Printing message to LCD");

  bool printToMonitor = false; // Option to print to the serial monitor for debugging
  int pauseTime = 5000; // Amount of time to pause between pages (5 seconds seems about right)
  int messageLength = message.length();
  int necessaryPages = (messageLength+80-1) / 80;   // Need to break message up into 80 character 'pages' (rounding up)

  // if(printToMonitor) { // Print data to monitor if desired
  //     Serial.print("Message is ");
  //     Serial.print(messageLength);
  //     Serial.print(" chars, ");
  //     Serial.print(messageLength/20);
  //     Serial.print(" lines, or ");
  //     Serial.print(necessaryPages);
  //     Serial.print(" pages long.\n");
  // }

  // This library wraps text incorrectly, so we need to do it ourselves

  for(int i = 0; i < necessaryPages; i++) {
    String page = message.substring(i*80,i*80+80); // Get the current page and its length
    int pageLength = page.length();
    int necessaryLines = (pageLength+20-1) / 20; // Get the number of necessary lines, rounding up

    LCD.clear();

    for(int j = 0; j < necessaryLines; j++) { // Iterate through each line
      String currentLine = page.substring(j*20, j*20+20); // Get the line
      LCD.setCursor(0,j); // Set the cursor to the first row
      LCD.print(currentLine); // Print the data

      // if(printToMonitor) { // Print data to monitor if desired
      //   Serial.print(currentLine);
      //   Serial.print('\n');
      // }
    }

    delay(pauseTime); // Pause before printing the next page for the user to read it 
  }
}

void waitForMessage() {
    while (!Serial.available()) {
        delay(10); // Prevent busy looping
    }
}