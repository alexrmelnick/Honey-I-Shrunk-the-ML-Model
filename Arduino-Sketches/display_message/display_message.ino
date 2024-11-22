/*
  This is the function for running the display. 
  Acknoledgements: This program uses the 'Extensible hd44780' library by Bill Perry (https://github.com/duinoWitchery/hd44780)
    Documentation: https://github.com/duinoWitchery/hd44780/wiki/ioClass:-hd44780_I2Cexp
*/

#include <Wire.h> 
#include <hd44780.h> // main hd44780 header
#include <hd44780ioClass/hd44780_I2Cexp.h> // i2c expander i/o class header
#include <string.h>

hd44780_I2Cexp LCD(0x27, 20, 4); // Create an LCD struct with I2C addr 0x27 that is 20 chars wide and 4 lines tall
// Verified that I2C addr is 0x27

void setup() {
  Serial.begin(9600);
  LCD.begin(20, 4);
  LCD.backlight();

  String msg1 = "Hello World! This is a long message to test the printMessage. I want it to be really long. That is why I am still typing. So far I am on... the 162nd character. Wow, I love modern IDEs. Isn't it crazy that the VIC-20 only had a 22x23 character display? Imagine trying to get real work done on a display as wide as ours!";
  printMessage(msg1);
}

void printMessage(String message) {
  bool printToMonitor = true; // Option to print to the serial monitor for debugging
  int pauseTime = 5000; // Amount of time to pause between pages (5 seconds seems about right)
  int messageLength = message.length();
  int necessaryPages = (messageLength+80-1) / 80;   // Need to break message up into 80 character 'pages' (rounding up)

  if(printToMonitor) { // Print data to monitor if desired
      Serial.print("Message is ");
      Serial.print(messageLength);
      Serial.print(" chars, ");
      Serial.print(messageLength/20);
      Serial.print(" lines, or ");
      Serial.print(necessaryPages);
      Serial.print(" pages long.\n");
  }

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

      if(printToMonitor) { // Print data to monitor if desired
        Serial.print(currentLine);
        Serial.print('\n');
      }
    }

    delay(pauseTime); // Pause before printing the next page for the user to read it 
  }
}

void loop() {

}