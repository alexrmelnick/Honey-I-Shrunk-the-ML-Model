/*
  This is the function for running the display. 
*/

#include <Wire.h> 
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C LCD(0x27, 20, 4); // Create an LCD struct with I2C addr 0x27 that is 20 chars wide and 4 lines tall
// Verified that I2C addr is 0x27

void setup() {
  Serial.begin(9600); // Set the baud rate to 9600 baud

  // Initialize the LCD
  LCD.begin(20, 4); // Need to pass the size of display (20x4 chars) in newer versions of this library
  delay(2); // Need a delay after each LCD command

  // // ------- Quick 3 blinks of backlight -------------
  // for(int i = 0; i< 3; i++)
  // {
  //   LCD.backlight();
  //   delay(250);
  //   LCD.noBacklight();
  //   delay(250);
  // }
  LCD.backlight(); // finish with backlight on
  delay(2); // Need a delay after each LCD command

  LCD.clear();
  delay(2); // Need a delay after each LCD command


  // Print a test message
  LCD.setCursor(0,0); // Set the cursor the char 0 on line 0
  delay(2); // Need a delay after each LCD command
  LCD.print("I am GROOT");
  delay(2); // Need a delay after each LCD command
}

void printMessage(String message) {
  // Set up the display
  LCD.clear();
  LCD.setCursor(0, 0);

  Serial.print(message);
  LCD.print(message);


  // // If message fits on 1 line, just print it
  // if(message_length < 20) {
  //   LCD.print(message);
  // }

  // // If message is multiple lines, we need to handle dividing it into multiple 
  // // lines and adding dashes to the end of each line that doesn't end in a space

}

void loop() {
  // put your main code here, to run repeatedly:
}