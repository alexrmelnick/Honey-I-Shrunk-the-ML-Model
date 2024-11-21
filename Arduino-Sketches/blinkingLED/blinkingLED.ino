/*
  This is a quick test program for making sure the board works on a basic level.
  // Functionality: LED light on board should blink twice the pause for half a second on a loop.  
*/

void setup() {
  // put your setup code here, to run once:
  pinMode(13, OUTPUT); // Set pin 13 to output (pin 13 is the LED)
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(13, HIGH); // Turn LED on
  delay(100); // Wait 0.1 seconds
  digitalWrite(13, LOW); // Turn LED off
  delay(100); // Wait 0.1 seconds
  digitalWrite(13, HIGH); // Turn LED on
  delay(100); // Wait 0.1 second
  digitalWrite(13, LOW); // Turn LED off
  delay(500); // Wait 0.5 seconds
}
