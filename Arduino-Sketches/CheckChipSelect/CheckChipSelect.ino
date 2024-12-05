// Flash Chips found on CS pin 6,

#include <SerialFlash.h>

const int totalPins = 55;  // Teensy 4.1 has 55 GPIO pins

void setup() {
  Serial.begin(9600);
  while (!Serial)
    ;  // Wait for Serial Monitor to open

  Serial.println("Testing all possible chip select pins...");

  for (int pin = 0; pin < totalPins; pin++) {
    Serial.printf("Testing CS pin: %d\n", pin);

    if (pin == 13) {  // Skip these pins
      Serial.printf("CS pin %d skipped\n", pin);
      continue;
    }

    if (SerialFlash.begin(pin)) {
      Serial.printf("!!!!Flash chip detected on CS pin %d\n", pin);
    } else {
      Serial.printf("No Flash chip found on CS pin %d\n", pin);
    }
  }

  Serial.println("Scan complete.");
}

void loop() {
  // Nothing to do
}
