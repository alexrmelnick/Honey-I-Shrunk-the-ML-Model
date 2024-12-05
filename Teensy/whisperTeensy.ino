#include <Arduino.h>
#include <Audio.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "model_data.h"  // this is the actual model coverted to .h

// Audio Configuration
#define SAMPLE_RATE 16000
#define RECORD_DURATION 20
#define BUFFER_SIZE (SAMPLE_RATE * RECORD_DURATION)

int16_t audio_buffer[BUFFER_SIZE];
volatile size_t buffer_index = 0;

// TensorFlow Lite Micro Configuration
constexpr int kTensorArenaSize = 64 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroMutableOpResolver<10> resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input_tensor;
TfLiteTensor* output_tensor;

// Display Configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Token-to-Text Mapping (Example)
const char* token_to_text[] = {
    "<|startoftext|>", "hello", "world", "this", "is", "a", "test", "<|endoftext|>"
};

// Setup Function
void setup() {
    Serial.begin(115200);
    AudioMemory(12);

    // Display Setup
    if (!display.begin(SSD1306_I2C_ADDRESS, 0x3C)) {
        Serial.println("OLED initialization failed!");
        while (1);
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.print("Starting...");
    display.display();

    // TensorFlow Lite Micro Setup
    const tflite::Model* model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        while (1);
    }

    resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
    resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());
    resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, tflite::ops::micro::Register_DEPTHWISE_CONV_2D());

    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter->AllocateTensors();

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.println("Setup complete.");
}

// Main Loop
void loop() {
    if (buffer_index < BUFFER_SIZE) {
        recordAudio();
    } else {
        computeMelSpectrogram(audio_buffer, BUFFER_SIZE);
        runInference();
        buffer_index = 0; // Reset for next recording
    }
}

// Record Audio
void recordAudio() {
    if (buffer_index < BUFFER_SIZE) {
        audio_buffer[buffer_index++] = analogRead(A0); // Example for analog microphone
    } else {
        Serial.println("Recording complete.");
    }
}

// Compute Mel Spectrogram
void computeMelSpectrogram(const int16_t* audio_samples, size_t sample_count) {
    // Placeholder: Implement FFT and mel filter bank application here
    // Compute the mel spectrogram and store it in input_tensor->data.f
    for (size_t i = 0; i < MEL_BINS; i++) {
        input_tensor->data.f[i] = log10f(1e-6 + i); // Example placeholder
    }
}

// Run Inference
void runInference() {
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
    }

    String transcription = decodeOutput(output_tensor);
    Serial.println("Transcription: " + transcription);
    displayText(transcription);
}

// Decode Model Output
String decodeOutput(TfLiteTensor* output_tensor) {
    String result = "";
    for (int i = 0; i < output_tensor->dims->data[1]; i++) {
        int token_id = static_cast<int>(output_tensor->data.f[i]);
        if (token_id == 7) { // End token
            break;
        }
        result += String(token_to_text[token_id]) + " ";
    }
    return result.trim();
}

// Display Transcription
void displayText(String text) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("Transcription:");
    display.setCursor(0, 10);
    display.println(text);
    display.display();
}
