#include <Arduino.h>
#include <Audio.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "model_data.h"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

AudioInputI2S i2s_mic;
AudioConnection patchCord1(i2s_mic, fft);
AudioAnalyzeFFT1024 fft;

constexpr int kTensorArenaSize = 64 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroMutableOpResolver<10> resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input_tensor;
TfLiteTensor* output_tensor;

int16_t audio_buffer[16000];
size_t audio_buffer_index = 0;

void setup() {
    Serial.begin(115200);
    AudioMemory(12);
    setupDisplay();

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
}

void loop() {
    captureAudio();
    if (audio_buffer_index == 0) {
        runInference();
    }
}

void captureAudio() {
    if (audio_buffer_index < 16000) {
        audio_buffer[audio_buffer_index++] = i2s_mic.read();
    } else {
        // Process when buffer is full
        audio_buffer_index = 0;
    }
}

void runInference() {
    // Assume mel spectrogram computation has been done
    memcpy(input_tensor->data.f, log_mel_spectrogram, MEL_BINS * sizeof(float));
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
    }

    String transcription = decodeOutput(output_tensor);
    Serial.println("Transcription: " + transcription);
    displayText(transcription);
}
