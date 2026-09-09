// Placeholder entry point.
//
// It exists so the project configures and builds while the real pipeline is
// written. What it prints is the one thing worth knowing at this stage: which
// wake word model got compiled in, so the phrase the host displays and the
// phrase the device listens for cannot drift apart.

#include <stdio.h>

#include "esp_log.h"
#include "sdkconfig.h"

static const char *TAG = "app";

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "respeaker-voicebot firmware, placeholder build");

#if CONFIG_IDF_TARGET_ESP32S3
    ESP_LOGI(TAG, "target: esp32s3");
#endif

    // The wake word is a compile-time choice, so the real firmware will print
    // the selected model here. Left out until the model is actually picked.
}
