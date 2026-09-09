// Step 2 of bring-up: stream both I2S channels to the host over USB CDC.
//
// The XVF3800 puts something different in each of the two 32-bit slots -- the
// levels move independently, so it is not one signal duplicated. Which slot
// carries the processed speech decides what WakeNet and the cloud recogniser
// get fed, and guessing wrong costs wake rate and accuracy. So: send both up,
// let the host run each through ASR, and believe whichever transcribes.
//
// Frames share the USB endpoint with the log output. That is deliberate --
// the host decoder finds frames by magic and skips anything else, so this
// doubles as a live test of the resync path.
//
// Pins and role are Seeed's, from their XVF3800 + XIAO example. The
// ReSpeaker Lite code this project started from has DOUT/DIN swapped and uses
// slave mode; both are wrong here.

#include <string.h>

#include "driver/i2s_std.h"
#include "driver/usb_serial_jtag.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "stream";

namespace {

constexpr gpio_num_t kPinBclk = GPIO_NUM_8;
constexpr gpio_num_t kPinWs   = GPIO_NUM_7;
constexpr gpio_num_t kPinDout = GPIO_NUM_44;  // XIAO -> XVF3800 (playback)
constexpr gpio_num_t kPinDin  = GPIO_NUM_43;  // XIAO <- XVF3800 (processed mic)

constexpr int    kSampleRate = 16000;
constexpr size_t kFrames     = 320;  // 20 ms

// Wire format, byte-for-byte with util/device_frame_helper.py.
constexpr uint16_t kMagic      = 0x5AA5;
constexpr uint8_t  kMsgAudioUp = 1;

struct __attribute__((packed)) FrameHeader {
    uint16_t magic;
    uint8_t  type;
    uint8_t  ctrl;
    uint32_t turn_id;
    uint32_t len;
};
static_assert(sizeof(FrameHeader) == 12, "FrameHeader must be 12 bytes");

i2s_chan_handle_t g_rx = nullptr;
i2s_chan_handle_t g_tx = nullptr;

void i2s_start() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num  = 6;
    chan_cfg.dma_frame_num = kFrames;
    chan_cfg.auto_clear    = true;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &g_tx, &g_rx));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = kPinBclk,
            .ws   = kPinWs,
            .dout = kPinDout,
            .din  = kPinDin,
            .invert_flags = {false, false, false},
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(g_rx, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(g_tx, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(g_rx));
    ESP_ERROR_CHECK(i2s_channel_enable(g_tx));
}

// Send one frame, blocking until the host drains it. A short timeout would
// silently drop audio and show up later as gaps nobody can explain.
void send_frame(const uint8_t *payload, size_t len) {
    FrameHeader header = {kMagic, kMsgAudioUp, 0, 0, (uint32_t)len};
    usb_serial_jtag_write_bytes(&header, sizeof(header), portMAX_DELAY);
    usb_serial_jtag_write_bytes(payload, len, portMAX_DELAY);
}

}  // namespace

extern "C" void app_main(void) {
    usb_serial_jtag_driver_config_t usb_cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_cfg.tx_buffer_size = 4096;
    usb_cfg.rx_buffer_size = 1024;
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_cfg));

    ESP_LOGI(TAG, "streaming stereo int16 @%d Hz, both channels interleaved", kSampleRate);
    i2s_start();

    static int32_t raw[kFrames * 2];
    static int16_t out[kFrames * 2];  // interleaved L,R

    while (true) {
        size_t got = 0;
        if (i2s_channel_read(g_rx, raw, sizeof(raw), &got, pdMS_TO_TICKS(200)) != ESP_OK) {
            continue;
        }
        const size_t frames = got / (2 * sizeof(int32_t));
        // Valid data is the top 16 bits of each 32-bit slot.
        for (size_t i = 0; i < frames * 2; ++i) {
            out[i] = (int16_t)(raw[i] >> 16);
        }
        send_frame((const uint8_t *)out, frames * 2 * sizeof(int16_t));
    }
}
