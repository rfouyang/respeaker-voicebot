// Step 3 of bring-up: full duplex, so the AEC can be tested.
//
// Uplink is unchanged -- both I2S slots, interleaved, to the host. What is new
// is downlink: AUDIO_DOWN frames from the host are written to I2S TX, which
// goes to the XVF3800 and out of its 3.5 mm jack.
//
// That routing is the whole point. The XVF3800 takes its AEC reference from
// what we send it over I2S, so the reply has to leave through the chip. Wire a
// speaker straight to the XIAO instead and the chip never sees the reference,
// the echo is never cancelled, and barge-in becomes impossible.
//
// The test this enables: play a long sentence out of the jack and watch both
// uplink channels. The one where the echo is gone is the processed output --
// which settles both open questions at once, since it also proves the AEC runs
// at all.

#include <math.h>
#include <string.h>

#include "driver/i2s_std.h"
#include "driver/usb_serial_jtag.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"

static const char *TAG = "duplex";

// Hardware loopback: write the received I2S words straight back out, with no
// USB, no ring buffer and no format conversion of any kind.
//
// This exists because too many things were being changed at once. If the
// speaker sounds clean here, the I2S TX path and the 32-bit format are fine
// and the noise lives in the host chain. If it is still noisy, the problem is
// below all of that. Either answer removes half the search space.
//
// Speak at the array and you hear yourself.
constexpr bool kLoopbackTest = false;

// Write digital silence to the speaker, forever.
//
// Loopback carried the voice but hissed loudly even while nobody spoke, and
// the captured microphone data is near zero in silence -- so near-zero input
// is coming out as loud noise. That points away from the data and at the
// analog side. If the speaker still hisses with nothing but zeros going out,
// the fault cannot be anything we send, and the answer is the AIC3104's own
// gain: Seeed's playback example initialises its DAC volume and line-out
// levels over I2C first, which this firmware has never done.
constexpr bool kSilenceTest = true;

// Local tone: generate a clean sine on the device and push it through exactly
// the same int16 -> (int32 << 16) construction the streamed audio uses, but
// with no USB and no ring buffer in the way.
//
// Loopback already proved the I2S path and the 32-bit format are fine, so the
// noise is somewhere on the host side. This splits that side in two: clean
// here means the sample construction is right and the fault is in USB or the
// ring; noisy here means the construction itself is wrong.
constexpr bool kToneTest = false;

namespace {

constexpr gpio_num_t kPinBclk = GPIO_NUM_8;
constexpr gpio_num_t kPinWs   = GPIO_NUM_7;
constexpr gpio_num_t kPinDout = GPIO_NUM_44;  // XIAO -> XVF3800 (and its AEC reference)
constexpr gpio_num_t kPinDin  = GPIO_NUM_43;  // XIAO <- XVF3800

// The bus runs at 16 kHz. Measured, the hard way.
//
// Running it at 48 kHz to match the playback path killed the uplink outright
// -- zero-crossing rate fell to 19/s, i.e. no signal -- so the XVF3800's I2S
// really is 16 kHz and its microphone output only comes out cleanly at that
// clock.
//
// The downlink is still wrong: a 1 kHz tone sent at 16 kHz comes back from the
// speaker at ~3 kHz. That is NOT explained yet, and guessing has cost enough;
// the next step is to read the chip's Audio Manager configuration over I2C
// rather than infer it from symptoms.
constexpr int    kSampleRate = 16000;
constexpr size_t kFrames     = 320;  // 20 ms at 16 kHz

constexpr uint16_t kMagic        = 0x5AA5;
constexpr uint8_t  kMsgAudioUp   = 1;
constexpr uint8_t  kMsgAudioDown = 2;
constexpr uint8_t  kMsgControl   = 3;

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

// Playback held here between the USB task and the I2S loop. Deliberately
// shallow: a deep buffer means audio already committed that the listener has
// not heard yet, and on a barge-in every one of those milliseconds is a
// sentence that keeps playing after they have started talking.
RingbufHandle_t g_playback = nullptr;
// Non-zero means the host outran the device. Reported so an underrun is a
// number rather than a noise.
volatile uint32_t g_dropped_bytes = 0;
volatile uint32_t g_underruns = 0;
constexpr size_t kPlaybackBytes = 16000 * 2 * 250 / 1000;  // 250 ms mono int16

void i2s_start() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num  = 6;
    chan_cfg.dma_frame_num = kFrames;
    chan_cfg.auto_clear    = true;  // send silence on underrun, not stale audio
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &g_tx, &g_rx));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate),
        // 32-bit slots. The capture side needs them: at 16-bit the uplink
        // collapsed to 68 zero-crossings per second and the channels swapped,
        // which is what frame misalignment looks like.
        //
        // The playback side wants 16-bit -- Seeed's two examples genuinely
        // disagree, because each of them only runs one direction. One bus
        // cannot be both, so the AIC3104 has to be told over I2C to accept
        // 32-bit words instead. That is the open item.
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

void send_frame(const uint8_t *payload, size_t len) {
    FrameHeader header = {kMagic, kMsgAudioUp, 0, 0, (uint32_t)len};
    usb_serial_jtag_write_bytes(&header, sizeof(header), portMAX_DELAY);
    usb_serial_jtag_write_bytes(payload, len, portMAX_DELAY);
}

// Read the host's byte stream and pull frames out of it. Same magic-first
// resync as the Python side: serial has no message boundaries, and starting
// mid-stream must not break the link permanently.
void usb_rx_task(void *) {
    static uint8_t chunk[512];
    static uint8_t buf[2048];
    size_t used = 0;

    while (true) {
        const int got = usb_serial_jtag_read_bytes(chunk, sizeof(chunk), pdMS_TO_TICKS(50));
        if (got <= 0) continue;
        const size_t room = sizeof(buf) - used;
        const size_t take = (size_t)got < room ? (size_t)got : room;
        memcpy(buf + used, chunk, take);
        used += take;

        size_t at = 0;
        while (used - at >= sizeof(FrameHeader)) {
            FrameHeader header;
            memcpy(&header, buf + at, sizeof(header));
            if (header.magic != kMagic || header.len > 4096) {
                at++;  // not a header; step one byte and keep hunting
                continue;
            }
            if (used - at < sizeof(FrameHeader) + header.len) break;  // still arriving

            const uint8_t *payload = buf + at + sizeof(FrameHeader);
            if (header.type == kMsgAudioDown && header.len > 0) {
                // Wait briefly rather than dropping. Dropping on a full ring
                // silently removes audio from the middle of a sentence, and
                // what comes out is a buzz nobody can trace back to here. A
                // short block pushes back on the host instead, which is the
                // behaviour that can actually be observed.
                if (xRingbufferSend(g_playback, payload, header.len,
                                    pdMS_TO_TICKS(40)) != pdTRUE) {
                    g_dropped_bytes += header.len;
                }
            }
            at += sizeof(FrameHeader) + header.len;
        }
        if (at > 0) {
            memmove(buf, buf + at, used - at);
            used -= at;
        }
        if (used == sizeof(buf)) used = 0;  // no frame in a full buffer: resync
    }
}

}  // namespace

extern "C" void app_main(void) {
    usb_serial_jtag_driver_config_t usb_cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_cfg.tx_buffer_size = 4096;
    usb_cfg.rx_buffer_size = 2048;
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_cfg));

    g_playback = xRingbufferCreate(kPlaybackBytes, RINGBUF_TYPE_BYTEBUF);
    ESP_ERROR_CHECK(g_playback ? ESP_OK : ESP_ERR_NO_MEM);

    ESP_LOGI(TAG, "full duplex @%d Hz; playback goes out through the XVF3800", kSampleRate);
    i2s_start();
    xTaskCreate(usb_rx_task, "usb_rx", 4096, nullptr, 5, nullptr);

    static int32_t raw[kFrames * 2];
    static int16_t up[kFrames * 2];
    static int32_t down[kFrames * 2];

    while (true) {
        size_t got = 0;
        if (i2s_channel_read(g_rx, raw, sizeof(raw), &got, pdMS_TO_TICKS(200)) != ESP_OK) {
            continue;
        }
        const size_t frames = got / (2 * sizeof(int32_t));

        if (kSilenceTest) {
            memset(down, 0, sizeof(down));
            size_t played = 0;
            i2s_channel_write(g_tx, down, frames * 2 * sizeof(int32_t), &played,
                              pdMS_TO_TICKS(50));
            continue;
        }

        if (kToneTest) {
            static float phase = 0.0f;
            const float step = 2.0f * 3.14159265f * 440.0f / kSampleRate;
            for (size_t i = 0; i < frames; ++i) {
                const int16_t sample = (int16_t)(8000.0f * sinf(phase));
                phase += step;
                if (phase > 2.0f * 3.14159265f) phase -= 2.0f * 3.14159265f;
                const int32_t v = (int32_t)sample << 16;
                down[i * 2] = v;
                down[i * 2 + 1] = v;
            }
            size_t played = 0;
            i2s_channel_write(g_tx, down, frames * 2 * sizeof(int32_t), &played,
                              pdMS_TO_TICKS(50));
            continue;
        }

        if (kLoopbackTest) {
            size_t echoed = 0;
            i2s_channel_write(g_tx, raw, got, &echoed, pdMS_TO_TICKS(50));
            continue;
        }

        // Uplink: valid data is the top 16 bits of each 32-bit slot.
        for (size_t i = 0; i < frames * 2; ++i) up[i] = (int16_t)(raw[i] >> 16);
        send_frame((const uint8_t *)up, frames * 2 * sizeof(int16_t));

        // Downlink: gather a WHOLE frame before playing any of it.
        //
        // xRingbufferReceiveUpTo routinely returns less than asked -- always so
        // when the data straddles the ring's wrap point. Zero-filling the
        // remainder puts a gap of silence inside every 20 ms block, which comes
        // out of the speaker as a buzz rather than as speech. So accumulate,
        // and if a full frame is not there yet, send clean silence instead of a
        // half-filled one.
        static int16_t pending[kFrames];
        static size_t held = 0;

        while (held < kFrames) {
            size_t have = 0;
            auto *part = (int16_t *)xRingbufferReceiveUpTo(
                g_playback, &have, 0, (kFrames - held) * sizeof(int16_t));
            if (!part) break;
            const size_t n = have / sizeof(int16_t);
            memcpy(pending + held, part, n * sizeof(int16_t));
            held += n;
            vRingbufferReturnItem(g_playback, part);
        }

        // Left slot only. XMOS is explicit that the far-end AEC reference goes
        // on the left (0) channel of the I2S input; the right slot is theirs to
        // interpret, and writing the same signal into it added a broad band of
        // hash on top of otherwise intelligible speech.
        memset(down, 0, sizeof(down));
        if (held < frames && held > 0) g_underruns++;
        if (held >= frames) {
            for (size_t i = 0; i < frames; ++i) {
                const int32_t v = (int32_t)pending[i] << 16;
                down[i * 2] = v;
                down[i * 2 + 1] = v;
            }
            held -= frames;
            if (held) memmove(pending, pending + frames, held * sizeof(int16_t));
        }

        size_t written = 0;
        i2s_channel_write(g_tx, down, frames * 2 * sizeof(int32_t), &written,
                          pdMS_TO_TICKS(50));

        static uint32_t ticks = 0;
        if (++ticks % 250 == 0 && (g_underruns || g_dropped_bytes)) {
            ESP_LOGW(TAG, "playback underruns=%lu dropped=%lu bytes",
                     g_underruns, g_dropped_bytes);
        }
    }
}
