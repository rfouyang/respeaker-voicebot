"""Project configuration, split by domain. Loaded once from .env.

Credentials come from `.env` and never from source. The XIAO firmware holds
none of them -- a flash dump would give them away.
"""

from __future__ import annotations

from pathlib import Path
import sys

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def use_utf8_output() -> None:
    """Make stdout/stderr able to print Chinese.

    Windows consoles default to a legacy codepage, so printing a device name or
    a transcript raises UnicodeEncodeError -- which, inside a try block, looks
    like the operation itself failed. Entry points call this first.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


class AudioConfig:
    """16 kHz / 16-bit / mono end to end.

    Every stage agrees on these numbers -- the XVF3800 pipeline, WakeNet on the
    XIAO, BytePlus ASR input and TTS output -- which is why not a single
    resample is needed anywhere in the chain. Do not hardcode them elsewhere.
    """

    SAMPLE_RATE = 16_000
    FRAME_MS = 20
    FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1_000
    FRAME_BYTES = FRAME_SAMPLES * 2


class LLMConfig:
    """DeepSeek chat completions.

    Carried over from services/robot-concierge, where this model and prompt
    were tuned against real conversations.
    """

    API_KEY_ENV = "DEEPSEEK_API_KEY"
    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-v4-flash"
    # Swap this to change what the bot is. wayfinding_stadium.md turns it
    # into a Stadium MRT (CC6) wayfinding assistant.
    PROMPT_PATH = BASE_DIR / "config" / "prompt" / "wayfinding_stadium.md"
    MEMORY_TURNS = 10
    TIMEOUT_SECONDS = 60


class BytePlusConfig:
    """BytePlus Seed Speech -- ASR and TTS.

    Single-key auth (`x-api-key`), ap-southeast-1. These endpoints and resource
    ids are the ones proven working in services/robot-concierge, not guesses.
    """

    API_KEY_ENV = "BYTEPLUS_API_KEY"
    REGION = "ap-southeast-1"

    ASR_URL = f"wss://voice.{REGION}.bytepluses.com/api/v3/sauc/bigmodel_async"
    ASR_RESOURCE_ID = "volc.seedasr.sauc.duration"
    ASR_MODEL = "bigmodel"
    ASR_UID = "respeaker-voicebot"
    # Audio arrives from the device in small frames; batch them before sending
    # so the cloud sees packets of a sensible size.
    ASR_FRAME_MS = 200
    # How long a recognised utterance may stay open after the last audio.
    ASR_END_WINDOW_MS = 1_000
    # The wait between the last audio frame and the final transcript -- the
    # delay the user actually feels after they stop talking.
    ASR_TAIL_GOOD_MS = 1_000

    # Unidirectional streaming: one HTTP request per piece of text, audio
    # streamed back as base64 JSON lines. It does NOT accept incremental text,
    # so a long reply is split at sentence boundaries by the caller.
    TTS_URL = f"https://voice.{REGION}.bytepluses.com/api/v3/tts/unidirectional"
    TTS_RESOURCE_ID = "seed-tts-2.0"
    SPEAKER = "zh_female_yingyujiaoxue_uranus_bigtts"
    # Ask for raw 16k PCM, not mp3 -- the XIAO should not spend cycles decoding
    # while it is also running WakeNet.
    TTS_FORMAT = "pcm"
    TTS_TIMEOUT_SECONDS = 60

    # First audio must arrive fast enough that the reply feels immediate.
    TTS_FIRST_CHUNK_GOOD_MS = 500


class OutputConfig:
    OUTPUT_DIR = BASE_DIR / "output"


class DeviceConfig:
    """The USB CDC link to the XIAO, and the audio format on it.

    The XIAO reaches the PC over its OWN USB-C port. The XVF3800 keeps its
    factory I2S firmware and never enumerates as a USB device at all -- it is
    an I2S slave that neither knows nor cares what the XIAO does with the
    audio afterwards.
    """

    PORT = "COM3"
    # ESP32-S3 native USB CDC is not a real UART, so the baud rate is ignored
    # by the hardware. pyserial still wants a number.
    BAUD = 921_600
    READ_TIMEOUT_SECONDS = 0.1

    # Frames are found in the byte stream by this magic. Serial has no message
    # boundaries, so without it a single lost byte desynchronises the link
    # permanently.
    MAGIC = 0x5AA5
    # A header that claims more than this is corrupt, not a real frame.
    MAX_PAYLOAD_BYTES = 4_096

    # Playback rate: UNRESOLVED, currently left equal to the capture rate.
    #
    # What is measured so far: sending 16 kHz over a 16 kHz bus yields speech a
    # listener can follow but that sounds wrong, and a 1 kHz tone returns at
    # ~3 kHz. Two attempted fixes both made it worse, and both are recorded
    # here so they are not tried again:
    #
    #   * Upsample to 48 kHz on the host, bus left at 16 kHz -- the device then
    #     receives three samples per one it can play, the ring overflows, two
    #     thirds are dropped, and it buzzes. Tell-tale: halving the volume did
    #     not lower the captured level.
    #   * Run the whole bus at 48 kHz -- this kills the uplink outright
    #     (zero-crossing rate 19/s, no signal), so the XVF3800's I2S really is
    #     16 kHz. Still buzzed.
    #
    # Next step is to read the chip's Audio Manager settings over I2C instead
    # of inferring them from symptoms.
    PLAYBACK_SAMPLE_RATE = AudioConfig.SAMPLE_RATE

    # One downlink packet. Larger and the device playback ring cannot absorb
    # it; smaller and the framing overhead starts to show.
    DOWNLINK_CHUNK_BYTES = 640  # 20 ms at 16 kHz mono 16-bit

    BYTES_PER_MS = AudioConfig.SAMPLE_RATE * 2 // 1_000  # 32


class WakeWordConfig:
    """WakeNet on the XIAO.

    The model is picked in the firmware's menuconfig, not here -- these values
    only exist so the host can log and display the same thing the device is
    listening for.

    Confirmed against esp-sr 2.5.3's own Kconfig: the option is
    `SR_WN_WN9_JARVIS_TTS`, labelled "Jarvis (wn9_jarvis_tts)". The phrase is
    the bare word; there is no "Hi Jarvis" model.

    Jarvis exists only as a WakeNet9 model. If the wake rate disappoints --
    plausible, since the XVF3800 gives one processed channel rather than the
    wake-word-tuned second stream a ReSpeaker Lite has -- the first thing to
    try is a WakeNet10 model such as `wn10_heynova` or `wn10_nihaoxiaozhi`,
    which generalise better. That is a one-line sdkconfig change.

    Espressif notes that wake-word brand names belong to their owners; Jarvis
    is a Marvel/Disney character, so swap the model before shipping anything
    public.
    """

    MODEL = "wn9_jarvis_tts"
    KCONFIG = "CONFIG_SR_WN_WN9_JARVIS_TTS"
    PHRASE = "Jarvis"

    # WakeNet has detection latency, so the first syllables after the wake word
    # would be lost without a pre-roll. The firmware keeps the same amount.
    PRE_ROLL_MS = 300


class DialogConfig:
    """Turn taking and barge-in.

    The barge-in numbers are carried over from services/robot-concierge, where
    they were tuned against this same XVF3800. Its AEC leaves a little of the
    reply in the processed channel, and these thresholds are what separates
    that residue from a real interruption.
    """

    HISTORY_MESSAGES = 12
    # Silence after which the session gives up and goes back to waiting for
    # the wake word.
    IDLE_TIMEOUT_SECONDS = 10
    MAX_UTTERANCE_SECONDS = 30

    # The reply is handed to TTS one sentence at a time, because the
    # unidirectional API takes a whole request per piece of text. Sentence
    # boundaries are the right unit: smaller pieces mean more round trips and
    # broken prosody, larger ones delay the first audio.
    SENTENCE_END = "[。！？!?；;\n]"
    MAX_SENTENCE_CHARS = 60

    # Converts delivered audio back into "how much text the user actually
    # heard". An estimate: BytePlus unidirectional TTS gives no per-sentence
    # audio offsets, so this is the best available. See Turn.spoken_text.
    CHARS_PER_SECOND = 5.0
    INTERRUPT_NOTE = "（被用户打断，未说完）"

    # A partial transcript this short is noise, not an interruption.
    BARGE_IN_MIN_CHARS = 1
    # Below this length an echo comparison is meaningless, so do not attempt it.
    BARGE_IN_ECHO_MIN_CHARS = 3
    # How close a partial has to be to what we are currently saying before it
    # is treated as our own voice leaking back rather than the user talking.
    BARGE_IN_ECHO_SIMILARITY = 0.72
    THREAD_STOP_SECONDS = 5
    # If the link goes quiet for this long, hand the recogniser silence
    # rather than blocking. A stalled link should end the utterance and
    # let the session time out, not hang the ASR thread with no clue why.
    AUDIO_STARVE_SECONDS = 0.5
