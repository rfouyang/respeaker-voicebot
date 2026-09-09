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
    PROMPT_PATH = BASE_DIR / "config" / "prompt" / "voicebot_system.md"
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
