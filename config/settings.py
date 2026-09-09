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
