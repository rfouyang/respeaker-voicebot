"""BytePlus Seed TTS unidirectional-streaming client.

Unidirectional means one HTTP request per piece of text: it will not accept
text incrementally. A long reply therefore has to be split at sentence
boundaries by the caller, so the first audio is not held back until the whole
answer is generated.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
import json
import time

import requests

from config.settings import (AudioConfig, BytePlusConfig, OutputConfig,
                             use_utf8_output)
from util.byteplus_helper import BytePlusHelper


class TTSHelper(BytePlusHelper):
    """Synthesize speech and yield 16 kHz mono PCM chunks as they arrive."""

    def __init__(
        self,
        speaker: str = BytePlusConfig.SPEAKER,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(api_key)
        self.speaker = speaker
        self.sample_rate = kwargs.get("sample_rate", AudioConfig.SAMPLE_RATE)
        self.timeout = kwargs.get("timeout", BytePlusConfig.TTS_TIMEOUT_SECONDS)

    def stream(self, text: str, **kwargs) -> Iterator[bytes]:
        payload = {
            "req_params": {
                "text": text,
                "speaker": kwargs.get("speaker", self.speaker),
                "audio_params": {
                    "format": BytePlusConfig.TTS_FORMAT,
                    "sample_rate": kwargs.get("sample_rate", self.sample_rate),
                },
            }
        }
        response = requests.post(
            BytePlusConfig.TTS_URL,
            headers={
                "x-api-key": self.api_key,
                "X-Api-Resource-Id": BytePlusConfig.TTS_RESOURCE_ID,
                "Content-Type": "application/json",
            },
            json=payload,
            stream=True,
            timeout=kwargs.get("timeout", self.timeout),
        )
        try:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                message = json.loads(line)
                if message.get("code") not in (0, 20_000_000):
                    raise RuntimeError(message.get("message", "BytePlus TTS failed"))
                if message.get("data"):
                    yield base64.b64decode(message["data"])
        finally:
            response.close()

    def synthesize(self, text: str, **kwargs) -> bytes:
        return b"".join(self.stream(text, **kwargs))

    def duration_seconds(self, pcm: bytes) -> float:
        return len(pcm) / 2 / self.sample_rate


TEXT = "你好，我是语音助手。今天有什么可以帮你的吗？"


def demo_synthesize() -> None:
    """Synthesize one sentence and write it out for listening."""
    tts = TTSHelper()
    audio = tts.synthesize(TEXT)
    OutputConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OutputConfig.OUTPUT_DIR / "tts_demo.pcm"
    path.write_bytes(audio)
    print(f"合成 {len(audio)} 字节，约 {tts.duration_seconds(audio):.2f} 秒 -> {path}")
    assert audio


def demo_first_chunk_latency() -> None:
    """Time to first audio chunk.

    This is the number the user actually feels -- total synthesis time is
    irrelevant when the audio is streamed out as it arrives.
    """
    tts = TTSHelper()
    started = time.monotonic()
    first_ms = None
    total = 0
    for chunk in tts.stream(TEXT):
        if first_ms is None:
            first_ms = (time.monotonic() - started) * 1_000
        total += len(chunk)
    budget = BytePlusConfig.TTS_FIRST_CHUNK_GOOD_MS
    verdict = "达标" if first_ms <= budget else "偏慢"
    print(f"首包 {first_ms:.0f} ms（目标 <{budget} ms，{verdict}），共 {total} 字节")


def main() -> None:
    use_utf8_output()
    demo_synthesize()
    demo_first_chunk_latency()


if __name__ == "__main__":
    main()
