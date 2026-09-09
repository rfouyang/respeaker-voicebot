"""BytePlus Seed ASR bidirectional-streaming client.

Ported unchanged from services/robot-concierge, where the framing, the
throttled draining of partial results and the overlapped audio read were all
settled against the live service. Do not "simplify" the drain loop or the
to_thread read: a sequential read-then-send fell behind real time and lost
short utterances.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import gzip
import json
import struct
import threading
import uuid

import aiohttp

from config.settings import (AudioConfig, BytePlusConfig, OutputConfig,
                             use_utf8_output)
from util.byteplus_helper import BytePlusHelper


class ASRHelper(BytePlusHelper):
    """Stream audio to BytePlus and return one finished utterance."""

    def __init__(self, api_key: str | None = None, **kwargs) -> None:
        super().__init__(api_key)
        self.frame_ms = kwargs.get("frame_ms", BytePlusConfig.ASR_FRAME_MS)
        self.end_window_ms = kwargs.get(
            "end_window_ms", BytePlusConfig.ASR_END_WINDOW_MS
        )
        self.responses_per_chunk = max(1, int(kwargs.get("responses_per_chunk", 1)))

    @staticmethod
    def _frame(message_type: int, sequence: int, payload: bytes, last: bool = False) -> bytes:
        flags = 3 if last else 1
        header = bytes([0x11, message_type << 4 | flags, 0x11, 0x00])
        compressed = gzip.compress(payload)
        sequence = -sequence if last else sequence
        return header + struct.pack(">iI", sequence, len(compressed)) + compressed

    @staticmethod
    def _parse(data: bytes) -> dict:
        """Unwrap one BytePlus frame: header, optional prefixes, then JSON.

        Each flag adds its own 4-byte prefix ahead of the payload, and the
        payload is only gzipped when the compression nibble says so. The
        handshake reply, for instance, is neither.
        """
        header_size = (data[0] & 0x0F) * 4
        message_type = data[1] >> 4
        flags = data[1] & 0x0F
        compression = data[2] & 0x0F
        payload = data[header_size:]

        if flags & 1:  # sequence number
            payload = payload[4:]
        if flags & 4:  # event code
            payload = payload[4:]

        if message_type == 9:  # full server response: 4-byte payload size
            payload = payload[4:]
        elif message_type == 15:  # error frame: code + size + message
            code = struct.unpack(">i", payload[:4])[0]
            size = struct.unpack(">I", payload[4:8])[0]
            message = payload[8 : 8 + size]
            if compression == 1:
                message = gzip.decompress(message)
            raise RuntimeError(
                f"BytePlus ASR error {code}: {message.decode(errors='replace')}"
            )

        if not payload:
            return {}
        if compression == 1:
            payload = gzip.decompress(payload)
        return json.loads(payload)

    @staticmethod
    def _text(payload: dict) -> str:
        result = payload.get("result", {})
        results = result if isinstance(result, list) else [result]
        return " ".join(item.get("text", "").strip() for item in results).strip()

    @staticmethod
    def _is_final(payload: dict) -> bool:
        result = payload.get("result", {})
        results = result if isinstance(result, list) else [result]
        return any(
            utterance.get("definite", False)
            for item in results
            for utterance in item.get("utterances", [])
        )

    def _headers(self) -> dict[str, str]:
        return {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": BytePlusConfig.ASR_RESOURCE_ID,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }

    def _config(self) -> dict:
        return {
            "user": {"uid": BytePlusConfig.ASR_UID},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": AudioConfig.SAMPLE_RATE,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": BytePlusConfig.ASR_MODEL,
                "enable_itn": True,
                "enable_punc": True,
                "show_utterances": True,
                "end_window_size": self.end_window_ms,
            },
        }

    async def listen_async(
        self,
        audio_source: Callable[[int], bytes],
        speech_timeout: float | None = 10,
        on_partial: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
        max_utterance_seconds: float | None = None,
    ) -> str:
        """Stream one utterance, reporting changing partial transcripts."""
        frame_samples = AudioConfig.SAMPLE_RATE * self.frame_ms // 1_000
        sequence = 2
        text = ""
        reported_text = ""
        speech_started = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + speech_timeout if speech_timeout is not None else None
        utterance_deadline: float | None = None

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                BytePlusConfig.ASR_URL, headers=self._headers()
            ) as socket:
                await socket.send_bytes(
                    self._frame(1, 1, json.dumps(self._config()).encode())
                )
                self._parse((await socket.receive()).data)

                # Read the next block while the current one is sent and server
                # responses are drained. Sequential read then network I/O fell
                # behind real time and lost short utterances.
                next_chunk = asyncio.create_task(
                    asyncio.to_thread(audio_source, frame_samples)
                )
                try:
                    while True:
                        chunk = await next_chunk
                        next_chunk = asyncio.create_task(
                            asyncio.to_thread(audio_source, frame_samples)
                        )
                        if stop_event and stop_event.is_set():
                            await socket.send_bytes(
                                self._frame(2, sequence, b"", last=True)
                            )
                            return ""
                        await socket.send_bytes(self._frame(2, sequence, chunk))
                        sequence += 1

                        # The service may continuously queue partial updates.
                        # Draining until a quiet gap throttles reads to roughly
                        # half real time, so consume a bounded number and return
                        # to audio immediately. Later iterations pick up any
                        # queued update, including the final result.
                        for _ in range(self.responses_per_chunk):
                            try:
                                message = await socket.receive(timeout=0.01)
                            except asyncio.TimeoutError:
                                break
                            if message.type in {
                                aiohttp.WSMsgType.CLOSE,
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            }:
                                raise ConnectionError(
                                    "BytePlus ASR socket closed before final text"
                                )
                            if message.type != aiohttp.WSMsgType.BINARY:
                                continue
                            payload = self._parse(message.data)
                            text = self._text(payload) or text
                            if text and not speech_started:
                                speech_started = True
                                if max_utterance_seconds is not None:
                                    utterance_deadline = (
                                        loop.time() + max_utterance_seconds
                                    )
                            if text and text != reported_text:
                                reported_text = text
                                if on_partial:
                                    on_partial(text)
                            if self._is_final(payload):
                                await socket.send_bytes(
                                    self._frame(2, sequence, b"", last=True)
                                )
                                return text

                        if (
                            deadline is not None
                            and not speech_started
                            and loop.time() >= deadline
                        ):
                            await socket.send_bytes(
                                self._frame(2, sequence, b"", last=True)
                            )
                            return ""
                        if (
                            utterance_deadline is not None
                            and loop.time() >= utterance_deadline
                        ):
                            await socket.send_bytes(
                                self._frame(2, sequence, b"", last=True)
                            )
                            return text
                finally:
                    # Cancelling to_thread does not stop its worker. Wait for
                    # the single in-flight read before a new ASR session starts.
                    try:
                        await next_chunk
                    except Exception:
                        pass

    def listen(
        self,
        audio_source: Callable[[int], bytes],
        speech_timeout: float | None = 10,
        on_partial: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
        max_utterance_seconds: float | None = None,
    ) -> str:
        return asyncio.run(
            self.listen_async(
                audio_source, speech_timeout, on_partial, stop_event,
                max_utterance_seconds,
            )
        )

    async def transcribe_async(self, pcm: bytes, realtime: bool = True) -> tuple[str, float]:
        """Recognize a PCM buffer; return the text and the tail latency.

        Tail latency is the wait between the last audio frame and the final
        transcript, the delay a user actually feels after they stop talking.
        ``realtime`` paces the upload like live speech so that number means the
        same thing it will on the device.
        """
        frame_samples = AudioConfig.SAMPLE_RATE * self.frame_ms // 1_000
        frame_bytes = frame_samples * 2
        sequence = 2
        text = ""
        loop = asyncio.get_running_loop()

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                BytePlusConfig.ASR_URL, headers=self._headers()
            ) as socket:
                await socket.send_bytes(
                    self._frame(1, 1, json.dumps(self._config()).encode())
                )
                self._parse((await socket.receive()).data)

                for offset in range(0, len(pcm), frame_bytes):
                    chunk = pcm[offset : offset + frame_bytes]
                    await socket.send_bytes(self._frame(2, sequence, chunk))
                    sequence += 1
                    if realtime:
                        await asyncio.sleep(self.frame_ms / 1_000)
                    while True:
                        try:
                            message = await socket.receive(timeout=0.001)
                        except asyncio.TimeoutError:
                            break
                        if message.type == aiohttp.WSMsgType.BINARY:
                            text = self._text(self._parse(message.data)) or text

                sent_at = loop.time()
                await socket.send_bytes(self._frame(2, sequence, b"", last=True))
                while True:
                    try:
                        message = await socket.receive(timeout=10)
                    except asyncio.TimeoutError:
                        break
                    if message.type != aiohttp.WSMsgType.BINARY:
                        break
                    payload = self._parse(message.data)
                    text = self._text(payload) or text
                    if self._is_final(payload) or message.data[1] & 2:
                        break
                return text, (loop.time() - sent_at) * 1_000

    def transcribe(self, pcm: bytes, realtime: bool = True) -> tuple[str, float]:
        return asyncio.run(self.transcribe_async(pcm, realtime))


SPOKEN = "今天天气怎么样？请帮我打开客厅的灯。"


def _synthesized_pcm(text: str) -> bytes:
    """TTS the sentence so ASR can be exercised without a microphone."""
    from util.tts_helper import TTSHelper

    return TTSHelper().synthesize(text)


def demo_transcribe_loopback() -> None:
    """TTS a known sentence, feed it back to ASR, compare.

    A closed loop that needs no microphone, so it still works with the XVF3800
    in I2S mode, where the PC has no audio device for it at all.
    """
    from difflib import SequenceMatcher

    pcm = _synthesized_pcm(SPOKEN)
    text, _tail_ms = ASRHelper().transcribe(pcm, realtime=False)
    ratio = SequenceMatcher(None, SPOKEN, text).ratio()
    print(f"原文: {SPOKEN}")
    print(f"识别: {text}")
    print(f"相似度 {ratio:.2f}")
    assert ratio > 0.6, f"识别偏差过大: {text!r}"


def demo_tail_latency() -> None:
    """How long after the last audio frame the final transcript arrives."""
    pcm = _synthesized_pcm(SPOKEN)
    text, tail_ms = ASRHelper().transcribe(pcm, realtime=True)
    budget = BytePlusConfig.ASR_TAIL_GOOD_MS
    verdict = "达标" if tail_ms <= budget else "偏慢"
    print(f"尾延迟 {tail_ms:.0f} ms（目标 <{budget} ms，{verdict}）-> {text}")


def demo_listen_from_buffer() -> None:
    """Drive listen() from a buffer, the call shape the device link will use."""
    pcm = _synthesized_pcm(SPOKEN)
    OutputConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OutputConfig.OUTPUT_DIR / "asr_demo.pcm").write_bytes(pcm)
    offset = 0

    def read(frames: int) -> bytes:
        nonlocal offset
        chunk = pcm[offset : offset + frames * 2]
        offset += frames * 2
        return chunk or b"\x00" * (frames * 2)  # silence once the buffer drains

    partials: list[str] = []
    text = ASRHelper().listen(read, speech_timeout=5, on_partial=partials.append)
    print(f"listen() 得到: {text}（{len(partials)} 次部分结果）")
    assert text


def main() -> None:
    use_utf8_output()
    demo_transcribe_loopback()
    demo_tail_latency()
    demo_listen_from_buffer()


if __name__ == "__main__":
    main()
