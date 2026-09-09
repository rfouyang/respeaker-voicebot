"""Step: device audio -> text.

This is the adapter between the device link and `ASRHelper`. The helper wants
a blocking `read(frames) -> bytes` callable, exactly as it had when it was
reading a sound card in services/robot-concierge; here that callable is served
from a buffer the link fills instead. Keeping that shape is deliberate -- the
recogniser did not have to change at all to move from USB audio to USB CDC.

Owns the ASRHelper. The orchestrating service owns none.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import threading
import time

from config.settings import AudioConfig, DialogConfig, use_utf8_output
from util.asr_helper import ASRHelper


class Transcription:
    """Buffer uplink PCM and recognise one utterance from it."""

    def __init__(self, asr: ASRHelper | None = None, **kwargs) -> None:
        self.asr = asr or ASRHelper(**kwargs)
        self.chunks: deque[bytes] = deque()
        self.pending = bytearray()
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.starve_seconds = kwargs.get(
            "starve_seconds", DialogConfig.AUDIO_STARVE_SECONDS
        )
        self.fed_bytes = 0
        # Non-zero means the link could not keep up with real time. Worth
        # logging on real hardware.
        self.starved_reads = 0

    def feed(self, pcm: bytes) -> None:
        """Called from the link thread as AUDIO_UP frames arrive."""
        with self.condition:
            self.chunks.append(pcm)
            self.fed_bytes += len(pcm)
            self.condition.notify_all()

    def reset(self) -> None:
        """Drop buffered audio so a new turn does not inherit the last one."""
        with self.condition:
            self.chunks.clear()
            self.pending.clear()
            self.stop_event.clear()

    def read(self, frames: int) -> bytes:
        """Blocking read of `frames` samples, the shape ASRHelper expects.

        Pads with silence instead of blocking forever, on stop and on
        starvation alike. The device streams continuously, so a gap means the
        link stalled -- and the right answer to that is to let the recogniser
        end the utterance and the session time out, not to hang this thread
        with nothing to show for it.
        """
        wanted = frames * 2
        deadline = time.monotonic() + self.starve_seconds
        with self.condition:
            while len(self.pending) < wanted and not self.stop_event.is_set():
                if self.chunks:
                    self.pending.extend(self.chunks.popleft())
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.starved_reads += 1
                    break
                self.condition.wait(min(0.05, remaining))
            if len(self.pending) >= wanted:
                out = bytes(self.pending[:wanted])
                del self.pending[:wanted]
                return out
            out = bytes(self.pending)
            self.pending.clear()
        return out + b"\x00" * (wanted - len(out))

    def stop(self) -> None:
        with self.condition:
            self.stop_event.set()
            self.condition.notify_all()

    def listen(
        self,
        speech_timeout: float | None = DialogConfig.IDLE_TIMEOUT_SECONDS,
        on_partial: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> str:
        """Recognise one utterance from whatever the device sends."""
        return self.asr.listen(
            self.read,
            speech_timeout=speech_timeout,
            on_partial=on_partial,
            stop_event=stop_event,
            max_utterance_seconds=DialogConfig.MAX_UTTERANCE_SECONDS,
        ).strip()


def _speech_pcm(text: str) -> bytes:
    from util.tts_helper import TTSHelper

    return TTSHelper().synthesize(text)


def demo_recognise_fed_audio() -> None:
    """Feed synthesized speech in device-sized frames; get the text back.

    No microphone and no hardware: this is the same path the link will drive,
    only the bytes come from TTS instead of the XVF3800.
    """
    spoken = "请帮我把客厅的灯打开。"
    pcm = _speech_pcm(spoken)
    transcription = Transcription()

    def pump() -> None:
        step = AudioConfig.FRAME_BYTES  # 20 ms, as the device will send
        for offset in range(0, len(pcm), step):
            transcription.feed(pcm[offset : offset + step])

    threading.Thread(target=pump, daemon=True).start()

    partials: list[str] = []
    text = transcription.listen(speech_timeout=8, on_partial=partials.append)
    print(f"喂入 {transcription.fed_bytes} 字节 -> 识别: {text}")
    print(f"（{len(partials)} 次部分结果）")
    assert text, "没有识别出任何文本"


def demo_reset_drops_stale_audio() -> None:
    """A new turn must not inherit the previous turn's leftover audio."""
    transcription = Transcription()
    transcription.feed(b"\x11\x22" * 800)
    assert transcription.fed_bytes == 1_600
    transcription.reset()
    transcription.feed(b"\x33\x44" * 10)
    got = transcription.read(10)
    assert got == b"\x33\x44" * 10, got[:8]
    print("reset() 后读到的是新音频，不是上一轮的残留")


def demo_read_does_not_hang_when_starved() -> None:
    """A stalled link must not hang the ASR thread."""
    transcription = Transcription(starve_seconds=0.2)
    started = time.monotonic()
    got = transcription.read(160)
    elapsed = time.monotonic() - started
    assert len(got) == 320 and got == b"\x00" * 320
    assert 0.15 < elapsed < 1.0, elapsed
    print(f"链路无数据时 {elapsed*1000:.0f}ms 后返回静音（饿死读 {transcription.starved_reads} 次）")


def demo_read_returns_silence_after_stop() -> None:
    """stop() must unblock read(), or the ASR thread never exits."""
    transcription = Transcription()
    transcription.stop()
    got = transcription.read(160)
    assert got == b"\x00" * 320 and len(got) == 320
    print("stop() 后 read() 立即返回静音，不会挂住")


def main() -> None:
    use_utf8_output()
    demo_recognise_fed_audio()
    demo_reset_drops_stale_audio()
    demo_read_does_not_hang_when_starved()
    demo_read_returns_silence_after_stop()


if __name__ == "__main__":
    main()
