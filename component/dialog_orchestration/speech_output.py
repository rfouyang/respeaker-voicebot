"""Step: reply text -> PCM on the device.

Owns the TTSHelper. Splits the reply at sentence boundaries because the
BytePlus unidirectional API takes a whole request per piece of text: smaller
pieces mean more round trips and broken prosody, larger ones delay the first
audio.

Two things here exist only for barge-in:

* The interrupt event is checked between every packet, not just between
  sentences. A sentence can be several seconds long, and a barge-in that waits
  for the current sentence to finish is not a barge-in.
* `delivered_ms` counts what actually went out on the wire, which is what
  `Turn.spoken_text` uses to decide how much of the reply the user heard.
"""

from __future__ import annotations

import re
import threading

from config.settings import DeviceConfig, DialogConfig, use_utf8_output
from component.dialog_orchestration.turn import Turn
from util.device_frame_helper import Ctrl
from util.tts_helper import TTSHelper


class SpeechOutput:
    """Synthesize a reply sentence by sentence and stream it to the device."""

    def __init__(self, tts: TTSHelper | None = None, **kwargs) -> None:
        self.tts = tts or TTSHelper(**kwargs)
        self.max_chars = kwargs.get("max_chars", DialogConfig.MAX_SENTENCE_CHARS)
        self.chunk_bytes = kwargs.get("chunk_bytes", DeviceConfig.DOWNLINK_CHUNK_BYTES)
        self.spoken_sentences = 0

    def split_sentences(self, text: str) -> list[str]:
        """Break a reply into synthesis-sized pieces.

        Punctuation is kept with the sentence it ends, and a run-on with no
        punctuation is cut at `max_chars` so the first audio is not held
        hostage to a full stop the model never wrote.
        """
        pieces: list[str] = []
        current = ""
        for char in text:
            current += char
            if re.match(DialogConfig.SENTENCE_END, char) or len(current) >= self.max_chars:
                if current.strip():
                    pieces.append(current.strip())
                current = ""
        if current.strip():
            pieces.append(current.strip())
        return pieces

    def speak(self, turn: Turn, text: str, link) -> bool:
        """Stream `text` to the device. Returns True if it was interrupted.

        Records queued text and delivered audio on the turn as it goes, so
        that a barge-in at any point leaves an accurate account of what the
        user actually heard.
        """
        for sentence in self.split_sentences(text):
            if turn.interrupt_event.is_set():
                return True
            turn.queued_text += sentence
            for chunk in self.tts.stream(sentence):
                for offset in range(0, len(chunk), self.chunk_bytes):
                    if turn.interrupt_event.is_set():
                        return True
                    packet = chunk[offset : offset + self.chunk_bytes]
                    link.transport.write(
                        link.frame.encode_audio_down(turn.id, packet)
                    )
                    turn.delivered_ms += len(packet) / DeviceConfig.BYTES_PER_MS
            self.spoken_sentences += 1
        link.send_control(Ctrl.TTS_END, turn.id)
        return False


def demo_split_sentences() -> None:
    output = SpeechOutput.__new__(SpeechOutput)  # no cloud client needed
    output.max_chars = 60
    pieces = output.split_sentences("你好。今天天气不错！要出门吗？")
    print("切句:", pieces)
    assert pieces == ["你好。", "今天天气不错！", "要出门吗？"]

    long_text = "啊" * 130
    pieces = output.split_sentences(long_text)
    print(f"无标点长句切成 {len(pieces)} 段，最长 {max(len(p) for p in pieces)} 字")
    assert all(len(p) <= 60 for p in pieces)


def demo_speak_to_fake_device() -> None:
    """A full reply reaches the device, and the turn accounts for it."""
    import time
    from util.device_link_helper import DeviceLinkHelper, FakeDevice

    host, device_link = DeviceLinkHelper.paired()
    device = FakeDevice(device_link)
    threading.Thread(target=device.run, daemon=True).start()

    turn = Turn(1)
    device.turn_id = 1
    interrupted = SpeechOutput().speak(turn, "你好，我是语音助手。", host)
    time.sleep(0.4)
    device_link.stop()

    print(f"送达设备 {device.received_audio_bytes} 字节，"
          f"turn 记账 {turn.delivered_ms:.0f} ms，被打断={interrupted}")
    assert not interrupted
    assert device.received_audio_bytes > 0
    assert (Ctrl.TTS_END, 1) in device.received_control


def demo_interrupt_stops_mid_reply() -> None:
    """Setting the interrupt event stops output part way, and the turn then
    reports only what was delivered."""
    import time
    from util.device_link_helper import DeviceLinkHelper, FakeDevice

    host, device_link = DeviceLinkHelper.paired()
    device = FakeDevice(device_link)
    threading.Thread(target=device.run, daemon=True).start()

    turn = Turn(2)
    device.turn_id = 2
    reply = "这是一段很长的回复。它有好几句话。用来验证打断能在中途生效。最后一句不该被说出来。"

    def interrupt_soon() -> None:
        time.sleep(0.8)
        turn.cancel()

    threading.Thread(target=interrupt_soon, daemon=True).start()
    interrupted = SpeechOutput().speak(turn, reply, host)
    time.sleep(0.3)
    device_link.stop()

    print(f"被打断={interrupted}，送进 TTS {len(turn.queued_text)} 字，"
          f"实际送出 {turn.delivered_ms:.0f} ms -> 听到约 {len(turn.spoken_text)} 字")
    print("回填历史:", turn.history_entry())
    assert interrupted
    assert len(turn.queued_text) < len(reply), "打断后不应继续合成剩余句子"


def main() -> None:
    use_utf8_output()
    demo_split_sentences()
    demo_speak_to_fake_device()
    demo_interrupt_stops_mid_reply()


if __name__ == "__main__":
    main()
