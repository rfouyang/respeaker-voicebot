"""Bring-up checks that need the real board on the other end of the link.

The question this answers first: the XVF3800 puts something different in each
of the two I2S slots, and only one of them is the processed speech. Feeding
the wrong one to WakeNet and the recogniser costs wake rate and accuracy, so
rather than guess, record both and let ASR say which is which.
"""

from __future__ import annotations

import array
import math
from pathlib import Path
import time
import wave

from config.settings import (AudioConfig, DeviceConfig, OutputConfig,
                             use_utf8_output)
from util.device_frame_helper import DeviceFrameHelper, MsgType


class DeviceProbeHelper:
    """Capture stereo PCM straight off the link and take it apart."""

    def __init__(self, **kwargs) -> None:
        self.port = kwargs.get("port", DeviceConfig.PORT)
        self.baud = kwargs.get("baud", DeviceConfig.BAUD)
        self.sample_rate = kwargs.get("sample_rate", AudioConfig.SAMPLE_RATE)
        self.out_dir = Path(kwargs.get("out_dir", OutputConfig.OUTPUT_DIR / "probe"))
        self.frame = DeviceFrameHelper()
        self.left = bytearray()
        self.right = bytearray()
        self.frames_seen = 0

    # ---- capture ----

    def capture(self, seconds: float = 8.0) -> float:
        """Read the link for a while, splitting the interleaved stereo.

        Returns the seconds of audio actually captured, which is the number
        worth checking: well under `seconds` means the link is not keeping up.
        """
        import serial

        link = serial.Serial(self.port, self.baud, timeout=0.1)
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                data = link.read(max(1, link.in_waiting))
                if not data:
                    continue
                for frame in self.frame.feed(data):
                    if frame.type == MsgType.AUDIO_UP:
                        self.frames_seen += 1
                        self.split(frame.payload)
        finally:
            link.close()
        return self.seconds

    def play_and_capture(self, pcm: bytes, tail: float = 2.0) -> float:
        """Send `pcm` down for the device to play, capturing uplink throughout.

        One port, both directions, so the echo and the audio that caused it are
        recorded against the same clock. Downlink is paced at real time rather
        than dumped: the device's playback ring is only 250 ms deep, and
        flooding it would just drop most of the audio.
        """
        import serial

        from util.device_link_helper import DeviceLinkHelper

        # The device plays at 48 kHz even though it records at 16 kHz.
        pcm = DeviceLinkHelper.to_playback_rate(
            pcm, self.sample_rate, DeviceConfig.PLAYBACK_SAMPLE_RATE
        )
        link = serial.Serial(self.port, self.baud, timeout=0.05)
        chunk = DeviceConfig.DOWNLINK_CHUNK_BYTES
        chunk_seconds = chunk / 2 / DeviceConfig.PLAYBACK_SAMPLE_RATE
        sender = DeviceFrameHelper()

        # Prime the ring before starting the clock. Beginning at exactly real
        # time leaves no slack at all, so the very first scheduling hiccup is
        # already an underrun -- which is heard as a stutter at the start of
        # every reply.
        offset = 0
        primed = int(DeviceConfig.PLAYBACK_SAMPLE_RATE * 2 * 0.15)  # 150 ms
        while offset < min(primed, len(pcm)):
            part = pcm[offset : offset + chunk]
            link.write(sender.encode_audio_down(0, part))
            offset += len(part)

        # Pace against a virtual clock, not against "now". Adding the interval
        # to the current time each round lets scheduling delays accumulate, and
        # the device falls further behind with every chunk.
        started = time.monotonic()
        sent_chunks = 0
        deadline = started + len(pcm) / 2 / DeviceConfig.PLAYBACK_SAMPLE_RATE + tail
        try:
            while time.monotonic() < deadline:
                now = time.monotonic()
                if offset < len(pcm) and now - started >= sent_chunks * chunk_seconds:
                    part = pcm[offset : offset + chunk]
                    link.write(sender.encode_audio_down(0, part))
                    offset += len(part)
                    sent_chunks += 1
                data = link.read(max(1, link.in_waiting))
                if data:
                    for frame in self.frame.feed(data):
                        if frame.type == MsgType.AUDIO_UP:
                            self.frames_seen += 1
                            self.split(frame.payload)
                else:
                    time.sleep(0.002)
        finally:
            link.close()
        return self.seconds

    @staticmethod
    def _deinterleave(interleaved: bytes) -> tuple[bytes, bytes]:
        """L,R,L,R... int16 little endian -> two mono buffers.

        Each sample is two bytes, so a stereo pair is four: L-low, L-high,
        R-low, R-high.
        """
        usable = len(interleaved) // 4 * 4
        data = interleaved[:usable]
        left = bytearray()
        right = bytearray()
        for offset in range(0, usable, 4):
            left += data[offset : offset + 2]
            right += data[offset + 2 : offset + 4]
        return bytes(left), bytes(right)

    def split(self, interleaved: bytes) -> None:
        left, right = self._deinterleave(interleaved)
        self.left.extend(left)
        self.right.extend(right)

    @property
    def seconds(self) -> float:
        return len(self.left) / 2 / self.sample_rate

    # ---- analysis ----

    @staticmethod
    def rms(pcm: bytes) -> float:
        samples = array.array("h")
        samples.frombytes(pcm[: len(pcm) // 2 * 2])
        if not samples:
            return 0.0
        return math.sqrt(sum(s * s for s in samples) / len(samples))

    def describe(self, pcm: bytes) -> dict:
        """Measurable properties of a recording, for when ASR just says "(空)".

        A failed transcript tells you something is wrong but not what. These
        three numbers separate the usual causes, and did so in practice:

        * `span_seconds` against what was played -- much shorter means it was
          played too fast, i.e. a sample-rate mismatch.
        * `gap_count` -- periodic near-silent runs mean buffer underruns.
        * `zero_crossing_rate` -- speech sits around 1000-3000 per second.
          Far above that is high-frequency hash riding on top of the speech,
          which is what a wrong slot or bit alignment produces.
        """
        samples = array.array("h")
        samples.frombytes(pcm[: len(pcm) // 2 * 2])
        if not samples:
            return {"seconds": 0.0, "span_seconds": 0.0, "gap_count": 0,
                    "zero_crossing_rate": 0.0}

        step = self.sample_rate // 20  # 50 ms
        envelope = []
        for start in range(0, len(samples) - step, step):
            window = samples[start : start + step]
            envelope.append(math.sqrt(sum(s * s for s in window) / len(window)))
        loud = [i for i, level in enumerate(envelope) if level > 300]

        gaps, run = 0, 0
        for sample in samples:
            if abs(sample) < 20:
                run += 1
            else:
                if run >= self.sample_rate // 1000:  # >= 1 ms
                    gaps += 1
                run = 0

        crossings = sum(
            1 for i in range(1, len(samples)) if (samples[i - 1] < 0) != (samples[i] < 0)
        )
        return {
            "seconds": len(samples) / self.sample_rate,
            "span_seconds": (loud[-1] - loud[0]) * 0.05 if loud else 0.0,
            "gap_count": gaps,
            "zero_crossing_rate": crossings / (len(samples) / self.sample_rate),
        }

    def write_wav(self, name: str, pcm: bytes) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / name
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm)
        return path

    def identify_speech_channel(self) -> dict:
        """Run each channel through ASR. The one that transcribes is the
        processed output; the other is whatever else the XVF3800 emits."""
        from util.asr_helper import ASRHelper

        results = {}
        for name, pcm in (("left", bytes(self.left)), ("right", bytes(self.right))):
            text, _tail = ASRHelper().transcribe(pcm, realtime=False)
            results[name] = {
                "text": text,
                "rms": round(self.rms(pcm), 1),
                "wav": str(self.write_wav(f"{name}.wav", pcm)),
            }
        return results


def demo_deinterleave() -> None:
    """The split has to be right before any conclusion drawn from it means
    anything."""
    probe = DeviceProbeHelper()
    # L = 1, 2, 3 ; R = -1, -2, -3
    pairs = array.array("h", [1, -1, 2, -2, 3, -3]).tobytes()
    left, right = probe._deinterleave(pairs)
    assert array.array("h", left).tolist() == [1, 2, 3]
    assert array.array("h", right).tolist() == [-1, -2, -3]
    print("去交织正确: L=[1,2,3] R=[-1,-2,-3]")

    # A payload cut mid-pair must not shift every later sample by one channel.
    left, right = probe._deinterleave(pairs + b"\x01")
    assert array.array("h", left).tolist() == [1, 2, 3]
    print("半个采样对的残尾被丢弃，不会错位")


def demo_capture_and_identify() -> None:
    """Needs the board flashed with the streaming firmware. Speak while it
    runs -- a silent recording proves nothing either way."""
    probe = DeviceProbeHelper()
    print(f"从 {probe.port} 采集 8 秒，请说话...")
    seconds = probe.capture(8.0)
    print(f"收到 {probe.frames_seen} 帧 / {seconds:.1f} 秒"
          f"（丢弃非帧字节 {probe.frame.dropped_bytes}，多半是日志文本）")
    if seconds < 1:
        print("音频太少，先确认固件是流式版本且板子已连接")
        return

    print(f"电平: L={probe.rms(bytes(probe.left)):.0f}  "
          f"R={probe.rms(bytes(probe.right)):.0f}")
    for name, info in probe.identify_speech_channel().items():
        print(f"  {name:5s} rms={info['rms']:8.1f}  识别: {info['text'] or '(空)'}")
        print(f"        -> {info['wav']}")
    print("能识别出文字的那一路就是处理后的语音通道。")


def demo_speak_and_identify() -> None:
    """Drive the whole check without a human: TTS out of the PC speaker, into
    the microphone array, back over the link, then ASR on each channel.

    This is the JBL playing the part of a person talking. It is NOT the
    playback path the product uses -- the reply has to leave through the
    XVF3800 so its AEC sees the echo reference. Here the point is the
    opposite: be an external voice the array has to pick up.
    """
    import threading

    import numpy as np
    import sounddevice as sd

    from util.tts_helper import TTSHelper

    # Deliberately inert. This gets played out loud into the room, so it must
    # not be something another listening device could act on -- no smart-home
    # commands, no wake words, nothing addressed to an assistant.
    spoken = "春天的湖面很平静，远处有三只白色的鸟慢慢飞过。"
    tts = TTSHelper()
    pcm = tts.synthesize(spoken)
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    # Loud enough to carry across the desk, short of clipping. Same reasoning
    # as VISITOR_SPEAKER_GAIN in services/robot-concierge.
    samples = np.clip(samples * 1.7, -32768, 32767).astype(np.int16)
    duration = len(samples) / tts.sample_rate

    probe = DeviceProbeHelper()
    device_name = sd.query_devices(kind="output")["name"]
    print(f"从扬声器播放 {duration:.1f} 秒语音（{device_name}），同时采集...")
    print(f"内容: {spoken}")

    def play() -> None:
        time.sleep(1.0)  # let capture settle, and bluetooth wake up
        sd.play(samples, tts.sample_rate)
        sd.wait()

    player = threading.Thread(target=play, daemon=True)
    player.start()
    probe.capture(duration + 2.5)
    player.join(timeout=5)

    print(f"收到 {probe.frames_seen} 帧 / {probe.seconds:.1f} 秒"
          f"（丢弃非帧字节 {probe.frame.dropped_bytes}）")
    print(f"电平: L={probe.rms(bytes(probe.left)):.0f}  "
          f"R={probe.rms(bytes(probe.right)):.0f}")

    results = probe.identify_speech_channel()
    for name, info in results.items():
        print(f"  {name:5s} rms={info['rms']:8.1f}  识别: {info['text'] or '(空)'}")
        print(f"        -> {info['wav']}")

    winners = [n for n, i in results.items() if i["text"]]
    if len(winners) == 1:
        print(f"\n处理后的语音通道 = {winners[0]}")
    elif len(winners) == 2:
        print("\n两路都能识别；比较文本质量和电平来决定用哪一路")
    else:
        print("\n两路都没识别出来。把音量调大、板子挪近扬声器再试")


def main() -> None:
    use_utf8_output()
    demo_deinterleave()
    demo_speak_and_identify()


if __name__ == "__main__":
    main()
