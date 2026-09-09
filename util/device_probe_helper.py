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


def main() -> None:
    use_utf8_output()
    demo_deinterleave()
    demo_capture_and_identify()


if __name__ == "__main__":
    main()
