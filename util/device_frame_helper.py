"""Wire format between this host and the XIAO, over USB CDC.

Must stay byte-for-byte identical to the firmware's `protocol.hpp`. If you
change one side, change the other in the same commit: a mismatch here shows up
as garbled audio rather than as an error.

Adapted from services/voicebot, which ran this framing over a WebSocket. The
one real difference is that a WebSocket preserves message boundaries and a
serial port does not, so the header carries a magic and decoding is a stream
reassembler rather than a per-message call. Without that, one lost byte
desynchronises the link forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct

from config.settings import DeviceConfig, use_utf8_output


class MsgType(IntEnum):
    AUDIO_UP = 1  # device -> host, processed PCM from the XVF3800
    AUDIO_DOWN = 2  # host -> device, TTS PCM
    CONTROL = 3


class Ctrl(IntEnum):
    NONE = 0
    WAKE_DETECTED = 1  # device -> host
    SPEECH_END = 2  # device -> host, endpoint called by the local VAD
    INTERRUPT = 3  # device -> host, the user barged in
    TTS_BEGIN = 10  # host -> device
    TTS_END = 11  # host -> device


@dataclass(frozen=True)
class DeviceFrame:
    type: MsgType
    ctrl: Ctrl
    turn_id: int
    payload: bytes


class DeviceFrameHelper:
    """Encoder and streaming decoder for the 12-byte framing.

    `<HBBII` = magic, type, ctrl, turn_id, len. Little endian, because the
    ESP32 is little endian and the firmware writes the struct out raw.
    """

    HEADER = struct.Struct("<HBBII")

    def __init__(self, **kwargs) -> None:
        self.magic = kwargs.get("magic", DeviceConfig.MAGIC)
        self.max_payload = kwargs.get("max_payload", DeviceConfig.MAX_PAYLOAD_BYTES)
        self.bytes_per_ms = kwargs.get("bytes_per_ms", DeviceConfig.BYTES_PER_MS)
        self.buffer = bytearray()
        # Every byte thrown away while hunting for a header. Non-zero after
        # startup means the link is losing data, so it is worth logging.
        self.dropped_bytes = 0
        assert self.HEADER.size == 12

    # ---- encoding ----

    def _pack(self, mtype: MsgType, ctrl: int, turn_id: int, payload: bytes) -> bytes:
        return self.HEADER.pack(self.magic, mtype, ctrl, turn_id, len(payload)) + payload

    def encode_audio_down(self, turn_id: int, pcm: bytes) -> bytes:
        return self._pack(MsgType.AUDIO_DOWN, 0, turn_id, pcm)

    def encode_control(self, ctrl: Ctrl, turn_id: int) -> bytes:
        return self._pack(MsgType.CONTROL, ctrl, turn_id, b"")

    def encode_audio_up(self, turn_id: int, pcm: bytes) -> bytes:
        """Only a fake device sends uplink audio from Python; the real one
        builds this header in C++."""
        return self._pack(MsgType.AUDIO_UP, 0, turn_id, pcm)

    # ---- decoding ----

    def feed(self, data: bytes) -> list[DeviceFrame]:
        """Add received bytes and return whatever complete frames they finish.

        Callers hand over whatever the serial read returned, at any chunk size
        and split at any offset. Partial frames stay buffered until the rest
        arrives.
        """
        self.buffer.extend(data)
        frames: list[DeviceFrame] = []
        while (frame := self._take_one()) is not None:
            frames.append(frame)
        return frames

    def _take_one(self) -> DeviceFrame | None:
        while True:
            if not self._seek_magic():
                return None
            if len(self.buffer) < self.HEADER.size:
                return None
            _magic, mtype, ctrl, turn_id, length = self.HEADER.unpack_from(self.buffer)

            if not self._plausible(mtype, ctrl, length):
                # The magic matched inside PCM by chance. Step over it and
                # keep hunting rather than trusting a nonsense header.
                self._drop(1)
                continue
            if len(self.buffer) < self.HEADER.size + length:
                return None  # payload still in flight

            start = self.HEADER.size
            payload = bytes(self.buffer[start : start + length])
            del self.buffer[: start + length]
            return DeviceFrame(MsgType(mtype), Ctrl(ctrl), turn_id, payload)

    def _seek_magic(self) -> bool:
        """Drop leading garbage until the buffer starts at a magic."""
        wanted = struct.pack("<H", self.magic)
        if self.buffer[:2] == wanted:
            return True
        index = self.buffer.find(wanted)
        if index < 0:
            # Keep the last byte: the magic may straddle this chunk boundary.
            self._drop(max(0, len(self.buffer) - 1))
            return False
        self._drop(index)
        return True

    def _drop(self, count: int) -> None:
        if count > 0:
            self.dropped_bytes += count
            del self.buffer[:count]

    def _plausible(self, mtype: int, ctrl: int, length: int) -> bool:
        if mtype not in tuple(MsgType) or ctrl not in tuple(Ctrl):
            return False
        if length > self.max_payload:
            return False
        return length == 0 or mtype != MsgType.CONTROL

    def decode(self, raw: bytes) -> DeviceFrame:
        """Decode exactly one complete frame. For tests; the link uses feed()."""
        frames = DeviceFrameHelper(magic=self.magic).feed(raw)
        if not frames:
            raise ValueError(f"不是一个完整的帧: {len(raw)} 字节")
        return frames[0]

    def audio_ms(self, payload: bytes) -> float:
        return len(payload) / self.bytes_per_ms


def demo_round_trip() -> None:
    helper = DeviceFrameHelper()

    pcm = bytes(range(256)) * 2
    frame = helper.decode(helper.encode_audio_down(7, pcm))
    assert frame.type is MsgType.AUDIO_DOWN and frame.turn_id == 7
    assert frame.payload == pcm
    print(f"音频帧往返 OK: turn={frame.turn_id}, {helper.audio_ms(frame.payload):.0f} ms")

    frame = helper.decode(helper.encode_control(Ctrl.INTERRUPT, 42))
    assert frame.type is MsgType.CONTROL and frame.ctrl is Ctrl.INTERRUPT
    print(f"控制帧往返 OK: {frame.ctrl.name} turn={frame.turn_id}")


def demo_header_layout() -> None:
    """The firmware static_asserts sizeof(FrameHeader) == 12. Mirror that."""
    helper = DeviceFrameHelper()
    raw = helper.encode_control(Ctrl.TTS_BEGIN, 0x01020304)
    assert len(raw) == 12, len(raw)
    assert raw[:2] == b"\xa5\x5a", "魔数必须是小端 0x5AA5"
    assert raw[2:4] == bytes([MsgType.CONTROL, Ctrl.TTS_BEGIN])
    assert raw[4:8] == b"\x04\x03\x02\x01", "turn_id 必须是小端"
    print("12 字节头、魔数、小端布局与固件约定一致")


def demo_stream_reassembly() -> None:
    """The serial-specific case: one frame split at every possible offset.

    A WebSocket would hand the whole message over at once. A serial read
    returns whatever happens to be in the buffer, so every split must work.
    """
    helper = DeviceFrameHelper()
    pcm = bytes(range(200))
    raw = helper.encode_audio_down(3, pcm) + helper.encode_control(Ctrl.TTS_END, 3)

    for split in range(1, len(raw)):
        reader = DeviceFrameHelper()
        frames = reader.feed(raw[:split]) + reader.feed(raw[split:])
        assert len(frames) == 2, f"split={split} 得到 {len(frames)} 帧"
        assert frames[0].payload == pcm
        assert frames[1].ctrl is Ctrl.TTS_END
    print(f"{len(raw) - 1} 种拆分位置全部正确重组")

    # Byte-at-a-time, the worst case a slow link can produce.
    reader = DeviceFrameHelper()
    frames = [f for byte in raw for f in reader.feed(bytes([byte]))]
    assert len(frames) == 2
    print("逐字节喂入也能重组")


def demo_resync_after_garbage() -> None:
    """Startup mid-stream, or a lost byte, must not break the link forever."""
    helper = DeviceFrameHelper()
    good = helper.encode_control(Ctrl.WAKE_DETECTED, 9)

    reader = DeviceFrameHelper()
    frames = reader.feed(b"\x00\x01\x02rubbish\xff" + good)
    assert len(frames) == 1 and frames[0].ctrl is Ctrl.WAKE_DETECTED
    print(f"前置垃圾后重同步 OK，丢弃 {reader.dropped_bytes} 字节")

    # A damaged header -- magic destroyed -- is skipped and the next real
    # frame is found.
    reader = DeviceFrameHelper()
    damaged = bytearray(helper.encode_audio_down(1, b"xyz"))
    damaged[0] ^= 0xFF
    frames = reader.feed(bytes(damaged) + good)
    assert len(frames) == 1 and frames[0].ctrl is Ctrl.WAKE_DETECTED
    print("帧头损坏后仍能找到下一帧")

    # What magic framing CANNOT catch: an intact header whose payload was cut
    # short. The length field is believed, so the next frame's first bytes get
    # eaten and one good frame is lost; the link resynchronises on the frame
    # after that. Detecting this needs a checksum, which is deliberately not
    # here -- USB CDC delivers bytes in order and does not silently drop them,
    # so this only happens if the firmware writes a wrong length, and a
    # checksum would hide that bug rather than fix it.
    reader = DeviceFrameHelper()
    frames = reader.feed(helper.encode_audio_down(1, b"xyz")[:-2] + good + good)
    assert [f.ctrl for f in frames][-1:] == [Ctrl.WAKE_DETECTED]
    print(f"载荷截断吞掉了下一帧，但之后恢复（得到 {len(frames)} 帧）")


def demo_magic_inside_payload() -> None:
    """PCM containing the magic by chance must not be mistaken for a header."""
    helper = DeviceFrameHelper()
    pcm = b"\xa5\x5a" * 64  # every other offset looks like a header start
    frame = helper.decode(helper.encode_audio_down(5, pcm))
    assert frame.payload == pcm, "载荷里的魔数被误当成帧头"
    print(f"载荷内含 {pcm.count(b'\xa5\x5a')} 处魔数，未被误判")


def main() -> None:
    use_utf8_output()
    demo_round_trip()
    demo_header_layout()
    demo_stream_reassembly()
    demo_resync_after_garbage()
    demo_magic_inside_payload()


if __name__ == "__main__":
    main()
