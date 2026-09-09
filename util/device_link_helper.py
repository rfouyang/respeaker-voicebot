"""The link to the XIAO: framed traffic over a byte transport.

The transport is deliberately just an object with read/write/close. Today that
is USB CDC via pyserial; a `_Pipe` pair stands in for it in the demos, and a
WiFi socket can be dropped in later without touching anything above this file.

Threading model matches services/robot-concierge, which is thread-based: the
caller runs `frames()` in its own thread and everything else stays on the
main one.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
import threading
import time

from config.settings import AudioConfig, DeviceConfig, use_utf8_output
from util.device_frame_helper import (Ctrl, DeviceFrame, DeviceFrameHelper,
                                      MsgType)


class _Pipe:
    """One direction of an in-memory byte stream, with blocking reads."""

    def __init__(self) -> None:
        self.chunks: deque[bytes] = deque()
        self.condition = threading.Condition()
        self.closed = False

    def write(self, data: bytes) -> None:
        with self.condition:
            self.chunks.append(bytes(data))
            self.condition.notify_all()

    def read(self, timeout: float) -> bytes:
        with self.condition:
            if not self.chunks and not self.closed:
                self.condition.wait(timeout)
            return self.chunks.popleft() if self.chunks else b""

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class _PipeTransport:
    """A transport built from two pipes, so two links can talk in-process."""

    def __init__(self, inbound: _Pipe, outbound: _Pipe) -> None:
        self.inbound = inbound
        self.outbound = outbound

    def read(self, timeout: float) -> bytes:
        return self.inbound.read(timeout)

    def write(self, data: bytes) -> None:
        self.outbound.write(data)

    def close(self) -> None:
        self.inbound.close()
        self.outbound.close()

    @classmethod
    def pair(cls) -> tuple[_PipeTransport, _PipeTransport]:
        host_to_device, device_to_host = _Pipe(), _Pipe()
        return (
            cls(device_to_host, host_to_device),  # host side
            cls(host_to_device, device_to_host),  # device side
        )


class _SerialTransport:
    """USB CDC via pyserial.

    The XIAO's own USB-C port, not the XVF3800's. Baud is ignored by native
    USB CDC but pyserial insists on a number.
    """

    def __init__(self, port: str, baud: int, timeout: float) -> None:
        import serial  # imported here so the demos run without hardware

        self.serial = serial.Serial(port, baud, timeout=timeout)

    def read(self, timeout: float) -> bytes:
        waiting = self.serial.in_waiting
        return self.serial.read(waiting if waiting else 1)

    def write(self, data: bytes) -> None:
        self.serial.write(data)

    def close(self) -> None:
        self.serial.close()


class DeviceLinkHelper:
    """Send and receive framed messages over one transport."""

    def __init__(self, transport=None, **kwargs) -> None:
        self.port = kwargs.get("port", DeviceConfig.PORT)
        self.baud = kwargs.get("baud", DeviceConfig.BAUD)
        self.read_timeout = kwargs.get("timeout", DeviceConfig.READ_TIMEOUT_SECONDS)
        self.chunk_bytes = kwargs.get("chunk_bytes", DeviceConfig.DOWNLINK_CHUNK_BYTES)
        self.transport = transport or _SerialTransport(
            self.port, self.baud, self.read_timeout
        )
        self.frame = DeviceFrameHelper()
        self.stop_event = threading.Event()
        self.sent_audio_bytes = 0

    @classmethod
    def paired(cls, **kwargs) -> tuple[DeviceLinkHelper, DeviceLinkHelper]:
        """A host link and a device link wired to each other, no hardware."""
        host_transport, device_transport = _PipeTransport.pair()
        return cls(host_transport, **kwargs), cls(device_transport, **kwargs)

    # ---- receiving ----

    def frames(self) -> Iterator[DeviceFrame]:
        """Yield frames until stop() is called. Run this in its own thread."""
        while not self.stop_event.is_set():
            data = self.transport.read(self.read_timeout)
            if not data:
                continue
            yield from self.frame.feed(data)

    # ---- sending ----

    def send_control(self, ctrl: Ctrl, turn_id: int) -> None:
        self.transport.write(self.frame.encode_control(ctrl, turn_id))

    @staticmethod
    def to_playback_rate(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
        """Resample mono int16 for the device's playback path.

        The XVF3800 reads back at 16 kHz but plays at 48 kHz -- measured, see
        DeviceConfig.PLAYBACK_SAMPLE_RATE. Sending 16 kHz straight through
        comes out exactly three times too fast.

        Linear interpolation rather than audioop.ratecv: audioop is deprecated
        and gone in 3.13, and at a whole-number ratio this is equivalent.
        """
        if source_rate == target_rate or not pcm:
            return pcm
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16)
        count = int(len(samples) * target_rate / source_rate)
        resampled = np.interp(
            np.linspace(0, len(samples) - 1, count),
            np.arange(len(samples)),
            samples.astype(np.float32),
        )
        return resampled.astype(np.int16).tobytes()

    def send_audio_down(self, turn_id: int, pcm: bytes) -> int:
        """Split a reply into device-sized packets. Returns bytes sent.

        Chunking is not cosmetic: the whole point is that the device's playback
        ring stays shallow, so that when the user barges in there is little
        audio already committed that they have not heard yet.
        """
        pcm = self.to_playback_rate(
            pcm, AudioConfig.SAMPLE_RATE, DeviceConfig.PLAYBACK_SAMPLE_RATE
        )
        for offset in range(0, len(pcm), self.chunk_bytes):
            chunk = pcm[offset : offset + self.chunk_bytes]
            self.transport.write(self.frame.encode_audio_down(turn_id, chunk))
            self.sent_audio_bytes += len(chunk)
        return len(pcm)

    def send_audio_up(self, turn_id: int, pcm: bytes) -> None:
        """Device side only; the real XIAO builds these frames in C++."""
        for offset in range(0, len(pcm), self.chunk_bytes):
            chunk = pcm[offset : offset + self.chunk_bytes]
            self.transport.write(self.frame.encode_audio_up(turn_id, chunk))

    def stop(self) -> None:
        self.stop_event.set()
        self.transport.close()


class FakeDevice:
    """Stands in for the XIAO until its firmware exists.

    It owns `turn_id` and increments it, exactly as the real device will. That
    ownership is the point: the decision to interrupt and the moment stale
    audio is dropped both live on the device, so no amount of link latency can
    put the two ends out of step.
    """

    def __init__(self, link: DeviceLinkHelper, **kwargs) -> None:
        self.link = link
        self.turn_id = 0
        self.speech_ms = kwargs.get("speech_ms", 400)
        self.interrupt_after_ms = kwargs.get("interrupt_after_ms")
        self.received_audio_bytes = 0
        self.received_control: list[tuple[Ctrl, int]] = []
        self.dropped_stale_bytes = 0

    def wake(self) -> int:
        """Wake word hit: claim the next turn and tell the host."""
        self.turn_id += 1
        self.link.send_control(Ctrl.WAKE_DETECTED, self.turn_id)
        return self.turn_id

    def speak(self, pcm: bytes | None = None) -> None:
        """Send uplink audio, then call the endpoint like the local VAD would."""
        audio = pcm if pcm is not None else bytes(
            AudioConfig.SAMPLE_RATE * 2 * self.speech_ms // 1_000
        )
        self.link.send_audio_up(self.turn_id, audio)
        self.link.send_control(Ctrl.SPEECH_END, self.turn_id)

    def interrupt(self) -> int:
        """Barge-in: flush playback, bump the turn, then tell the host."""
        self.turn_id += 1
        self.link.send_control(Ctrl.INTERRUPT, self.turn_id)
        return self.turn_id

    def run(self) -> None:
        """Consume host frames, dropping any that belong to a retired turn."""
        playing_since: float | None = None
        for frame in self.link.frames():
            if frame.type == MsgType.AUDIO_DOWN:
                # The one gate that stops late audio from a turn the user
                # already interrupted from reaching the speaker.
                if frame.turn_id < self.turn_id:
                    self.dropped_stale_bytes += len(frame.payload)
                    continue
                self.received_audio_bytes += len(frame.payload)
                if playing_since is None:
                    playing_since = time.monotonic()
                if (
                    self.interrupt_after_ms is not None
                    and (time.monotonic() - playing_since) * 1_000
                    >= self.interrupt_after_ms
                ):
                    self.interrupt_after_ms = None
                    self.interrupt()
            elif frame.type == MsgType.CONTROL:
                self.received_control.append((frame.ctrl, frame.turn_id))


def _collect(link: DeviceLinkHelper, into: list[DeviceFrame]) -> threading.Thread:
    thread = threading.Thread(target=lambda: into.extend(link.frames()), daemon=True)
    thread.start()
    return thread


def demo_control_exchange() -> None:
    """A wake on the device shows up as a wake on the host, same turn id."""
    host, device_link = DeviceLinkHelper.paired()
    heard: list[DeviceFrame] = []
    _collect(host, heard)

    device = FakeDevice(device_link)
    turn = device.wake()
    time.sleep(0.2)
    host.stop()

    assert heard and heard[0].ctrl is Ctrl.WAKE_DETECTED
    assert heard[0].turn_id == turn
    print(f"唤醒送达主机: {heard[0].ctrl.name} turn={heard[0].turn_id}")


def demo_audio_up() -> None:
    """Uplink PCM arrives intact and chunked, then the endpoint follows."""
    host, device_link = DeviceLinkHelper.paired()
    heard: list[DeviceFrame] = []
    _collect(host, heard)

    device = FakeDevice(device_link, speech_ms=400)
    device.wake()
    device.speak()
    time.sleep(0.3)
    host.stop()

    audio = [f for f in heard if f.type == MsgType.AUDIO_UP]
    total = sum(len(f.payload) for f in audio)
    assert heard[-1].ctrl is Ctrl.SPEECH_END
    print(f"上行 {len(audio)} 个包 / {total} 字节 "
          f"({total / DeviceConfig.BYTES_PER_MS:.0f} ms)，以 SPEECH_END 收尾")
    assert total == AudioConfig.SAMPLE_RATE * 2 * 400 // 1_000


def demo_audio_down() -> None:
    """A reply reaches the device split into 20 ms packets."""
    host, device_link = DeviceLinkHelper.paired()
    device = FakeDevice(device_link)
    thread = threading.Thread(target=device.run, daemon=True)
    thread.start()

    reply = bytes(DeviceConfig.BYTES_PER_MS * 500)  # 500 ms at 16 kHz
    sent = host.send_audio_down(1, reply)
    host.send_control(Ctrl.TTS_END, 1)
    time.sleep(0.3)
    device_link.stop()

    # send_audio_down resamples to the device's 48 kHz playback rate, so what
    # arrives is three times what was handed in. That ratio IS the check.
    print(f"送入 {len(reply)} 字节 @16k -> 送达 {device.received_audio_bytes} 字节 @48k "
          f"(x{device.received_audio_bytes / len(reply):.1f})")
    assert device.received_audio_bytes == sent
    assert sent == len(reply) * 3, sent
    assert (Ctrl.TTS_END, 1) in device.received_control


def demo_barge_in_drops_stale_audio() -> None:
    """The reason turn_id exists.

    The device interrupts mid-reply and bumps the turn. Audio the host had
    already put on the wire for the old turn must be discarded at the device,
    not played after the user has moved on.
    """
    host, device_link = DeviceLinkHelper.paired()
    device = FakeDevice(device_link)
    thread = threading.Thread(target=device.run, daemon=True)
    thread.start()

    device.turn_id = 1
    host.send_audio_down(1, bytes(DeviceConfig.BYTES_PER_MS * 200))
    time.sleep(0.15)
    new_turn = device.interrupt()
    late = host.send_audio_down(1, bytes(DeviceConfig.BYTES_PER_MS * 300))  # turn 1
    time.sleep(0.3)
    device_link.stop()

    print(f"打断后 turn {new_turn}；丢弃迟到音频 {device.dropped_stale_bytes} 字节")
    assert new_turn == 2
    assert device.dropped_stale_bytes == late


def demo_list_serial_ports() -> None:
    """What the host can actually see right now."""
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    if not ports:
        print("没有串口设备")
        return
    for port in ports:
        mark = "  <-- XIAO" if "303A" in (port.hwid or "").upper() else ""
        print(f"{port.device}: {port.description} [{port.hwid}]{mark}")


def main() -> None:
    use_utf8_output()
    demo_control_exchange()
    demo_audio_up()
    demo_audio_down()
    demo_barge_in_drops_stale_audio()
    demo_list_serial_ports()


if __name__ == "__main__":
    main()
