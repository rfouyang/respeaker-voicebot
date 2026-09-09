"""Facade and command line for the host side.

    uv run python -m component.voicebot_application            # 真设备，COM3
    uv run python -m component.voicebot_application --fake     # 假设备，无需硬件
    uv run python -m component.voicebot_application --list     # 列出串口

The device is what does not exist yet, so `--fake` is the mode that matters
today: it speaks the same 12-byte protocol the firmware will, which is what
lets the whole host stack be finished and tested before any C++ is written.
"""

from __future__ import annotations

import argparse
import threading
import time

from loguru import logger

from component.dialog_orchestration.service import (DialogOrchestrationService,
                                                    VoiceState)
from config.settings import DeviceConfig, WakeWordConfig, use_utf8_output
from util.device_link_helper import DeviceLinkHelper, FakeDevice
from util.logging_helper import configure_logging


class VoicebotApplication:
    """Wire a device link to the dialog service and run it."""

    def __init__(self, fake: bool = False, **kwargs) -> None:
        self.fake = fake
        self.port = kwargs.get("port", DeviceConfig.PORT)
        self.device: FakeDevice | None = None

        if fake:
            host_link, device_link = DeviceLinkHelper.paired()
            self.device = FakeDevice(device_link)
            threading.Thread(target=self.device.run, daemon=True).start()
        else:
            host_link = DeviceLinkHelper(port=self.port)

        self.link = host_link
        self.service = DialogOrchestrationService(host_link, **kwargs)

    def run(self, max_sessions: int | None = None) -> None:
        logger.info("链路：{}", "假设备（内存管道）" if self.fake else self.port)
        logger.info("唤醒词：{}（由 XIAO 上的 WakeNet 判定）", WakeWordConfig.PHRASE)
        try:
            self.service.run(max_sessions=max_sessions)
        except KeyboardInterrupt:
            logger.info("停止")
        finally:
            self.service.stop()

    def stop(self) -> None:
        self.service.stop()


def list_ports() -> None:
    from serial.tools import list_ports as tools

    ports = list(tools.comports())
    if not ports:
        print("没有串口设备")
        return
    for port in ports:
        mark = "  <-- XIAO" if "303A" in (port.hwid or "").upper() else ""
        print(f"{port.device}: {port.description}{mark}")


def demo_full_turn() -> None:
    """One scripted exchange end to end against the fake device."""
    from util.tts_helper import TTSHelper

    app = VoicebotApplication(fake=True, idle_timeout=6)
    threading.Thread(target=lambda: app.run(max_sessions=1), daemon=True).start()
    device = app.device

    _wait(lambda: app.service.state == VoiceState.WAITING_WAKE)
    device.wake()
    _wait(lambda: app.service.state == VoiceState.LISTENING)
    device.speak(TTSHelper().synthesize("你好，请用一句话介绍你自己。"))

    assert _wait(lambda: app.service.state == VoiceState.SPEAKING), "没有进入播放"
    assert _wait(lambda: app.service.state != VoiceState.SPEAKING), "播放没有结束"
    time.sleep(0.5)
    played = device.received_audio_bytes
    app.stop()

    seconds = played / DeviceConfig.BYTES_PER_MS / 1_000
    print(f"一轮完成：设备收到 {played} 字节回复音频（{seconds:.1f} 秒）")
    assert played > 0


def demo_barge_in_then_continue() -> None:
    """Interrupt mid-reply, then check the session is ready for the next turn
    rather than stuck in SPEAKING."""
    from util.tts_helper import TTSHelper

    app = VoicebotApplication(fake=True, idle_timeout=5)
    threading.Thread(target=lambda: app.run(max_sessions=1), daemon=True).start()
    device = app.device

    _wait(lambda: app.service.state == VoiceState.WAITING_WAKE)
    device.wake()
    _wait(lambda: app.service.state == VoiceState.LISTENING)
    device.speak(TTSHelper().synthesize("请详细讲讲你能帮我做哪些事情。"))

    assert _wait(lambda: app.service.state == VoiceState.SPEAKING)
    time.sleep(1.2)
    device.interrupt()
    assert _wait(lambda: app.service.state != VoiceState.SPEAKING), "打断没有生效"
    # After a barge-in the session must go back to listening for the next
    # thing the user says, not end.
    back = _wait(lambda: app.service.state == VoiceState.LISTENING, 15)
    app.stop()

    print(f"打断后回到聆听={back}，已播 {device.received_audio_bytes} 字节")
    assert back


def _wait(predicate, timeout: float = 90) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def main() -> None:
    use_utf8_output()
    parser = argparse.ArgumentParser(description="respeaker-voicebot 主机侧")
    parser.add_argument("--fake", action="store_true", help="用假设备，不需要硬件")
    parser.add_argument("--demo", action="store_true", help="跑完整一轮和打断两个场景")
    parser.add_argument("--list", action="store_true", help="列出串口")
    parser.add_argument("--port", default=DeviceConfig.PORT)
    args = parser.parse_args()

    if args.list:
        list_ports()
        return

    configure_logging()
    if args.demo:
        demo_full_turn()
        # stop() sets the flags, but a session already inside an ASR call
        # unwinds a moment later. Without this pause the two demos' logs
        # interleave and the output is unreadable.
        time.sleep(2)
        logger.info("--- 下一个场景：打断 ---")
        demo_barge_in_then_continue()
        return

    VoicebotApplication(fake=args.fake, port=args.port).run()


if __name__ == "__main__":
    main()
