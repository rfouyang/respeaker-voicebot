"""The conversation state machine, ported from services/robot-concierge.

Six states, same as `WakeVoiceBot` there:

    WAITING_WAKE -> LISTENING -> THINKING -> SPEAKING -> (INTERRUPTED)
                 <- IDLE_TIMEOUT <-

What changed in the port is only where the events come from. There, the host
ran the wake word locally and decided barge-in from a second ASR session. Here
the XIAO owns both, and sends WAKE_DETECTED / SPEECH_END / INTERRUPT down the
link. That ownership matters: the device holds `turn_id`, so the decision to
interrupt and the moment stale audio is dropped are on the same side of the
link and no amount of latency can put them out of step.

This module orchestrates and holds no util helper of its own; each step owns
the one it needs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
import threading
from typing import Any

from loguru import logger

from component.dialog_orchestration.response_generation import ResponseGeneration
from component.dialog_orchestration.speech_output import SpeechOutput
from component.dialog_orchestration.transcription import Transcription
from component.dialog_orchestration.turn import Turn
from config.settings import (DeviceConfig, DialogConfig, OutputConfig,
                             WakeWordConfig, use_utf8_output)
from util.device_frame_helper import Ctrl, MsgType


class VoiceState(StrEnum):
    WAITING_WAKE = "waiting_wake"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    IDLE_TIMEOUT = "idle_timeout"


class DialogOrchestrationService:
    """Follow the device's state machine and drive the cloud services."""

    def __init__(
        self,
        link,
        transcription: Transcription | None = None,
        response_generation: ResponseGeneration | None = None,
        speech_output: SpeechOutput | None = None,
        **kwargs,
    ) -> None:
        self.link = link
        self.transcription = transcription or Transcription()
        self.response_generation = response_generation or ResponseGeneration()
        self.speech_output = speech_output or SpeechOutput()

        self.state = VoiceState.WAITING_WAKE
        self.turn: Turn | None = None
        self.idle_timeout = kwargs.get(
            "idle_timeout", DialogConfig.IDLE_TIMEOUT_SECONDS
        )

        # Set by the reader thread when the device reports each event.
        self.wake_event = threading.Event()
        self.speech_end_event = threading.Event()
        self.device_turn_id = 0
        self.stop_event = threading.Event()
        self.reader: threading.Thread | None = None

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.debug_dir = Path(
            kwargs.get("debug_dir", OutputConfig.OUTPUT_DIR / "sessions" / run_id)
        )
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.debug_dir / "events.jsonl"
        self._log_lock = threading.Lock()

    # ---- bookkeeping ----

    def _log(self, event: str, **details: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "state": self.state.value,
            **details,
        }
        with self._log_lock:
            with self.events_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _set_state(self, state: VoiceState) -> None:
        self.state = state
        self._log("state_changed")
        logger.info("状态：{}", state.value)

    # ---- device events ----

    def start(self) -> None:
        self.reader = threading.Thread(
            target=self._reader_loop, name="device-reader", daemon=True
        )
        self.reader.start()

    def _reader_loop(self) -> None:
        for frame in self.link.frames():
            if self.stop_event.is_set():
                return
            self._on_frame(frame)

    def _on_frame(self, frame) -> None:
        if frame.type == MsgType.AUDIO_UP:
            # Only audio belonging to the turn in progress is worth feeding to
            # the recogniser; anything older is from a turn already retired.
            if frame.turn_id >= self.device_turn_id:
                self.transcription.feed(frame.payload)
            return
        if frame.type != MsgType.CONTROL:
            return

        if frame.ctrl == Ctrl.WAKE_DETECTED:
            self.device_turn_id = frame.turn_id
            self._log("wake_detected", turn_id=frame.turn_id)
            logger.success("检测到唤醒词：{}", WakeWordConfig.PHRASE)
            self.wake_event.set()
        elif frame.ctrl == Ctrl.SPEECH_END:
            self._log("speech_end", turn_id=frame.turn_id)
            self.speech_end_event.set()
        elif frame.ctrl == Ctrl.INTERRUPT:
            self.device_turn_id = frame.turn_id
            self._log("barge_in_detected", turn_id=frame.turn_id)
            logger.warning("用户打断，立即停止播放")
            if self.turn:
                self.turn.cancel()

    # ---- the loop ----

    def wait_for_wake(self) -> bool:
        self._set_state(VoiceState.WAITING_WAKE)
        logger.info("等待唤醒词：{}", WakeWordConfig.PHRASE)
        while not self.stop_event.is_set():
            if self.wake_event.wait(0.2):
                self.wake_event.clear()
                return True
        return False

    def listen_turn(self) -> str:
        self._set_state(VoiceState.LISTENING)
        logger.info("等待讲话；{} 秒无声返回待唤醒", self.idle_timeout)
        self.transcription.reset()
        self.speech_end_event.clear()
        message = self.transcription.listen(
            speech_timeout=self.idle_timeout,
            on_partial=lambda text: logger.debug("ASR partial：{}", text),
        )
        self._log("asr_final", text=message)
        if message:
            logger.success("ASR final：{}", message)
        return message

    def process_turn(self, message: str) -> dict[str, Any]:
        turn = Turn(self.device_turn_id)
        self.turn = turn

        self._set_state(VoiceState.THINKING)
        self._log("turn_started", message=message, turn_id=turn.id)
        answer = self.response_generation.generate(message)
        self._log("llm_answer", answer=answer)
        logger.info("LLM：{}", answer)

        if turn.cancelled:
            # Interrupted while thinking. Nothing was spoken, so the reply
            # leaves no trace beyond the memory entry.
            self.response_generation.discard_last_turn()
            self._set_state(VoiceState.INTERRUPTED)
            result = {"status": "superseded", "message": message, "answer": answer}
            self._log("turn_finished", **result)
            return result

        self._set_state(VoiceState.SPEAKING)
        interrupted = self.speech_output.speak(turn, answer, self.link)
        if interrupted or turn.cancelled:
            self.response_generation.discard_last_turn()
            self._set_state(VoiceState.INTERRUPTED)
            status = "interrupted"
            logger.warning("播放被打断，听到约 {} 字", len(turn.spoken_text))
        else:
            turn.done = True
            status = "completed"
            logger.success("播放完成，{:.0f} ms", turn.delivered_ms)

        result = {
            "status": status,
            "message": message,
            "answer": answer,
            "heard": turn.spoken_text,
            "history_entry": turn.history_entry(),
        }
        self._log("turn_finished", **result)
        self.turn = None
        return result

    def run_session(self) -> int:
        """One wake-to-timeout session. Returns how many turns it held."""
        self.response_generation.clear_memory()
        turns = 0
        while not self.stop_event.is_set():
            message = self.listen_turn()
            if not message:
                self._set_state(VoiceState.IDLE_TIMEOUT)
                self._log("session_ended", reason="idle_timeout", turns=turns)
                logger.info("会话超时，需要重新唤醒")
                return turns
            self.process_turn(message)
            turns += 1
        return turns

    def run(self, max_sessions: int | None = None) -> None:
        self.start()
        sessions = 0
        while max_sessions is None or sessions < max_sessions:
            if not self.wait_for_wake():
                break
            self.run_session()
            sessions += 1

    def stop(self) -> None:
        self.stop_event.set()
        self.transcription.stop()
        if self.turn:
            self.turn.cancel()
        self.link.stop()


def _fake_stack(**kwargs):
    """A service wired to a FakeDevice, running its session loop.

    No hardware. The cloud services are real -- they are the part worth
    exercising, and the device is the part that does not exist yet.
    """
    from util.device_link_helper import DeviceLinkHelper, FakeDevice

    host, device_link = DeviceLinkHelper.paired()
    service = DialogOrchestrationService(host, **kwargs)
    device = FakeDevice(device_link)
    threading.Thread(target=device.run, daemon=True).start()
    threading.Thread(
        target=lambda: service.run(max_sessions=1), daemon=True
    ).start()
    return service, device


def _wait_for(predicate, timeout: float = 90) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def demo_one_turn() -> None:
    """Wake, speak, get a reply, then time out back to waiting."""
    import time
    from util.tts_helper import TTSHelper

    service, device = _fake_stack(idle_timeout=6)
    assert _wait_for(lambda: service.state == VoiceState.WAITING_WAKE)

    device.wake()
    assert _wait_for(lambda: service.state == VoiceState.LISTENING), "没有进入聆听"
    device.speak(TTSHelper().synthesize("北京今天天气怎么样？"))

    assert _wait_for(lambda: service.state == VoiceState.SPEAKING), "没有进入播放"
    assert _wait_for(lambda: service.state != VoiceState.SPEAKING), "播放没有结束"
    time.sleep(0.5)
    heard = device.received_audio_bytes
    service.stop()

    print(f"设备收到回复音频 {heard} 字节 "
          f"({heard / DeviceConfig.BYTES_PER_MS / 1000:.1f} 秒)")
    print(f"事件流: {service.events_path}")
    assert heard > 0


def demo_barge_in() -> None:
    """The device interrupts mid-reply; the host stops and accounts for it."""
    import time
    from util.tts_helper import TTSHelper

    service, device = _fake_stack(idle_timeout=6)
    assert _wait_for(lambda: service.state == VoiceState.WAITING_WAKE)

    device.wake()
    assert _wait_for(lambda: service.state == VoiceState.LISTENING)
    device.speak(TTSHelper().synthesize("请详细介绍一下你能做什么。"))

    assert _wait_for(lambda: service.state == VoiceState.SPEAKING), "没有进入播放"
    time.sleep(1.0)  # let a bit of the reply play
    played_before = device.received_audio_bytes
    device.interrupt()
    assert _wait_for(lambda: service.state != VoiceState.SPEAKING), "打断后没有离开播放"
    time.sleep(0.5)
    stale = device.dropped_stale_bytes
    service.stop()

    print(f"打断前已播 {played_before} 字节；打断后设备丢弃迟到音频 {stale} 字节")
    assert played_before > 0


def demo_idle_timeout_returns_to_wake() -> None:
    """Silence after the wake word sends the session back to waiting."""
    import time

    service, device = _fake_stack(idle_timeout=2)
    assert _wait_for(lambda: service.state == VoiceState.WAITING_WAKE)
    device.wake()
    reached_timeout = _wait_for(lambda: service.state == VoiceState.IDLE_TIMEOUT, 40)
    reached = service.state
    service.stop()
    assert reached_timeout, reached

    print(f"无人说话 -> {reached.value}")
    assert reached == VoiceState.IDLE_TIMEOUT


def main() -> None:
    use_utf8_output()
    from util.logging_helper import configure_logging

    configure_logging()
    demo_one_turn()
    demo_barge_in()
    demo_idle_timeout_returns_to_wake()


if __name__ == "__main__":
    main()
