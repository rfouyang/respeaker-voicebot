"""One turn of the dialog, and its cancellation handles.

Every piece of barge-in cleanup is collected here. Do not scatter aborts
across the LLM and TTS clients: sooner or later one gets missed, and it shows
up either as tokens still being billed after an interrupt, or as half a
sentence of stale audio surfacing later.
"""

from __future__ import annotations

from collections.abc import Callable
import threading

from config.settings import DialogConfig, use_utf8_output


class Turn:
    """State for one exchange, from wake (or barge-in) to reply finished."""

    def __init__(self, turn_id: int, **kwargs) -> None:
        self.id = turn_id
        self.chars_per_second = kwargs.get(
            "chars_per_second", DialogConfig.CHARS_PER_SECOND
        )
        self.interrupt_note = kwargs.get("interrupt_note", DialogConfig.INTERRUPT_NOTE)

        self.transcript = ""
        self.cancelled = False
        self.done = False  # finished speaking normally; no longer cancellable

        # All the text handed to TTS, and the audio actually delivered to the
        # device. These agree only when the turn completes; a barge-in makes
        # them diverge, and the history has to follow the second one.
        self.queued_text = ""
        self.delivered_ms = 0.0

        self.cancels: list[Callable[[], None]] = []
        self.interrupt_event = threading.Event()

    def on_cancel(self, fn: Callable[[], None]) -> None:
        """Register a cleanup: abort the LLM request, stop the TTS stream."""
        self.cancels.append(fn)

    def cancel(self) -> None:
        if self.cancelled or self.done:
            return
        self.cancelled = True
        self.interrupt_event.set()
        for fn in self.cancels:
            fn()
        self.cancels.clear()

    @property
    def spoken_text(self) -> str:
        """The part of the reply the user actually heard.

        The point is that this is derived from delivered audio, not from the
        text handed to TTS. On a barge-in the latter runs ten or twenty
        characters ahead: those were abandoned mid-synthesis and never reached
        anyone. Writing them into the history makes the model believe it
        already said them, and the next turn is quietly wrong.
        """
        heard = int(self.delivered_ms / 1_000 * self.chars_per_second)
        return self.queued_text[:heard]

    def history_entry(self) -> str | None:
        """What to write back into the history once this turn has ended."""
        text = self.spoken_text
        if not text:
            return None
        if self.cancelled:
            return text + self.interrupt_note
        return self.queued_text


def demo_spoken_text_diverges() -> None:
    """The bug this class exists to prevent: 23 characters queued, 800 ms of
    audio delivered, so only about 4 were heard."""
    turn = Turn(1)
    turn.queued_text = "好的，我来说明一下。这是一段刻意写得比较长的回复"
    turn.delivered_ms = 800
    turn.cancel()
    print(f"送进 TTS {len(turn.queued_text)} 字，实际听到 {len(turn.spoken_text)} 字")
    print("回填历史:", turn.history_entry())
    assert len(turn.spoken_text) < len(turn.queued_text)


def demo_completed_turn_is_not_cancelled() -> None:
    """A turn that finished must not be tagged as interrupted, or the note
    gets appended to a reply the user heard in full."""
    turn = Turn(2)
    turn.queued_text = "今天多云，二十四度"
    turn.delivered_ms = 10_000
    turn.done = True
    turn.cancel()
    assert not turn.cancelled
    print("正常完成的 turn 回填:", turn.history_entry())


def demo_cancel_runs_callbacks() -> None:
    """Both remote ends get told to stop, in the order they registered."""
    stopped: list[str] = []
    turn = Turn(3)
    turn.on_cancel(lambda: stopped.append("llm"))
    turn.on_cancel(lambda: stopped.append("tts"))
    turn.cancel()
    assert stopped == ["llm", "tts"], stopped
    assert turn.interrupt_event.is_set()
    print("取消时已通知的远端:", stopped)

    turn.cancel()  # idempotent: a second barge-in must not re-fire callbacks
    assert stopped == ["llm", "tts"]
    print("重复取消不会重复触发")


def demo_nothing_heard_writes_nothing() -> None:
    """Interrupted before any audio played: there is no history entry at all,
    not an empty one."""
    turn = Turn(4)
    turn.queued_text = "我正要开始说话"
    turn.delivered_ms = 0
    turn.cancel()
    assert turn.history_entry() is None
    print("一个字都没听到 -> 不回填历史")


def main() -> None:
    use_utf8_output()
    demo_spoken_text_diverges()
    demo_completed_turn_is_not_cancelled()
    demo_cancel_runs_callbacks()
    demo_nothing_heard_writes_nothing()


if __name__ == "__main__":
    main()
