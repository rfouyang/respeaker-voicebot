"""Step: transcript -> reply text.

Owns the LLMHelper. Kept non-streaming, exactly as in
services/robot-concierge: BytePlus TTS is unidirectional and takes a whole
sentence per request anyway, so streaming tokens out of the model would buy
nothing until the synthesiser can accept them incrementally.

Barge-in during generation is handled the way it is there too -- the reply is
generated, then dropped from memory if the user has already moved on. Nothing
was spoken, so nothing needs unwinding beyond the memory entry.
"""

from __future__ import annotations

import threading

from config.settings import use_utf8_output
from util.llm_helper import LLMHelper


class ResponseGeneration:
    """Turn one user utterance into one reply, with bounded memory."""

    def __init__(self, llm: LLMHelper | None = None, **kwargs) -> None:
        self.llm = llm or LLMHelper(**kwargs)
        self.generated = 0

    def clear_memory(self) -> None:
        """Called when a new session starts after the wake word."""
        self.llm.clear_memory()

    def discard_last_turn(self) -> None:
        """Drop a reply the user never heard, so the model does not believe
        it already said something it did not."""
        self.llm.discard_last_turn()

    def generate(self, message: str, superseded: threading.Event | None = None) -> str:
        """Produce a reply. If `superseded` is already set, the caller will
        discard it -- generation itself cannot be aborted mid-request."""
        answer = self.llm.chat(message)
        self.generated += 1
        return answer

    @property
    def memory_turns(self) -> int:
        return len(self.llm.memory)


def demo_generate() -> None:
    generation = ResponseGeneration()
    answer = generation.generate("用一句话说说今天适合做什么。")
    print("回复:", answer)
    assert answer and generation.memory_turns == 1


def demo_memory_across_turns() -> None:
    """The session keeps context until the wake word starts a new one."""
    generation = ResponseGeneration()
    generation.generate("我叫小明，请记住我的名字。")
    answer = generation.generate("我叫什么名字？只回答名字。")
    print("记忆回答:", answer)
    assert "小明" in answer, answer
    assert generation.memory_turns == 2


def demo_clear_memory_on_new_session() -> None:
    """A new wake word means a fresh conversation, not a continuation."""
    generation = ResponseGeneration()
    generation.generate("我叫小红。")
    generation.clear_memory()
    assert generation.memory_turns == 0
    print("新会话已清空记忆")


def demo_discard_superseded_reply() -> None:
    """The user spoke again while we were thinking: the unheard reply must
    leave no trace in memory."""
    generation = ResponseGeneration()
    superseded = threading.Event()
    superseded.set()
    answer = generation.generate("介绍一下你自己。", superseded)
    generation.discard_last_turn()
    print(f"生成了 {len(answer)} 字但被丢弃，剩余记忆 {generation.memory_turns} 轮")
    assert generation.memory_turns == 0


def main() -> None:
    use_utf8_output()
    demo_generate()
    demo_memory_across_turns()
    demo_clear_memory_on_new_session()
    demo_discard_superseded_reply()


if __name__ == "__main__":
    main()
