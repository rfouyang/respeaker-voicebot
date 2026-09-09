"""DeepSeek chat client with bounded memory.

Ported from services/robot-concierge. Tool calling was left behind on purpose:
this project starts as a plain voice assistant, and the tool registry there is
specific to the museum robot. Add it back as a `component/` concern if needed.
"""

from __future__ import annotations

from collections import deque
import os
from typing import Any

from openai import OpenAI

from config.settings import LLMConfig, use_utf8_output


class LLMHelper:
    """Hold a conversation and return one reply per turn."""

    def __init__(
        self,
        model: str = LLMConfig.MODEL,
        api_key: str | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if client is None:
            key = api_key or os.getenv(LLMConfig.API_KEY_ENV)
            if not key:
                raise RuntimeError(
                    f"Set {LLMConfig.API_KEY_ENV} in .env before using the LLM"
                )
            client = OpenAI(
                api_key=key,
                base_url=LLMConfig.BASE_URL,
                timeout=kwargs.get("timeout", LLMConfig.TIMEOUT_SECONDS),
            )
        self.client = client
        self.model = model
        self.memory: deque[tuple[str, str]] = deque(
            maxlen=kwargs.get("memory_turns", LLMConfig.MEMORY_TURNS)
        )
        self.prompt = LLMConfig.PROMPT_PATH.read_text(encoding="utf-8").strip()

    def clear_memory(self) -> None:
        self.memory.clear()

    def discard_last_turn(self) -> None:
        """Drop a reply the user never heard.

        Called after a barge-in: the model generated it, but playback was cut
        short, so leaving it in memory makes the model believe it already said
        something the user never received.
        """
        if self.memory:
            self.memory.pop()

    def _conversation(self, message: str) -> list[dict[str, Any]]:
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt}
        ]
        for user_text, assistant_text in self.memory:
            conversation.append({"role": "user", "content": user_text})
            conversation.append({"role": "assistant", "content": assistant_text})
        conversation.append({"role": "user", "content": message})
        return conversation

    def chat(self, message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._conversation(message),
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("LLM returned an empty response")
        self.memory.append((message, answer))
        return answer


def demo_chat() -> None:
    """One exchange -- proves credentials, model name and prompt all load."""
    llm = LLMHelper()
    question = "你好，用一句话介绍你自己。"
    print(f"[{llm.model}]")
    print("User:", question)
    print("Bot :", llm.chat(question))


def demo_memory() -> None:
    """Two turns -- the second only works if the first stayed in memory."""
    llm = LLMHelper()
    llm.chat("我叫小明，请记住我的名字。")
    answer = llm.chat("我叫什么名字？只回答名字。")
    print("记忆回答:", answer)
    assert "小明" in answer, f"memory lost: {answer!r}"


def demo_discard_last_turn() -> None:
    """A barge-in drops the unheard reply, so the model must not recall it."""
    llm = LLMHelper()
    llm.chat("我叫小红，请记住我的名字。")
    llm.discard_last_turn()
    print("丢弃后剩余轮数:", len(llm.memory))
    assert len(llm.memory) == 0


def demo_interactive() -> None:
    """Type at it. Not called from main() -- run it by hand when wanted."""
    llm = LLMHelper()
    print(f"Chat with {llm.model}. Type 'quit' to stop.")
    while (message := input("You: ").strip()).lower() not in {"quit", "exit"}:
        if message:
            print("Bot:", llm.chat(message))


def main() -> None:
    use_utf8_output()
    demo_chat()
    demo_memory()
    demo_discard_last_turn()


if __name__ == "__main__":
    main()
