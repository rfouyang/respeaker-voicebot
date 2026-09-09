"""End-to-end test over the air, with no human in the loop.

A synthesized visitor speaks through the PC speaker; the microphone array
hears it, the XVF3800 processes it, the XIAO ships it over USB CDC, and the
host recognises it, answers and speaks back. Every stage that will exist in
the product except the wake word is exercised, and it runs on one command.

The PC speaker is playing the part of a person. It is NOT the product's
playback path: the real reply has to leave through the XVF3800 so its AEC sees
the echo reference.

Whatever is played aloud must be inert. It goes into a real room where other
listening devices may be in earshot, so: no smart-home commands, no wake
words, nothing addressed to an assistant.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time

from config.settings import LLMConfig, OutputConfig, use_utf8_output
from util.asr_helper import ASRHelper
from util.device_probe_helper import DeviceProbeHelper
from util.llm_helper import LLMHelper
from util.tts_helper import TTSHelper


class AcousticProbeHelper:
    """Play a question at the array, and run what comes back through the bot."""

    def __init__(self, channel: str = "right", **kwargs) -> None:
        self.channel = channel  # which I2S slot to recognise from
        self.gain = kwargs.get("gain", 1.7)
        self.lead_in = kwargs.get("lead_in", 1.0)
        self.tail = kwargs.get("tail", 2.0)
        self.tts = TTSHelper()
        self.asr = ASRHelper()
        self.llm = LLMHelper(**kwargs)
        self.out_dir = Path(kwargs.get("out_dir", OutputConfig.OUTPUT_DIR / "acoustic"))

    def _play(self, pcm: bytes) -> None:
        import numpy as np
        import sounddevice as sd

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        samples = np.clip(samples * self.gain, -32768, 32767).astype(np.int16)
        time.sleep(self.lead_in)
        sd.play(samples, self.tts.sample_rate)
        sd.wait()

    def ask_aloud(self, question: str) -> dict:
        """Speak `question` into the room and return what the bot made of it."""
        pcm = self.tts.synthesize(question)
        duration = len(pcm) / 2 / self.tts.sample_rate

        probe = DeviceProbeHelper(out_dir=self.out_dir)
        player = threading.Thread(target=self._play, args=(pcm,), daemon=True)
        player.start()
        probe.capture(self.lead_in + duration + self.tail)
        player.join(timeout=5)

        heard_pcm = bytes(probe.right if self.channel == "right" else probe.left)
        heard, _tail_ms = self.asr.transcribe(heard_pcm, realtime=False)
        answer = self.llm.chat(heard) if heard else ""
        return {
            "asked": question,
            "heard": heard,
            "answer": answer,
            "seconds": probe.seconds,
            "rms": round(probe.rms(heard_pcm), 1),
        }

    def speak(self, text: str) -> None:
        self._play(self.tts.synthesize(text))


# Questions a person might actually stop and ask at Stadium station. All inert:
# nothing here could be acted on by another device that overhears it.
VISITOR_QUESTIONS = (
    "请问去新加坡室内体育馆要走哪个出口？",
    "Excuse me, is there a shopping mall near this station?",
    "我想去国家体育场看比赛，从这里怎么走？",
)


def demo_one_question() -> None:
    """The whole loop for a single question."""
    probe = AcousticProbeHelper()
    print(f"提示词: {LLMConfig.PROMPT_PATH.name}")
    result = probe.ask_aloud(VISITOR_QUESTIONS[0])
    print(f"\n播放  : {result['asked']}")
    print(f"听到  : {result['heard'] or '(空)'}   [rms {result['rms']}, {result['seconds']:.1f}s]")
    print(f"回答  : {result['answer'] or '(无)'}")
    assert result["heard"], "没有听到任何内容；检查音量和板子位置"


def demo_conversation() -> None:
    """Several questions in a row, keeping context between them."""
    probe = AcousticProbeHelper()
    for index, question in enumerate(VISITOR_QUESTIONS, 1):
        result = probe.ask_aloud(question)
        print(f"\n--- 第 {index} 问 ---")
        print(f"路人: {result['heard'] or '(没听清)'}")
        print(f"机器人: {result['answer'] or '(无)'}")
        if result["answer"]:
            probe.speak(result["answer"])


def demo_aec() -> None:
    """The test everything else depends on.

    Play a sentence out of the XVF3800's own jack while recording both uplink
    slots. The chip takes its echo reference from exactly what we sent it, so
    on the processed channel that sentence should be largely gone. Whichever
    channel still contains it is the unprocessed one.

    Two answers from one measurement: which slot to feed WakeNet and the
    recogniser, and whether the AEC works at all. Without a working AEC the
    microphone is deaf while the bot is talking, and barge-in is impossible.
    """
    from difflib import SequenceMatcher

    spoken = ("这是一段用来测试回声消除的朗读内容，长度足够让麦克风持续听到，"
              "而处理后的通道里它应该基本消失。")
    tts = TTSHelper()
    pcm = tts.synthesize(spoken)
    seconds = len(pcm) / 2 / tts.sample_rate

    probe = DeviceProbeHelper(out_dir=OutputConfig.OUTPUT_DIR / "aec")
    print(f"经 XVF3800 从 3.5mm 播放 {seconds:.1f} 秒，同时录上行两路...")
    probe.play_and_capture(pcm)
    print(f"收到 {probe.frames_seen} 帧 / {probe.seconds:.1f} 秒"
          f"（丢弃非帧字节 {probe.frame.dropped_bytes}）")

    asr = ASRHelper()
    report = {}
    for name, buf in (("left", bytes(probe.left)), ("right", bytes(probe.right))):
        rms = probe.rms(buf)
        text, _tail = asr.transcribe(buf, realtime=False)
        similarity = SequenceMatcher(None, spoken, text).ratio() if text else 0.0
        report[name] = (rms, text, similarity)
        shape = probe.describe(buf)
        print(f"  {name:5s} rms={rms:8.1f}  回声残留相似度={similarity:.2f}")
        print(f"        识别: {text[:40] or '(空)'}")
        print(f"        有声跨度 {shape['span_seconds']:.1f}s  "
              f"缺口 {shape['gap_count']}  "
              f"过零率 {shape['zero_crossing_rate']:.0f}/s（语音 1000-3000）")
        print(f"        -> {probe.write_wav(f'{name}.wav', buf)}")

    if probe.seconds < 1:
        print("\n没收到上行音频；先确认固件是全双工版本")
        return

    quiet = min(report, key=lambda n: report[n][2])
    loud = max(report, key=lambda n: report[n][2])
    gap = report[loud][2] - report[quiet][2]
    print(f"\n{loud} 保留了回声（相似度 {report[loud][2]:.2f}），"
          f"{quiet} 抑制得更多（{report[quiet][2]:.2f}）")
    if report[loud][2] < 0.2:
        print("两路都几乎没听到回声 —— 音箱可能没出声，或音量太低，无法判定")
    elif gap > 0.3:
        print(f"结论：{quiet} 是 AEC 处理后的通道，喂给 WakeNet 和 ASR 用它。")
    else:
        print("两路差距不明显，AEC 是否生效存疑；先确认音箱确实在响")


def main() -> None:
    use_utf8_output()
    demo_aec()


if __name__ == "__main__":
    main()
