# ReSpeaker XVF3800 + XIAO ESP32S3 语音助手

唤醒词跑在 XIAO 上，ASR / LLM / TTS 跑在电脑上。支持打断（barge-in），
会话静默超时后回到待唤醒状态。

```
[4 麦阵列] → XVF3800 ──I2S──→ XIAO ESP32S3 ──→ 电脑
[3.5mm 音箱] ← ────┘   ↑  (WakeNet 唤醒词)      ├→ BytePlus ASR
                       │                        ├→ DeepSeek LLM
              AEC 参考 ┘                        └→ BytePlus TTS
```

## 来源

| 来自 | 内容 |
|---|---|
| `services/robot-concierge` | 对话状态机、打断判定、回声抑制 —— **在同一颗 XVF3800 上实测调通过** |
| `services/voicebot` | 设备传输层设计、XIAO 固件骨架 |

## 目录

```
config/     settings.py（各域 XxxConfig）+ prompt/*.md
util/       技术原子能力，不认识业务概念
component/  业务逻辑 + 命令行入口
```

依赖方向严格 `component → util → config`。

## AEC：不能踩的坑

XVF3800 的 AEC 参考信号取自**送进它的 I2S/USB 输入的左声道**。所以 TTS
回放路径必须是：

```
电脑 → XIAO → I2S 左声道 → XVF3800 → 3.5mm 音箱
```

只要让 XIAO 外接 DAC 直接推喇叭，XVF3800 就拿不到参考信号，AEC 失效，
播放期间麦克风全是自己的声音。**没有 AEC，打断无从谈起。**

## 跑起来

```bash
uv sync
cp .env.example .env   # 填 DEEPSEEK_API_KEY / BYTEPLUS_API_KEY
```

每个 `.py` 底部都有 `demo_xxx()`，由 `main()` 调用，可以单独运行：

```bash
uv run python util/llm_helper.py       # 直接跑
uv run python -m util.llm_helper       # 或按模块跑
```

直接跑脚本依赖 `.venv/Lib/site-packages/_project_root.pth`（指向项目根）。
`uv sync` 重建 venv 后若丢失，重新写入即可；PyCharm 右键运行不依赖它。

## 进度

- [x] `config/settings.py`
- [x] `util/llm_helper.py` — DeepSeek 对话 + 记忆 + 打断丢弃
- [x] `util/tts_helper.py` — BytePlus Seed TTS 流式合成（首包实测 313ms）
- [x] `util/asr_helper.py` — BytePlus 流式识别（实测尾延迟 46ms）
- [x] `util/device_frame_helper.py` — USB CDC 线格式，流式重组 + 重同步
- [x] `util/device_link_helper.py` — 串口传输 + `FakeDevice`
- [ ] `component/` 状态机（移植 robot-concierge 的 WakeVoiceBot）
- [ ] `firmware-xiao/` XIAO 固件（ESP-IDF + WakeNet）
