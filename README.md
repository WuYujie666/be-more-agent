# 睡前陪伴机器人 🌙

**一个跑在树莓派上、全程离线的中文语音睡前陪伴机器人**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red) ![License](https://img.shields.io/badge/License-MIT-green)

> 计算机体系结构课程项目。基于树莓派，把「语音识别 → 大模型 → 语音合成」的完整链路全部塞进一台本地设备，不联网、不上云。

睡前打开它，它会安静地听你说话，温柔地接住你这一天的情绪，陪你聊一会儿；聊够了就主动收尾、道一声晚安，并放上一段助眠白噪音或轻音乐，定时自动关机。第二天晚上再开机，它会用一句话回顾「昨天你聊了什么、心情如何」，再问你今天想分享些什么。

---

## ✨ 它能做什么

* **全离线、纯中文**：语音识别、对话、语音合成全部在树莓派本机完成，无需联网、无 API 费用、对话内容不出设备。
* **免唤醒词持续监听**：用 VAD（语音活动检测）自动断句录音，检测到你开口就录、停顿就停，不用喊唤醒词、不用按键。
* **睡前情绪梳理**：人格被设定为温和、简短，只「接住」你当下这句话，轻轻引导你说出想法与感受，但**不出主意、不深挖、不在睡前帮你解决烦心事**——提到烦心事就先「寄存」，明天再想。
* **主动收尾 + 助眠音频**：聊够设定轮数、或听出你困了，就温柔收尾，询问（或直接播放）白噪音 / 轻音乐，并在自然放完或到达时间上限后自动关机。
* **隔夜记忆**：每晚把对话压缩成一句话摘要存档，第二天开场自动回顾，营造「认识你」的连续感。
* **反应式面部动画**：tkinter 界面根据状态（待机 / 聆听 / 思考 / 说话 / 问候 / 睡眠 / 预热）切换面部动画。
* **可随时打断**：说到一半你想插话，按空格即可立即打断当前发言。

---

## 🧠 工作原理

### 单轮对话数据流水线（全本机、不联网）

```text
麦克风 → webrtcvad 断句录音 → whisper.cpp 语音转文字 (STT)
       → Ollama 大模型流式生成回复 (LLM)
       → sherpa-onnx / piper 文字转语音 (TTS) → 扬声器播放
```

### 线程模型

* **GUI 主线程**：tkinter 界面，负责面部动画、状态文字、按键事件。
* **后台主循环线程**：串起「录音 → 识别 → 对话」一整轮，循环往复。
* **TTS 两级流水线线程**：合成线程（`_synth_worker`）与播放线程（`_play_worker`）靠队列解耦——当前句正在播放时，下一句已经在提前合成，消除句间空挡。

### 睡前引导状态机

对话由 [chat_engine.py](chat_engine.py) 里的 `ChatEngine` 驱动，分三个阶段流转：

```text
chat（聊天中）── 听出睡意 / 聊满轮数 ──► ask_audio（询问白噪音还是轻音乐）
                                              │
                                              ▼
                                       playing（播放助眠音频，到时自动关机）
```

大模型回复里可夹带内部控制标签 `[AUDIO:white]` / `[AUDIO:music]` 来决定播哪种助眠音频，标签会被程序剥离、绝不读给用户。

### 交互按键

| 按键 | 作用 |
| --- | --- |
| `Space` | 打断当前思考 / 发言 |
| `Esc` | 全屏 / 窗口切换 |
| `Ctrl+Q` 或 Exit 按钮 | 退出并存档（生成当日摘要、卸载模型） |
| 点击画面 | 切换聊天气泡 / 状态栏的显隐 |

---

## 📂 项目结构

```text
be-more-agent/
├── agent.py                 # 主程序：tkinter 界面 + 录音/识别/合成播放流水线
├── chat_engine.py           # 对话核心 ChatEngine（无硬件依赖：状态机/LLM/记忆/摘要）
├── config.py                # 配置加载、关键词识别、摘要存取、音频设备选择、常量
├── debug_chat.py            # 文字版对话调试工具（开发机上调 prompt，免硬件）
├── test_agent.py            # pytest 单元测试
├── config.json              # 用户配置（模型、whisper、VAD、睡前引导话术…）
├── summaries.json           # 每日一句话摘要存档（运行时生成）
├── requirements.txt         # Python 依赖
├── start_agent.sh           # 启动脚本（激活 venv 并运行 agent.py）
├── be-more-agent.desktop    # 桌面快捷方式（开机自启场景）
├── whisper.cpp/             # 语音转文字引擎与模型
├── sherpa-models/           # sherpa-onnx 中文 TTS 模型（vits-zh-aishell3）
├── piper/                   # Piper TTS 引擎（备用 / 英文嗓）
├── sounds/
│   └── relaxation/          # 助眠音频
│       ├── white_noise/     # 白噪音（雨声、海浪、风声…），默认类型
│       └── music/           # 轻音乐（钢琴、纯音乐…）
└── faces/                   # 面部动画帧（每个状态一个子目录，内含 PNG 序列）
    ├── idle/  listening/  thinking/  speaking/
    └── greeting/  sleep/  warmup/
```

> 三层职责清晰分离：`config.py`（配置与纯函数）→ `chat_engine.py`（与硬件无关的对话逻辑）→ `agent.py`（硬件/GUI）。因为对话核心不依赖任何硬件，[debug_chat.py](debug_chat.py) 才能复用同一套逻辑、在开发机上不接麦克风/扬声器就调 prompt。

---

## 🛠️ 硬件要求

* **树莓派 5**（推荐）或树莓派 4（建议 ≥ 4GB 内存）
* USB 麦克风
* 扬声器
* （可选）DSI / HDMI 屏幕，用于显示面部动画

---

## 🚀 安装与运行

> ⚠️ 仓库里的 [setup.sh](setup.sh) 是上游英文项目遗留的脚本（下载英文嗓、唤醒词、拉取 `gemma3:1b` 等），**与本项目的中文配置不一致**，仅供参考，请勿直接照搬。下面是本项目实际需要的依赖。

### 1. 系统依赖

```bash
sudo apt update
sudo apt install -y python3-tk python3-dev libasound2-dev portaudio19-dev cmake build-essential git
```

### 2. 安装 Ollama 并拉取文本模型

本项目用 [Ollama](https://ollama.com) 在本机运行大模型。

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
```

### 3. 编译 whisper.cpp 并下载中文模型

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build && cmake --build build --config Release
# 下载中文识别模型（程序默认用 ggml-small.bin）
bash ./models/download-ggml-model.sh small
cd ..
```

程序通过 `./whisper.cpp/build/bin/whisper-cli` 调用，模型放在 `./whisper.cpp/models/` 下。

### 4. 下载 sherpa-onnx 中文 TTS 模型

下载 `vits-zh-aishell3` 模型，解压到 `sherpa-models/vits-zh-aishell3/`（需包含 `vits-aishell3.onnx`、`lexicon.txt`、`tokens.txt` 以及 `*.fst` / `rule.far` 规则文件）。模型可从 [sherpa-onnx 预训练模型](https://github.com/k2-fsa/sherpa-onnx/releases) 获取。

### 5. 安装 Python 依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install sherpa-onnx        # 中文 TTS 引擎（requirements.txt 未含）
```

### 6. 准备助眠音频

把 **16-bit PCM `.wav`** 音频按类型放进 `sounds/relaxation/white_noise/` 和 `sounds/relaxation/music/`，详见 [sounds/relaxation/README.md](sounds/relaxation/README.md)。

### 7. 运行

```bash
source venv/bin/activate
python agent.py
```

开发机上调试对话 / prompt（只需 Ollama，免硬件）：

```bash
python debug_chat.py                         # 交互打字逐轮对话
python debug_chat.py --script <脚本文件>     # 一键回放脚本化对话
```

运行单元测试：

```bash
pytest
```

---

## ⚙️ 配置说明（`config.json`）

程序启动时以 `config.py` 里的 `DEFAULT_CONFIG` 为底，再用 `config.json` 覆盖。主要字段：

| 字段 | 说明 |
| --- | --- |
| `text_model` | Ollama 文本模型，如 `qwen2.5:1.5b` |
| `tts_engine` | TTS 引擎：`sherpa`（中文，推荐）或留空走 `piper` |
| `sherpa_model_dir` | sherpa-onnx 模型目录，如 `sherpa-models/vits-zh-aishell3` |
| `voice_model` | piper 备用嗓模型路径 |
| `whisper_model` / `whisper_lang` | whisper 模型与语言（如 `ggml-small.bin` / `zh`） |
| `whisper_beam_size` / `whisper_best_of` | 解码参数，`1` = greedy，在树莓派上最快 |
| `vad_aggressiveness` / `vad_silence_ms` / `vad_preroll_ms` … | VAD 断句灵敏度与时序 |
| `max_chat_turns` | 聊满几轮后主动收尾过渡（默认 5） |
| `default_audio` | 默认助眠音频类型：`white_noise` / `music` |
| `relaxation_gain` / `relaxation_max_minutes` | 助眠音频音量与最长播放分钟数（到点自动关机） |
| `summaries_file` / `summary_recent_days` | 每日摘要存档文件与「昨日摘要」有效天数 |
| `system_prompt` | 机器人人格与睡前引导规则的唯一来源 |
| `transition_prompt` / `max_turns_prompt` / `decline_audio_prompt` / `goodnight_prompt` / `story_prompt` / `summary_prompt` | 各阶段一次性引导话术 |
| `debug_prompt` | 打开后每次调 LLM 前打印完整 messages，便于调 prompt |

---

## 🎨 自定义

* **助眠音频**：往 `sounds/relaxation/<类型>/` 放 `.wav` 即可，详见 [sounds/relaxation/README.md](sounds/relaxation/README.md)。
* **面部动画**：往 `faces/<状态>/` 放 PNG 帧序列，程序按文件名排序循环播放（250ms/帧）；缺帧时回退到 `idle`。
* **人格与话术**：改 `config.json` 的 `system_prompt` 与各 `*_prompt`，无需动代码即可调整机器人的语气和睡前引导方式。

---

## 🙏 致谢与许可

本项目最初 fork 自 [brenpoly/be-more-agent](https://github.com/brenpoly/be-more-agent)，并改造为中文睡前陪伴场景。感谢以下开源项目：

* [Ollama](https://ollama.com) —— 本地大模型运行时
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp) —— 语音转文字
* [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) —— 中文语音合成
* [Piper](https://github.com/rhasspy/piper)（Rhasspy 项目）—— 备用语音合成

源代码基于 [MIT License](LICENSE) 开源。
