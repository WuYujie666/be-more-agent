# 睡前陪伴机器人 🌙

**一个跑在树莓派上的中文语音睡前陪伴机器人**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red) ![License](https://img.shields.io/badge/License-MIT-green)

> 计算机体系结构课程项目。基于树莓派，语音识别和语音合成在本地运行，文本模型支持 DeepSeek API 或本地后端。

睡前打开它，它会安静地听你说话，温柔地接住你这一天的情绪，陪你聊一会儿；聊够了就主动收尾、道一声晚安，并放上一段助眠白噪音或轻音乐，定时自动关机。第二天晚上再开机，它会用一句话回顾「昨天你聊了什么、心情如何」，再问你今天想分享些什么。

---

## ✨ 它能做什么

* **中文语音交互**：语音识别与合成本地运行；当前对话使用 DeepSeek API，会发送文本并产生费用，也可配置本地文本模型。
* **免唤醒词持续监听**：用 VAD（语音活动检测）自动断句录音，检测到你开口就录、停顿就停，不用喊唤醒词、不用按键。
* **睡前情绪梳理**：人格被设定为温和、简短，只「接住」你当下这句话，轻轻引导你说出想法与感受，但**不出主意、不深挖、不在睡前帮你解决烦心事**——提到烦心事就先「寄存」，明天再想。
* **主动收尾 + 助眠音频**：聊够设定轮数、或听出你困了，就温柔收尾，询问（或直接播放）白噪音 / 轻音乐，并在自然放完或到达时间上限后自动关机。
* **隔夜记忆**：每晚把对话压缩成一句话摘要存档，第二天开场自动回顾，营造「认识你」的连续感。
* **反应式面部动画**：tkinter 界面根据状态（待机 / 聆听 / 思考 / 说话 / 问候 / 睡眠 / 预热）切换面部动画。
* **可随时打断**：说到一半你想插话，按空格即可立即打断当前发言。

---

## 🧠 工作原理

### 单轮对话数据流水线

```text
麦克风 → webrtcvad 断句录音 → SenseVoice / whisper.cpp 语音转文字 (STT)
       → DeepSeek API / llama.cpp / Ollama 大模型流式生成回复 (LLM)
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
# 下载中文识别模型，并生成当前 config.json 使用的 Q8_0 版本
bash ./models/download-ggml-model.sh small
./build/bin/whisper-quantize models/ggml-small.bin models/ggml-small-q8_0.bin q8_0
cd ..
```

程序通过 `./whisper.cpp/build/bin/whisper-cli` 调用，模型放在 `./whisper.cpp/models/` 下。

Whisper 回退配置使用 `ggml-small-q8_0.bin`；原始 `ggml-small.bin` 保留用于对照或回退。

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

性能实验结果与人工核对的 CER 记录见 [experiments/PERFORMANCE.txt](experiments/PERFORMANCE.txt)。

### 本地大模型后端：llama.cpp（回退选项）

将 `config.json` 中 `llm_backend` 设为 `llama.cpp` 可使用本地后端。通过 `start_agent.sh` 或
`venv/bin/python agent.py` 启动时，会自动启动本机服务，监听 `127.0.0.1:18080`。
对话、预热和摘要共用后端；退出时关闭本程序启动的服务，外部启动的服务不受影响。
运行日志在 `cache/llama-server.log`。

本机运行包：`runtime/llama-b11796/`；模型：
`cache/backend-benchmark/qwen2.5-1.5b-q4_k_m.bin`（内容为 GGUF）。
这两项是本地文件，不随 Git 提交；新机器需要从官方来源准备它们并配置路径。
运行包来源：https://github.com/ggml-org/llama.cpp/releases/tag/b11796
模型 SHA-256：`183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4`。

配置项：`llama_binary`、`llama_model` 为项目相对路径或绝对路径；
`llama_context` 当前 4096；`llama_autostart` 当前 true；`llama_url` 当前
`http://127.0.0.1:18080`。连接自行管理的服务时关闭 `llama_autostart`。
服务不可用会明确报错，不会悄悄切换后端。

回退：把 `llm_backend` 改为 `ollama` 后重启，继续使用 `text_model` 指定的模型。
两种本机后端均绕过 HTTP 代理，不修改系统代理配置。

### 当前大模型后端：DeepSeek Flash API

在 Git 忽略的 `.env` 中填写 `CLOUD_LLM_API_KEY`、
`CLOUD_LLM_BASE_URL=https://api.deepseek.com`、`CLOUD_LLM_MODEL=deepseek-flash`。
同名环境变量优先于文件；该文件只读取简单 KEY=VALUE，不执行 shell 展开。
选择云端时将 `config.json` 的 `llm_backend` 改为 `deepseek` 并重启。
对话和摘要会发送到 DeepSeek；默认关闭深度思考，以降低语音等待。
外部请求沿用现有网络代理，本地后端继续绕过代理。密钥不打印到日志。
当前正式配置已设为 `deepseek`；测试结果见 `experiments/deepseek_results.json`。
手动运行 `experiments/test_deepseek_live.py` 会调用付费 API（合成测试输入）。

### 当前识别后端：SenseVoice

正式链路为 SenseVoice → DeepSeek Flash → 本地 sherpa TTS，提示词保持原样。
`asr_backend` 当前为 `sensevoice`，模型目录为 `sherpa-models/sensevoice-int8`，
包含 `model.int8.onnx`、`tokens.txt` 和 LICENSE。识别器启动加载一次，4线程常驻。
下载来源：https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html
输入使用录音流程产生的单声道16位PCM WAV；空音频返回空文字。
回退识别：把 `asr_backend` 改为 `whisper` 后重启，继续使用保留的 Q8 模型。
回退文本模型：把 `llm_backend` 改为 `llama.cpp` 后重启。两项可以独立切换。

自动完整链路测试（会调用所配置的云端API并播放音频）：
`venv/bin/python -B experiments/smoke_voice_pipeline.py`
它使用固定3秒录音，不录麦克风，不读取历史摘要，不写入测试记忆。
