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

## 调试日志

使用现有 `config.json` 开关：`debug_prompt: true` 打印模型输入、决策、工具
开始/结果/耗时，并区分自动记忆检索；`whisper_debug` 控制 whisper-cli 诊断。
在终端正常执行 `bash start_agent.sh` 即可查看。修改配置后重启生效，
不需要额外 CLI 参数、调试启动器或独立日志窗口。

## Agent 与记忆操作（当前版本）

默认 `config.json` 中的 `agent_backend` 是 `native`：使用现有 OpenAI SDK，
模型返回结构化决定，程序执行并核对结果。可改为 `langchain` 或 `pydantic`，
重启生效；后两者先安装 `experiments/requirements-agents.txt`。三者共用业务层，
不用迁移 Mem0 数据。旧 `legacy` 模式保留作兼容，它不提供这些操作工具。

- 普通记忆支持搜索、纠错和选择遗忘；目标不明确时应追问。
- “以后回复短一点、少主动问问题”等明确偏好，以及日期明确的计划，在本次采用，
  会话结束后统一写入 Mem0。到期计划在后续开场关心一次，不等同于闹钟。
- “今晚音量30%、放10分钟”调整本次助眠播放；只影响白噪音/轻音乐，
  不调整系统或TTS音量，播放过程中尚不支持语音控制。
- “本次聊天不要保存”由模型按语义决定是否调用遗忘操作，不使用固定句式匹配。
  当前遗忘范围是整段尚未保存会话；不支持精确删除其中一句。纠错/删除还会清除
  当前上下文与摘要，避免旧内容在结束时重新写入；其他长期记录保持不变。

操作确认来自真实执行结果。模型仍可能理解错意图，因此这不是任意自然表述都
可靠的保证。完整结构化决定校验后才送入TTS，日志中的“首段文本就绪”不等于
底层首token延迟；TTS分段和合成/播放流水线仍保留。

云端网络阶段超时为12秒，SDK不自动重试；它不是整轮总时限。框架/请求异常会
回退本地 llama.cpp/Qwen文字回复，已执行操作不重放。本地回退不开放修改工具。
测试基准禁用回退，避免将Qwen的回复算作被测框架成功。

三方案已完成本机对比，当前建议先用 `native`：功能相同，现阶段没有测出引入
框架的必要收益。详细数据、失败实验、成本范围和局限见
[Agent 对比报告](experiments/agent_compare/REPORT.txt)。框架均可在本机运行，
以后流程变复杂可以继续使用保留的独立适配器。

主要代码：`agent_adapters.py` 管取消与回退，`agent_decisions.py` 管结构化决定与
执行回执，`agent_backends/` 放三个可替换适配器，`companion_tools.py` 管业务动作，
`memory_backend.py` 管 Mem0 持久化。最终版本不依赖 LangGraph 状态图。

## 🧠 工作原理

### 单轮对话数据流水线

```text
麦克风 → webrtcvad 断句录音 → SenseVoice / whisper.cpp 语音转文字 (STT)
       → DeepSeek API / llama.cpp 大模型流式生成回复 (LLM)
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
├── runtime/memory/          # Mem0 事实、偏好与计划（开场和聊天共用）
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

### 2. 配置文本模型

当前使用 DeepSeek API，在 `.env` 配置密钥；本地 llama.cpp/Qwen 用于故障回退。
具体配置与本地模型路径见下方后端说明。

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

开发机上调试对话 / prompt（配置文本后端即可，免硬件）：

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
| `text_model` | 文本模型标识；本地权重由 `llama_model` 指定 |
| `tts_engine` | TTS 引擎：`sherpa`（中文，推荐）或留空走 `piper` |
| `sherpa_model_dir` | sherpa-onnx 模型目录，如 `sherpa-models/vits-zh-aishell3` |
| `voice_model` | piper 备用嗓模型路径 |
| `whisper_model` / `whisper_lang` | whisper 模型与语言（如 `ggml-small.bin` / `zh`） |
| `whisper_beam_size` / `whisper_best_of` | 解码参数，`1` = greedy，在树莓派上最快 |
| `vad_aggressiveness` / `vad_silence_ms` / `vad_preroll_ms` … | VAD 断句灵敏度与时序 |
| `max_chat_turns` | 聊满几轮后主动收尾过渡（默认 5） |
| `default_audio` | 默认助眠音频类型：`white_noise` / `music` |
| `relaxation_gain` / `relaxation_max_minutes` | 助眠音频音量与最长播放分钟数（到点自动关机） |
| `system_prompt` | 机器人人格与睡前引导规则的唯一来源 |
| `transition_prompt` / `max_turns_prompt` / `decline_audio_prompt` / `goodnight_prompt` / `story_prompt` | 各阶段一次性引导话术 |
| `debug_prompt` | 打开后每次调 LLM 前打印完整 messages，便于调 prompt |

---

## 🎨 自定义

* **助眠音频**：往 `sounds/relaxation/<类型>/` 放 `.wav` 即可，详见 [sounds/relaxation/README.md](sounds/relaxation/README.md)。
* **面部动画**：往 `faces/<状态>/` 放 PNG 帧序列，程序按文件名排序循环播放（250ms/帧）；缺帧时回退到 `idle`。
* **人格与话术**：改 `config.json` 的 `system_prompt` 与各 `*_prompt`，无需动代码即可调整机器人的语气和睡前引导方式。

---

## 🙏 致谢与许可

本项目最初 fork 自 [brenpoly/be-more-agent](https://github.com/brenpoly/be-more-agent)，并改造为中文睡前陪伴场景。感谢以下开源项目：

* [llama.cpp](https://github.com/ggml-org/llama.cpp) —— 本地大模型运行时
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
本地服务也不可用时会明确报错。

手动选择本地：把 `llm_backend` 改为 `llama.cpp` 后重启。
本机后端绕过 HTTP 代理，不修改系统代理配置。

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

### 本地长期记忆（Mem0 最小版）

与语音模块共用 `venv`。依赖固定为 Mem0 2.2.1、FastEmbed 0.9.0、
OpenAI SDK 2.54.0、protobuf 6.33.6；无需独立服务或独立虚拟环境。
安装项目依赖后，首次执行 `venv/bin/python setup_memory.py` 下载约 90 MB
的 BGE-small-zh-v1.5。正常运行从本地加载模型，不自动下载。

`config.json` 中 `long_term_memory` 控制启用，`memory_dir` 指定目录
（默认 `runtime/memory`）。该目录的 Qdrant 保存记忆正文和向量，
`history.db` 是 SQLite 操作历史及最近提取上下文；每个目录只对应一个用户。
提取使用 `.env` 中已有的 DeepSeek 配置，会增加 API 调用。
把本次会话原文交给提取模型，不导入旧摘要、不存原始录音。

每轮回答前检索最多四条相关记忆；本次会话结束（进入助眠或退出）后，
将整段会话一次性提交给后台提取，聊天过程中不逐句写入。助手发言仅用于
理解用户上下文，提取指令要求只保存用户事实；开场问候不进入提取。
后台忙时最多等待锁 150 ms，然后本轮跳过长期检索，继续使用会话历史。
这不是整个检索过程的硬超时。初始化或写入失败会打印 `[MEMORY]` 提示，
不会阻断正常聊天；后台写入可能尚未完成，异常退出可能丢失未完成写入。
正常退出等待后台任务，超时会记录提示。

可说“清空记忆”或“请清空记忆”：清除长期记录、SQLite 历史/提取上下文、
每日摘要和当前会话上下文。先等待已接受的写入，再清空，避免旧任务恢复记忆。
失败时不会声称已经清空。此功能不保证删除既有日志、备份或云端服务记录。

暂停语音程序后，可用以下命令审核记录（同一 Qdrant 目录不能同时打开）：

```bash
venv/bin/python memory_tool.py list
venv/bin/python memory_tool.py search '我喜欢什么声音？'
venv/bin/python memory_tool.py history <memory_id>
venv/bin/python memory_tool.py update <memory_id> '修订后的事实'
venv/bin/python memory_tool.py delete <memory_id>
```

工具的单条删除也清除该 ID 的 SQLite 历史及提取最近上下文；不会语义追踪
其他记录或每日摘要中的同一事实。工具 `clear` 仅清空长期库；完整清空请用
对话中的“清空记忆”。旧版 `memory.json` 当前不加载，也不属于本模块数据。

注意：这个 Mem0 版本的自动提取采用追加方式，变化前后的记录可能同时返回；
回答结合记录时间和变化描述判断。显式 `update` 才是改写并记录历史。
目前不是 proposal 中“唯一当前版本”的严格事实库，复杂纠错仍需继续评测。
第一版使用中文向量检索，关闭额外的英文 BM25 编码器，不装 spaCy/知识图谱。

验证：`venv/bin/python -m unittest test_memory -v`（离线），以及
`venv/bin/python experiments/check_memory.py --live`（虚构对话，会调用 DeepSeek）。
结果见 `experiments/memory_results.json`，测试不接触真实用户记忆。

25 秒录音记忆验收：`venv/bin/python -B experiments/check_memory_audio.py --live`。
使用独立临时库，跨进程验证保存、追问和清空；会调用付费 API，不运行 GUI/TTS。
结果见 `experiments/memory_audio_results.json`。LoCoMo 小样本结果见
`experiments/locomo_session/REPORT.txt`，自动评分较宽松，不代表完整回答准确率。

### 云端故障回退

DeepSeek 请求失败（包括 HTTP 错误、连接/读取超时、空回复、流意外结束）时，
聊天、预热和摘要自动尝试本地 llama.cpp/Qwen；云端连接超时5秒、读取超时12秒。
读取超时是等待下一批数据的时限，不是整个回复的总时限。下一次调用仍优先云端。
流式回复已经输出的部分保留，并让本地模型续写；模型仍可能重复或改变语气。
用户主动打断不会触发回退。首次回退需加载本地模型，响应会比正常云端慢。
本地服务也失败则保留错误提示，不无限重试。Mem0 提取的 API 失败也尝试本地
模型；它的超时为30秒。小模型的结构化提取质量和4096 token上下文仍有限，
过长会话或无效输出可能无法保存，不能保证所有云端失败都能完成记忆写入。
Ollama 已从项目依赖、运行路径及安装脚本移除；历史实验数据保留。

开场与跨会话连续性：优先关心 Mem0 中日期过去 1～7 天且尚未关心的计划；否则使用普通问候。不推断计划已完成。“少主动提问”时使用普通问候。会话结束只提交 Mem0，不再生成独立开场摘要；旧 summaries.json 留作历史文件，程序不再读取。

### VAD 误触发诊断

默认开启 `vad_diagnostics_enabled`，每次完成 VAD 录音后，在相对运行目录的
`diagnostics/vad/` 保存一对同名 `.wav` 和 `.json`。WAV 与送入 ASR 的音频一致；
JSON 在 ASR 完成后补上 `asr_text`，异常时记录 `asr_error`。文件名时间为 UTC。
程序无法自动判断是否误触发，因此正常说话和疑似误触发都会保存，便于对照回听。

JSON 包含 `duration_ms`、`vad_speech_ms`、`vad_speech_ratio`、`rms_dbfs`、
逐帧 `speech` / `rms_dbfs` / `peak_dbfs`、停止原因、采样率、VAD 配置和输入溢出次数。
`noise_floor_dbfs` 是触发前最近最多约 5 秒的 VAD 非语音帧 RMS 中位数，
没有这种帧时为 null；`speech_above_noise_db` 是语音帧音量中位数与该估计值之差，
不是经过校准的真实信噪比。VAD 标记本身可能出错，需结合 WAV 判断。

`config.json` 可覆盖：`vad_diagnostics_enabled`（默认 true）、
`vad_diagnostics_dir`（默认 `diagnostics/vad`）、
`vad_diagnostics_max_captures`（默认 100，至少保留 1 次）。
每次保存后自动清理最旧样本及其 JSON。默认目录已被 Git 忽略，内容包含本地录音和识别文本。
发现误触发时，按终端 `[VAD DIAG]` 路径找到对应文件；需要长期保留的样本请复制到其他目录。
诊断保存失败仅打印日志，不中断原有识别流程。

计划在会话结束的 Mem0 写入队列中统一提取，不依赖聊天模型当场调用 plan。提取以用户发言时间解释相对日期，综合改期、取消和不保存要求；结构校验失败最多纠正重试一次。普通事实继续走 Mem0 原有提取，并排除计划，避免两个来源同时维护计划。相比原流程，结束整理通常增加一次模型请求；不是数据库事务，部分步骤失败会报告未完整保存。

### 正式 VAD 后端

默认使用本机 CPU 上的 Silero ONNX，`config.json` 中设置：

```json
{
  "vad_backend": "silero",
  "silero_vad_model": "models/silero-vad/silero_vad.onnx",
  "silero_vad_threshold": 0.5
}
```

首次安装运行 `venv/bin/python setup_vad.py`（`setup.sh` 也会调用）。
下载会检查已验证模型的 SHA-256；模型缺失时启动会明确报错，不会悄悄切回其他检测器。
运行时不联网下载模型。模型会话复用，每轮监听重置状态；16kHz 输入按 512 个采样点
（32ms）处理，非 16kHz 输入先进行抗混叠重采样。
`vad_start_ms=150` 对应连续 5 帧（160ms）；`vad_silence_ms=900` 对应 29 帧（928ms）。
前置缓冲至少覆盖启动确认窗口，以保留短回应开头。诊断 JSON 记录后端、阈值及逐帧
`speech_probability`。需要对照旧后端时把 `vad_backend` 改为 `webrtc` 并重启。

验证：`venv/bin/python -m unittest test_vad_backend test_vad_diagnostics -v`。
本地录音回归用例需要保留 `diagnostics/vad/` 中本次采集的两段样本；没有样本时跳过。
此阶段仍是轮流收听与说话；Silero 本身不提供外放回声消除或语音打断。

### 语音打断（快速版）

`voice_interrupt_enabled` 默认 true。独立模块 `voice_interrupt.py` 使用持续全双工
音频流：实际播放参考 → SpeexDSP 基础回声消除 → Silero → 录音队列。
普通收听仍按 `vad_start_ms` 启动；回复期间使用 `barge_in_threshold=0.7` 和
`barge_in_start_ms=256` 确认用户开口，然后停止旧播放、清空待播队列。
合成中的旧音频带轮次编号，完成后也不会混入新回复。打断录音保留前置缓冲，
说完后走原来的 ASR 和对话逻辑。开场问候也支持打断；助眠音频阶段维持原来的停止监听行为。

此版本依赖系统 `libspeexdsp1`（已加入 setup.sh），默认输入输出设备均使用 48kHz，
需要设备支持同时录放。`aec_reference_delay_ms=10` 是参考信号额外提前量，
基础 AEC 的实际表现受音量、设备延迟和摆放影响，仍需真人同时说话验证。
诊断 WAV 是回声处理后的、实际送给 ASR 的音频。

关闭打断并回到轮流录放：在 `config.json` 设置 `"voice_interrupt_enabled": false` 后重启。
语音调试启动：`venv/bin/python -u debug_voice.py`，保留麦克风/TTS，动画窗口最小化。
验证：`venv/bin/python -m unittest test_voice_interrupt test_vad_backend test_vad_diagnostics -v`。
