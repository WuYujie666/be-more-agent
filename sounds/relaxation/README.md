# 助眠放松音频

把你的助眠音频按类型放进这里，程序会从对应子目录取第一个 `.wav` 播放：

```
sounds/relaxation/
├── white_noise/   # 白噪音（雨声、海浪、风声…），默认类型
└── music/         # 轻音乐（钢琴、纯音乐…）
```

要求：

- 格式为 **16-bit PCM `.wav`**（程序用标准库 `wave` 读取）。
- 单条音频即可：程序播放一次，**自然放完或满 45 分钟**（见 `config.json` 的 `relaxation_max_minutes`）后自动关机。
- 音量已在程序里按 `relaxation_gain`（默认 0.6）做过平衡，比语音更柔和。

相关配置（`config.json`）：

- `default_audio`：默认音频类型（`white_noise` / `music`）。
- 用户在对话里说「想听雨声 / 轻音乐」，或大模型输出 `[AUDIO:rain]` / `[AUDIO:music]` 标签，会切换类型。
