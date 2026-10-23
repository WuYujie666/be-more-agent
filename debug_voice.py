"""Voice debugging with console logs; keep the animated window minimized."""
from config import CURRENT_CONFIG
CURRENT_CONFIG['debug_prompt'] = True

import tkinter as tk
from agent import BotGUI

if __name__ == '__main__':
    print('=== 语音调试：Silero + ASR + 对话 + TTS ===', flush=True)
    print('动画窗口默认最小化；终端显示识别、模型和播放日志。', flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    root.after(500, root.iconify)
    root.mainloop()
