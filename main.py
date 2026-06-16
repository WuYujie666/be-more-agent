# =========================================================================
#  Be More Agent · 睡前情绪梳理机器人入口
#  用 SleepFlow 状态机替换原有的开放式问答主循环。
#
#  依赖: agent.py (BotGUI), flow.py (SleepFlow), config.py, prompts.py
#
#  启动方式: python main.py
#  注: 保留 agent.py 不动，agent.py 的 if __name__ == "__main__" 在新流程中不再使用。
# =========================================================================

import tkinter as tk
import threading

from config import CURRENT_CONFIG
from agent import BotGUI


def main():
    print("--- 睡前情绪梳理机器人 STARTING ---", flush=True)

    # 1. 创建 Tk 窗口和 GUI
    #    autostart=False：不起 BotGUI 自带的开放式对话循环，改由 SleepFlow 驱动，
    #    避免两个线程同时抢麦克风。
    root = tk.Tk()
    app = BotGUI(root, autostart=False)

    # 2. 创建 SleepFlow 状态机，传入 BotGUI 实例
    #    CHAT 状态委派给 app.run_chat_phase，复用 BotGUI 的录音/STT/LLM/TTS 内核。
    from flow import SleepFlow
    flow = SleepFlow(config=CURRENT_CONFIG, gui=app)

    # 3. 在后台线程启动状态机（不阻塞 Tk 主循环）
    threading.Thread(target=flow.start, daemon=True).start()

    # 4. Tk 主循环在前台运行（响应用户按键等）
    root.mainloop()


# =========================================================================
# 修改说明
#
# 原 entry point (agent.py 末尾):
#   if __name__ == "__main__":
#       root = tk.Tk()
#       app = BotGUI(root)
#       root.mainloop()
#
# 改为:
#   python main.py   ← SleepFlow 状态机自动接管
#
# 关键变化:
#   1. BotGUI(autostart=False)：不启动 safe_main_execution 开放式对话循环
#   2. SleepFlow 编排宏观状态；CHAT 委派 BotGUI.run_chat_phase 复用聊天内核
#   3. BotGUI 的 set_state / run_chat_phase / warm_up / play_sound 等被 SleepFlow 调用
#   4. 原有唤醒词 / PTT 停用；开放式对话仅在 python agent.py 独立运行时启用
#   5. agent.py 仍可 python agent.py 独立运行（autostart 默认 True）
# =========================================================================

if __name__ == "__main__":
    main()
