# =========================================================================
#  Be More Agent · 文字版对话调试工具
# =========================================================================
"""不依赖麦克风/扬声器/whisper.cpp/sherpa，在开发机上用打字或脚本测 prompt。

只需要 ollama + 目标文本模型（config.json 的 text_model）就能跑。复用
chat_engine.ChatEngine 的全部真实逻辑（多轮状态机、prompt 拼装、[AUDIO:x]
标签剥离、每日摘要），只把显示/TTS/播放换成控制台实现，因此调出来的 prompt
和回复与住应用完全一致。

用法：
    python debug_chat.py                                   # 交互打字，逐轮对话
    python debug_chat.py --script scenarios/bedtime5.txt   # 一键跑脚本化对话

启动时强制打开 debug_prompt，每次调 LLM 前会打印完整 messages。
交互模式下输入空行或 quit / exit / q 结束。
"""

import sys

# 强制打开 prompt 打印——必须在 import chat_engine 之前改 CURRENT_CONFIG，
# 因为 _dump_messages 在调用时读 CURRENT_CONFIG，这里早改晚改都生效，放前面更稳。
from config import CURRENT_CONFIG
CURRENT_CONFIG["debug_prompt"] = True

from chat_engine import ChatEngine


class ConsoleIO:
    """把 ChatEngine 的 8 个交互钩子接到控制台：流式打印回复，不出声、不放音频。"""

    def set_state(self, state, msg=""):
        if msg:
            print(f"  ·[{state}] {msg}", flush=True)

    def bot_start(self):
        print("机器人: ", end="", flush=True)

    def token(self, piece):
        print(piece, end="", flush=True)

    def speak(self, sentence):
        pass   # 控制台不出声；回复内容已由 token() 实时打出

    def wait_speech(self):
        pass   # 控制台无音频，无需等待

    def is_interrupted(self):
        return False   # 文字调试不支持打断

    def stage(self, msg):
        print(f"  [STAGE] {msg}", flush=True)

    def play_relaxation_audio(self, audio_type):
        print(f"\n  [助眠] 此处会播放：{audio_type}（控制台模式不实际播放）", flush=True)


def _dispatch(engine, user_text):
    """喂一轮用户输入，打印分隔与回复。"""
    print(f"\n用户: {user_text}", flush=True)
    engine.handle_turn(user_text)
    print()   # 回复流式结束后补个换行


def run_interactive(engine):
    print("=== 文字对话调试（交互模式）===")
    print("直接打字回车发送；空行或 quit/exit/q 结束。\n", flush=True)
    while True:
        if engine.phase == "playing":
            print("[已进入助眠播放阶段，对话结束]", flush=True)
            break
        try:
            user_text = input("你说> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in ("", "quit", "exit", "q"):
            break
        _dispatch(engine, user_text)
    _finish(engine)


def run_script(engine, path):
    print(f"=== 文字对话调试（脚本模式：{path}）===\n", flush=True)
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.lstrip().startswith("#")]
    for user_text in lines:
        if engine.phase == "playing":
            print("[已进入助眠播放阶段，脚本剩余轮次跳过]", flush=True)
            break
        _dispatch(engine, user_text)
    _finish(engine)


def _finish(engine):
    """对话结束后确保每日摘要生成并打印（脚本/交互通用）。
    若已进入 ask_audio 应答（handle_audio_answer 已异步起摘要线程），等它跑完——
    否则直接退出会把 daemon 线程杀掉，摘要来不及生成；其余情况同步补生成一次。"""
    print("\n=== 生成每日摘要 ===", flush=True)
    if engine.summary_thread is not None:
        engine.summary_thread.join()
    else:
        engine.finalize_session_summary()


def main(argv):
    # Windows 控制台默认 gbk，中文 prompt/回复会乱码；强制 stdout 用 UTF-8。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    engine = ChatEngine(ConsoleIO())
    if len(argv) >= 2 and argv[0] in ("--script", "-s"):
        run_script(engine, argv[1])
    elif argv:
        print(f"未知参数：{argv}\n用法：python debug_chat.py [--script <file>]", flush=True)
        return 1
    else:
        run_interactive(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
