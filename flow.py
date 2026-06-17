# =========================================================================
#  Be More Agent · 睡前情绪梳理状态机
#  档2: 单向状态机编排，从 BOOT 到 SHUTDOWN 共 6 个状态。
#  依赖 config / prompts，与 BotGUI 实例协作完成语音交互。
# =========================================================================

import os
import time
import wave
import traceback
from enum import Enum

import numpy as np

# 录音/STT/LLM/TTS 全部交给 BotGUI（agent.py），flow 只编排宏观状态。
# 这里仅需要放松音频播放用的 sounddevice，以及轮次引导 prompt。

# 放松音频播放（缺失时静等超时，不影响状态流转）
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    _HAS_SOUNDDEVICE = False

try:
    from prompts import get_chat_prompt
    _HAS_CONFIG = True
except (ImportError, ModuleNotFoundError) as e:
    # 无硬件依赖的开发环境中 config/prompts 可能因 import sounddevice 失败 → 纯日志模式
    print(f"[FLOW] prompts.py import 失败: {e}", flush=True)
    print("[FLOW] 纯日志模式（无轮次引导 prompt）", flush=True)
    get_chat_prompt = None  # type: ignore
    _HAS_CONFIG = False


# =========================================================================
# 1. STATE ENUM
# =========================================================================

class SleepState(Enum):
    """睡前情绪梳理机器人状态枚举，严格单向转移，不可逆"""
    BOOT       = "boot"
    CHAT       = "chat"
    TRANSITION = "transition"
    AUDIO      = "audio"
    SLEEP      = "sleep"
    SHUTDOWN   = "shutdown"


# =========================================================================
# 2. 硬件检查
# =========================================================================

def check_hardware():
    """检查放松音频播放依赖是否可用，缺失时打印警告（录音/STT/LLM 由 BotGUI 负责）"""
    if not _HAS_SOUNDDEVICE:
        print("[HARDWARE] sounddevice 未安装，放松音频播放将退化为静等", flush=True)


# =========================================================================
# 3. 放松音频生成
# =========================================================================

def _generate_white_noise(duration_sec, sample_rate=44100):
    """生成白噪音音频 (samples, rate)"""
    n = int(sample_rate * duration_sec)
    noise = np.random.randn(n).astype(np.float32) * 0.3
    return noise, sample_rate


# =========================================================================
# 4. SLEEP FLOW 状态机
# =========================================================================

class SleepFlow:
    """
    睡前情绪梳理机器人状态机编排。

    单向状态转移（不可逆）::

        BOOT → CHAT → TRANSITION → AUDIO → SLEEP → SHUTDOWN

    使用方式::

        flow = SleepFlow(config=CURRENT_CONFIG, gui=bot_gui)
        flow.start()          # 阻塞直到 SHUTDOWN

    CHAT 状态把控制权交给 ``gui.run_chat_phase``（录音/STT/LLM/TTS 全由 BotGUI 负责），
    flow 自身只编排 BOOT/TRANSITION/AUDIO/SLEEP/SHUTDOWN 与固定话术缓存。

    参数:
        config:  配置字典（通常是 ``CURRENT_CONFIG``）
        gui:     BotGUI 实例（提供 warm_up / run_chat_phase / set_state / play_sound / _render）
                 为 None 时进入纯日志模式（无音频 I/O），适合调试。
    """

    def __init__(self, config, gui=None):
        self.cfg = config
        self.sleep_cfg = config.get("sleep_flow", {})
        self.gui = gui

        # --- 硬件检查 ---
        check_hardware()

        # --- 状态 ---
        self.state = SleepState.BOOT
        self.chat_round = 0
        self._exit_flag = False

        # --- 预合成缓存目录 ----------------------------------------------------
        self.cache_dir = self.sleep_cfg.get("cache_dir", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # --- 预合成固定话术 ---
        self._presynth_all()

        print(f"[FLOW] SleepFlow initialized  |  cache={os.path.abspath(self.cache_dir)}", flush=True)

    # =========================================================================
    # 4a.  缓存管理
    # =========================================================================

    def _cache_path(self, key):
        """返回缓存 wav 的完整路径"""
        return os.path.join(self.cache_dir, f"{key}.wav")

    def _presynth_all(self):
        """遍历 pre_synthesize_texts，缺失的调用 gui._render 合成并落盘。

        固定话术只走缓存一条路：渲染失败大声报错（不退回实时 TTS）。
        gui=None（纯日志模式）下跳过，由 _play_cached 容错。
        """
        texts = self.sleep_cfg.get("pre_synthesize_texts", {})
        if not texts:
            return

        renderer = None
        if self.gui and hasattr(self.gui, "_render"):
            renderer = self.gui._render

        if renderer is None:
            print("[CACHE] 无 renderer（gui=None 或缺 _render），跳过预合成", flush=True)
            return

        for key, text in texts.items():
            path = self._cache_path(key)
            if os.path.exists(path):
                print(f"[CACHE] Hit: {key}", flush=True)
                continue

            print(f"[CACHE] Synthesizing: {key} ...", flush=True)
            try:
                result = renderer(text)
                if result is not None:
                    samples, rate = result
                    _save_float32_wav(samples, rate, path)
                    print(f"[CACHE] Saved: {key} -> {path}", flush=True)
                else:
                    print(f"[CACHE][ERROR] 渲染返回空，话术 '{key}' 将无音频！", flush=True)
            except Exception as e:
                print(f"[CACHE][ERROR] 合成 '{key}' 失败: {e}", flush=True)
                traceback.print_exc()

    def _play_cached(self, key):
        """播放缓存 wav。缺失则大声报错（不实时合成兜底）。返回 True 表示已播放。"""
        path = self._cache_path(key)
        if os.path.exists(path) and self.gui:
            print(f"[AUDIO] Playing cached: {key}", flush=True)
            self.gui.play_sound(path)
            return True

        # 缓存缺失：固定话术按设计必须预合成成功，这里只报错不兜底。
        print(f"[AUDIO][ERROR] 缓存缺失，话术 '{key}' 未播放（应在初始化时预合成）", flush=True)
        return False

    # =========================================================================
    # 4b.  GUI / 日志辅助
    # =========================================================================

    def _set_state(self, state_name, msg=""):
        """更新 GUI 状态显示；无 GUI 时仅打印日志"""
        if self.gui:
            self.gui.set_state(state_name, msg)
        print(f"[FLOW] [{self.state.value}] {msg}", flush=True)

    # =========================================================================
    # 4c.  放松音频播放
    # =========================================================================

    def _play_relax_audio(self, audio_type, timeout):
        """
        播放放松音频（白噪音/轻音乐/冥想），持续 timeout 秒或音频放完。

        优先从 ``sounds/relax/<type>.wav`` 循环播放；
        文件不存在则生成白噪音。
        """
        audio_path = os.path.join("sounds", "relax", f"{audio_type}.wav")
        start = time.time()

        if os.path.exists(audio_path) and self.gui:
            print(f"[AUDIO] Playing file: {audio_path}", flush=True)
            while time.time() - start < timeout and not self._exit_flag:
                self.gui.play_sound(audio_path)
            return

        # 无文件 → 生成白噪音
        if not _HAS_SOUNDDEVICE:
            print(f"[AUDIO] sounddevice 不可用，无法播放音频，静等超时", flush=True)
            time.sleep(min(timeout, 10))
            return
        print(f"[AUDIO] No file at {audio_path}, generating white noise", flush=True)
        try:
            SAMPLE_RATE = 44100
            # 每次生成最长 5 分钟，循环播放直到超时
            chunk_sec = min(timeout, 300)
            noise, rate = _generate_white_noise(chunk_sec, SAMPLE_RATE)

            while time.time() - start < timeout and not self._exit_flag:
                remaining = timeout - (time.time() - start)
                play_sec = min(remaining, chunk_sec)
                end = int(rate * play_sec)
                sd.play(noise[:end], rate)
                sd.wait()
        except Exception as e:
            print(f"[AUDIO] Playback error: {e}", flush=True)
            # 音频播放失败 → 静等剩余时间
            elapsed = time.time() - start
            remaining = timeout - elapsed
            if remaining > 0:
                time.sleep(min(remaining, 10))

    # =========================================================================
    # 5.  PUBLIC API
    # =========================================================================

    def start(self):
        """
        启动状态机主循环，阻塞直到 SHUTDOWN。

        状态转移: BOOT → CHAT → TRANSITION → AUDIO → SLEEP → SHUTDOWN
        """
        print("=" * 50, flush=True)
        print("  睡前情绪梳理机器人  Sleep Flow", flush=True)
        print("=" * 50, flush=True)

        while self.state != SleepState.SHUTDOWN and not self._exit_flag:
            try:
                if self.state == SleepState.BOOT:
                    self._run_boot()
                elif self.state == SleepState.CHAT:
                    self._run_chat()
                elif self.state == SleepState.TRANSITION:
                    self._run_transition()
                elif self.state == SleepState.AUDIO:
                    self._run_audio()
                elif self.state == SleepState.SLEEP:
                    self._run_sleep()
                else:
                    break
            except Exception as e:
                print(f"[FLOW CRITICAL] 状态 {self.state.value} 异常: {e}", flush=True)
                traceback.print_exc()
                # 出错后强制推进到下一状态，防止卡死
                self._force_next()

        self._run_shutdown()

    def stop(self):
        """安全停止状态机（可在另一线程调用）"""
        self._exit_flag = True
        # 同时通知 BotGUI 退出，让正在跑的 run_chat_phase / record_voice_vad 及时返回
        if self.gui is not None:
            try:
                self.gui.exiting = True
            except Exception:
                pass

    # =========================================================================
    # 6.  各状态执行方法
    # =========================================================================

    # ------------------------------------------------------------------
    # BOOT
    # ------------------------------------------------------------------

    def _run_boot(self):
        """预热模型/TTS 流水线，播放开机问候，自动转入 CHAT"""
        self._set_state("warmup", "预热中…")
        # LLM 前缀预热 + 启动 TTS worker（CHAT 委派 run_chat_phase 前必须先就绪）
        if self.gui and hasattr(self.gui, "warm_up"):
            self.gui.warm_up()
        self._set_state("greeting", "晚上好")
        self._play_cached("greeting")
        self._transition_to(SleepState.CHAT)

    # ------------------------------------------------------------------
    # CHAT
    # ------------------------------------------------------------------

    def _run_chat(self):
        """
        多轮对话状态：把控制权交给 BotGUI 现成的聊天内核。

        录音/STT/LLM/流式 TTS 全由 ``gui.run_chat_phase`` 负责，flow 只传入
        轮次上限与静音超时，并按它返回的原因播放对应收尾话术后转入 TRANSITION：

            - "silence"    静音超时 → soft_close
            - "max_rounds" 聊满轮次 → round5_close
        """
        max_rounds = self.sleep_cfg.get("max_chat_rounds", 5)
        silence_timeout = self.sleep_cfg.get("silence_timeout_chat", 75)

        if self.gui is None:
            # 纯日志模式：无聊天内核可委派，直接过渡，便于验证状态流转
            print("[CHAT] gui=None，跳过聊天阶段（纯日志模式）", flush=True)
            self._transition_to(SleepState.TRANSITION)
            return

        self._set_state("listening", "我在听…")
        reason = self.gui.run_chat_phase(
            max_rounds=max_rounds,
            silence_timeout=silence_timeout,
            get_system_prompt=get_chat_prompt,
        )
        print(f"[CHAT] 聊天阶段结束，原因: {reason}", flush=True)

        if self._exit_flag or reason == "exit":
            self._transition_to(SleepState.TRANSITION)
            return

        # 按结束原因播放收尾话术
        self._play_cached("soft_close" if reason == "silence" else "round5_close")
        self._transition_to(SleepState.TRANSITION)

    # ------------------------------------------------------------------
    # TRANSITION
    # ------------------------------------------------------------------

    def _run_transition(self):
        """播放过渡脚本，转入 AUDIO"""
        self._set_state("greeting", "准备放松")
        self._play_cached("transition")
        self._transition_to(SleepState.AUDIO)

    # ------------------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------------------

    def _run_audio(self):
        """
        播放放松音频，用户在此阶段入眠。

        - 音频类型由 ``sleep_flow.audio_type`` 指定
        - 持续 ``sleep_flow.shutdown_timeout`` 秒后自动结束
        - 期间不检测用户说话，不打扰
        """
        self._set_state("sleep", "放松中…")
        audio_type = self.sleep_cfg.get("audio_type", "white_noise")
        timeout = self.sleep_cfg.get("shutdown_timeout", 2700)  # 45 分钟

        print(f"[AUDIO] 开始播放: {audio_type}  持续时间: {timeout}s", flush=True)
        self._play_relax_audio(audio_type, timeout)

        self._transition_to(SleepState.SLEEP)

    # ------------------------------------------------------------------
    # SLEEP
    # ------------------------------------------------------------------

    def _run_sleep(self):
        """短暂停留后进入 SHUTDOWN"""
        self._set_state("sleep", "晚安")
        print("[SLEEP] 用户已在放松音频中入眠，5 秒后关机", flush=True)
        time.sleep(5)
        self._transition_to(SleepState.SHUTDOWN)

    # =========================================================================
    # 7.  状态转移 / 关机
    # =========================================================================

    def _transition_to(self, new_state):
        """单向状态转移（打印日志 + 记入内存）"""
        prev = self.state.value
        self.state = new_state
        print(f"[FLOW] {prev} -> {new_state.value}", flush=True)

    def _force_next(self):
        """出错时的紧急推进：无论如何跳到下一个状态"""
        order = list(SleepState)
        try:
            idx = order.index(self.state)
            if idx < len(order) - 1:
                self.state = order[idx + 1]
                print(f"[FLOW] ⚠ 强制前进到 {self.state.value}", flush=True)
            else:
                self.state = SleepState.SHUTDOWN
        except ValueError:
            self.state = SleepState.SHUTDOWN

    def _run_shutdown(self):
        """关机 / 退出"""
        shutdown_enabled = self.sleep_cfg.get("shutdown_enabled", False)

        if shutdown_enabled:
            print("[SHUTDOWN] 执行系统关机...", flush=True)
            self._set_state("sleep", "关机中")
            try:
                import subprocess
                subprocess.run(["sudo", "halt"], check=True, timeout=10)
            except Exception as e:
                print(f"[SHUTDOWN] 关机命令失败: {e}", flush=True)
                print("[SHUTDOWN] 回退: 退出进程", flush=True)
        else:
            print("[SHUTDOWN] 开发模式：此处会执行关机", flush=True)
            print("[SHUTDOWN] 生产环境请设置 shutdown_enabled=true", flush=True)

        # 安全退出 GUI
        if self.gui:
            try:
                self.gui.safe_exit()
            except Exception as e:
                print(f"[SHUTDOWN] safe_exit: {e}", flush=True)

        print("[FLOW] 晚安 🌙", flush=True)


# =========================================================================
# 5. 工具函数
# =========================================================================

def _save_float32_wav(samples, rate, path):
    """将 float32 [-1,1] 音频保存为 16-bit wav 文件"""
    samples_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples_int16.tobytes())


# =========================================================================
# 6. 独立测试入口
# =========================================================================

if __name__ == "__main__":
    # 纯日志模式（无 GUI、无音频硬件），验证初始化与状态流转
    print(">>> 测试 SleepFlow 纯日志模式 <<<")
    test_cfg = {
        "text_model": "gemma3:1b",
        "whisper_lang": "zh",
        "sleep_flow": {
            "max_chat_rounds": 2,
            "silence_timeout_chat": 10,
            "audio_type": "white_noise",
            "shutdown_timeout": 30,
            "shutdown_enabled": False,
            "cache_dir": "cache",
            "pre_synthesize_texts": {
                "greeting":    "晚上好，今天过得怎么样？有什么想说的吗？",
                "transition":  "好的，已经记下了。现在让我们慢慢放松，准备休息吧。",
                "soft_close":  "如果你没什么想说的了，我们就开始放松吧。",
                "round5_close":"我们先到这里，准备休息吧。",
            },
        },
    }
    flow = SleepFlow(config=test_cfg, gui=None)
    print(">>> 初始化完成（无异常）<<<")
    print(">>> cache/ 目录已创建，预合成结果如下：")
    for f in sorted(os.listdir(flow.cache_dir)):
        print(f"    {f}")
    print(">>> 测试结束 <<<")
