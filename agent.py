# =========================================================================
#  Be More Agent 🤖
#  A Local, Offline-First AI Agent for Raspberry Pi
#
#  Copyright (c) 2026 brenpoly
#  Licensed under the MIT License
#  Source: https://github.com/brenpoly/be-more-agent
#
#  DISCLAIMER:
#  This software is provided "as is", without warranty of any kind.
#  This project is a generic framework and includes no copyrighted assets.
# =========================================================================

"""本地离线语音助手主程序。

一轮对话的数据流水线（全部在本机运行，不联网）：
    麦克风 → VAD 断句录音 → whisper.cpp 语音转文字（STT）
          → Ollama 大模型流式生成回复（LLM）
          → sherpa/piper 文字转语音（TTS）→ 扬声器播放

线程模型：
    - 主线程：tkinter GUI（动画、状态文字、按键事件）。
    - 后台主循环线程 safe_main_execution：串起「录音→识别→对话」一整轮。
    - TTS 两级流水线线程 _synth_worker（合成）/ _play_worker（播放），
      靠 tts_queue / audio_queue 解耦，使下一句在当前句播放时就提前合成。

交互：Space 打断当前发言，Esc 切换全屏，Ctrl+Q / Exit 按钮退出并存档。
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import json
import os
import subprocess
import re
import traceback
import atexit
import wave
import collections

# Core dependencies
import sounddevice as sd
import numpy as np
import scipy.signal
import webrtcvad

# --- AI ENGINES ---
import ollama

# =========================================================================
# 1. 配置 / 提示词从内部模块导入
# =========================================================================
from config import (
    MEMORY_FILE,
    INPUT_DEVICE_NAME, OLLAMA_OPTIONS, CURRENT_CONFIG, TEXT_MODEL,
    BotStates, timed_block, choose_input_samplerate,
)
from prompts import SYSTEM_PROMPT

# =========================================================================
# 2. GUI CLASS
# =========================================================================

class BotGUI:
    """助手的全部状态与行为：tkinter 界面 + 录音/识别/对话/合成播放流水线。

    一个进程只创建一个实例，由 __main__ 启动。GUI 在主线程，重活
    （录音、LLM、TTS）跑在后台线程，通过线程安全队列与 Event 协调。
    """

    def __init__(self, master):
        """搭好界面、加载历史与模型、起动后台主循环线程。"""
        self.master = master
        master.title("Pi Assistant")
        master.geometry("800x480")
        master.minsize(480, 300)
        self.is_fullscreen = False
        master.bind('<Escape>', self.toggle_fullscreen)        # 全屏/窗口切换
        master.bind('<Control-q>', lambda e: self.safe_exit())  # 退出程序
        master.bind('<Configure>', self.on_window_resize)      # 窗口尺寸变化时重新居中

        # Inputs
        master.bind('<space>', self.handle_speaking_interrupt)
        atexit.register(self.safe_exit)
        master.focus_force()   # 抢焦点，确保 Escape 等按键能被窗口收到
        master.configure(bg='#1a1a2e')   # 动画区外背景色

        # 动画区固定 800×450（5 倍像素缩放），在窗口内居中
        self.anim_w, self.anim_h = 800, 450
        self.anim_x = (800 - self.anim_w) // 2   # = 0
        self.anim_y = (480 - self.anim_h) // 2   # = 15

        # State
        self.current_state = BotStates.WARMUP
        self.current_volume = 0
        self.animations = {}
        self.current_frame_index = 0

        self.permanent_memory = self.load_chat_history()
        self.session_memory = []

        self.interrupted = threading.Event()

        self.tts_queue = []
        self.tts_queue_lock = threading.Lock()
        self.tts_active = threading.Event()
        self.current_audio_process = None

        # --- TTS 两级流水线：合成线程提前渲染，播放线程只管播，消除句间空挡 ---
        self.audio_queue = []                    # 已渲染音频：(samples float32, rate)
        self.audio_queue_lock = threading.Lock()
        self.audio_queue_max = 2                 # 提前渲染深度（背压上限）
        self.synth_active = threading.Event()    # 合成线程正在合成某句
        self.play_active = threading.Event()     # 播放线程正在播某句
        self.synth_thread = None
        self.play_thread = None
        self.exiting = False

        # --- SHERPA TTS INITIALIZATION ---
        self.sherpa_tts = None
        if CURRENT_CONFIG.get("tts_engine") == "sherpa":
            self._init_sherpa_tts()

        # GUI Setup
        self.background_label = tk.Label(master)
        self.background_label.place(x=self.anim_x, y=self.anim_y, width=self.anim_w, height=self.anim_h)
        self.background_label.bind('<Button-1>', self.toggle_hud_visibility)

        # --- 聊天气泡系统（Canvas + 自动滚动，初始隐藏）---
        self.chat_canvas = tk.Canvas(master, bg='#ffffff', highlightthickness=0)
        self.chat_scrollbar = ttk.Scrollbar(master, orient='vertical', command=self.chat_canvas.yview)
        self.chat_inner = tk.Frame(self.chat_canvas, bg='#ffffff')
        self.chat_inner.bind('<Configure>',
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox('all')))
        self.chat_canvas.create_window((0, 0), window=self.chat_inner, anchor='nw')
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        self._last_bot_var = None   # 流式追加时跟踪最后一条 bot 气泡
        self._chat_height = 130     # 聊天气泡面板默认高度

        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(master, textvariable=self.status_var, background="#2e2e2e", foreground="white")

        self.exit_button = ttk.Button(master, text="Exit & Save", command=self.safe_exit)
        self.fullscreen_button = ttk.Button(master, text="全屏", command=self.toggle_fullscreen_btn)

        self.load_animations()
        self.update_animation()

        threading.Thread(target=self.safe_main_execution, daemon=True).start()

    # --- HELPERS ---

    def safe_exit(self):
        """统一退出口：停音频、存对话历史、卸载模型、关窗口。可重入只执行一次。"""
        if self.exiting:
            return
        self.exiting = True
        print("\n--- SHUTDOWN SEQUENCE ---", flush=True)
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=1)
            except: pass

        self.tts_active.clear()

        self.save_chat_history()

        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=0)
        except: pass
        try:
            sd.stop()
        except: pass

        try:
            self.master.quit()
        except Exception:
            pass

    def toggle_fullscreen(self, event=None):
        # 全屏 / 窗口化切换（程序继续运行；退出请用 Ctrl+Q 或 Exit 按钮）。
        # 退出全屏时恢复 800×480 窗口，并同步按钮文字。
        self.is_fullscreen = not self.is_fullscreen
        self.master.attributes('-fullscreen', self.is_fullscreen)
        if not self.is_fullscreen:
            self.master.geometry("800x480")
        if hasattr(self, 'fullscreen_button'):
            self.fullscreen_button.config(text="窗口" if self.is_fullscreen else "全屏")

    def toggle_fullscreen_btn(self):
        """「全屏/窗口」按钮回调，与按 Esc 等效。"""
        self.toggle_fullscreen()

    def toggle_hud_visibility(self, event=None):
        """点击画面：在「只显示动画」和「显示聊天气泡/状态/按钮」之间切换。"""
        try:
            if self.chat_canvas.winfo_ismapped():
                self.chat_canvas.place_forget()
                self.chat_scrollbar.place_forget()
                self.status_label.place_forget()
                self.exit_button.place_forget()
                self.fullscreen_button.place_forget()
            else:
                ch = self.master.winfo_height()
                cw = self.master.winfo_width()
                chat_h = min(self._chat_height, ch // 3)
                chat_w = cw - 40
                chat_y = ch - chat_h - 22
                self.chat_canvas.place(x=20, y=chat_y, width=chat_w, height=chat_h)
                self.chat_scrollbar.place(x=cw-20, y=chat_y, height=chat_h)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
                self.fullscreen_button.place(x=105, y=10)
        except tk.TclError: pass

    def handle_speaking_interrupt(self, event=None):
        """Space 打断：思考/发言时立即清空两条队列、停掉播放，回到 IDLE。"""
        if self.current_state == BotStates.SPEAKING or self.current_state == BotStates.THINKING:
            self.interrupted.set()
            with self.tts_queue_lock:
                self.tts_queue.clear()
            with self.audio_queue_lock:
                self.audio_queue.clear()
            if self.current_audio_process:
                try: self.current_audio_process.terminate()
                except: pass
            try: sd.stop()
            except: pass
            self.set_state(BotStates.IDLE, "Interrupted.")

    def load_animations(self):
        """从 faces/<状态>/ 预加载各状态的 PNG 帧序列；缺帧时回退到 idle 或纯色占位。"""
        base_path = "faces"
        states = ["idle", "listening", "thinking", "speaking", "greeting", "sleep", "warmup"]
        for state in states:
            folder = os.path.join(base_path, state)
            self.animations[state] = []
            if os.path.exists(folder):
                files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
                for f in files:
                    img = Image.open(os.path.join(folder, f)).resize((self.anim_w, self.anim_h), Image.NEAREST)
                    self.animations[state].append(ImageTk.PhotoImage(img))
            if not self.animations[state]:
                if "idle" in self.animations and self.animations["idle"]:
                     self.animations[state] = self.animations["idle"]
                else:
                    blank = Image.new('RGB', (800, 450), color='#1a1a2e')
                    self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        """每 250ms 切到当前状态的下一帧，用 after 自我调度形成循环播放。"""
        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if not frames:
            self.master.after(250, self.update_animation)
            return

        self.current_frame_index = (self.current_frame_index + 1) % len(frames)
        self.background_label.config(image=frames[self.current_frame_index])

        self.master.after(250, self.update_animation)

    def set_state(self, state, msg=""):
        """切换动画状态并更新底部状态栏文字；经 after 投递到 GUI 线程，可跨线程调用。"""
        def _update():
            if msg: print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg: self.status_var.set(msg)
        self.master.after(0, _update)

    def add_message(self, role, text):
        """新增一条聊天气泡。role='user'（右对齐黄底）或 'bot'（左对齐灰底）。
        bot 气泡会记下其 StringVar，供 stream_to_bubble 逐字追加。跨线程安全。"""
        def _update():
            is_user = role == 'user'
            bg = '#fff3cd' if is_user else '#f0f0f0'
            side = 'right' if is_user else 'left'

            frame = tk.Frame(self.chat_inner, bg='#ffffff')
            frame.pack(fill='x', padx=8, pady=2, expand=True)

            cw = self.chat_canvas.winfo_width()
            max_w = int(cw * 0.7) if cw > 100 else 350

            var = tk.StringVar(value=text)
            lbl = tk.Label(frame, textvariable=var, wraplength=max_w,
                           bg=bg, font=('Arial', 11), padx=10, pady=5,
                           justify='left', anchor='w')
            lbl.pack(side=side)

            # 记录 bot 气泡的 StringVar，供 stream_to_bubble 流式追加
            self._last_bot_var = None if is_user else var

            self.chat_canvas.yview_moveto(1.0)
        self.master.after(0, _update)

    def stream_to_bubble(self, chunk):
        """把 LLM 流式吐出的片段实时追加到最后一条 bot 气泡，实现逐字显示。"""
        def _update():
            if self._last_bot_var is None:
                return
            self._last_bot_var.set(self._last_bot_var.get() + chunk)
            self.chat_canvas.yview_moveto(1.0)
        self.master.after(0, _update)

    def on_window_resize(self, event):
        """窗口尺寸变化时，重新把动画区和（可见的）聊天面板居中/定位。"""
        if event.widget != self.master:
            return
        w = event.width
        h = event.height
        self.anim_x = (w - self.anim_w) // 2
        self.anim_y = (h - self.anim_h) // 2
        self.background_label.place(x=self.anim_x, y=self.anim_y)
        # 聊天面板当前可见时一并重新定位
        try:
            if self.chat_canvas.winfo_ismapped():
                chat_h = min(self._chat_height, h // 3)
                chat_w = w - 40
                chat_y = h - chat_h - 22
                self.chat_canvas.place(x=20, y=chat_y, width=chat_w, height=chat_h)
                self.chat_scrollbar.place(x=w-20, y=chat_y, height=chat_h)
        except tk.TclError:
            pass

    # =========================================================================
    # 4. CORE LOGIC
    # =========================================================================

    def safe_main_execution(self):
        """后台主循环：预热 → 起动 TTS 流水线 → 反复「录音→识别→对话」直到退出。
        整个循环包在 try 里，任何未预料异常都转成 ERROR 状态而非让线程静默崩掉。"""
        try:
            self.warm_up_logic()
            self.synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
            self.synth_thread.start()
            self.play_thread = threading.Thread(target=self._play_worker, daemon=True)
            self.play_thread.start()

            while True:
                if self.exiting:
                    break
                self.set_state(BotStates.LISTENING, "我在听…")
                audio_file = self.record_voice_vad()

                if self.interrupted.is_set():
                    self.interrupted.clear()
                    self.set_state(BotStates.IDLE, "Resetting...")
                    continue

                if not audio_file:
                    # 没听到，安静地继续监听（不报错停顿，符合助眠场景）
                    continue

                user_text = self.transcribe_audio(audio_file)
                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue

                self.add_message("user", user_text)
                self.interrupted.clear()
                with timed_block("完整一轮对话"):
                    self.chat_and_respond(user_text)

        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.IDLE, f"Fatal Error: {str(e)[:40]}")

    def warm_up_logic(self):
        """开机预热：先空跑一次 LLM 摊销首轮延迟，再播放开场问候（顺带预热 TTS）。"""
        self.set_state(BotStates.WARMUP, "Warming up brains...")
        # 不只是载入权重，还要把第1轮真实会话要用的 KV 前缀（system prompt + 历史）
        # 提前评估一遍，否则首轮 prompt-eval 会拖慢 LLM 首 Token（实测 ~16s）。
        # 跑一次真实 ollama.chat，丢弃输出、不写入 memory，让真实第1轮退化成"第2轮"速度。
        try:
            with timed_block("LLM warmup (prefix)"):
                warmup_messages = self.permanent_memory + [
                    {"role": "user", "content": "你好"}
                ]
                ollama.chat(
                    model=TEXT_MODEL,
                    messages=warmup_messages,
                    stream=False,
                    options=OLLAMA_OPTIONS,
                    keep_alive=-1,
                )
        except Exception as e:
            print(f"Failed to load {TEXT_MODEL}: {e}", flush=True)
        self.set_state(BotStates.GREETING, "晚上好")
        self.speak("你好，我在。今天过得怎么样？")
        print("Models loaded.", flush=True)

    def record_voice_vad(self, filename="input.wav"):
        """webrtcvad 持续监听，检测到人声起始自动开始录音，尾部静音自动停止。
        阻塞直到捕获完整一句话，返回 wav 路径；没听到则返回 None。"""
        VAD_RATE = 16000
        FRAME_MS = 30
        frame_samples = int(VAD_RATE * FRAME_MS / 1000)  # 16000Hz×30ms = 480

        aggressiveness = int(CURRENT_CONFIG.get("vad_aggressiveness", 2))
        start_frames   = max(1, int(CURRENT_CONFIG.get("vad_start_ms", 150)   / FRAME_MS))
        silence_frames = max(1, int(CURRENT_CONFIG.get("vad_silence_ms", 900) / FRAME_MS))
        max_frames     = max(1, int(CURRENT_CONFIG.get("vad_max_record_ms", 30000) / FRAME_MS))
        preroll_frames = max(0, int(CURRENT_CONFIG.get("vad_preroll_ms", 300) / FRAME_MS))
        skip_frames    = int(200 / FRAME_MS)  # 丢弃头部 ~200ms，避开上一句 TTS 的房间回声尾巴

        vad = webrtcvad.Vad(aggressiveness)

        # webrtcvad 只吃 8/16/32/48kHz。优先 16000Hz 直采；设备只能跑 44100/48000 时
        # 按原生率采集，再用最近邻重采样把每帧降到 480 个样本。
        input_rate = choose_input_samplerate(INPUT_DEVICE_NAME, VAD_RATE)
        use_resampling = (input_rate != VAD_RATE)
        read_size = int(input_rate * FRAME_MS / 1000) if use_resampling else frame_samples

        buffer = []                                          # 已确认录音的帧（int16, 16000Hz）
        preroll = collections.deque(maxlen=preroll_frames)   # 起始前回看缓冲
        recording = False
        voiced_run = 0
        silence_run = 0
        total_frames = 0

        try:
            # 释放硬件，避免 Pi 上音频争用死锁
            sd.stop()
            time.sleep(0.2)
            with sd.InputStream(samplerate=input_rate, channels=1, dtype='int16',
                                blocksize=read_size, device=INPUT_DEVICE_NAME) as stream:
                print("[VAD] Listening...", flush=True)
                while True:
                    if self.exiting:
                        return None

                    data, _overflow = stream.read(read_size)
                    frame = np.frombuffer(data, dtype=np.int16)
                    if frame.ndim > 1:
                        frame = frame.flatten()

                    if use_resampling:
                        step = len(frame) / frame_samples
                        idx = np.arange(0, len(frame), step)[:frame_samples].astype(int)
                        frame = frame[idx]
                    if len(frame) != frame_samples:   # webrtcvad 要求帧长精确，长度不对就跳过
                        continue

                    if skip_frames > 0:
                        skip_frames -= 1
                        continue

                    is_speech = vad.is_speech(frame.tobytes(), VAD_RATE)

                    if not recording:
                        preroll.append(frame.copy())
                        if is_speech:
                            voiced_run += 1
                            if voiced_run >= start_frames:
                                recording = True
                                buffer.extend(preroll)   # 预缓冲并入开头，避免吞掉第一个字
                                preroll.clear()
                                total_frames = len(buffer)
                                silence_run = 0
                                print("[VAD] Speech detected, recording...", flush=True)
                        else:
                            voiced_run = 0
                    else:
                        buffer.append(frame.copy())
                        total_frames += 1
                        if is_speech:
                            silence_run = 0
                        else:
                            silence_run += 1
                            if silence_run >= silence_frames:
                                print("[VAD] Trailing silence, stop.", flush=True)
                                break
                        if total_frames >= max_frames:
                            print("[VAD] Max record time reached, stop.", flush=True)
                            break
        except Exception as e:
            print(f"[AUDIO ERROR] VAD Recording Failed: {e}", flush=True)
            return None

        if not buffer:
            return None
        return self.save_audio_buffer(buffer, filename, samplerate=VAD_RATE, already_int16=True)

    def save_audio_buffer(self, buffer, filename, samplerate=16000, already_int16=False):
        if not buffer: return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        if already_int16:
            # VAD 路径的 buffer 已是 int16 PCM，直接落盘，跳过 float×32767 换算。
            audio_data = audio_data.astype(np.int16)
        else:
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        return filename

    def transcribe_audio(self, filename):
        print("Transcribing...", flush=True)
        whisper_model = CURRENT_CONFIG.get("whisper_model", "ggml-base.en.bin")
        whisper_lang  = CURRENT_CONFIG.get("whisper_lang", "en")
        try:
            with timed_block("STT whisper-cli"):
                result = subprocess.run(
                    ["./whisper.cpp/build/bin/whisper-cli",
                     "-m", f"./whisper.cpp/models/{whisper_model}",
                     "-l", whisper_lang, "-t", "4", "-f", filename],
                    capture_output=True, text=True
                )
            transcription_lines = result.stdout.strip().split('\n')
            if transcription_lines and transcription_lines[-1].strip():
                last_line = transcription_lines[-1].strip()
                if ']' in last_line: transcription = last_line.split("]")[1].strip()
                else: transcription = last_line
            else: transcription = ""
            print(f"Heard: '{transcription}'", flush=True)
            return transcription.strip()
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""

    # =========================================================================
    # 5. CHAT & RESPOND
    # =========================================================================

    def chat_and_respond(self, text):
        if "forget everything" in text.lower() or "reset memory" in text.lower() \
                or "清空记忆" in text or "忘记一切" in text:
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.save_chat_history()
            with self.tts_queue_lock:
                self.tts_queue.append("好的，我把记忆清空了。")
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return

        self.set_state(BotStates.THINKING, "Thinking...")

        lang = CURRENT_CONFIG.get("whisper_lang", "en")
        lang_hint = "请用中文回答。" if lang == "zh" else ""
        user_msg = {"role": "user", "content": text + ("\n" + lang_hint if lang_hint else "")}
        messages = self.permanent_memory + self.session_memory + [user_msg]

        full_response_buffer = ""
        sentence_buffer = ""

        try:
            stream = ollama.chat(model=TEXT_MODEL, messages=messages, stream=True, options=OLLAMA_OPTIONS)

            _t_llm = time.perf_counter()
            _ttft_logged = False

            for chunk in stream:
                if self.interrupted.is_set(): break
                content = chunk['message']['content']
                if not _ttft_logged:
                    print(f"[TIMER] LLM 首Token延迟 {time.perf_counter()-_t_llm:.2f}s", flush=True)
                    _ttft_logged = True
                full_response_buffer += content

                if self.current_state != BotStates.SPEAKING:
                    self.set_state(BotStates.SPEAKING, "Speaking...")
                    self.add_message("bot", "")

                self.stream_to_bubble(content)

                sentence_buffer += content
                if any(punct in content for punct in ".!?\n。！？"):
                    clean_sentence = sentence_buffer.strip()
                    if clean_sentence and re.search(r'[\w一-鿿]', clean_sentence):
                        with self.tts_queue_lock: self.tts_queue.append(clean_sentence)
                    sentence_buffer = ""

            if sentence_buffer.strip() and re.search(r'[\w一-鿿]', sentence_buffer):
                with self.tts_queue_lock: self.tts_queue.append(sentence_buffer.strip())
            self.session_memory.append({"role": "assistant", "content": full_response_buffer})

            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")

        except Exception as e:
            print(f"LLM Error: {e}")
            self.set_state(BotStates.IDLE, "Brain Freeze!")

    def wait_for_tts(self):
        # 两级都空闲才算"说完"：两个队列空，且合成/播放线程都不忙。
        while (self.tts_queue or self.audio_queue
               or self.synth_active.is_set() or self.play_active.is_set()):
            if self.interrupted.is_set(): break
            time.sleep(0.1)

    def _synth_worker(self):
        # 阶段一：从 tts_queue 取文本，提前合成成音频缓冲，推入 audio_queue。
        # 这样第 N 句播放期间第 N+1 句已在合成，句间空挡被消除。
        while True:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue:
                    self.synth_active.set()   # 先置忙再出队，避免 wait_for_tts 抢到"空队列+未置忙"
                    text = self.tts_queue.pop(0)
            if text is None:
                time.sleep(0.05)
                continue
            try:
                if self.interrupted.is_set():
                    continue
                rendered = self._render(text)        # (samples, rate) 或 None
                if rendered is None or self.interrupted.is_set():
                    continue
                # 背压：audio_queue 满则等播放线程消化，避免提前渲染堆积过多。
                while not self.interrupted.is_set():
                    with self.audio_queue_lock:
                        if len(self.audio_queue) < self.audio_queue_max:
                            self.audio_queue.append(rendered)
                            break
                    time.sleep(0.02)
            finally:
                self.synth_active.clear()

    def _play_worker(self):
        # 阶段二：从 audio_queue 取已渲染音频并播放。
        while True:
            item = None
            with self.audio_queue_lock:
                if self.audio_queue:
                    self.play_active.set()    # 先置忙再出队，理由同上
                    item = self.audio_queue.pop(0)
            if item is None:
                time.sleep(0.05)
                continue
            try:
                if not self.interrupted.is_set():
                    self._play_samples(*item)
            finally:
                self.play_active.clear()

    def _init_sherpa_tts(self):
        try:
            import sherpa_onnx
            model_dir = CURRENT_CONFIG.get("sherpa_model_dir", "sherpa-models/vits-zh-aishell3")
            num_threads = CURRENT_CONFIG.get("sherpa_num_threads", 4)
            print(f"[INIT] Sherpa num_threads (from config) = {num_threads}", flush=True)
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=f"{model_dir}/vits-aishell3.onnx",
                        lexicon=f"{model_dir}/lexicon.txt",
                        tokens=f"{model_dir}/tokens.txt",
                    ),
                    # 默认单线程合成在 Pi 上慢到 ~0.5s/字；吃满多核可砍掉一半以上耗时。
                    num_threads=num_threads,
                    provider="cpu",
                ),
                rule_fsts=(
                    f"{model_dir}/date.fst,"
                    f"{model_dir}/number.fst,"
                    f"{model_dir}/phone.fst,"
                    f"{model_dir}/new_heteronym.fst"
                ),
                rule_fars=f"{model_dir}/rule.far",
                max_num_sentences=1,
            )
            self.sherpa_tts = sherpa_onnx.OfflineTts(cfg)
            print("[INIT] Sherpa TTS loaded.", flush=True)
        except Exception as e:
            print(f"[INIT] Sherpa TTS load failed: {e}. Falling back to piper.", flush=True)
            self.sherpa_tts = None

    def speak(self, text):
        # 同步合成并播放一句（阻塞）。用于开场问候等流水线 worker 启动前的场景。
        rendered = self._render(text)
        if rendered is not None:
            self._play_samples(*rendered)

    # --- 合成阶段：文本 → (samples float32 [-1,1], rate)，不播放 ---

    def _render(self, text):
        clean = re.sub(r"[^\w\s,.!?:-，。！？、；：]", "", text)
        if not clean.strip(): return None
        if self.sherpa_tts is not None:
            return self._render_sherpa(clean)
        return self._render_piper(clean)

    def _fit_samplerate(self, samples, rate):
        # 设备支持模型原生采样率就直接用；否则用多相重采样（比 FFT 法 resample 快很多）。
        try:
            sd.check_output_settings(samplerate=rate)
            return samples, rate
        except Exception:
            try:
                native_rate = int(sd.query_devices(kind='output')['default_samplerate'])
            except Exception:
                native_rate = 48000
            resampled = scipy.signal.resample_poly(samples, native_rate, rate).astype(np.float32)
            return resampled, native_rate

    def _render_sherpa(self, text):
        with timed_block(f"TTS sherpa synth [{text[:15]}...]"):
            print(f"[SHERPA TTS] '{text}'", flush=True)
            try:
                audio = self.sherpa_tts.generate(
                    text,
                    sid=CURRENT_CONFIG.get("sherpa_speaker_id", 0),
                    speed=CURRENT_CONFIG.get("sherpa_speed", 1.0),
                )
                samples = np.array(audio.samples, dtype=np.float32)
                # 归一到 [-1,1]，让偏小的模型输出以满音量播放。
                max_val = np.max(np.abs(samples))
                if max_val > 0:
                    samples /= max_val
                return self._fit_samplerate(samples, audio.sample_rate)
            except Exception as e:
                print(f"[SHERPA TTS ERROR] {e}, falling back to piper")
                return self._render_piper(text)

    def _render_piper(self, text):
        with timed_block(f"TTS piper synth [{text[:15]}...]"):
            print(f"[PIPER SPEAKING] '{text}'", flush=True)
            voice_model = CURRENT_CONFIG.get("voice_model", "piper/en_GB-semaine-medium.onnx")
            try:
                proc = subprocess.Popen(
                    ["./piper/piper", "--model", voice_model, "--output-raw"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                raw, _ = proc.communicate(text.encode() + b'\n')
                if not raw: return None
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                return self._fit_samplerate(samples, 22050)
            except Exception as e:
                print(f"Audio Error: {e}")
                return None

    # --- 播放阶段：消费已渲染的音频缓冲 ---

    def _play_samples(self, samples, rate):
        with timed_block(f"TTS play [{rate}Hz {len(samples)}smp]"):
            try:
                sd.play(samples, rate)
                while True:
                    if self.interrupted.is_set():
                        sd.stop()
                        break
                    try:
                        if not sd.get_stream().active:
                            sd.stop()
                            break
                    except Exception:
                        break
                    time.sleep(0.05)
                time.sleep(0.1)
            except Exception as e:
                print(f"Audio playback error: {e}")
            finally:
                self.current_volume = 0

    def load_chat_history(self):
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    turns = json.load(f)
                # memory.json 只存对话轮次，不存 system message
                turns = [t for t in turns if t.get("role") != "system"]
                return [system_msg] + turns
            except: pass
        return [system_msg]

    def save_chat_history(self):
        full = self.permanent_memory + self.session_memory
        # 只保存 user/assistant 轮次，system prompt 是配置不是历史
        turns = [t for t in full if t.get("role") != "system"]
        if len(turns) > 10: turns = turns[-10:]
        with open(MEMORY_FILE, "w") as f:
            json.dump(turns, f, indent=4)

if __name__ == "__main__":
    print("--- SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()
