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
from functools import partial

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
    INPUT_DEVICE_NAME, OLLAMA_OPTIONS, CURRENT_CONFIG, TEXT_MODEL,
    BotStates, timed_block, choose_input_samplerate,
)
from chat_engine import ChatEngine

# =========================================================================
# 2. GUI CLASS
# =========================================================================

class GuiIO:
    """把 ChatEngine 需要的 8 个交互钩子映射到 BotGUI 的现有方法。
    引擎只认这套接口，因此换成控制台实现（debug_chat.py）即可脱离硬件调 prompt。"""

    def __init__(self, gui):
        self.gui = gui

    def set_state(self, state, msg=""):
        self.gui.set_state(state, msg)

    def bot_start(self):
        self.gui.add_message("bot", "")

    def token(self, piece):
        self.gui.stream_to_bubble(piece)

    def speak(self, sentence):
        self.gui._queue_sentence(sentence)

    def wait_speech(self):
        self.gui.wait_for_tts()

    def is_interrupted(self):
        return self.gui.interrupted.is_set()

    def stage(self, msg):
        self.gui._stage(msg)

    def play_relaxation_audio(self, audio_type):
        # audio_type 已写在 engine.audio_type 上，start_relaxation_audio 自行读取
        self.gui.start_relaxation_audio()


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
        master.configure(bg='#000000')   # 动画区外背景色

        # 动画区固定 800×450（5 倍像素缩放），在窗口内居中
        self.anim_w, self.anim_h = 800, 450
        self.anim_x = (800 - self.anim_w) // 2   # = 0
        self.anim_y = (480 - self.anim_h) // 2   # = 15

        # State
        self.current_state = BotStates.WARMUP
        self.current_volume = 0
        self.animations = {}
        self.current_frame_index = 0

        self.interrupted = threading.Event()

        # 对话核心（记忆 / 睡前引导状态机 / LLM 流式回复 / 每日摘要）全部在
        # ChatEngine 里，与硬件无关；GUI 通过 GuiIO 适配器把显示/TTS/播放接进去。
        # 对话状态（permanent_memory / phase / audio_type 等）由 engine 持有。
        self.engine = ChatEngine(GuiIO(self))

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

        # --- 聊天气泡系统（Canvas 绘制圆角矩形 + 文字，初始隐藏）---
        # 外框只显示上下两条 #b9451d 边线，左右隐藏
        self.chat_outer = tk.Frame(master, bg='#b9451d', highlightthickness=0, bd=0)
        self.chat_canvas = tk.Canvas(self.chat_outer, bg='#000000',
                                     highlightthickness=0, bd=0)
        self.chat_canvas.place(x=0, y=1, relwidth=1, relheight=1, height=-2)
        self.chat_canvas.bind('<MouseWheel>', self._on_chat_mousewheel)
        self.chat_canvas.bind('<Button-4>', lambda e: self.chat_canvas.yview_scroll(-3, 'units'))
        self.chat_canvas.bind('<Button-5>', lambda e: self.chat_canvas.yview_scroll(3, 'units'))
        self.chat_canvas.bind('<Configure>', self._on_chat_canvas_configure)
        self._bubbles = []           # 每项: {"role":str, "text_id":int, "rect_id":int, "text":str}
        self._bubble_y = 10          # 下一条气泡的起始 Y
        self._last_bot_text_id = None
        self._last_bot_rect_id = None
        self._chat_height = 130      # 聊天气泡面板默认高度

        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = tk.Label(master, textvariable=self.status_var,
                                     bg='#000000', fg='#b9451d', font=('Arial', 10))

        self.exit_button = tk.Button(master, text="Exit & Save", command=self.safe_exit,
            bg='#000000', fg='#b9451d', bd=0,
            highlightthickness=1, highlightbackground='#b9451d', highlightcolor='#b9451d',
            font=('Arial', 9), padx=6, pady=1)
        self.exit_button.bind('<Enter>', lambda e: self.exit_button.config(bg='#3a1509'))
        self.exit_button.bind('<Leave>', lambda e: self.exit_button.config(bg='#000000'))
        self.fullscreen_button = tk.Button(master, text="全屏", command=self.toggle_fullscreen_btn,
            bg='#000000', fg='#b9451d', bd=0,
            highlightthickness=1, highlightbackground='#b9451d', highlightcolor='#b9451d',
            font=('Arial', 9), padx=6, pady=1)
        self.fullscreen_button.bind('<Enter>', lambda e: self.fullscreen_button.config(bg='#3a1509'))
        self.fullscreen_button.bind('<Leave>', lambda e: self.fullscreen_button.config(bg='#000000'))

        self.load_animations()
        self.update_animation()

        threading.Thread(target=self.safe_main_execution, daemon=True).start()

    # --- HELPERS ---

    @staticmethod
    def _interp_color(c1, c2, t):
        """在俩 hex 颜色间插值，t∈[0,1] 返回 #rrggbb。"""
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        return f'#{int(r1+(r2-r1)*t):02x}{int(g1+(g2-g1)*t):02x}{int(b1+(b2-b1)*t):02x}'

    def _fade_in_widget(self, widget, attr, start_c, end_c, remaining, total, interval=20):
        """透明度渐变弹入：从 visible-dark 到目标色，linear 插值。"""
        if remaining <= 0:
            return
        raw = 1.0 - (remaining / total)
        t = max(0.0, min(1.0, 0.15 + 0.85 * raw))
        color = self._interp_color(start_c, end_c, t)
        try:
            widget.config(**{attr: color})
        except tk.TclError:
            pass
        self.master.after(interval,
            partial(self._fade_in_widget, widget, attr, start_c, end_c,
                    remaining-1, total, interval))

    def _fade_out_widget(self, widget, attr, start_c, end_c, remaining, total=6, interval=15):
        """透明度渐变弹出：从目标色渐变到黑。"""
        if remaining <= 0:
            return
        raw = 1.0 - (remaining / total)
        color = self._interp_color(start_c, end_c, raw)
        try:
            widget.config(**{attr: color})
        except tk.TclError:
            pass
        self.master.after(interval,
            partial(self._fade_out_widget, widget, attr, start_c, end_c,
                    remaining-1, total, interval))

    def _fade_out_bubbles(self, remaining, total=6, interval=15):
        """淡出时把所有气泡的文字/边框渐变为黑。"""
        if remaining <= 0:
            return
        raw = 1.0 - (remaining / total)
        for b in self._bubbles:
            role = b["role"]
            tt = '#b9451d' if role != 'user' else '#c8c8c8'
            ft = '#000000' if role != 'user' else '#3a1509'
            try:
                self.chat_canvas.itemconfigure(b["text_id"],
                    fill=self._interp_color(tt, '#000000', raw))
                self.chat_canvas.itemconfigure(b["rect_id"],
                    outline=self._interp_color('#b9451d', '#000000', raw),
                    fill=self._interp_color(ft, '#000000', raw))
            except tk.TclError:
                pass
        self.master.after(interval,
            partial(self._fade_out_bubbles, remaining-1, total, interval))

    def _fade_in_bubbles(self, remaining, total, interval=20):
        """恢复旧气泡颜色：15%→100%（淡出时变暗的泡泡渐变成全亮）。"""
        if remaining <= 0:
            return
        raw = 1.0 - (remaining / total)
        t = 0.15 + 0.85 * raw
        for b in self._bubbles:
            role = b["role"]
            tt = '#b9451d' if role != 'user' else '#c8c8c8'
            ft = '#000000' if role != 'user' else '#3a1509'
            try:
                self.chat_canvas.itemconfigure(b["text_id"],
                    fill=self._interp_color('#000000', tt, t))
                self.chat_canvas.itemconfigure(b["rect_id"],
                    outline=self._interp_color('#000000', '#b9451d', t),
                    fill=self._interp_color('#000000', ft, t))
            except tk.TclError:
                pass
        self.master.after(interval,
            partial(self._fade_in_bubbles, remaining-1, total, interval))

    def _do_slide(self, widget, start, end, steps=10, interval=20, total=None, callback=None):
        """将 widget 从 start(y) 平滑滑动到 end(y)，ease-out 立方缓动：快入慢停。"""
        if total is None:
            total = steps
        if steps <= 0:
            if callback:
                callback()
            return
        raw = 1.0 - (steps / total)
        t = 1.0 - (1.0 - raw) ** 3   # ease-out cubic
        y = int(start + (end - start) * t)
        try:
            if widget.winfo_exists():
                widget.place(y=y)
        except tk.TclError:
            pass
        self.master.after(interval,
            partial(self._do_slide, widget, start, end, steps-1, interval, total, callback))

    def _slide_in_hud(self):
        """弹入：按钮从上往下、聊天区从下往上滑入（200ms），伴随透明度渐变。"""
        ch = self.master.winfo_height()
        cw = self.master.winfo_width()
        chat_h = min(self._chat_height, ch // 3)
        chat_w = cw - 40
        chat_target = ch - chat_h - 22
        STEPS, INT = 10, 20  # 200ms

        self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
        self.status_label.config(fg='#1b0a04')
        self.exit_button.place(x=10, y=-30);  self.exit_button.config(fg='#1b0a04')
        btn_w = 50
        self.fullscreen_button.place(x=cw-10-btn_w, y=-30, width=btn_w)
        self.fullscreen_button.config(fg='#1b0a04')
        self.chat_outer.place(x=20, y=ch, width=chat_w, height=chat_h)

        self._do_slide(self.exit_button, -30, 10, STEPS, INT)
        self._do_slide(self.fullscreen_button, -30, 10, STEPS, INT)
        self._do_slide(self.chat_outer, ch, chat_target, STEPS, INT)

        self._fade_in_widget(self.exit_button, 'fg', '#1b0a04', '#b9451d',
                             STEPS, STEPS, INT)
        self._fade_in_widget(self.fullscreen_button, 'fg', '#1b0a04', '#b9451d',
                             STEPS, STEPS, INT)
        self._fade_in_widget(self.status_label, 'fg', '#1b0a04', '#b9451d',
                             STEPS, STEPS, INT)
        # 恢复旧气泡颜色（淡出时变暗的泡泡渐变成全亮）
        self._fade_in_bubbles(STEPS, STEPS, INT)

    def _slide_out_hud(self):
        """弹出：同弹入步数/时长（200ms），颜色反向变化，像倒放。"""
        ch = self.master.winfo_height()
        chat_target = ch - min(self._chat_height, ch // 3) - 22
        STEPS, INT = 10, 20  # 与弹入完全一致

        self._do_slide(self.exit_button, 10, -30, STEPS, INT)
        self._do_slide(self.fullscreen_button, 10, -30, STEPS, INT)
        self._do_slide(self.chat_outer, chat_target, ch, STEPS, INT,
                       callback=self._forget_hud)

        # 淡出：全亮 → 暗色（mirror 弹入）
        self._fade_out_widget(self.exit_button, 'fg', '#b9451d', '#1b0a04',
                              STEPS, STEPS, INT)
        self._fade_out_widget(self.fullscreen_button, 'fg', '#b9451d', '#1b0a04',
                              STEPS, STEPS, INT)
        self._fade_out_widget(self.status_label, 'fg', '#b9451d', '#1b0a04',
                              STEPS, STEPS, INT)
        self._fade_out_bubbles(STEPS, STEPS, INT)

    def _forget_hud(self):
        """隐藏所有 HUD 元素。"""
        for w in (self.chat_outer, self.status_label,
                  self.exit_button, self.fullscreen_button):
            try:
                w.place_forget()
            except tk.TclError:
                pass

    def _on_chat_mousewheel(self, event):
        """鼠标滚轮滚动聊天区。"""
        try:
            if event.delta > 0:
                self.chat_canvas.yview_scroll(-3, 'units')
            else:
                self.chat_canvas.yview_scroll(3, 'units')
        except tk.TclError:
            pass

    def _animate_bubble_up(self, text_id, rect_id, total_dist, steps=8, is_user=False):
        """新气泡短距离弹入（10px） + 从 visible-dark 渐变到目标色（280ms）。
        位置 ease-out cubic 稳稳落定，颜色 linear 从 15% 到 100%。"""
        text_target = '#b9451d' if not is_user else '#c8c8c8'
        fill_target = '#000000' if not is_user else '#3a1509'
        steps_data = []
        for i in range(1, steps + 1):
            p = i / steps
            # 位置：ease-out cubic（最后稳稳落定）
            cum = int(total_dist * (1.0 - (1.0 - p) ** 3))
            prev = int(total_dist * (1.0 - (1.0 - (i-1)/steps) ** 3)) if i > 1 else 0
            # 颜色：15% → 100% linear（全程肉眼可见）
            fade = 0.15 + 0.85 * p
            tc = self._interp_color('#000000', text_target, fade)
            oc = self._interp_color('#000000', '#b9451d', fade)
            fc = self._interp_color('#000000', fill_target, fade)
            steps_data.append((cum - prev, tc, oc, fc))
        for i, (d, tc, oc, fc) in enumerate(steps_data):
            self.master.after(i * 35,  # 280ms / 8 steps
                partial(self._step_bubble_in, text_id, rect_id, d, tc, oc, fc))

    def _step_bubble_in(self, text_id, rect_id, dy, text_color, outline_color, fill_color):
        """单步气泡弹入：上移 + 更新颜色。"""
        try:
            self.chat_canvas.move(text_id, 0, -dy)
            self.chat_canvas.move(rect_id, 0, -dy)
            self.chat_canvas.itemconfigure(text_id, fill=text_color)
            self.chat_canvas.itemconfigure(rect_id, fill=fill_color, outline=outline_color)
        except tk.TclError:
            pass

    def _on_chat_canvas_configure(self, event):
        """聊天 Canvas 尺寸变化时，重排所有气泡的位置和换行宽度。"""
        cw = event.width
        if cw < 20:
            return
        pad, ipad = 15, 10
        max_w = max(60, int(cw * 0.65) - ipad * 2)
        for b in self._bubbles:
            self.chat_canvas.itemconfigure(b["text_id"], width=max_w)
            if b["role"] == "user":
                coords = self.chat_canvas.coords(b["text_id"])
                if coords:
                    self.chat_canvas.coords(b["text_id"], cw - pad - ipad, coords[1])
        self.master.update_idletasks()
        self._redraw_all_rects()

    def _redraw_all_rects(self):
        """删除并重绘所有气泡的背景圆角矩形（配合 text 换行后的新尺寸）。"""
        cw = self.chat_canvas.winfo_width()
        if cw < 20:
            cw = 760
        pad, ipad, cr = 15, 10, 10
        min_w = 60
        for b in self._bubbles:
            self.chat_canvas.delete(b["rect_id"])
            bbox = self.chat_canvas.bbox(b["text_id"])
            if not bbox:
                continue
            role = b["role"]
            fill = '#3a1509' if role == 'user' else '#000000'
            if role == "user":
                rx1 = bbox[0] - ipad
                rx2 = cw - pad + ipad
            else:
                rx1 = pad - ipad
                rx2 = bbox[2] + ipad
            if rx2 - rx1 < min_w:
                if role == "user":
                    rx1 = rx2 - min_w
                else:
                    rx2 = rx1 + min_w
            ry1 = bbox[1] - ipad
            ry2 = bbox[3] + ipad
            new_rect = self._round_rect(
                self.chat_canvas, rx1, ry1, rx2, ry2, cr,
                fill=fill, outline='#b9451d', width=1)
            self.chat_canvas.tag_lower(new_rect, b["text_id"])
            b["rect_id"] = new_rect

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

        # 退出时把今天对话压成一句话摘要存档（幂等；不保留原始转录）
        self.engine.finalize_session_summary()

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
        """点击画面：弹入/弹出聊天气泡/状态/按钮（带滑入滑出动画）。"""
        try:
            if self.chat_outer.winfo_ismapped():
                self._slide_out_hud()
            else:
                self._slide_in_hud()
        except tk.TclError:
            pass

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

    def _stage(self, msg):
        """打印睡前引导流程的阶段调试信息，便于跟踪走到了哪一步。"""
        print(f"[STAGE] {msg}", flush=True)

    @staticmethod
    def _round_rect(canvas, x1, y1, x2, y2, r, **kwargs):
        """在 Canvas 上绘制圆角矩形多边形，返回 item id。"""
        points = [
            x1+r, y1, x2-r, y1,
            x2, y1, x2, y1+r,
            x2, y2-r, x2, y2,
            x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r,
            x1, y1+r, x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def add_message(self, role, text):
        """在 Canvas 上新增一条圆角聊天气泡。role='user'（右对齐+底色叠加）或 'bot'（左对齐黑底）。
        跨线程安全（经 after 投递到 GUI 线程）。"""
        def _update():
            is_user = role == 'user'
            cw = self.chat_canvas.winfo_width()
            if cw < 20:
                cw = 760

            pad = 15          # Canvas 边缘留白
            ipad = 10         # 文字在气泡内的内边距
            cr = 10           # 圆角半径
            max_w = int(cw * 0.65) - ipad * 2
            if max_w < 60:
                max_w = 60

            # 文字位置
            if is_user:
                anchor = 'ne'
                tx = cw - pad - ipad
                tx_color = '#c8c8c8'
            else:
                anchor = 'nw'
                tx = pad + ipad
                tx_color = '#b9451d'

            ty = self._bubble_y + ipad

            text_id = self.chat_canvas.create_text(
                tx, ty, text=text, font=('Arial', 11), fill=tx_color,
                width=max_w, anchor=anchor, justify='left')

            # 测量文字实际占用的 bbox
            self.master.update_idletasks()
            bbox = self.chat_canvas.bbox(text_id)
            if not bbox:
                bbox = (tx, ty, tx + (max_w if is_user else 80), ty + 18)

            # 背景圆角矩形
            fill = '#3a1509' if is_user else '#000000'
            min_bubble_w = 60

            if is_user:
                rx1 = bbox[0] - ipad
                rx2 = cw - pad + ipad
            else:
                rx1 = pad - ipad
                rx2 = bbox[2] + ipad

            # 保证最小宽度
            if rx2 - rx1 < min_bubble_w:
                if is_user:
                    rx1 = rx2 - min_bubble_w
                else:
                    rx2 = rx1 + min_bubble_w

            ry1 = bbox[1] - ipad
            ry2 = bbox[3] + ipad

            rect_id = self._round_rect(
                self.chat_canvas, rx1, ry1, rx2, ry2, cr,
                fill=fill, outline='#b9451d', width=1)

            self.chat_canvas.tag_lower(rect_id, text_id)

            # 记录气泡
            self._bubbles.append({
                "role": role, "text_id": text_id, "rect_id": rect_id, "text": text})

            # 更新下一个 Y 位置
            self._bubble_y = ry2 + 8

            # 流式追加跟踪
            if is_user:
                self._last_bot_text_id = None
                self._last_bot_rect_id = None
            else:
                self._last_bot_text_id = text_id
                self._last_bot_rect_id = rect_id

            # 初始设为目标色 15% 亮度（在黑底上隐约可见），弹入时渐变成 100%
            tx_color_init = self._interp_color('#000000', tx_color, 0.15)
            outline_init = self._interp_color('#000000', '#b9451d', 0.15)
            fill_init = self._interp_color('#000000', fill, 0.15)
            self.chat_canvas.itemconfigure(text_id, fill=tx_color_init)
            self.chat_canvas.itemconfigure(rect_id, fill=fill_init, outline=outline_init)

            # 短距离弹入 + 透明度渐变（280ms）
            anim_offset = 10
            self.chat_canvas.move(text_id, 0, anim_offset)
            self.chat_canvas.move(rect_id, 0, anim_offset)
            self._animate_bubble_up(text_id, rect_id, anim_offset, steps=8, is_user=is_user)

            # 更新滚动区域 & 滚到底
            self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox('all'))
            self.chat_canvas.yview_moveto(1.0)

        self.master.after(0, _update)

    def stream_to_bubble(self, chunk):
        """把 LLM 流式片段实时追加到最后一条 bot 气泡的 Canvas 文字项，
        并重绘背景圆角矩形以适配新尺寸。"""
        def _update():
            text_id = self._last_bot_text_id
            rect_id = self._last_bot_rect_id
            if text_id is None:
                return

            # 追加文字
            current = self.chat_canvas.itemcget(text_id, 'text')
            self.chat_canvas.itemconfigure(text_id, text=current + chunk)

            # 重绘背景矩形
            self.master.update_idletasks()
            bbox = self.chat_canvas.bbox(text_id)
            if not bbox:
                return

            self.chat_canvas.delete(rect_id)

            ipad = 10
            cr = 10
            pad = 15
            cw = self.chat_canvas.winfo_width()
            if cw < 20:
                cw = 760

            rx1 = pad - ipad
            rx2 = bbox[2] + ipad
            if rx2 - rx1 < 60:
                rx2 = rx1 + 60
            ry1 = bbox[1] - ipad
            ry2 = bbox[3] + ipad

            new_rect = self._round_rect(
                self.chat_canvas, rx1, ry1, rx2, ry2, cr,
                fill='#000000', outline='#b9451d', width=1)
            self.chat_canvas.tag_lower(new_rect, text_id)
            self._last_bot_rect_id = new_rect

            # 更新气泡记录
            for b in self._bubbles:
                if b["text_id"] == text_id:
                    b["rect_id"] = new_rect
                    b["text"] = current + chunk
                    break

            self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox('all'))
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
            if self.chat_outer.winfo_ismapped():
                chat_h = min(self._chat_height, h // 3)
                chat_w = w - 40
                chat_y = h - chat_h - 22
                self.chat_outer.place(x=20, y=chat_y, width=chat_w, height=chat_h)
                btn_w = 50
                self.fullscreen_button.place(x=w-10-btn_w, y=10, width=btn_w)
        except tk.TclError:
            pass

    # =========================================================================
    # 4. CORE LOGIC
    # =========================================================================

    def safe_main_execution(self):
        """后台主循环：预热 → 起动 TTS 流水线 → 反复「录音→识别→对话」直到退出。
        整个循环包在 try 里，任何未预料异常都转成 ERROR 状态而非让线程静默崩掉。"""
        try:
            self.play_startup_sound()
            self.warm_up_logic()
            self.synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
            self.synth_thread.start()
            self.play_thread = threading.Thread(target=self._play_worker, daemon=True)
            self.play_thread.start()

            while True:
                if self.exiting:
                    break

                # 播放助眠音频期间不再监听，等看门狗线程放完/超时后退出
                if self.engine.phase == "playing":
                    time.sleep(0.5)
                    continue

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
                    self.engine.handle_turn(user_text)

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
                ollama.chat(
                    model=TEXT_MODEL,
                    messages=self.engine.permanent_memory + [{"role": "user", "content": "你好"}],
                    stream=False,
                    options=OLLAMA_OPTIONS,
                    keep_alive=-1,
                )
        except Exception as e:
            print(f"Failed to load {TEXT_MODEL}: {e}", flush=True)

        # 开场问候（载入每日摘要、拼成可念出的句子、存入会话记忆）统一在 engine 里。
        greeting, has_summary = self.engine.build_greeting()
        self._stage("问候阶段" + ("（带昨日摘要）" if has_summary else "（无摘要）"))
        self.set_state(BotStates.GREETING, "晚上好")
        self.speak(greeting)
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
        whisper_threads = CURRENT_CONFIG.get("whisper_threads", 4)
        # beam search / best-of：whisper-cli 默认 5/5，较慢；设为 1/1 即 greedy，
        # 在树莓派上能省下大量解码时间，短句精度几乎不受影响。
        whisper_beam_size = CURRENT_CONFIG.get("whisper_beam_size", 1)
        whisper_best_of   = CURRENT_CONFIG.get("whisper_best_of", 1)
        cmd = ["./whisper.cpp/build/bin/whisper-cli",
               "-m", f"./whisper.cpp/models/{whisper_model}",
               "-l", whisper_lang, "-t", str(whisper_threads),
               "-bs", str(whisper_beam_size), "-bo", str(whisper_best_of),
               "-f", filename]
        # 初始提示偏置：给一句简体示例，引导 whisper 输出简体而非繁体。
        whisper_prompt = CURRENT_CONFIG.get("whisper_prompt", "以下是普通话的句子。")
        if whisper_prompt:
            cmd += ["--prompt", whisper_prompt]
        whisper_debug = CURRENT_CONFIG.get("whisper_debug", False)
        try:
            with timed_block("STT whisper-cli"):
                result = subprocess.run(cmd, capture_output=True, text=True)
            # whisper-cli 把诊断信息（加载模型、检测语言、各阶段耗时、报错）都打到 stderr。
            # 默认全部捕获不显示；调试时打开 whisper_debug 看完整输出定位问题。
            if result.returncode != 0:
                print(f"[whisper] WARNING: whisper-cli 退出码 {result.returncode}", flush=True)
            if whisper_debug:
                print(f"[whisper] cmd: {' '.join(cmd)}", flush=True)
                print(f"[whisper] returncode: {result.returncode}", flush=True)
                print(f"[whisper] --- stderr ---\n{result.stderr}", flush=True)
                print(f"[whisper] --- stdout ---\n{result.stdout}", flush=True)
            # whisper-cli 每个语音片段输出一行，形如：
            #   [00:00:00.000 --> 00:00:06.800]  片段文本
            # 一句话被切成多段时会有多行，必须把所有片段拼起来，
            # 只取最后一行会丢掉前半句（曾导致"识别不准"）。
            segments = []
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 去掉行首的 [时间戳] 前缀；没有前缀的行原样保留
                seg = line.split("]", 1)[1].strip() if line.startswith("[") and "]" in line else line
                if seg:
                    segments.append(seg)
            transcription = "".join(segments)
            print(f"Heard: '{transcription}'", flush=True)
            return transcription.strip()
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""

    # =========================================================================
    # 5. TTS 入队（对话逻辑见 chat_engine.ChatEngine）
    # =========================================================================

    def _queue_sentence(self, sentence):
        """把一句完整文本推进 TTS 队列（过滤掉没有可读字符的空句）。"""
        s = sentence.strip()
        if s and re.search(r'[\w一-鿿]', s):
            with self.tts_queue_lock:
                self.tts_queue.append(s)

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

    def play_startup_sound(self):
        """开机音效：放一段固定的 wav（连接提示音）。缺文件/出错都静默跳过，
        绝不阻断开机流程。同步播放（sd.wait），紧接着的 LLM 预热不放音频，不抢设备。"""
        path = CURRENT_CONFIG.get(
            "startup_sound", "sounds/start/connection-sound-for-software.wav")
        if not os.path.isfile(path):
            return
        try:
            samples, rate = self._load_wav(path)
            gain = float(CURRENT_CONFIG.get("startup_gain", 0.8))
            samples = (samples * gain).astype(np.float32)
            samples, rate = self._fit_samplerate(samples, rate)
            self._stage(f"开机音效：{path}")
            sd.play(samples, rate)
            sd.wait()
        except Exception as e:
            print(f"[STARTUP SOUND] 播放失败 {path}: {e}", flush=True)

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

    # =========================================================================
    # 6. 助眠音频 / 每日摘要
    # =========================================================================

    def start_relaxation_audio(self):
        """加载并播放一条助眠音频（白噪音/轻音乐），做音量平衡；
        音频自然放完或满 relaxation_max_minutes 分钟后自动关机。"""
        path = self._pick_relaxation_file()
        if not path:
            self._stage(f"助眠音频决定：类型={self.engine.audio_type}，但未找到对应音频文件")
            print("[RELAX] 未找到助眠音频，直接进入关机。", flush=True)
            self.set_state(BotStates.SLEEP, "晚安")
            self.safe_exit()
            return
        self._stage(f"助眠音频决定：类型={self.engine.audio_type}，文件={path}")
        try:
            samples, rate = self._load_wav(path)
        except Exception as e:
            print(f"[RELAX] 读取音频失败 {path}: {e}", flush=True)
            self.safe_exit()
            return
        gain = float(CURRENT_CONFIG.get("relaxation_gain", 0.6))
        samples = (samples * gain).astype(np.float32)
        samples, rate = self._fit_samplerate(samples, rate)
        self.set_state(BotStates.SLEEP, "晚安，好梦")
        threading.Thread(target=self._relaxation_watchdog,
                         args=(samples, rate, path), daemon=True).start()

    def _pick_relaxation_file(self):
        """从 sounds/relaxation/<audio_type>/ 取第一个 .wav；目录为空或不存在返回 None。"""
        base = CURRENT_CONFIG.get("relaxation_audio_dir", "sounds/relaxation")
        folder = os.path.join(base, self.engine.audio_type)
        if not os.path.isdir(folder):
            return None
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))
        return os.path.join(folder, files[0]) if files else None

    def _load_wav(self, path):
        """读取 16-bit PCM wav → (float32 [-1,1] 单声道, 采样率)。"""
        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            n_ch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)   # 混成单声道
        return samples, rate

    def _relaxation_watchdog(self, samples, rate, path):
        """后台播放助眠音频并看门：到时间上限或自然放完即停止并关机。"""
        max_minutes = float(CURRENT_CONFIG.get("relaxation_max_minutes", 45))
        deadline = time.time() + max_minutes * 60
        print(f"[RELAX] 播放 {path}，上限 {max_minutes} 分钟。", flush=True)
        try:
            sd.play(samples, rate)
            while not self.exiting:
                if time.time() >= deadline:
                    print("[RELAX] 到达时间上限，停止。", flush=True)
                    sd.stop()
                    break
                try:
                    st = sd.get_stream()
                    if st is None or not st.active:
                        print("[RELAX] 音频自然放完。", flush=True)
                        break
                except Exception:
                    break
                time.sleep(0.5)
        except Exception as e:
            print(f"[RELAX] 播放出错: {e}", flush=True)
        self.safe_exit()

if __name__ == "__main__":
    print("--- SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    # 测试：展示 UI 效果（多轮对话演示）
    # def demo_ui():
    #     app.toggle_hud_visibility()
    #     app.add_message("user", "你好呀，今天过得怎么样？")
    #     root.after(300, lambda: app.add_message("bot", "晚上好呀～今天陪你聊天真开心！有什么想聊的话题吗？"))
    #     root.after(600, lambda: app.add_message("user", "我想听一个睡前故事"))
    #     root.after(900, lambda: app.add_message("bot", "好啊，我给你讲一个小王子的故事吧。从前有一个小王子，他住在一个很小的星球上，那个星球比一座房子大不了多少。他每天清理猴面包树的幼苗，看日落，照顾一朵骄傲的玫瑰花。"))
    #     root.after(1500, lambda: app.add_message("user", "玫瑰花后来怎么样了？"))
    #     root.after(1800, lambda: app.add_message("bot", "玫瑰花后来明白了，小王子对她是独一无二的，就像她对他也是独一无二的。小王子尽管走遍了各个星球，见到了各种奇奇怪怪的大人，但他心里始终惦记着他的玫瑰花。"))
    #     root.after(2400, lambda: app.add_message("user", "真好啊，我也想养一朵玫瑰花"))
    #     root.after(2700, lambda: app.add_message("bot", "那你可以从现在开始种一颗种子呀。每天给它浇水、陪它说话，等到开花的时候，你会发现这朵花是全世界最特别的——因为那是属于你的玫瑰花🌹"))
    #     root.after(3300, lambda: app.add_message("user", "嗯，那晚安啦"))
    #     root.after(3600, lambda: app.add_message("bot", "晚安，好梦。愿你在梦里也能遇见属于你的小王子。明天见～"))
    # root.after(500, demo_ui)
    root.mainloop()
