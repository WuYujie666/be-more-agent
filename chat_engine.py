# =========================================================================
#  Be More Agent · 对话引擎（无硬件依赖）
# =========================================================================
"""睡前陪伴的对话核心：多轮状态机 + LLM 流式回复 + 每日摘要。

本模块刻意不 import tkinter / sounddevice / webrtcvad / PIL / sherpa，
只依赖 ollama 和 config，因此可在任意开发机上跑（不需要麦克风/扬声器/
whisper.cpp/sherpa 模型）。GUI 与文字版调试工具共用这同一套逻辑，避免
prompt 拼装、标签剥离、状态机出现两份实现而漂移。

与外界（显示 / TTS / 助眠播放 / 打断）的交互全部通过注入的 `io` 对象
（ChatIO 协议）完成：
    set_state(state, msg)        切换状态/动画文字
    bot_start()                  新开一条机器人气泡（流式追加前调一次）
    token(piece)                 把流式片段追加到当前气泡
    speak(sentence)              把一句完整文本送去合成播放
    wait_speech()                阻塞直到 TTS 放完
    is_interrupted() -> bool     是否被用户打断
    stage(msg)                   打印流程阶段调试信息
    play_relaxation_audio(type)  进入助眠音频播放阶段
"""

import re
import time
import threading

import ollama

from config import (
    CURRENT_CONFIG, OLLAMA_OPTIONS, TEXT_MODEL, BotStates, SYSTEM_PROMPT,
    detect_story_intent, extract_audio_tag,
    match_audio_word, match_sleep_word, match_yesno_word,
    append_summary, load_recent_summary,
)


def _dump_messages(messages):
    """debug_prompt 打开时，打印这次发给 LLM 的完整 messages，便于调 prompt。"""
    if not CURRENT_CONFIG.get("debug_prompt"):
        return
    print("===== PROMPT =====", flush=True)
    for m in messages:
        print(f"[{m['role']}]\n{m['content']}\n", flush=True)
    print("==================", flush=True)


class ChatEngine:
    """对话状态机与 LLM 交互的全部逻辑，不持有任何硬件/GUI 资源。"""

    def __init__(self, io):
        self.io = io

        # --- 对话记忆 ---
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.chat_user_texts = []   # 仅 CHAT 阶段的用户发言，供每日摘要取材

        # --- 睡前引导状态机 ---
        # phase: "chat"（聊天中）/ "ask_audio"（已问是否助眠，等是/否）/ "playing"（放助眠音频）
        self.phase = "chat"
        self.turn_count = 0          # CHAT 阶段用户发言计数
        self.audio_type = CURRENT_CONFIG.get("default_audio", "white_noise")
        self.max_chat_turns = int(CURRENT_CONFIG.get("max_chat_turns", 5))
        self.summary_saved = False   # 每日摘要幂等保护
        self.summary_thread = None   # 异步摘要线程句柄，供调用方需要时 join

    # =========================================================================
    #  状态机分派
    # =========================================================================

    def handle_turn(self, user_text):
        """按当前 phase 把一轮用户输入分派给对应处理函数。"""
        if self.phase == "ask_audio":
            self.handle_audio_answer(user_text)
        else:
            self.handle_chat_turn(user_text)

    def handle_chat_turn(self, user_text):
        """CHAT 阶段一轮：更新轮次与音频类型，判断是否进入收尾过渡，再生成回复。
        触发过渡的条件：用户说了想睡/想结束的关键词，或聊满 max_chat_turns 轮。"""
        self.turn_count += 1
        self.chat_user_texts.append(user_text)   # 只记 CHAT 阶段用户发言，供摘要取材
        self.io.stage(f"第 {self.turn_count} 轮对话：{user_text}")

        t, audio_word = match_audio_word(user_text)
        if t:
            self.audio_type = t
            self.io.stage(f"检测到助眠类型关键词「{audio_word}」→ {t}")

        sleep_word = match_sleep_word(user_text)
        turns_reached = self.turn_count >= self.max_chat_turns
        transition = bool(sleep_word) or turns_reached

        extra = None
        if transition:
            if sleep_word:
                self.io.stage(f"检测到睡意关键词「{sleep_word}」，进入睡前询问阶段")
            else:
                self.io.stage(f"已聊满 {self.max_chat_turns} 轮，进入睡前询问阶段")
            extra = CURRENT_CONFIG.get("transition_prompt", "")
        elif detect_story_intent(user_text):
            self.io.stage("检测到讲故事意图，放宽本轮回复长度")
            extra = CURRENT_CONFIG.get("story_prompt", "")

        self.chat_and_respond(user_text, extra_instruction=extra)

        if transition:
            self.phase = "ask_audio"

    def handle_audio_answer(self, user_text):
        """ASK_AUDIO 阶段：用户回答想听白噪音还是轻音乐。
        拒绝则先安抚、说睡觉恢复精力的好处、再晚安，之后仍照常播放；
        其余（含听不清）默认按愿意处理。收尾句播完后异步生成今日摘要，再进入助眠音频播放。"""
        t, audio_word = match_audio_word(user_text)
        if t:
            self.audio_type = t
            self.io.stage(f"检测到助眠类型关键词「{audio_word}」→ {t}")

        decision, yesno_word = match_yesno_word(user_text)
        if decision == "no":
            self.io.stage(f"睡前询问应答：命中「{yesno_word}」→ 拒绝助眠")
            extra = CURRENT_CONFIG.get("decline_audio_prompt", "")
        else:
            hit = f"命中「{yesno_word}」" if yesno_word else "听不清/默认"
            self.io.stage(f"睡前询问应答：{hit} → 接受助眠")
            extra = CURRENT_CONFIG.get("goodnight_prompt", "")

        self.chat_and_respond(user_text, extra_instruction=extra)

        self.start_session_summary()   # 异步压缩今日对话，不阻塞助眠
        self.phase = "playing"
        self.io.play_relaxation_audio(self.audio_type)

    # =========================================================================
    #  LLM 流式回复
    # =========================================================================

    def chat_and_respond(self, text, extra_instruction=None):
        """流式生成一段回复：实时显示气泡、按句送 TTS，并剥掉 [AUDIO:x] 控制标签。
        extra_instruction 作为系统提示拼到本轮 user 消息后（用于过渡/收尾指令）。
        返回去掉标签后的完整回复文本。"""
        if "forget everything" in text.lower() or "reset memory" in text.lower() \
                or "清空记忆" in text or "忘记一切" in text:
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.io.speak("好的，我把记忆清空了。")
            self.io.set_state(BotStates.IDLE, "Memory Wiped")
            return ""

        self.io.set_state(BotStates.THINKING, "Thinking...")

        lang = CURRENT_CONFIG.get("whisper_lang", "en")
        lang_hint = "请用中文回答。" if lang == "zh" else ""
        parts = [text]
        if extra_instruction:
            parts.append("[系统提示]" + extra_instruction)
        if lang_hint:
            parts.append(lang_hint)
        # 本轮发给 LLM 的 user 消息带上临时的系统/语言提示；但存入历史的只放干净
        # 原文，避免「过渡/收尾」这类一次性指令污染后续轮次的上下文。
        call_user_msg = {"role": "user", "content": "\n".join(parts)}
        messages = self.permanent_memory + self.session_memory + [call_user_msg]
        _dump_messages(messages)
        # 用户原文进会话记忆，保证后续轮次 LLM 能看到完整的多轮历史（不只是机器人自己说过的话）。
        self.session_memory.append({"role": "user", "content": text})

        full_raw = ""          # LLM 原始输出（含标签），用于最终解析与记忆
        pending = ""           # 显示缓冲：拦住可能跨 chunk 的 [AUDIO:x] 标签
        sentence_buffer = ""   # 攒成整句再送 TTS
        spoke = False
        TAG_START = "[AUDIO:"
        COMPLETE_TAG = re.compile(r'^\[AUDIO:\w+\]', re.IGNORECASE)

        def emit(piece):
            nonlocal sentence_buffer, spoke
            if not piece:
                return
            if not spoke:
                self.io.set_state(BotStates.SPEAKING, "Speaking...")
                self.io.bot_start()
                spoke = True
            self.io.token(piece)
            sentence_buffer += piece
            if any(p in piece for p in ".!?\n。！？"):
                self.io.speak(sentence_buffer)
                sentence_buffer = ""

        def drain(final=False):
            # 把 pending 里「确认安全」的文本送出去，遇到可能的 [AUDIO:x] 标签先拦住。
            nonlocal pending
            while pending:
                i = pending.find('[')
                if i == -1:
                    emit(pending); pending = ""; return
                if i > 0:
                    emit(pending[:i]); pending = pending[i:]
                m = COMPLETE_TAG.match(pending)
                if m:
                    _, tg = extract_audio_tag(m.group(0))
                    if tg:
                        self.audio_type = tg
                    pending = pending[m.end():]; continue
                looks_like_prefix = (TAG_START.startswith(pending[:len(TAG_START)])
                                     or (pending.startswith(TAG_START) and ']' not in pending))
                if final:
                    if looks_like_prefix:
                        pending = ""; return   # 收尾时丢掉被截断的标签残尾
                elif looks_like_prefix:
                    return                     # 可能是跨 chunk 的标签前缀，等后续 chunk
                emit('['); pending = pending[1:]   # 只是个普通 '['

        try:
            stream = ollama.chat(model=TEXT_MODEL, messages=messages, stream=True, options=OLLAMA_OPTIONS)

            _t_llm = time.perf_counter()
            _ttft_logged = False

            for chunk in stream:
                if self.io.is_interrupted(): break
                content = chunk['message']['content']
                if not _ttft_logged:
                    print(f"[TIMER] LLM 首Token延迟 {time.perf_counter()-_t_llm:.2f}s", flush=True)
                    _ttft_logged = True
                full_raw += content
                pending += content
                drain()

            drain(final=True)
            if sentence_buffer.strip():
                self.io.speak(sentence_buffer)

            clean_full, tag_type = extract_audio_tag(full_raw)
            clean_full = re.sub(r'\[AUDIO:?\w*\]?', '', clean_full).strip()
            if tag_type:
                self.audio_type = tag_type
            self.session_memory.append({"role": "assistant", "content": clean_full})

            self.io.wait_speech()
            self.io.set_state(BotStates.IDLE, "Ready")
            return clean_full

        except Exception as e:
            print(f"LLM Error: {e}")
            self.io.set_state(BotStates.IDLE, "Brain Freeze!")
            return ""

    # =========================================================================
    #  每日摘要
    # =========================================================================

    def start_session_summary(self):
        """后台线程把今天对话压成一句话摘要，不阻塞助眠流程。
        线程句柄存到 self.summary_thread，便于调试工具等它跑完再退出。"""
        self.summary_thread = threading.Thread(target=self.finalize_session_summary, daemon=True)
        self.summary_thread.start()

    def finalize_session_summary(self):
        """把今日对话压成一句话摘要并存档（幂等）。原始转录不保留。"""
        if self.summary_saved:
            return
        self.summary_saved = True   # 先占位，避免异步线程与 safe_exit 重复生成
        texts = [t for t in self.chat_user_texts if t.strip()]
        if not texts:
            return
        transcript = "\n".join(texts)
        summary_prompt = CURRENT_CONFIG.get("summary_prompt", "")
        try:
            messages = [{"role": "user", "content": summary_prompt + "\n" + transcript}]
            _dump_messages(messages)
            resp = ollama.chat(
                model=TEXT_MODEL,
                messages=messages,
                stream=False, options=OLLAMA_OPTIONS,
            )
            summary, _ = extract_audio_tag(resp["message"]["content"])
            summary = summary.strip().replace("\n", " ")
            if summary:
                append_summary(summary)
                print(f"[SUMMARY] saved: {summary}", flush=True)
        except Exception as e:
            print(f"[SUMMARY] generate failed: {e}", flush=True)

    def load_chat_history(self):
        """跨会话不再回灌原始转录；仅以系统 prompt 起步，连续性靠每日摘要。"""
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def build_greeting(self):
        """开场问候：晚上好 →（昨天做的事＝每日摘要）→ 今天有什么想分享的吗。
        「昨天做的事」直接拼接摘要——摘要已存成可念出的第二人称句，无需再加工。
        问候作为 assistant 消息存入会话记忆，让后续对话上下文连贯。
        返回 (greeting, has_summary)，调用方负责念出/打印。"""
        summary = load_recent_summary()
        if summary:
            greeting = "晚上好。昨天" + summary + "。今天有什么想分享的吗？"
        else:
            greeting = "晚上好。今天有什么想分享的吗？"
        self.session_memory.append({"role": "assistant", "content": greeting})
        return greeting, bool(summary)
