# =========================================================================
#  Be More Agent · 配置与常量
# =========================================================================

import os
import re
import json
import time
import datetime
import contextlib

# sounddevice 只在音频设备查询/录音播放时用到。容错导入：在没装该库的开发机上
# 也能 import config（供 chat_engine / debug_chat 等无硬件链路使用）。
try:
    import sounddevice as sd
except ImportError:
    sd = None

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.5

# HARDWARE SETTINGS
INPUT_DEVICE_NAME = None

DEFAULT_CONFIG = {
    "text_model": "gemma3:1b",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "chat_memory": True,
    "input_device": None,
    "input_sample_rate": None,
    "whisper_model": "ggml-base.en.bin",
    "whisper_lang": "en",
    "whisper_prompt": "以下是普通话的句子。",   # 初始提示，偏置 whisper 输出简体中文
    "whisper_threads": 4,      # whisper.cpp 推理线程数，建议设为物理核心数（树莓派 4/5 为 4）
    "whisper_debug": False,    # 调试用：打印 whisper-cli 的 returncode/stderr/stdout 完整输出
    "whisper_beam_size": 1,    # beam search 宽度，1 = greedy（最快）；whisper-cli 默认 5
    "whisper_best_of": 1,      # best-of 采样数，1 = greedy（最快）；whisper-cli 默认 5
    # --- VAD（免手持续监听）---
    "vad_aggressiveness": 3,   # webrtcvad 灵敏度 0~3，越大越严格（越不易把噪声当人声）
    "vad_start_ms": 150,       # 连续多少毫秒判定为人声才算"开始说话"（防瞬时噪声误触发）
    "vad_silence_ms": 900,     # 尾部静音多久判定"说完"
    "vad_max_record_ms": 30000,# 单次最长录音
    "vad_preroll_ms": 300,     # 起始前回看缓冲，避免吞掉第一个字
    # --- 睡前引导 / 放松音频 ---
    "max_chat_turns": 5,                       # 聊满几轮后主动收尾过渡
    "default_audio": "white_noise",            # 默认助眠音频类型：white_noise / music
    "relaxation_audio_dir": "sounds/relaxation",  # 放松音频根目录（下分 white_noise/ 和 music/）
    "relaxation_gain": 0.6,                     # 助眠音频音量（相对 TTS 满音量），做音量平衡
    "relaxation_max_minutes": 45,              # 助眠音频最长播放分钟数，到点自动关机
    "summaries_file": "summaries.json",        # 每日一句话摘要存档
    "summary_recent_days": 2,                  # 开场时「昨天摘要」允许的最大天数
    "transition_prompt": "用户表达了睡意，但没有指定想听哪种助眠声音。请先用一句话简短回应用户这句话，再温柔收尾，并问用户想听白噪音还是轻音乐。不要出主意、不要解决问题。",
    "max_turns_prompt": "对话已达到本次睡前陪伴轮数上限。回复必须分两步：第一句必须直接承接用户最后一句话，并明确提到用户最后一句里的核心内容；如果用户最后一句是在提问，不要给建议或展开解决，只温柔接住并说先记下。第二句再温柔收尾，说今天聊了很多就到这里、该晚安了，并说明马上播放默认白噪音。不要再询问白噪音还是轻音乐，不要出主意、不要解决问题。",
    "decline_audio_prompt": "用户还不想睡或不想听助眠声音。请先用一句温柔的话安抚，表示理解他现在还不太想睡；再说一句好好睡觉能帮身体恢复精力的好处；最后道一句晚安、祝好梦。",
    "goodnight_prompt": "请只说一句温柔的晚安、祝好梦。",
    "story_prompt": "用户想听故事。可以讲一个温柔、适合睡前的短故事，长度可以比平时长一些（几句到一小段即可），讲完轻声收尾。",
    "summary_prompt": "你是睡前陪伴机器人，请对用户说一句开场白：先说“晚上好。”，再用一句话回顾用户昨天聊了什么、心情如何，最后问“今天有什么想分享的吗？”。整段只输出这一句话，不要引号、不要换行。用户昨天说：",
    "debug_prompt": False,                      # 打开后每次调 LLM 前打印完整 messages，便于调 prompt
}

# LLM SETTINGS
OLLAMA_OPTIONS = {
    'keep_alive': '-1',
    'num_thread': 4,
    'temperature': 0.7,
    'top_k': 40,
    'top_p': 0.9
}


@contextlib.contextmanager
def timed_block(label):
    t0 = time.perf_counter()
    print(f"[TIMER] >>> {label}", flush=True)
    try:
        yield
    finally:
        print(f"[TIMER] <<< {label}  {time.perf_counter()-t0:.2f}s", flush=True)


# =========================================================================
# 1b. 睡前引导：关键词识别 / 标签解析 / 摘要存取（纯函数，便于单测）
# =========================================================================

# 音频类型识别用的关键词
_WHITE_NOISE_WORDS = ("白噪", "白噪音", "雨声", "下雨", "雨", "海浪", "海", "风声", "流水", "溪")
_MUSIC_WORDS = ("轻音乐", "音乐", "钢琴", "旋律", "曲子", "乐曲")
# 想结束 / 想睡的意图关键词
_SLEEP_WORDS = ("睡", "困", "晚安", "累了", "不聊了", "不想聊", "结束", "休息", "睡觉", "该睡")
# 应答里的否定 / 肯定关键词（先判否定，避免「不想」「不要」被当成肯定）
_NO_WORDS = ("不", "别", "算了", "无所谓")
_YES_WORDS = ("好", "可以", "行", "嗯", "要", "想", "当然", "来吧", "OK", "ok", "play")
# 想听故事的意图关键词（命中则允许本轮回复适当延长）
_STORY_WORDS = ("讲故事", "讲个故事", "睡前故事", "故事", "讲一个", "讲一段")
# 明确请求播放助眠音频的动作词。需同时命中音频类型，才会直接进入播放阶段。
_AUDIO_REQUEST_WORDS = ("播放", "放", "播", "想听", "要听", "来点", "来一段", "给我", "开一下", "打开")
_AUDIO_REQUEST_NEGATIONS = ("不想听", "不要听", "别听", "不播放", "不要播放", "别播放", "别放", "不放")

# LLM 回复里偷偷夹带的音频控制标签，例如 [AUDIO:white] / [AUDIO:music]。
# 容错匹配类似 [想要白噪音时加AUDIO:white] 的啰嗦输出，避免泄露给用户。
_AUDIO_TAG_RE = re.compile(r"\[[^\]]*AUDIO:\s*(\w+)[^\]]*\]", re.IGNORECASE)
_TAG_TO_TYPE = {
    "white": "white_noise",
    "music": "music",
}

# 音频类型 → 询问句里用的中文名
AUDIO_TYPE_LABELS = {"white_noise": "白噪音", "music": "轻音乐"}


def _first_hit(text, words):
    """返回 text 中命中的第一个关键词，没命中返回 None。"""
    for w in words:
        if w in text:
            return w
    return None


def match_audio_word(text):
    """识别助眠音频类型；命中返回 (type, 命中词)，否则 (None, None)。"""
    if not text:
        return None, None
    # 先判轻音乐（'白噪音' 不含 '音乐'，两类关键词不冲突）
    w = _first_hit(text, _MUSIC_WORDS)
    if w:
        return "music", w
    w = _first_hit(text, _WHITE_NOISE_WORDS)
    if w:
        return "white_noise", w
    return None, None


def match_sleep_word(text):
    """返回命中的睡意 / 想结束关键词，没命中返回 None。"""
    return _first_hit(text, _SLEEP_WORDS) if text else None


def match_yesno_word(text):
    """识别是否答应听助眠音频：返回 (decision, 命中词)；先判否定，无法判断返回 (None, None)。"""
    if not text:
        return None, None
    w = _first_hit(text, _NO_WORDS)
    if w:
        return "no", w
    w = _first_hit(text, _YES_WORDS)
    if w:
        return "yes", w
    return None, None


def detect_audio_type(text):
    """从用户文本里识别想要的助眠音频类型；命中返回 'white_noise'/'music'，否则 None。"""
    return match_audio_word(text)[0]


def detect_audio_request(text):
    """用户是否明确要求现在播放某种助眠音频。"""
    if not text or match_audio_word(text)[0] is None:
        return False
    if _first_hit(text, _AUDIO_REQUEST_NEGATIONS):
        return False
    return _first_hit(text, _AUDIO_REQUEST_WORDS) is not None


def detect_sleep_intent(text):
    """用户是否表达了想结束对话 / 想睡的意图。"""
    return match_sleep_word(text) is not None


def detect_yes_no(text):
    """识别是否答应听助眠音频：'yes' / 'no' / None（无法判断）。先判否定。"""
    return match_yesno_word(text)[0]


def detect_story_intent(text):
    """用户是否想听故事（命中则允许本轮回复适当延长）。"""
    return _first_hit(text, _STORY_WORDS) is not None if text else False


def extract_audio_tag(text):
    """剥掉 LLM 回复里的 [AUDIO:x] 控制标签，返回 (clean_text, audio_type 或 None)。"""
    if not text:
        return text, None
    audio_type = None
    for m in _AUDIO_TAG_RE.finditer(text):
        mapped = _TAG_TO_TYPE.get(m.group(1).lower())
        if mapped:
            audio_type = mapped
    clean = _AUDIO_TAG_RE.sub("", text)
    return clean, audio_type


def load_recent_summary(summaries_file=None, recent_days=None):
    """读取最近一条每日摘要：日期距今 ≤ recent_days 天则返回摘要文本，否则 None。"""
    summaries_file = summaries_file or CURRENT_CONFIG.get("summaries_file", "summaries.json")
    recent_days = recent_days if recent_days is not None \
        else CURRENT_CONFIG.get("summary_recent_days", 2)
    if not os.path.exists(summaries_file):
        return None
    try:
        with open(summaries_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not items:
            return None
        last = items[-1]
        d = datetime.date.fromisoformat(last["date"])
        if (datetime.date.today() - d).days <= recent_days:
            return last.get("summary") or None
    except Exception as e:
        print(f"[SUMMARY] load failed: {e}", flush=True)
    return None


def append_summary(summary, summaries_file=None, keep=5):
    """把一条带当天日期的摘要追加进存档，只保留最近 keep 条。"""
    summaries_file = summaries_file or CURRENT_CONFIG.get("summaries_file", "summaries.json")
    items = []
    if os.path.exists(summaries_file):
        try:
            with open(summaries_file, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            items = []
    items.append({"date": datetime.date.today().isoformat(), "summary": summary})
    items = items[-keep:]
    with open(summaries_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            print(f"Config Error: {e}. Using defaults.")
    return config

CURRENT_CONFIG = load_config()
TEXT_MODEL = CURRENT_CONFIG["text_model"]
# 系统 prompt 唯一来源：config.json 的 system_prompt
SYSTEM_PROMPT = CURRENT_CONFIG["system_prompt"]


def resolve_input_device(config):
    requested = config.get("input_device")
    if requested in (None, "", "default"):
        return None

    if sd is None:
        print("[AUDIO] sounddevice 未安装，无法解析输入设备（无硬件调试链路可忽略）", flush=True)
        return None

    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[AUDIO] Device query failed: {e}", flush=True)
        return None

    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        index = int(requested)
        if 0 <= index < len(devices):
            return index
        print(f"[AUDIO] Input device index not found: {index}", flush=True)
        return None

    requested_lower = str(requested).lower()
    for idx, dev in enumerate(devices):
        print(f"[AUDIO DEBUG] Index {idx}: {dev.get('name')} (In: {dev.get('max_input_channels')})", flush=True) # DEBUG LINE
        if dev.get("max_input_channels", 0) > 0 and requested_lower in dev.get("name", "").lower():
            return idx

    print(f"[AUDIO] Input device name not found: {requested}", flush=True)
    return None

INPUT_DEVICE_NAME = resolve_input_device(CURRENT_CONFIG)
if INPUT_DEVICE_NAME is not None:
    try:
        device_info = sd.query_devices(INPUT_DEVICE_NAME)
        print(f"[AUDIO] Using input device: {device_info.get('name', INPUT_DEVICE_NAME)}", flush=True)
    except Exception:
        print(f"[AUDIO] Using input device index: {INPUT_DEVICE_NAME}", flush=True)

def choose_input_samplerate(device, preferred=None):
    candidates = []
    if preferred:
        candidates.append(preferred)
    try:
        device_info = sd.query_devices(device)
        print(f"[AUDIO DEBUG] Device Info: {device_info}", flush=True) # DEBUG
        if "default_samplerate" in device_info:
            candidates.append(int(device_info["default_samplerate"]))
    except Exception as e:
        print(f"[AUDIO DEBUG] Query failed: {e}", flush=True)
        pass

    candidates.extend([48000, 44100, 32000, 16000])
    seen = set()
    for rate in candidates:
        if not rate or rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue

    return int(candidates[0]) if candidates else 44100


class BotStates:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    WARMUP = "warmup"
    GREETING = "greeting"
    SLEEP = "sleep"
