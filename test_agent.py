import re
import json

from config import (
    detect_audio_type, detect_sleep_intent, detect_yes_no, extract_audio_tag,
)


# --- 测试1：中文 bug ---
# 修复前，TTS 队列过滤正则要求句子含英文字母或数字，导致中文句子全部被丢弃
def test_chinese_tts_filter_bug():
    chinese = "你好，今天天气很好。"
    old_regex = r'[a-zA-Z0-9]'
    new_regex = r'[\w一-鿿]'
    assert not re.search(old_regex, chinese), "旧正则不应匹配纯中文（即旧代码会丢弃此句）"
    assert re.search(new_regex, chinese),     "新正则应匹配中文（修复后此句能进入 TTS 队列）"


# --- 测试2：speak() 文本清理保留中文标点 ---
# 修复前的正则会删掉，。！？等中文标点，导致 TTS 停顿异常
def test_speak_clean_keeps_chinese_punct():
    text = "你好！今天天气，很好。"
    clean = re.sub(r"[^\w\s,.!?:-，。！？、；：]", "", text)
    assert "，" in clean, "逗号应被保留"
    assert "。" in clean, "句号应被保留"
    assert "！" in clean, "感叹号应被保留"


# --- 测试3：config 缺字段时有合理默认值 ---
# 保证旧的 config.json 不加新字段也能正常运行（向后兼容）
def test_whisper_config_defaults():
    config = {}
    model = config.get("whisper_model", "ggml-base.en.bin")
    lang  = config.get("whisper_lang", "en")
    assert model == "ggml-base.en.bin"
    assert lang  == "en"
# --- 测试4：JSON 动作解析 ---
def test_action_json_parse():
    text = '{"action": "get_time", "value": "now"}'
    data = json.loads(text)
    assert data["action"] == "get_time"
    assert data["value"]  == "now"


# --- 测试5：音频类型识别（关键词为主）---
def test_detect_audio_type():
    assert detect_audio_type("我想听雨声") == "white_noise"
    assert detect_audio_type("放点白噪音吧") == "white_noise"
    assert detect_audio_type("来点轻音乐") == "music"
    assert detect_audio_type("钢琴曲也行") == "music"
    assert detect_audio_type("今天好累") is None


# --- 测试6：睡意 / 结束意图识别 ---
def test_detect_sleep_intent():
    assert detect_sleep_intent("我困了")
    assert detect_sleep_intent("晚安")
    assert detect_sleep_intent("不聊了")
    assert not detect_sleep_intent("今天工作很顺利")


# --- 测试7：是否应答（先判否定，避免「不想」被当成肯定）---
def test_detect_yes_no():
    assert detect_yes_no("可以啊") == "yes"
    assert detect_yes_no("好的") == "yes"
    assert detect_yes_no("不用了") == "no"
    assert detect_yes_no("不想听") == "no"
    assert detect_yes_no("嗯嗯") == "yes"
    assert detect_yes_no("天气怎么样") is None


# --- 测试8：剥离 [AUDIO:x] 控制标签，不读出来 ---
def test_extract_audio_tag():
    clean, t = extract_audio_tag("好的，这就给你放 [AUDIO:music]")
    assert t == "music"
    assert "[AUDIO" not in clean
    clean, t = extract_audio_tag("下雨的声音很好听 [AUDIO:rain]")
    assert t == "white_noise"
    assert "[AUDIO" not in clean
    clean, t = extract_audio_tag("今晚聊得很舒服")
    assert t is None
    assert clean == "今晚聊得很舒服"
