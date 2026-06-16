# =========================================================================
#  Be More Agent · 提示词
#  从 agent.py 抽出。档2: 每状态窄 prompt + few-shot 范例后续在此扩展。
# =========================================================================

from config import CURRENT_CONFIG

# --- SYSTEM PROMPT ---
# 权威值是 config.json 的 `system_prompt`。下面的 BASE_SYSTEM_PROMPT 仅作降级兜底：
# 只在 config.json 缺失 / 损坏 / 没有 system_prompt 键时启用。
# 它【允许与 config.json 漂移、无需同步】——改人设请改 config.json，不要改这里。
BASE_SYSTEM_PROMPT = """你是一个睡前陪伴机器人，帮用户在睡前梳理情绪。
说话温和、简短，每次回应不超过两句话。
只负责接住用户当下这一句，不出主意、不深挖、不在睡前帮用户解决烦心事。"""

SYSTEM_PROMPT = CURRENT_CONFIG.get("system_prompt", BASE_SYSTEM_PROMPT) + "\n\n" + CURRENT_CONFIG.get("system_prompt_extras", "")

# =========================================================================
#  档2: 睡前情绪梳理 flow 专用 prompt 工厂
#  每次返回完整 system prompt（含 base + extras + 轮次信息），
#  供 flow.py 的 _call_llm 使用。
# =========================================================================


def get_chat_prompt(round_num, max_rounds=5):
    """
    返回第 N 轮对话的 system prompt（含收尾引导）。

    参数:
        round_num:  当前轮次（从 0 开始）
        max_rounds: 最大对话轮次

    返回:
        完整 system prompt 字符串
    """
    lines = [SYSTEM_PROMPT]

    # 越接近结束，越明确提示收尾
    if round_num >= max_rounds - 1:
        lines.append("\n【注意】这是本轮最后一次交流，请自然收尾，告诉用户准备放松休息。")
    elif round_num >= max_rounds - 2:
        lines.append("\n【注意】对话接近尾声，可以开始引导用户放松了。")

    lines.append(f"\n（当前第 {round_num+1}/{max_rounds} 轮）")
    return "".join(lines)


def get_transition_prompt():
    """
    返回过渡步骤的 prompt（TTS 说一句过渡词，不需要 LLM 参与）。
    此处保留函数签名，未来可扩展为 LLM 生成个性化过渡语。
    """
    # 当前固定使用预合成缓存，无需 LLM
    return ""


def get_boot_prompt():
    """
    返回开机问候语。当前固定使用预合成缓存，无需 LLM。
    未来可扩展为根据日期/天气生成个性化问候。
    """
    return ""
