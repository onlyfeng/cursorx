"""Cloud 请求检测工具模块

提供统一的 Cloud 请求检测和处理函数，被以下模块使用：
- run.py: 统一入口脚本
- scripts/run_iterate.py: 自我迭代脚本
- cursor/client.py: Cursor Agent 客户端

边界情况处理:
- None 或空字符串返回 False
- 只有 & 的字符串返回 False（无实际内容）
- 只有空白字符返回 False
- & 后面需要有实际内容才认为是 cloud request
"""

# Cloud 前缀（& 开头表示使用 Cloud 执行）
CLOUD_PREFIX = "&"


def is_cloud_request(prompt: str) -> bool:
    """检测是否为 Cloud 请求（以 & 开头）

    边界情况处理:
    - None 或空字符串返回 False
    - 只有 & 的字符串返回 False（无实际内容）
    - 只有空白字符返回 False
    - & 后面需要有实际内容才认为是 cloud request

    Args:
        prompt: 任务 prompt

    Returns:
        是否为云端请求
    """
    # 处理 None 和非字符串类型
    if not prompt or not isinstance(prompt, str):
        return False

    stripped = prompt.strip()

    # 空字符串
    if not stripped:
        return False

    # 检查是否以 & 开头
    if not stripped.startswith(CLOUD_PREFIX):
        return False

    # 确保 & 后面有实际内容（不只是空白）
    content_after_prefix = stripped[len(CLOUD_PREFIX):].strip()
    return len(content_after_prefix) > 0


def strip_cloud_prefix(prompt: str) -> str:
    """去除 Cloud 前缀 &

    Args:
        prompt: 可能带 & 前缀的 prompt

    Returns:
        去除前缀后的 prompt
    """
    if not prompt or not isinstance(prompt, str):
        return prompt or ""

    stripped = prompt.strip()
    if stripped.startswith(CLOUD_PREFIX):
        return stripped[len(CLOUD_PREFIX):].strip()
    return prompt
