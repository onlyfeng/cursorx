"""Cloud 请求检测工具模块 - 权威实现

本模块提供「单条 prompt 是否 Cloud 请求」的**唯一判定函数**与「剥离前缀」的**唯一实现**。
所有需要检测或处理 Cloud 请求的模块都应使用本模块的函数，而非自行实现。

使用模块列表（代理到此实现）:
- run.py: 统一入口脚本（直接导入）
- scripts/run_iterate.py: 自我迭代脚本（直接导入）
- cursor/client.py: CursorAgentClient._is_cloud_request / _strip_cloud_prefix（代理）
- cursor/cloud/client.py: CursorCloudClient.is_cloud_request / strip_cloud_prefix（代理）

Cloud 请求判定规则:
- prompt 以 '&' 开头（允许前导空白）
- '&' 后面必须有实际内容（非空白字符）

边界情况处理:
- None 或空字符串返回 False
- 只有 & 的字符串返回 False（无实际内容）
- 只有空白字符返回 False
- & 后面需要有实际内容才认为是 cloud request
- 多个 & 只识别第一个，后续 & 视为内容

示例:
    >>> is_cloud_request("& 任务")
    True
    >>> is_cloud_request("&任务")
    True
    >>> is_cloud_request("&")
    False
    >>> is_cloud_request("& ")
    False
    >>> is_cloud_request("任务 & 描述")
    False
    >>> is_cloud_request("& 包含 & 符号")
    True
"""

# Cloud 前缀（& 开头表示使用 Cloud 执行）
CLOUD_PREFIX = "&"


def is_cloud_request(prompt: object | None) -> bool:
    """检测是否为 Cloud 请求（以 & 开头）

    这是检测 Cloud 请求的**权威实现**，所有其他模块应代理到此函数。

    判定规则:
    - prompt 以 '&' 开头（允许前导空白）
    - '&' 后面必须有实际内容（非空白字符）

    边界情况处理:
    - None 或空字符串返回 False
    - 只有 & 的字符串返回 False（无实际内容）
    - 只有空白字符返回 False
    - 多个 & 只识别第一个，后续 & 视为内容

    Args:
        prompt: 任务 prompt

    Returns:
        是否为云端请求

    Examples:
        >>> is_cloud_request("& 任务")
        True
        >>> is_cloud_request("&")
        False
        >>> is_cloud_request("任务 & 描述")
        False
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
    content_after_prefix = stripped[len(CLOUD_PREFIX) :].strip()
    return len(content_after_prefix) > 0


def strip_cloud_prefix(prompt: object | None) -> str:
    """去除 Cloud 前缀 &

    这是剥离 Cloud 前缀的**权威实现**，所有其他模块应代理到此函数。

    处理规则:
    - 如果 prompt 以 '&' 开头，去除 '&' 前缀及其后的空白
    - 如果 prompt 不以 '&' 开头，保持原样返回（不做 strip）
    - None 返回空字符串

    Args:
        prompt: 可能带 & 前缀的 prompt

    Returns:
        去除前缀后的 prompt

    Examples:
        >>> strip_cloud_prefix("& 任务")
        '任务'
        >>> strip_cloud_prefix("&任务")
        '任务'
        >>> strip_cloud_prefix("普通任务")
        '普通任务'
    """
    if not prompt or not isinstance(prompt, str):
        return ""

    stripped = prompt.strip()
    if stripped.startswith(CLOUD_PREFIX):
        return stripped[len(CLOUD_PREFIX) :].strip()
    return prompt


def parse_cloud_request(prompt: object | None) -> tuple[bool, str]:
    """解析 Cloud 请求，返回是否为 Cloud 请求及剥离后的 prompt

    这是一个便捷函数，组合了 is_cloud_request 和 strip_cloud_prefix 的功能，
    适用于需要同时判定并剥离的场景。

    Args:
        prompt: 任务 prompt

    Returns:
        元组 (is_cloud, clean_prompt):
        - is_cloud: 是否为 Cloud 请求
        - clean_prompt: 剥离前缀后的 prompt（如果不是 Cloud 请求则返回原 prompt）

    Examples:
        >>> parse_cloud_request("& 任务")
        (True, '任务')
        >>> parse_cloud_request("普通任务")
        (False, '普通任务')
        >>> parse_cloud_request("&")
        (False, '&')
    """
    is_cloud = is_cloud_request(prompt)
    if is_cloud:
        return True, strip_cloud_prefix(prompt)
    # 非 Cloud 请求时，返回原始 prompt（处理 None / 非字符串情况）
    if isinstance(prompt, str):
        return False, prompt
    return False, ""
