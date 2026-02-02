"""Cooldown 信息字段常量定义

最小依赖模块，仅包含 CooldownInfoFields 类与字段集合常量。
此模块不依赖 execution_policy，避免循环导入问题。

其他模块应从此处导入字段常量：
- core/output_contract.py: re-export 供外部使用
- core/execution_policy.py: CooldownInfo.to_dict() 使用常量
"""

from __future__ import annotations


class CooldownInfoFields:
    """cooldown_info 子字段常量

    定义 cooldown_info 结构中的所有字段名。
    CloudResultFields 和 IterateResultFields 的 COOLDOWN_INFO 字段
    均指向同一 cooldown_info 契约定义。

    字段分类：
    - 稳定字段（MINIMUM_STABLE）：必须存在的核心字段
    - 兼容字段（COMPAT）：向后兼容保留的废弃字段
    - 扩展字段：允许额外字段，消费方应忽略未知字段

    **消费逻辑规范**:
    - 消费逻辑应优先使用稳定字段常量（KIND/REASON/USER_MESSAGE）
    - 对可能缺失的字段使用 .get(..., default) 模式
    - 测试断言应使用 CooldownInfoFields 常量而非字符串字面量
    """

    # 稳定字段（必须存在）
    USER_MESSAGE = "user_message"
    KIND = "kind"
    REASON = "reason"
    RETRYABLE = "retryable"
    RETRY_AFTER = "retry_after"
    IN_COOLDOWN = "in_cooldown"
    REMAINING_SECONDS = "remaining_seconds"
    FAILURE_COUNT = "failure_count"

    # 兼容字段（向后兼容，新代码应使用稳定字段）
    FALLBACK_REASON = "fallback_reason"  # reason 的兼容别名
    ERROR_TYPE = "error_type"  # kind 的兼容映射（旧版分类）
    FAILURE_KIND = "failure_kind"  # kind 的兼容别名

    # 扩展字段（可选，用于未来扩展）
    MESSAGE_LEVEL = "message_level"  # 消息级别（"warning" 或 "info"）
    SKIP_REASON = "skip_reason"  # 跳过 Cloud 的原因


# ============================================================
# cooldown_info 字段集合常量
# ============================================================

COOLDOWN_INFO_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        CooldownInfoFields.USER_MESSAGE,
        CooldownInfoFields.KIND,
        CooldownInfoFields.REASON,
        CooldownInfoFields.RETRYABLE,
        CooldownInfoFields.RETRY_AFTER,
        CooldownInfoFields.IN_COOLDOWN,
        CooldownInfoFields.REMAINING_SECONDS,
        CooldownInfoFields.FAILURE_COUNT,
    }
)
"""cooldown_info 必需字段集合

这些字段在 cooldown_info 结构中必须存在（值可为 None）。
用于验证 cooldown_info 结构的完整性。
"""

COOLDOWN_INFO_COMPAT_FIELDS: frozenset[str] = frozenset(
    {
        CooldownInfoFields.FALLBACK_REASON,
        CooldownInfoFields.ERROR_TYPE,
        CooldownInfoFields.FAILURE_KIND,
    }
)
"""cooldown_info 兼容字段集合

这些字段为向后兼容保留，新代码应使用对应的稳定字段。
- fallback_reason → 使用 reason
- error_type → 使用 kind（旧版分类映射）
- failure_kind → 使用 kind（等同别名）
"""

COOLDOWN_INFO_MINIMUM_STABLE_FIELDS: frozenset[str] = COOLDOWN_INFO_REQUIRED_FIELDS
"""cooldown_info 最小稳定字段集合

契约稳定保证的核心字段集。等同于 COOLDOWN_INFO_REQUIRED_FIELDS。
消费方可依赖这些字段始终存在于 cooldown_info 结构中。
"""

COOLDOWN_INFO_EXTENSION_FIELDS: frozenset[str] = frozenset(
    {
        CooldownInfoFields.MESSAGE_LEVEL,
        CooldownInfoFields.SKIP_REASON,
    }
)
"""cooldown_info 扩展字段集合

当前已知的扩展字段。cooldown_info 允许包含额外字段以支持未来扩展，
消费方应忽略未知字段。常见扩展字段示例：skip_reason（跳过原因）等。
"""

COOLDOWN_INFO_ALL_KNOWN_FIELDS: frozenset[str] = (
    COOLDOWN_INFO_REQUIRED_FIELDS | COOLDOWN_INFO_COMPAT_FIELDS | COOLDOWN_INFO_EXTENSION_FIELDS
)
"""cooldown_info 所有已知字段集合

包含必需字段、兼容字段和扩展字段的并集。
用于文档生成和完整性检查。
"""
