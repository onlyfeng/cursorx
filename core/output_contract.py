"""执行结果输出契约

定义 run.py 和 scripts/run_iterate.py 返回值的统一结构，确保所有分支返回一致的字段。

================================================================================
通用结果字段集合 (ResultFields)
================================================================================

所有执行模式的结果都应包含以下核心字段：

| 字段名       | 类型          | 必需 | 说明                                      |
|--------------|---------------|------|-------------------------------------------|
| success      | bool          | 是   | 执行是否成功                              |
| goal         | str           | 是   | 原始任务目标                              |
| mode         | str           | 是   | 执行模式 (cloud/iterate/plan/ask/basic)   |
| error        | Optional[str] | 否   | 错误信息（仅失败时有值）                  |

================================================================================
Cloud 结果字段 (CloudResultFields)
================================================================================

Cloud 模式的完整字段：

| 字段名                | 类型          | 说明                                      |
|-----------------------|---------------|-------------------------------------------|
| success               | bool          | 执行是否成功                              |
| goal                  | str           | 原始任务目标                              |
| mode                  | str           | 固定为 "cloud"                            |
| background            | bool          | 是否后台模式                              |
| output                | Optional[str] | 执行输出（仅成功时有值）                  |
| error                 | Optional[str] | 错误信息（仅失败时有值）                  |
| session_id            | Optional[str] | 会话 ID（用于恢复）                       |
| resume_command        | Optional[str] | 恢复命令                                  |
| files_modified        | list          | 修改的文件列表                            |
| has_ampersand_prefix  | bool          | 语法检测层面，prompt 是否包含 & 前缀      |
| prefix_routed         | bool          | 策略决策层面，& 前缀是否成功触发 Cloud    |
| triggered_by_prefix   | bool          | prefix_routed 的兼容别名                  |
| failure_kind          | Optional[str] | 失败类型枚举值                            |
| retry_after           | Optional[int] | 建议重试等待时间（秒）                    |
| retryable             | bool          | 是否可重试                                |
| cooldown_info         | Optional[dict]| 冷却信息（回退详情，见下方子字段）        |

cooldown_info 子字段（参见 CooldownInfoFields 常量类）：

**字段契约原则**：
- **稳定字段必须存在**：消费方可依赖这些字段始终存在（值可为 None）
- **允许扩展字段**：cooldown_info 允许包含额外字段，消费方应忽略未知字段
- **dedup_key 不属于契约**：消息去重由入口脚本通过 compute_message_dedup_key() 计算

**稳定字段（COOLDOWN_INFO_MINIMUM_STABLE_FIELDS）**：
这些字段在任何 cooldown_info 结构中必须存在（可为 None），是契约稳定保证的核心。

| 子字段名              | 类型              | 必需 | 说明                                      |
|-----------------------|-------------------|------|-------------------------------------------|
| user_message          | str | None        | 是   | 用户可见的回退消息                        |
| kind                  | str | None        | 是   | 失败类型（与 failure_kind 一致）          |
| reason                | str | None        | 是   | 回退原因描述                              |
| retryable             | bool              | 是   | 是否可重试                                |
| retry_after           | int | None        | 是   | 建议重试等待时间（秒）                    |
| in_cooldown           | bool              | 是   | 是否处于冷却期                            |
| remaining_seconds     | float | None      | 是   | 冷却剩余秒数（浮点数以支持精确计时）      |
| failure_count         | int               | 是   | 连续失败次数                              |

**兼容字段（COOLDOWN_INFO_COMPAT_FIELDS）**：
这些字段为向后兼容保留，新代码应使用对应的稳定字段。

| 子字段名              | 类型              | 说明                                      |
|-----------------------|-------------------|-------------------------------------------|
| fallback_reason       | str | None        | reason 的兼容别名（已废弃，使用 reason）  |
| error_type            | str | None        | kind 的兼容映射（旧版分类，使用 kind）    |
| failure_kind          | str | None        | kind 的兼容别名（等同于 kind）            |

**扩展字段（允许额外字段）**：
cooldown_info 结构允许包含额外字段以支持未来扩展，但消费方应忽略未知字段。
常见扩展字段示例：skip_reason（跳过原因）等。以下为当前已知的扩展字段：

| 子字段名              | 类型              | 说明                                      |
|-----------------------|-------------------|-------------------------------------------|
| message_level         | str | None        | 消息级别（"warning" 或 "info"），控制入口脚本输出方式 |

================================================================================
Iterate 结果字段 (IterateResultFields)
================================================================================

Iterate 模式的结果字段：

| 字段名               | 类型          | 说明                                      |
|----------------------|---------------|-------------------------------------------|
| success              | bool          | 执行是否成功                              |
| error                | Optional[str] | 错误信息（仅失败时有值）                  |
| dry_run              | bool          | 是否为 dry-run 模式                       |
| minimal              | bool          | 是否为 minimal 模式                       |
| side_effects         | dict          | 副作用控制策略                            |
| summary              | Optional[str] | 执行摘要                                  |
| goal_length          | Optional[int] | 目标长度                                  |
| cooldown_info        | Optional[dict]| 冷却信息（回退详情，结构同 Cloud）        |
| has_ampersand_prefix | bool          | 语法检测，原始 prompt 是否有 & 前缀       |
| prefix_routed        | bool          | 策略决策，& 前缀是否成功触发 Cloud 模式   |
| triggered_by_prefix  | bool          | prefix_routed 的兼容别名                  |
| requested_mode       | Optional[str] | 用户请求的执行模式（CLI 或 config.yaml）  |
| effective_mode       | Optional[str] | 有效执行模式（经过决策后实际使用）        |
| orchestrator         | Optional[str] | 编排器类型 (mp/basic)                     |

**执行决策字段契约原则**：
- **稳定字段必须存在**：这些字段在所有返回分支（成功/失败/dry-run）中必须存在
- **值可为 None/False**：当决策信息不可用时，字段值可为 None 或默认布尔值
- **语义与 build_execution_decision 一致**：字段语义与 ExecutionDecision 类定义一致

================================================================================
使用方式
================================================================================

    from core.output_contract import (
        # 字段常量集合
        ResultFields,
        CloudResultFields,
        IterateResultFields,
        # cooldown_info 子字段常量
        CooldownInfoFields,
        COOLDOWN_INFO_REQUIRED_FIELDS,
        COOLDOWN_INFO_COMPAT_FIELDS,
        COOLDOWN_INFO_MINIMUM_STABLE_FIELDS,
        COOLDOWN_INFO_EXTENSION_FIELDS,
        COOLDOWN_INFO_ALL_KNOWN_FIELDS,
        # Cloud 结果构建器
        build_cloud_result,
        build_cloud_result_defaults,
        build_cloud_success_result,
        build_cloud_error_result,
        # Iterate 结果构建器
        build_iterate_result_defaults,
        build_iterate_success_result,
        build_iterate_error_result,
    )

    # 使用字段常量确保字段名一致
    result = {
        ResultFields.SUCCESS: True,
        ResultFields.GOAL: "任务描述",
        ResultFields.MODE: "cloud",
    }

    # 使用构建器确保字段完整
    result = build_cloud_success_result(goal="任务", output="结果")

    # ===== cooldown_info 构建方式 =====
    #
    # 【推荐】使用权威函数构建 cooldown_info（生产代码必须使用此方式）
    from core.execution_policy import build_cooldown_info, CloudFailureInfo, CloudFailureKind
    failure_info = CloudFailureInfo(
        kind=CloudFailureKind.NO_KEY,
        message="未设置 CURSOR_API_KEY",
        retryable=False,
        retry_after=None,
    )
    cooldown_info = build_cooldown_info(
        failure_info=failure_info,
        fallback_reason="未设置 CURSOR_API_KEY",
        requested_mode="auto",
        has_ampersand_prefix=False,
        mode_source="config",
    )

    # 【仅用于测试验证】使用 CooldownInfoFields 手动构建（禁止在生产代码中使用）
    # 此示例仅展示如何验证字段完整性，不应直接复制到生产代码
    cooldown_info_for_test = {
        CooldownInfoFields.USER_MESSAGE: "Cloud 不可用，已回退到本地 CLI",
        CooldownInfoFields.KIND: "NO_KEY",
        CooldownInfoFields.REASON: "未设置 CURSOR_API_KEY",
        CooldownInfoFields.RETRYABLE: True,
        CooldownInfoFields.RETRY_AFTER: None,
        CooldownInfoFields.IN_COOLDOWN: False,
        CooldownInfoFields.REMAINING_SECONDS: None,
        CooldownInfoFields.FAILURE_COUNT: 1,
    }

    # 验证 cooldown_info 包含所有必需字段
    missing = COOLDOWN_INFO_REQUIRED_FIELDS - set(cooldown_info_for_test.keys())
    assert not missing, f"缺少必需字段: {missing}"
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

# ============================================================
# 从 contract_fields 模块导入并 re-export CooldownInfoFields 及字段集合常量
# ============================================================
from core.contract_fields import (
    COOLDOWN_INFO_ALL_KNOWN_FIELDS,
    COOLDOWN_INFO_COMPAT_FIELDS,
    COOLDOWN_INFO_EXTENSION_FIELDS,
    COOLDOWN_INFO_MINIMUM_STABLE_FIELDS,
    COOLDOWN_INFO_REQUIRED_FIELDS,
    CooldownInfoFields,
)
from core.execution_policy import CloudFailureKind, compute_message_dedup_key

# 显式 re-export（供 from core.output_contract import ... 使用）
__all__ = [
    # 从 contract_fields 模块 re-export
    "CooldownInfoFields",
    "COOLDOWN_INFO_REQUIRED_FIELDS",
    "COOLDOWN_INFO_COMPAT_FIELDS",
    "COOLDOWN_INFO_MINIMUM_STABLE_FIELDS",
    "COOLDOWN_INFO_EXTENSION_FIELDS",
    "COOLDOWN_INFO_ALL_KNOWN_FIELDS",
    # 本模块定义
    "ResultFields",
    "CloudResultFields",
    "IterateResultFields",
    "build_cloud_result",
    "build_cloud_result_defaults",
    "build_cloud_success_result",
    "build_cloud_error_result",
    "build_iterate_result_defaults",
    "build_iterate_success_result",
    "build_iterate_error_result",
    "MessageOutput",
    "prepare_cooldown_message",
]

# ============================================================
# 通用结果字段常量
# ============================================================


class ResultFields:
    """通用结果字段常量

    所有执行模式结果都应包含的核心字段名。
    使用常量可确保字段名在代码中保持一致。
    """

    SUCCESS = "success"
    GOAL = "goal"
    MODE = "mode"
    ERROR = "error"
    OUTPUT = "output"


class CloudResultFields(ResultFields):
    """Cloud 结果字段常量

    Cloud 执行模式特有的字段名。
    继承自 ResultFields，包含所有通用字段。
    """

    BACKGROUND = "background"
    SESSION_ID = "session_id"
    RESUME_COMMAND = "resume_command"
    FILES_MODIFIED = "files_modified"
    # & 前缀路由相关字段
    HAS_AMPERSAND_PREFIX = "has_ampersand_prefix"
    PREFIX_ROUTED = "prefix_routed"
    TRIGGERED_BY_PREFIX = "triggered_by_prefix"  # 兼容别名
    # 失败信息字段
    FAILURE_KIND = "failure_kind"
    RETRY_AFTER = "retry_after"
    RETRYABLE = "retryable"
    # 冷却信息字段（统一回退信息结构，子字段定义见 CooldownInfoFields）
    COOLDOWN_INFO = "cooldown_info"  # 引用 CooldownInfoFields 契约定义


class IterateResultFields(ResultFields):
    """Iterate 结果字段常量

    Iterate 执行模式特有的字段名。
    继承自 ResultFields，包含所有通用字段。
    """

    DRY_RUN = "dry_run"
    MINIMAL = "minimal"
    SIDE_EFFECTS = "side_effects"
    SUMMARY = "summary"
    GOAL_LENGTH = "goal_length"
    HINT = "hint"
    DRY_RUN_STATS = "dry_run_stats"
    # 冷却信息字段（统一回退信息结构，子字段定义见 CooldownInfoFields）
    # 与 CloudResultFields.COOLDOWN_INFO 指向同一契约定义
    COOLDOWN_INFO = "cooldown_info"
    # 执行决策相关字段（与 build_execution_decision 语义一致）
    # 这些字段在所有返回分支（成功/失败/dry-run）中必须存在
    HAS_AMPERSAND_PREFIX = "has_ampersand_prefix"  # 语法检测：原始 prompt 是否有 & 前缀
    PREFIX_ROUTED = "prefix_routed"  # 策略决策：& 前缀是否成功触发 Cloud 模式
    TRIGGERED_BY_PREFIX = "triggered_by_prefix"  # 兼容别名（prefix_routed 的别名）
    REQUESTED_MODE = "requested_mode"  # 用户请求的执行模式（来自 CLI 或 config.yaml）
    EFFECTIVE_MODE = "effective_mode"  # 有效执行模式（经过决策后实际使用）
    ORCHESTRATOR = "orchestrator"  # 编排器类型 (mp/basic)


def build_cloud_result_defaults(
    goal: str,
    background: bool = False,
    has_ampersand_prefix: bool = False,
    prefix_routed: bool = False,
) -> dict[str, Any]:
    """构建 Cloud 结果的默认字段模板

    Args:
        goal: 原始任务目标
        background: 是否后台模式
        has_ampersand_prefix: 语法检测层面，prompt 是否包含 & 前缀
        prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud

    Returns:
        包含所有字段默认值的字典
    """
    return {
        # 核心字段
        "success": False,
        "goal": goal,
        "mode": "cloud",
        "background": background,
        # 输出/错误字段
        "output": None,
        "error": None,
        # 会话恢复字段
        "session_id": None,
        "resume_command": None,
        # 文件修改信息
        "files_modified": [],
        # 前缀路由相关字段（字段语义说明见模块文档）
        "has_ampersand_prefix": has_ampersand_prefix,
        "prefix_routed": prefix_routed,
        "triggered_by_prefix": prefix_routed,  # 兼容别名，由 prefix_routed 派生
        # 失败信息字段
        "failure_kind": None,
        "retry_after": None,
        "retryable": False,
        # 冷却信息（统一回退信息结构，缺省为 None）
        "cooldown_info": None,
    }


def build_cloud_result(
    goal: str,
    background: bool = False,
    has_ampersand_prefix: bool = False,
    prefix_routed: bool = False,
    *,
    # 可覆盖字段
    success: bool = False,
    output: str | None = None,
    error: str | None = None,
    session_id: str | None = None,
    resume_command: str | None = None,
    files_modified: list | None = None,
    failure_kind: str | None = None,
    retry_after: int | None = None,
    retryable: bool = False,
    cooldown_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建统一的 Cloud 执行结果

    先填充默认字段，再按参数覆盖具体值。

    Args:
        goal: 原始任务目标
        background: 是否后台模式
        has_ampersand_prefix: 语法检测层面，prompt 是否包含 & 前缀
        prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud
        success: 执行是否成功
        output: 执行输出（仅成功时有值）
        error: 错误信息（仅失败时有值）
        session_id: 会话 ID
        resume_command: 恢复命令
        files_modified: 修改的文件列表
        failure_kind: 失败类型（CloudFailureKind 枚举值或字符串）
        retry_after: 建议重试等待时间（秒）
        retryable: 是否可重试
        cooldown_info: 冷却信息字典（回退详情）

    Returns:
        完整的结果字典，包含所有必需字段
    """
    # 从默认模板开始
    result = build_cloud_result_defaults(
        goal=goal,
        background=background,
        has_ampersand_prefix=has_ampersand_prefix,
        prefix_routed=prefix_routed,
    )

    # 覆盖核心字段
    result["success"] = success

    # 覆盖输出/错误字段
    if output is not None:
        result["output"] = output
    if error is not None:
        result["error"] = error

    # 覆盖会话字段
    if session_id is not None:
        result["session_id"] = session_id
        result["resume_command"] = resume_command or f"agent --resume {session_id}"
    elif resume_command is not None:
        result["resume_command"] = resume_command

    # 覆盖文件修改信息
    if files_modified is not None:
        result["files_modified"] = files_modified

    # 覆盖失败信息字段
    if failure_kind is not None:
        # 支持 CloudFailureKind 枚举或字符串
        if isinstance(failure_kind, CloudFailureKind):
            result["failure_kind"] = failure_kind.value
        else:
            result["failure_kind"] = failure_kind
    if retry_after is not None:
        result["retry_after"] = retry_after
    result["retryable"] = retryable

    # 覆盖冷却信息字段
    if cooldown_info is not None:
        result["cooldown_info"] = cooldown_info

    return result


def build_cloud_success_result(
    goal: str,
    output: str,
    session_id: str | None = None,
    files_modified: list | None = None,
    background: bool = False,
    has_ampersand_prefix: bool = False,
    prefix_routed: bool = False,
    failure_kind: str | None = None,
    retry_after: int | None = None,
    cooldown_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建成功的 Cloud 执行结果

    便捷函数，用于成功分支。

    Args:
        goal: 原始任务目标
        output: 执行输出
        session_id: 会话 ID
        files_modified: 修改的文件列表
        background: 是否后台模式
        has_ampersand_prefix: 语法检测层面
        prefix_routed: 策略决策层面
        failure_kind: 失败类型（可能为 None，表示无失败）
        retry_after: 建议重试等待时间（秒）
        cooldown_info: 冷却信息字典（回退详情）

    Returns:
        成功结果字典
    """
    return build_cloud_result(
        goal=goal,
        background=background,
        has_ampersand_prefix=has_ampersand_prefix,
        prefix_routed=prefix_routed,
        success=True,
        output=output,
        session_id=session_id,
        files_modified=files_modified,
        failure_kind=failure_kind,
        retry_after=retry_after,
        cooldown_info=cooldown_info,
    )


def build_cloud_error_result(
    goal: str,
    error: str,
    failure_kind: str,
    session_id: str | None = None,
    background: bool = False,
    has_ampersand_prefix: bool = False,
    prefix_routed: bool = False,
    retry_after: int | None = None,
    retryable: bool = False,
    cooldown_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建失败的 Cloud 执行结果

    便捷函数，用于错误分支。

    Args:
        goal: 原始任务目标
        error: 错误信息
        failure_kind: 失败类型（必需）
        session_id: 会话 ID（如有）
        background: 是否后台模式
        has_ampersand_prefix: 语法检测层面
        prefix_routed: 策略决策层面
        retry_after: 建议重试等待时间（秒），None 表示无需等待
        retryable: 是否可重试
        cooldown_info: 冷却信息字典（回退详情）

    Returns:
        失败结果字典
    """
    return build_cloud_result(
        goal=goal,
        background=background,
        has_ampersand_prefix=has_ampersand_prefix,
        prefix_routed=prefix_routed,
        success=False,
        error=error,
        session_id=session_id,
        failure_kind=failure_kind,
        retry_after=retry_after,
        retryable=retryable,
        cooldown_info=cooldown_info,
    )


# ============================================================
# Iterate 结果构建器
# ============================================================


def build_iterate_result_defaults(
    minimal: bool = False,
    side_effects: dict | None = None,
    *,
    # 执行决策相关字段（与 build_execution_decision 语义一致）
    has_ampersand_prefix: bool = False,
    prefix_routed: bool = False,
    requested_mode: str | None = None,
    effective_mode: str | None = None,
    orchestrator: str | None = None,
) -> dict[str, Any]:
    """构建 Iterate 结果的默认字段模板

    Args:
        minimal: 是否为 minimal 模式
        side_effects: 副作用控制策略字典
        has_ampersand_prefix: 语法检测，原始 prompt 是否有 & 前缀
        prefix_routed: 策略决策，& 前缀是否成功触发 Cloud 模式
        requested_mode: 用户请求的执行模式（来自 CLI 或 config.yaml）
        effective_mode: 有效执行模式（经过决策后实际使用）
        orchestrator: 编排器类型 (mp/basic)

    Returns:
        包含所有字段默认值的字典
    """
    return {
        # 核心字段
        IterateResultFields.SUCCESS: False,
        IterateResultFields.ERROR: None,
        # 模式标记
        IterateResultFields.DRY_RUN: False,
        IterateResultFields.MINIMAL: minimal,
        IterateResultFields.SIDE_EFFECTS: side_effects or {},
        # 可选字段
        IterateResultFields.SUMMARY: None,
        IterateResultFields.GOAL_LENGTH: None,
        IterateResultFields.HINT: None,
        IterateResultFields.DRY_RUN_STATS: None,
        # 冷却信息（统一回退信息结构，缺省为 None）
        IterateResultFields.COOLDOWN_INFO: None,
        # 执行决策相关字段（必须存在，值可为 None/False）
        IterateResultFields.HAS_AMPERSAND_PREFIX: has_ampersand_prefix,
        IterateResultFields.PREFIX_ROUTED: prefix_routed,
        IterateResultFields.TRIGGERED_BY_PREFIX: prefix_routed,  # 兼容别名，由 prefix_routed 派生
        IterateResultFields.REQUESTED_MODE: requested_mode,
        IterateResultFields.EFFECTIVE_MODE: effective_mode,
        IterateResultFields.ORCHESTRATOR: orchestrator,
    }


def build_iterate_success_result(
    minimal: bool = False,
    side_effects: dict | None = None,
    *,
    dry_run: bool = False,
    summary: str | None = None,
    goal_length: int | None = None,
    dry_run_stats: dict | None = None,
    cooldown_info: dict[str, Any] | None = None,
    # 执行决策相关字段
    has_ampersand_prefix: bool = False,
    prefix_routed: bool = False,
    requested_mode: str | None = None,
    effective_mode: str | None = None,
    orchestrator: str | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    """构建成功的 Iterate 执行结果

    便捷函数，用于成功分支。

    Args:
        minimal: 是否为 minimal 模式
        side_effects: 副作用控制策略字典
        dry_run: 是否为 dry-run 模式
        summary: 执行摘要
        goal_length: 目标长度
        dry_run_stats: dry-run 统计信息
        cooldown_info: 冷却信息字典（回退详情）
        has_ampersand_prefix: 语法检测，原始 prompt 是否有 & 前缀
        prefix_routed: 策略决策，& 前缀是否成功触发 Cloud 模式
        requested_mode: 用户请求的执行模式（来自 CLI 或 config.yaml）
        effective_mode: 有效执行模式（经过决策后实际使用）
        orchestrator: 编排器类型 (mp/basic)
        **extra_fields: 其他额外字段

    Returns:
        成功结果字典
    """
    result = build_iterate_result_defaults(
        minimal=minimal,
        side_effects=side_effects,
        has_ampersand_prefix=has_ampersand_prefix,
        prefix_routed=prefix_routed,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        orchestrator=orchestrator,
    )
    result[IterateResultFields.SUCCESS] = True
    result[IterateResultFields.DRY_RUN] = dry_run

    if summary is not None:
        result[IterateResultFields.SUMMARY] = summary
    if goal_length is not None:
        result[IterateResultFields.GOAL_LENGTH] = goal_length
    if dry_run_stats is not None:
        result[IterateResultFields.DRY_RUN_STATS] = dry_run_stats
    if cooldown_info is not None:
        result[IterateResultFields.COOLDOWN_INFO] = cooldown_info

    # 添加额外字段（覆盖默认值）
    result.update(extra_fields)

    return result


def build_iterate_error_result(
    error: str,
    minimal: bool = False,
    side_effects: dict | None = None,
    *,
    hint: str | None = None,
    cooldown_info: dict[str, Any] | None = None,
    # 执行决策相关字段
    has_ampersand_prefix: bool = False,
    prefix_routed: bool = False,
    requested_mode: str | None = None,
    effective_mode: str | None = None,
    orchestrator: str | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    """构建失败的 Iterate 执行结果

    便捷函数，用于错误分支。

    Args:
        error: 错误信息
        minimal: 是否为 minimal 模式
        side_effects: 副作用控制策略字典
        hint: 提示信息
        cooldown_info: 冷却信息字典（回退详情）
        has_ampersand_prefix: 语法检测，原始 prompt 是否有 & 前缀
        prefix_routed: 策略决策，& 前缀是否成功触发 Cloud 模式
        requested_mode: 用户请求的执行模式（来自 CLI 或 config.yaml）
        effective_mode: 有效执行模式（经过决策后实际使用）
        orchestrator: 编排器类型 (mp/basic)
        **extra_fields: 其他额外字段

    Returns:
        失败结果字典
    """
    result = build_iterate_result_defaults(
        minimal=minimal,
        side_effects=side_effects,
        has_ampersand_prefix=has_ampersand_prefix,
        prefix_routed=prefix_routed,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        orchestrator=orchestrator,
    )
    result[IterateResultFields.SUCCESS] = False
    result[IterateResultFields.ERROR] = error

    if hint is not None:
        result[IterateResultFields.HINT] = hint
    if cooldown_info is not None:
        result[IterateResultFields.COOLDOWN_INFO] = cooldown_info

    # 添加额外字段（覆盖默认值）
    result.update(extra_fields)

    return result


# ============================================================
# Cooldown 消息输出工具（纯函数）
# ============================================================


class MessageOutput(NamedTuple):
    """消息输出数据

    封装 cooldown 消息的去重和输出级别信息。
    入口脚本使用此结构实现统一的消息去重和级别选择。

    Attributes:
        user_message: 用户可见的消息文本
        dedup_key: 稳定哈希去重标识（由 compute_message_dedup_key 计算）
        level: 消息级别，"warning" 或 "info"
    """

    user_message: str
    dedup_key: str
    level: Literal["warning", "info"]


def prepare_cooldown_message(
    cooldown_info: dict[str, Any] | None,
) -> MessageOutput | None:
    """准备 cooldown 消息输出（纯函数）

    读取 cooldown_info.user_message、计算 dedup_key、按 message_level 选择输出级别。
    不管理去重集合，不实际打印消息——仅返回准备好的数据供入口脚本使用。

    ================================================================================
    设计原则
    ================================================================================

    1. 纯函数：不修改任何外部状态，不管理去重集合
    2. 单一职责：仅计算输出数据，由入口脚本决定是否打印
    3. 保守默认：message_level 缺失时默认按 "info" 处理

    ================================================================================
    使用方式
    ================================================================================

    入口脚本调用此函数后，使用返回的 MessageOutput 进行去重和打印：

        from core.output_contract import prepare_cooldown_message, IterateResultFields

        msg_output = prepare_cooldown_message(result.get(IterateResultFields.COOLDOWN_INFO))
        if msg_output and msg_output.dedup_key not in _shown_messages:
            _shown_messages.add(msg_output.dedup_key)
            if msg_output.level == "warning":
                print_warning(msg_output.user_message)
            else:
                print_info(msg_output.user_message)

    Args:
        cooldown_info: 冷却信息字典，应包含 user_message 和可选的 message_level 字段

    Returns:
        MessageOutput 元组，包含 user_message、dedup_key 和 level；
        如果 cooldown_info 为 None 或无 user_message 则返回 None
    """
    if not cooldown_info:
        return None

    user_message = cooldown_info.get(CooldownInfoFields.USER_MESSAGE)
    if not user_message:
        return None

    dedup_key = compute_message_dedup_key(user_message)
    # message_level 缺失时默认按 info 处理（与 execution_policy 保守默认一致）
    raw_level = cooldown_info.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
    level: Literal["warning", "info"] = "warning" if raw_level == "warning" else "info"

    return MessageOutput(
        user_message=user_message,
        dedup_key=dedup_key,
        level=level,
    )
