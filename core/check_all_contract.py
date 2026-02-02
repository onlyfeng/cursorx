"""check_all.sh JSON 输出契约定义

轻量模块，仅包含常量与简单校验函数，无重依赖。
用于 check_all.sh --json 输出的 schema 定义和验证。

使用方:
- tests/test_check_all_json_contract.py
- tests/test_ci_summary_renderer.py
- scripts/render_check_all_summary.py
"""

from __future__ import annotations

from typing import Any

# ==================== 状态常量 ====================

# 有效的检查状态枚举
VALID_STATUSES = ("pass", "fail", "warn", "skip", "info")

# 状态对应的 emoji 映射
STATUS_EMOJI_MAP = {
    "pass": "✅",
    "fail": "❌",
    "warn": "⚠️",
    "skip": "⏭️",
    "info": "ℹ️",
}

# 未知状态的默认 emoji
DEFAULT_STATUS_EMOJI = "❓"


# ==================== 字段定义 ====================

# 顶层必需字段及其类型
REQUIRED_FIELDS = {
    "success": bool,
    "exit_code": int,
    "summary": dict,
    "checks": list,
}

# 顶层可选字段及其类型
OPTIONAL_FIELDS = {
    "ci_mode": bool,
    "fail_fast": bool,
    "full_check": bool,
    "diagnose_hang": bool,
    "timeout_backend": str,
    "log_dir": str,
    "project_root": str,
    "timestamp": str,
    "durations": list,
}

# summary 子结构字段
SUMMARY_FIELDS = {
    "passed": int,
    "failed": int,
    "warnings": int,
    "skipped": int,
    "total": int,
}

# check 项必需字段
CHECK_REQUIRED_FIELDS = {
    "section": str,
    "name": str,
    "status": str,
}

# check 项可选字段
CHECK_OPTIONAL_FIELDS = {
    "message": str,
    "duration_ms": int,
    "log_file": str,
    "command": str,
    "last_test": str,
}

# duration 项结构
DURATION_FIELDS = {
    "name": str,
    "duration_ms": int,
}


# ==================== 工具函数 ====================


def status_emoji(status: str) -> str:
    """获取状态对应的 emoji

    Args:
        status: 状态字符串 (pass/fail/warn/skip/info)

    Returns:
        对应的 emoji 字符串，未知状态返回 ❓
    """
    return STATUS_EMOJI_MAP.get(status, DEFAULT_STATUS_EMOJI)


def get_expected_schema() -> dict[str, Any]:
    """返回 check_all.sh --json 输出的预期 schema

    这是 output_json_result 函数输出的契约定义。

    Returns:
        完整的 schema 定义字典
    """
    return {
        "required_fields": REQUIRED_FIELDS.copy(),
        "optional_fields": OPTIONAL_FIELDS.copy(),
        "summary_fields": SUMMARY_FIELDS.copy(),
        "check_required_fields": CHECK_REQUIRED_FIELDS.copy(),
        "check_optional_fields": CHECK_OPTIONAL_FIELDS.copy(),
        "valid_statuses": list(VALID_STATUSES),
        "duration_fields": DURATION_FIELDS.copy(),
    }


def validate_json_against_schema(data: dict[str, Any], schema: dict[str, Any] | None = None) -> list[str]:
    """验证 JSON 数据是否符合 schema

    Args:
        data: 要验证的 JSON 数据
        schema: 预期的 schema 定义，为 None 时使用默认 schema

    Returns:
        错误消息列表，空列表表示验证通过
    """
    if schema is None:
        schema = get_expected_schema()

    errors = []

    # 验证顶层必需字段
    for field, expected_type in schema["required_fields"].items():
        if field not in data:
            errors.append(f"缺少必需字段: {field}")
        elif not isinstance(data[field], expected_type):
            errors.append(f"字段 {field} 类型错误: 期望 {expected_type.__name__}, 实际 {type(data[field]).__name__}")

    # 验证顶层可选字段类型（如果存在）
    for field, expected_type in schema["optional_fields"].items():
        if field in data and data[field] is not None and not isinstance(data[field], expected_type):
            errors.append(
                f"可选字段 {field} 类型错误: 期望 {expected_type.__name__}, 实际 {type(data[field]).__name__}"
            )

    # 验证 summary 结构
    if "summary" in data and isinstance(data["summary"], dict):
        summary = data["summary"]
        for field, expected_type in schema["summary_fields"].items():
            if field not in summary:
                errors.append(f"summary 缺少字段: {field}")
            elif not isinstance(summary[field], expected_type):
                errors.append(
                    f"summary.{field} 类型错误: 期望 {expected_type.__name__}, 实际 {type(summary[field]).__name__}"
                )

    # 验证 checks 数组
    if "checks" in data and isinstance(data["checks"], list):
        for i, check in enumerate(data["checks"]):
            if not isinstance(check, dict):
                errors.append(f"checks[{i}] 应为 dict，实际为 {type(check).__name__}")
                continue

            # 验证必需字段
            for field, expected_type in schema["check_required_fields"].items():
                if field not in check:
                    errors.append(f"checks[{i}] 缺少必需字段: {field}")
                elif not isinstance(check[field], expected_type):
                    errors.append(
                        f"checks[{i}].{field} 类型错误: 期望 {expected_type.__name__}, "
                        f"实际 {type(check[field]).__name__}"
                    )

            # 验证 status 值
            if "status" in check and check["status"] not in schema["valid_statuses"]:
                errors.append(f"checks[{i}].status 值无效: {check['status']}, 允许值: {schema['valid_statuses']}")

            # 验证可选字段类型（如果存在）
            for field, expected_type in schema["check_optional_fields"].items():
                if field in check and check[field] is not None and not isinstance(check[field], expected_type):
                    errors.append(
                        f"checks[{i}].{field} 类型错误: 期望 {expected_type.__name__}, "
                        f"实际 {type(check[field]).__name__}"
                    )

    # 验证 durations 数组
    if "durations" in data and isinstance(data["durations"], list):
        for i, duration in enumerate(data["durations"]):
            if not isinstance(duration, dict):
                errors.append(f"durations[{i}] 应为 dict，实际为 {type(duration).__name__}")
                continue

            for field, expected_type in schema["duration_fields"].items():
                if field not in duration:
                    errors.append(f"durations[{i}] 缺少字段: {field}")
                elif not isinstance(duration[field], expected_type):
                    errors.append(
                        f"durations[{i}].{field} 类型错误: 期望 {expected_type.__name__}, "
                        f"实际 {type(duration[field]).__name__}"
                    )

    return errors
