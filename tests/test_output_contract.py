"""输出契约测试

验证 core/output_contract.py 定义的契约字段与实际生成结果一致。

重点测试:
1. build_cooldown_info() 返回的字段集合与 COOLDOWN_INFO_ALL_KNOWN_FIELDS 一致
2. CooldownInfo.to_dict() 输出与契约定义匹配
"""
from __future__ import annotations

import pytest

from core.execution_policy import (
    CloudFailureInfo,
    CloudFailureKind,
    CooldownInfo,
    build_cooldown_info,
)
from core.output_contract import (
    CooldownInfoFields,
    COOLDOWN_INFO_REQUIRED_FIELDS,
    COOLDOWN_INFO_COMPAT_FIELDS,
    COOLDOWN_INFO_EXTENSION_FIELDS,
    COOLDOWN_INFO_ALL_KNOWN_FIELDS,
    COOLDOWN_INFO_MINIMUM_STABLE_FIELDS,
)


class TestCooldownInfoContractConsistency:
    """cooldown_info 契约一致性测试

    验证 build_cooldown_info() 和 CooldownInfo.to_dict() 的输出
    与 core/output_contract.py 定义的契约字段集合一致。
    """

    def test_build_cooldown_info_contains_all_known_fields(self) -> None:
        """build_cooldown_info() 返回值至少包含 COOLDOWN_INFO_ALL_KNOWN_FIELDS"""
        # 构造 CloudFailureInfo
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.RATE_LIMIT,
            message="Rate limit exceeded",
            retry_after=60,
            retryable=True,
        )

        # 调用 build_cooldown_info
        result = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason="Test fallback reason",
            requested_mode="auto",
            has_ampersand_prefix=True,
            in_cooldown=True,
            remaining_seconds=30.5,
            failure_count=2,
            mode_source="cli",
        )

        # 验证返回类型为 dict
        assert isinstance(result, dict), "build_cooldown_info() 应返回 dict"

        # 验证至少包含所有已知字段
        result_keys = set(result.keys())
        missing_fields = COOLDOWN_INFO_ALL_KNOWN_FIELDS - result_keys
        assert not missing_fields, (
            f"build_cooldown_info() 返回值缺少契约字段: {missing_fields}\n"
            f"返回的字段: {result_keys}\n"
            f"期望的字段: {COOLDOWN_INFO_ALL_KNOWN_FIELDS}"
        )

    def test_build_cooldown_info_contains_required_fields(self) -> None:
        """build_cooldown_info() 返回值包含所有必需字段（COOLDOWN_INFO_REQUIRED_FIELDS）"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.NO_KEY,
            message="No API key configured",
            retry_after=None,
            retryable=False,
        )

        result = build_cooldown_info(
            failure_info=failure_info,
        )

        result_keys = set(result.keys())
        missing_required = COOLDOWN_INFO_REQUIRED_FIELDS - result_keys
        assert not missing_required, (
            f"build_cooldown_info() 缺少必需字段: {missing_required}"
        )

    def test_build_cooldown_info_contains_compat_fields(self) -> None:
        """build_cooldown_info() 返回值包含兼容字段（COOLDOWN_INFO_COMPAT_FIELDS）"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.AUTH,
            message="Authentication failed",
            retry_after=None,
            retryable=False,
        )

        result = build_cooldown_info(failure_info=failure_info)

        result_keys = set(result.keys())
        missing_compat = COOLDOWN_INFO_COMPAT_FIELDS - result_keys
        assert not missing_compat, (
            f"build_cooldown_info() 缺少兼容字段: {missing_compat}"
        )

    def test_build_cooldown_info_contains_extension_fields(self) -> None:
        """build_cooldown_info() 返回值包含扩展字段（COOLDOWN_INFO_EXTENSION_FIELDS）"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.NETWORK,
            message="Network error",
            retry_after=30,
            retryable=True,
        )

        result = build_cooldown_info(failure_info=failure_info)

        result_keys = set(result.keys())
        missing_extension = COOLDOWN_INFO_EXTENSION_FIELDS - result_keys
        assert not missing_extension, (
            f"build_cooldown_info() 缺少扩展字段: {missing_extension}"
        )

    def test_cooldown_info_to_dict_matches_contract(self) -> None:
        """CooldownInfo.to_dict() 输出字段与契约定义一致"""
        # 直接创建 CooldownInfo 实例
        info = CooldownInfo(
            kind="rate_limit",
            user_message="Speed limit message",
            retryable=True,
            retry_after=45,
            reason="Rate limit exceeded",
            in_cooldown=True,
            remaining_seconds=20.0,
            failure_count=3,
            message_level="warning",
        )

        result = info.to_dict()

        # 验证输出字段集合
        result_keys = set(result.keys())
        missing_fields = COOLDOWN_INFO_ALL_KNOWN_FIELDS - result_keys
        assert not missing_fields, (
            f"CooldownInfo.to_dict() 缺少契约字段: {missing_fields}"
        )

    def test_cooldown_info_fields_class_covers_all_fields(self) -> None:
        """CooldownInfoFields 类常量覆盖所有契约字段"""
        # 获取 CooldownInfoFields 类中所有大写常量（字段名）
        field_attrs = {
            attr: getattr(CooldownInfoFields, attr)
            for attr in dir(CooldownInfoFields)
            if attr.isupper() and not attr.startswith("_")
        }
        field_values = set(field_attrs.values())

        # 验证与 COOLDOWN_INFO_ALL_KNOWN_FIELDS 一致
        assert field_values == COOLDOWN_INFO_ALL_KNOWN_FIELDS, (
            f"CooldownInfoFields 常量与 COOLDOWN_INFO_ALL_KNOWN_FIELDS 不一致\n"
            f"CooldownInfoFields 常量值: {field_values}\n"
            f"COOLDOWN_INFO_ALL_KNOWN_FIELDS: {COOLDOWN_INFO_ALL_KNOWN_FIELDS}\n"
            f"差异: {field_values ^ COOLDOWN_INFO_ALL_KNOWN_FIELDS}"
        )

    def test_minimum_stable_fields_equals_required_fields(self) -> None:
        """COOLDOWN_INFO_MINIMUM_STABLE_FIELDS 等于 COOLDOWN_INFO_REQUIRED_FIELDS"""
        assert COOLDOWN_INFO_MINIMUM_STABLE_FIELDS == COOLDOWN_INFO_REQUIRED_FIELDS, (
            "COOLDOWN_INFO_MINIMUM_STABLE_FIELDS 应等于 COOLDOWN_INFO_REQUIRED_FIELDS"
        )


class TestCooldownInfoFieldValues:
    """cooldown_info 字段值测试

    验证 build_cooldown_info() 返回的字段值语义正确。
    """

    @pytest.mark.parametrize(
        "failure_kind,expect_kind_value",
        [
            (CloudFailureKind.NO_KEY, "no_key"),
            (CloudFailureKind.AUTH, "auth"),
            (CloudFailureKind.RATE_LIMIT, "rate_limit"),
            (CloudFailureKind.NETWORK, "network"),
            (CloudFailureKind.TIMEOUT, "timeout"),
        ],
    )
    def test_kind_field_matches_failure_kind_value(
        self,
        failure_kind: CloudFailureKind,
        expect_kind_value: str,
    ) -> None:
        """kind 字段值与 CloudFailureKind 枚举值一致"""
        failure_info = CloudFailureInfo(
            kind=failure_kind,
            message="Test message",
            retry_after=None,
            retryable=False,
        )

        result = build_cooldown_info(failure_info=failure_info)

        assert result[CooldownInfoFields.KIND] == expect_kind_value
        # 兼容字段 failure_kind 应与 kind 一致
        assert result[CooldownInfoFields.FAILURE_KIND] == expect_kind_value

    def test_user_message_not_empty(self) -> None:
        """user_message 字段值非空"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.RATE_LIMIT,
            message="Rate limit exceeded",
            retry_after=60,
            retryable=True,
        )

        result = build_cooldown_info(failure_info=failure_info)

        user_message = result[CooldownInfoFields.USER_MESSAGE]
        assert user_message is not None, "user_message 不应为 None"
        assert len(user_message) > 0, "user_message 不应为空字符串"

    def test_fallback_reason_equals_reason(self) -> None:
        """fallback_reason 与 reason 值相同（兼容别名）"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.NO_KEY,
            message="No API key",
            retry_after=None,
            retryable=False,
        )

        result = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason="Custom fallback reason",
        )

        assert result[CooldownInfoFields.REASON] == result[CooldownInfoFields.FALLBACK_REASON]

    def test_cooldown_state_fields_passed_through(self) -> None:
        """冷却状态字段正确传递"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.RATE_LIMIT,
            message="Rate limit",
            retry_after=60,
            retryable=True,
        )

        result = build_cooldown_info(
            failure_info=failure_info,
            in_cooldown=True,
            remaining_seconds=45.5,
            failure_count=5,
        )

        assert result[CooldownInfoFields.IN_COOLDOWN] is True
        assert result[CooldownInfoFields.REMAINING_SECONDS] == 45.5
        assert result[CooldownInfoFields.FAILURE_COUNT] == 5

    @pytest.mark.parametrize(
        "mode_source,has_ampersand_prefix,expect_level",
        [
            ("cli", False, "warning"),
            ("cli", True, "warning"),
            ("config", False, "info"),
            ("config", True, "warning"),
            (None, False, "info"),
            (None, True, "warning"),
        ],
    )
    def test_message_level_strategy(
        self,
        mode_source: str | None,
        has_ampersand_prefix: bool,
        expect_level: str,
    ) -> None:
        """message_level 字段值符合策略规则"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.NO_KEY,
            message="No key",
            retry_after=None,
            retryable=False,
        )

        result = build_cooldown_info(
            failure_info=failure_info,
            mode_source=mode_source,
            has_ampersand_prefix=has_ampersand_prefix,
        )

        assert result[CooldownInfoFields.MESSAGE_LEVEL] == expect_level, (
            f"mode_source={mode_source}, has_ampersand_prefix={has_ampersand_prefix}: "
            f"期望 message_level={expect_level}, 实际={result[CooldownInfoFields.MESSAGE_LEVEL]}"
        )
