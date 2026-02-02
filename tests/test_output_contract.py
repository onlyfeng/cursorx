"""输出契约测试

验证 core/output_contract.py 定义的契约字段与实际生成结果一致。

重点测试:
1. build_cooldown_info() 返回的字段集合与 COOLDOWN_INFO_ALL_KNOWN_FIELDS 一致
2. CooldownInfo.to_dict() 输出与契约定义匹配
3. build_iterate_*_result() 返回值包含执行决策相关稳定字段
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
    COOLDOWN_INFO_ALL_KNOWN_FIELDS,
    COOLDOWN_INFO_COMPAT_FIELDS,
    COOLDOWN_INFO_EXTENSION_FIELDS,
    COOLDOWN_INFO_MINIMUM_STABLE_FIELDS,
    COOLDOWN_INFO_REQUIRED_FIELDS,
    # Cloud 结果构建器
    CloudResultFields,
    CooldownInfoFields,
    # Iterate 结果构建器
    IterateResultFields,
    build_cloud_error_result,
    build_cloud_result_defaults,
    build_cloud_success_result,
    build_iterate_error_result,
    build_iterate_result_defaults,
    build_iterate_success_result,
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
        assert not missing_required, f"build_cooldown_info() 缺少必需字段: {missing_required}"

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
        assert not missing_compat, f"build_cooldown_info() 缺少兼容字段: {missing_compat}"

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
        assert not missing_extension, f"build_cooldown_info() 缺少扩展字段: {missing_extension}"

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
        assert not missing_fields, f"CooldownInfo.to_dict() 缺少契约字段: {missing_fields}"

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


# ============================================================
# Iterate 结果执行决策字段契约测试
# ============================================================

# Iterate 结果必须包含的执行决策字段（稳定字段）
ITERATE_EXECUTION_DECISION_FIELDS: frozenset[str] = frozenset(
    {
        IterateResultFields.HAS_AMPERSAND_PREFIX,
        IterateResultFields.PREFIX_ROUTED,
        IterateResultFields.TRIGGERED_BY_PREFIX,
        IterateResultFields.REQUESTED_MODE,
        IterateResultFields.EFFECTIVE_MODE,
        IterateResultFields.ORCHESTRATOR,
    }
)
"""Iterate 结果必须包含的执行决策字段

这些字段在所有返回分支（默认/成功/失败/dry-run）中必须存在。
与 build_execution_decision / _build_execution_decision_fields() 的语义一致。
"""


# ============================================================
# Cloud 结果构建器 cooldown_info 键存在性测试
# ============================================================


class TestCloudResultCooldownInfoKeyPresence:
    """Cloud 结果构建器 cooldown_info 键存在性测试

    验证 build_cloud_result_defaults/build_cloud_success_result/build_cloud_error_result
    返回的 dict 始终含 cooldown_info 键：
    - 当不传入时，值为 None
    - 当传入 cooldown_info 时，值按原样透传
    """

    def test_defaults_always_contains_cooldown_info_key(self) -> None:
        """build_cloud_result_defaults() 返回的 dict 始终含 cooldown_info 键"""
        result = build_cloud_result_defaults(goal="测试任务")

        assert CloudResultFields.COOLDOWN_INFO in result, (
            "build_cloud_result_defaults() 返回的 dict 必须包含 cooldown_info 键"
        )

    def test_defaults_cooldown_info_is_none_when_not_passed(self) -> None:
        """build_cloud_result_defaults() 不传入 cooldown_info 时，值为 None"""
        result = build_cloud_result_defaults(goal="测试任务")

        assert result[CloudResultFields.COOLDOWN_INFO] is None, "不传入 cooldown_info 时，值应为 None"

    def test_success_result_always_contains_cooldown_info_key(self) -> None:
        """build_cloud_success_result() 返回的 dict 始终含 cooldown_info 键"""
        result = build_cloud_success_result(goal="测试任务", output="成功输出")

        assert CloudResultFields.COOLDOWN_INFO in result, (
            "build_cloud_success_result() 返回的 dict 必须包含 cooldown_info 键"
        )

    def test_success_result_cooldown_info_is_none_when_not_passed(self) -> None:
        """build_cloud_success_result() 不传入 cooldown_info 时，值为 None"""
        result = build_cloud_success_result(goal="测试任务", output="成功输出")

        assert result[CloudResultFields.COOLDOWN_INFO] is None, "不传入 cooldown_info 时，值应为 None"

    def test_success_result_cooldown_info_passthrough(self) -> None:
        """build_cloud_success_result() 传入 cooldown_info 时，按原样透传"""
        test_cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "测试消息",
            CooldownInfoFields.KIND: "test_kind",
            CooldownInfoFields.REASON: "测试原因",
            CooldownInfoFields.RETRYABLE: True,
            CooldownInfoFields.RETRY_AFTER: 60,
            CooldownInfoFields.IN_COOLDOWN: True,
            CooldownInfoFields.REMAINING_SECONDS: 30.5,
            CooldownInfoFields.FAILURE_COUNT: 2,
            "extra_field": "extra_value",  # 测试扩展字段也被透传
        }

        result = build_cloud_success_result(
            goal="测试任务",
            output="成功输出",
            cooldown_info=test_cooldown_info,
        )

        assert result[CloudResultFields.COOLDOWN_INFO] is test_cooldown_info, (
            "传入的 cooldown_info 应按原样透传（同一对象引用）"
        )
        assert result[CloudResultFields.COOLDOWN_INFO]["extra_field"] == "extra_value", "扩展字段也应被透传"

    def test_error_result_always_contains_cooldown_info_key(self) -> None:
        """build_cloud_error_result() 返回的 dict 始终含 cooldown_info 键"""
        result = build_cloud_error_result(
            goal="测试任务",
            error="测试错误",
            failure_kind="test_failure",
        )

        assert CloudResultFields.COOLDOWN_INFO in result, (
            "build_cloud_error_result() 返回的 dict 必须包含 cooldown_info 键"
        )

    def test_error_result_cooldown_info_is_none_when_not_passed(self) -> None:
        """build_cloud_error_result() 不传入 cooldown_info 时，值为 None"""
        result = build_cloud_error_result(
            goal="测试任务",
            error="测试错误",
            failure_kind="test_failure",
        )

        assert result[CloudResultFields.COOLDOWN_INFO] is None, "不传入 cooldown_info 时，值应为 None"

    def test_error_result_cooldown_info_passthrough(self) -> None:
        """build_cloud_error_result() 传入 cooldown_info 时，按原样透传"""
        test_cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "错误消息",
            CooldownInfoFields.KIND: "rate_limit",
            CooldownInfoFields.REASON: "速率限制",
            CooldownInfoFields.RETRYABLE: True,
            CooldownInfoFields.RETRY_AFTER: 120,
            CooldownInfoFields.IN_COOLDOWN: False,
            CooldownInfoFields.REMAINING_SECONDS: None,
            CooldownInfoFields.FAILURE_COUNT: 1,
        }

        result = build_cloud_error_result(
            goal="测试任务",
            error="测试错误",
            failure_kind="rate_limit",
            cooldown_info=test_cooldown_info,
        )

        assert result[CloudResultFields.COOLDOWN_INFO] is test_cooldown_info, (
            "传入的 cooldown_info 应按原样透传（同一对象引用）"
        )

    @pytest.mark.parametrize(
        "builder_func,builder_kwargs",
        [
            (build_cloud_result_defaults, {"goal": "任务"}),
            (build_cloud_success_result, {"goal": "任务", "output": "输出"}),
            (build_cloud_error_result, {"goal": "任务", "error": "错误", "failure_kind": "test"}),
        ],
        ids=["defaults", "success", "error"],
    )
    def test_all_cloud_builders_contain_cooldown_info_key(
        self,
        builder_func,
        builder_kwargs,
    ) -> None:
        """参数化测试：所有 Cloud 构建器返回值包含 cooldown_info 键"""
        result = builder_func(**builder_kwargs)

        assert CloudResultFields.COOLDOWN_INFO in result, f"{builder_func.__name__}() 返回值必须包含 cooldown_info 键"
        assert result[CloudResultFields.COOLDOWN_INFO] is None, (
            f"{builder_func.__name__}() 不传入 cooldown_info 时，值应为 None"
        )


# ============================================================
# Iterate 结果构建器 cooldown_info 键存在性测试
# ============================================================


class TestIterateResultCooldownInfoKeyPresence:
    """Iterate 结果构建器 cooldown_info 键存在性测试

    验证 build_iterate_result_defaults/build_iterate_success_result/build_iterate_error_result
    返回的 dict 始终含 cooldown_info 键：
    - 当不传入时，值为 None
    - 当传入 cooldown_info 时，值按原样透传
    """

    def test_defaults_always_contains_cooldown_info_key(self) -> None:
        """build_iterate_result_defaults() 返回的 dict 始终含 cooldown_info 键"""
        result = build_iterate_result_defaults()

        assert IterateResultFields.COOLDOWN_INFO in result, (
            "build_iterate_result_defaults() 返回的 dict 必须包含 cooldown_info 键"
        )

    def test_defaults_cooldown_info_is_none_when_not_passed(self) -> None:
        """build_iterate_result_defaults() 不传入 cooldown_info 时，值为 None"""
        result = build_iterate_result_defaults()

        assert result[IterateResultFields.COOLDOWN_INFO] is None, "不传入 cooldown_info 时，值应为 None"

    def test_success_result_always_contains_cooldown_info_key(self) -> None:
        """build_iterate_success_result() 返回的 dict 始终含 cooldown_info 键"""
        result = build_iterate_success_result()

        assert IterateResultFields.COOLDOWN_INFO in result, (
            "build_iterate_success_result() 返回的 dict 必须包含 cooldown_info 键"
        )

    def test_success_result_cooldown_info_is_none_when_not_passed(self) -> None:
        """build_iterate_success_result() 不传入 cooldown_info 时，值为 None"""
        result = build_iterate_success_result()

        assert result[IterateResultFields.COOLDOWN_INFO] is None, "不传入 cooldown_info 时，值应为 None"

    def test_success_result_cooldown_info_passthrough(self) -> None:
        """build_iterate_success_result() 传入 cooldown_info 时，按原样透传"""
        test_cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "迭代成功消息",
            CooldownInfoFields.KIND: "info",
            CooldownInfoFields.REASON: "执行完成",
            CooldownInfoFields.RETRYABLE: False,
            CooldownInfoFields.RETRY_AFTER: None,
            CooldownInfoFields.IN_COOLDOWN: False,
            CooldownInfoFields.REMAINING_SECONDS: None,
            CooldownInfoFields.FAILURE_COUNT: 0,
            "custom_field": 12345,  # 测试扩展字段也被透传
        }

        result = build_iterate_success_result(cooldown_info=test_cooldown_info)

        assert result[IterateResultFields.COOLDOWN_INFO] is test_cooldown_info, (
            "传入的 cooldown_info 应按原样透传（同一对象引用）"
        )
        assert result[IterateResultFields.COOLDOWN_INFO]["custom_field"] == 12345, "扩展字段也应被透传"

    def test_error_result_always_contains_cooldown_info_key(self) -> None:
        """build_iterate_error_result() 返回的 dict 始终含 cooldown_info 键"""
        result = build_iterate_error_result(error="测试错误")

        assert IterateResultFields.COOLDOWN_INFO in result, (
            "build_iterate_error_result() 返回的 dict 必须包含 cooldown_info 键"
        )

    def test_error_result_cooldown_info_is_none_when_not_passed(self) -> None:
        """build_iterate_error_result() 不传入 cooldown_info 时，值为 None"""
        result = build_iterate_error_result(error="测试错误")

        assert result[IterateResultFields.COOLDOWN_INFO] is None, "不传入 cooldown_info 时，值应为 None"

    def test_error_result_cooldown_info_passthrough(self) -> None:
        """build_iterate_error_result() 传入 cooldown_info 时，按原样透传"""
        test_cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "迭代失败消息",
            CooldownInfoFields.KIND: "network",
            CooldownInfoFields.REASON: "网络连接失败",
            CooldownInfoFields.RETRYABLE: True,
            CooldownInfoFields.RETRY_AFTER: 30,
            CooldownInfoFields.IN_COOLDOWN: True,
            CooldownInfoFields.REMAINING_SECONDS: 25.0,
            CooldownInfoFields.FAILURE_COUNT: 3,
        }

        result = build_iterate_error_result(
            error="测试错误",
            cooldown_info=test_cooldown_info,
        )

        assert result[IterateResultFields.COOLDOWN_INFO] is test_cooldown_info, (
            "传入的 cooldown_info 应按原样透传（同一对象引用）"
        )

    @pytest.mark.parametrize(
        "builder_func,builder_kwargs",
        [
            (build_iterate_result_defaults, {}),
            (build_iterate_success_result, {}),
            (build_iterate_success_result, {"dry_run": True}),
            (build_iterate_error_result, {"error": "错误"}),
        ],
        ids=["defaults", "success", "success-dryrun", "error"],
    )
    def test_all_iterate_builders_contain_cooldown_info_key(
        self,
        builder_func,
        builder_kwargs,
    ) -> None:
        """参数化测试：所有 Iterate 构建器返回值包含 cooldown_info 键"""
        result = builder_func(**builder_kwargs)

        assert IterateResultFields.COOLDOWN_INFO in result, f"{builder_func.__name__}() 返回值必须包含 cooldown_info 键"
        assert result[IterateResultFields.COOLDOWN_INFO] is None, (
            f"{builder_func.__name__}() 不传入 cooldown_info 时，值应为 None"
        )


class TestIterateResultExecutionDecisionFields:
    """Iterate 结果执行决策字段契约测试

    验证 build_iterate_*_result() 返回值包含执行决策相关稳定字段。
    这些字段在默认/成功/失败分支都必须存在（值可为 None/False）。
    """

    def test_defaults_contains_all_decision_fields(self) -> None:
        """build_iterate_result_defaults() 返回值包含所有执行决策字段"""
        result = build_iterate_result_defaults()

        result_keys = set(result.keys())
        missing_fields = ITERATE_EXECUTION_DECISION_FIELDS - result_keys
        assert not missing_fields, (
            f"build_iterate_result_defaults() 缺少执行决策字段: {missing_fields}\n返回的字段: {result_keys}"
        )

    def test_success_result_contains_all_decision_fields(self) -> None:
        """build_iterate_success_result() 返回值包含所有执行决策字段"""
        result = build_iterate_success_result()

        result_keys = set(result.keys())
        missing_fields = ITERATE_EXECUTION_DECISION_FIELDS - result_keys
        assert not missing_fields, (
            f"build_iterate_success_result() 缺少执行决策字段: {missing_fields}\n返回的字段: {result_keys}"
        )

    def test_error_result_contains_all_decision_fields(self) -> None:
        """build_iterate_error_result() 返回值包含所有执行决策字段"""
        result = build_iterate_error_result(error="测试错误")

        result_keys = set(result.keys())
        missing_fields = ITERATE_EXECUTION_DECISION_FIELDS - result_keys
        assert not missing_fields, (
            f"build_iterate_error_result() 缺少执行决策字段: {missing_fields}\n返回的字段: {result_keys}"
        )

    def test_success_result_with_decision_fields(self) -> None:
        """build_iterate_success_result() 正确传递执行决策字段值"""
        result = build_iterate_success_result(
            has_ampersand_prefix=True,
            prefix_routed=True,
            requested_mode="auto",
            effective_mode="cloud",
            orchestrator="basic",
        )

        assert result[IterateResultFields.HAS_AMPERSAND_PREFIX] is True
        assert result[IterateResultFields.PREFIX_ROUTED] is True
        assert result[IterateResultFields.TRIGGERED_BY_PREFIX] is True  # 兼容别名
        assert result[IterateResultFields.REQUESTED_MODE] == "auto"
        assert result[IterateResultFields.EFFECTIVE_MODE] == "cloud"
        assert result[IterateResultFields.ORCHESTRATOR] == "basic"

    def test_error_result_with_decision_fields(self) -> None:
        """build_iterate_error_result() 正确传递执行决策字段值"""
        result = build_iterate_error_result(
            error="测试错误",
            has_ampersand_prefix=False,
            prefix_routed=False,
            requested_mode="cli",
            effective_mode="cli",
            orchestrator="mp",
        )

        assert result[IterateResultFields.HAS_AMPERSAND_PREFIX] is False
        assert result[IterateResultFields.PREFIX_ROUTED] is False
        assert result[IterateResultFields.TRIGGERED_BY_PREFIX] is False  # 兼容别名
        assert result[IterateResultFields.REQUESTED_MODE] == "cli"
        assert result[IterateResultFields.EFFECTIVE_MODE] == "cli"
        assert result[IterateResultFields.ORCHESTRATOR] == "mp"

    def test_triggered_by_prefix_equals_prefix_routed(self) -> None:
        """triggered_by_prefix 始终等于 prefix_routed（兼容别名）"""
        # 测试 prefix_routed=True
        result_true = build_iterate_success_result(prefix_routed=True)
        assert result_true[IterateResultFields.TRIGGERED_BY_PREFIX] == result_true[IterateResultFields.PREFIX_ROUTED]

        # 测试 prefix_routed=False
        result_false = build_iterate_success_result(prefix_routed=False)
        assert result_false[IterateResultFields.TRIGGERED_BY_PREFIX] == result_false[IterateResultFields.PREFIX_ROUTED]

    def test_default_values_are_safe(self) -> None:
        """默认值安全：布尔字段默认 False，可选字段默认 None"""
        result = build_iterate_result_defaults()

        # 布尔字段默认 False
        assert result[IterateResultFields.HAS_AMPERSAND_PREFIX] is False
        assert result[IterateResultFields.PREFIX_ROUTED] is False
        assert result[IterateResultFields.TRIGGERED_BY_PREFIX] is False

        # 可选字段默认 None
        assert result[IterateResultFields.REQUESTED_MODE] is None
        assert result[IterateResultFields.EFFECTIVE_MODE] is None
        assert result[IterateResultFields.ORCHESTRATOR] is None

    @pytest.mark.parametrize(
        "builder_func,builder_kwargs",
        [
            (build_iterate_result_defaults, {}),
            (build_iterate_success_result, {}),
            (build_iterate_success_result, {"dry_run": True}),
            (build_iterate_error_result, {"error": "错误"}),
        ],
        ids=[
            "defaults",
            "success",
            "success-dryrun",
            "error",
        ],
    )
    def test_all_branches_contain_decision_fields(
        self,
        builder_func,
        builder_kwargs,
    ) -> None:
        """参数化测试：所有构建器返回值包含执行决策字段"""
        result = builder_func(**builder_kwargs)

        result_keys = set(result.keys())
        missing_fields = ITERATE_EXECUTION_DECISION_FIELDS - result_keys
        assert not missing_fields, f"{builder_func.__name__}({builder_kwargs}) 缺少执行决策字段: {missing_fields}"

    def test_iterate_result_fields_class_covers_decision_fields(self) -> None:
        """IterateResultFields 类常量覆盖所有执行决策字段"""
        # 获取 IterateResultFields 类中的执行决策相关常量
        decision_attrs = {
            "HAS_AMPERSAND_PREFIX": IterateResultFields.HAS_AMPERSAND_PREFIX,
            "PREFIX_ROUTED": IterateResultFields.PREFIX_ROUTED,
            "TRIGGERED_BY_PREFIX": IterateResultFields.TRIGGERED_BY_PREFIX,
            "REQUESTED_MODE": IterateResultFields.REQUESTED_MODE,
            "EFFECTIVE_MODE": IterateResultFields.EFFECTIVE_MODE,
            "ORCHESTRATOR": IterateResultFields.ORCHESTRATOR,
        }
        field_values = set(decision_attrs.values())

        # 验证与 ITERATE_EXECUTION_DECISION_FIELDS 一致
        assert field_values == ITERATE_EXECUTION_DECISION_FIELDS, (
            f"IterateResultFields 决策常量与 ITERATE_EXECUTION_DECISION_FIELDS 不一致\n"
            f"IterateResultFields 常量值: {field_values}\n"
            f"ITERATE_EXECUTION_DECISION_FIELDS: {ITERATE_EXECUTION_DECISION_FIELDS}\n"
            f"差异: {field_values ^ ITERATE_EXECUTION_DECISION_FIELDS}"
        )


# ============================================================
# 典型失败类型的结果构建与契约一致性测试
# ============================================================

# 典型失败类型定义
TYPICAL_FAILURE_KINDS = [
    CloudFailureKind.NO_KEY,
    CloudFailureKind.AUTH,
    CloudFailureKind.RATE_LIMIT,
    CloudFailureKind.NETWORK,
    CloudFailureKind.TIMEOUT,
]
"""典型失败类型列表

覆盖最常见的 Cloud 执行失败场景:
- NO_KEY: 未配置 API Key
- AUTH: 认证失败
- RATE_LIMIT: 速率限制
- NETWORK: 网络连接错误
- TIMEOUT: 请求超时
"""


class TestTypicalFailureKindsCooldownInfoContract:
    """典型失败类型的 cooldown_info 契约测试

    验证所有典型失败类型构造的 CloudFailureInfo 走 build_cooldown_info() 后，
    返回的 cooldown_info 结构满足契约要求。
    """

    @pytest.mark.parametrize(
        "failure_kind",
        TYPICAL_FAILURE_KINDS,
        ids=[fk.value for fk in TYPICAL_FAILURE_KINDS],
    )
    def test_build_cooldown_info_contains_stable_fields(
        self,
        failure_kind: CloudFailureKind,
    ) -> None:
        """各典型失败类型的 cooldown_info 包含所有稳定字段"""
        # 构造 CloudFailureInfo
        failure_info = CloudFailureInfo(
            kind=failure_kind,
            message=f"Test message for {failure_kind.value}",
            retry_after=60 if failure_kind == CloudFailureKind.RATE_LIMIT else None,
            retryable=failure_kind in {CloudFailureKind.RATE_LIMIT, CloudFailureKind.NETWORK, CloudFailureKind.TIMEOUT},
        )

        # 调用 build_cooldown_info
        cooldown_info = build_cooldown_info(failure_info=failure_info)

        # 断言包含所有稳定字段
        missing_fields = COOLDOWN_INFO_MINIMUM_STABLE_FIELDS - set(cooldown_info.keys())
        assert not missing_fields, f"失败类型 {failure_kind.value} 的 cooldown_info 缺少稳定字段: {missing_fields}"

    @pytest.mark.parametrize(
        "failure_kind",
        TYPICAL_FAILURE_KINDS,
        ids=[fk.value for fk in TYPICAL_FAILURE_KINDS],
    )
    def test_build_cooldown_info_kind_field_matches(
        self,
        failure_kind: CloudFailureKind,
    ) -> None:
        """kind 字段值与 CloudFailureKind 枚举值一致"""
        failure_info = CloudFailureInfo(
            kind=failure_kind,
            message=f"Error: {failure_kind.value}",
            retry_after=None,
            retryable=False,
        )

        cooldown_info = build_cooldown_info(failure_info=failure_info)

        assert cooldown_info[CooldownInfoFields.KIND] == failure_kind.value
        # 兼容字段 failure_kind 也应一致
        assert cooldown_info[CooldownInfoFields.FAILURE_KIND] == failure_kind.value


class TestTypicalFailureKindsCloudResultContract:
    """典型失败类型的 Cloud 结果契约测试

    验证各典型失败类型走 build_cloud_error_result() 构建的结果，
    顶层字段与 cooldown_info 子字段一致。
    """

    @pytest.mark.parametrize(
        "failure_kind",
        TYPICAL_FAILURE_KINDS,
        ids=[fk.value for fk in TYPICAL_FAILURE_KINDS],
    )
    def test_cloud_error_result_contains_all_fields(
        self,
        failure_kind: CloudFailureKind,
    ) -> None:
        """Cloud 错误结果包含所有必需字段"""
        # 构造 CloudFailureInfo
        retry_after = 60 if failure_kind == CloudFailureKind.RATE_LIMIT else None
        retryable = failure_kind in {CloudFailureKind.RATE_LIMIT, CloudFailureKind.NETWORK, CloudFailureKind.TIMEOUT}

        failure_info = CloudFailureInfo(
            kind=failure_kind,
            message=f"Error: {failure_kind.value}",
            retry_after=retry_after,
            retryable=retryable,
        )

        # 构建 cooldown_info
        cooldown_info = build_cooldown_info(failure_info=failure_info)

        # 使用 build_cloud_error_result 构建结果
        result = build_cloud_error_result(
            goal="Test task",
            error=failure_info.message,
            failure_kind=failure_kind.value,
            retry_after=retry_after,
            retryable=retryable,
            cooldown_info=cooldown_info,
        )

        # 断言顶层字段存在
        assert result[CloudResultFields.SUCCESS] is False
        assert result[CloudResultFields.GOAL] == "Test task"
        assert result[CloudResultFields.MODE] == "cloud"
        assert result[CloudResultFields.FAILURE_KIND] == failure_kind.value
        assert result[CloudResultFields.RETRYABLE] == retryable
        assert result[CloudResultFields.COOLDOWN_INFO] is not None

    @pytest.mark.parametrize(
        "failure_kind",
        TYPICAL_FAILURE_KINDS,
        ids=[fk.value for fk in TYPICAL_FAILURE_KINDS],
    )
    def test_cloud_result_toplevel_matches_cooldown_info(
        self,
        failure_kind: CloudFailureKind,
    ) -> None:
        """顶层失败字段与 cooldown_info 子字段一致"""
        retry_after = 60 if failure_kind == CloudFailureKind.RATE_LIMIT else None
        retryable = failure_kind in {CloudFailureKind.RATE_LIMIT, CloudFailureKind.NETWORK, CloudFailureKind.TIMEOUT}

        failure_info = CloudFailureInfo(
            kind=failure_kind,
            message=f"Error: {failure_kind.value}",
            retry_after=retry_after,
            retryable=retryable,
        )

        cooldown_info = build_cooldown_info(failure_info=failure_info)

        result = build_cloud_error_result(
            goal="Test task",
            error=failure_info.message,
            failure_kind=failure_kind.value,
            retry_after=retry_after,
            retryable=retryable,
            cooldown_info=cooldown_info,
        )

        # 顶层字段与 cooldown_info 子字段一致性断言
        assert result[CloudResultFields.FAILURE_KIND] == cooldown_info[CooldownInfoFields.KIND]
        assert result[CloudResultFields.RETRY_AFTER] == cooldown_info[CooldownInfoFields.RETRY_AFTER]
        assert result[CloudResultFields.RETRYABLE] == cooldown_info[CooldownInfoFields.RETRYABLE]


class TestTypicalFailureKindsIterateResultContract:
    """典型失败类型的 Iterate 结果契约测试

    验证各典型失败类型走 build_iterate_error_result() 构建的结果，
    cooldown_info 结构满足契约要求。
    """

    @pytest.mark.parametrize(
        "failure_kind",
        TYPICAL_FAILURE_KINDS,
        ids=[fk.value for fk in TYPICAL_FAILURE_KINDS],
    )
    def test_iterate_error_result_with_cooldown_info(
        self,
        failure_kind: CloudFailureKind,
    ) -> None:
        """Iterate 错误结果的 cooldown_info 包含所有稳定字段"""
        retry_after = 60 if failure_kind == CloudFailureKind.RATE_LIMIT else None
        retryable = failure_kind in {CloudFailureKind.RATE_LIMIT, CloudFailureKind.NETWORK, CloudFailureKind.TIMEOUT}

        failure_info = CloudFailureInfo(
            kind=failure_kind,
            message=f"Error: {failure_kind.value}",
            retry_after=retry_after,
            retryable=retryable,
        )

        cooldown_info = build_cooldown_info(failure_info=failure_info)

        result = build_iterate_error_result(
            error=failure_info.message,
            cooldown_info=cooldown_info,
            requested_mode="auto",
            effective_mode="cli",
        )

        # 断言 cooldown_info 存在且包含所有稳定字段
        assert result[IterateResultFields.COOLDOWN_INFO] is not None
        result_cooldown = result[IterateResultFields.COOLDOWN_INFO]
        missing_fields = COOLDOWN_INFO_MINIMUM_STABLE_FIELDS - set(result_cooldown.keys())
        assert not missing_fields, (
            f"失败类型 {failure_kind.value} 的 iterate 结果 cooldown_info 缺少稳定字段: {missing_fields}"
        )


class TestRateLimitRetryAfterContract:
    """RateLimit 的 retry_after 字段契约测试

    验证 retry_after 的传递与夹逼策略不影响契约字段存在性。
    """

    @pytest.mark.parametrize(
        "retry_after_value",
        [None, 0, 30, 60, 120, 300],
        ids=["none", "zero", "30s", "60s", "120s", "300s"],
    )
    def test_retry_after_passed_through_cooldown_info(
        self,
        retry_after_value: int | None,
    ) -> None:
        """retry_after 值正确传递到 cooldown_info"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.RATE_LIMIT,
            message="Rate limit exceeded",
            retry_after=retry_after_value,
            retryable=True,
        )

        cooldown_info = build_cooldown_info(failure_info=failure_info)

        # 断言 retry_after 字段存在（契约要求）
        assert CooldownInfoFields.RETRY_AFTER in cooldown_info, "cooldown_info 必须包含 retry_after 字段"
        # 断言值正确传递
        assert cooldown_info[CooldownInfoFields.RETRY_AFTER] == retry_after_value

    @pytest.mark.parametrize(
        "retry_after_value",
        [None, 0, 30, 60, 120, 300],
        ids=["none", "zero", "30s", "60s", "120s", "300s"],
    )
    def test_retry_after_passed_through_cloud_result(
        self,
        retry_after_value: int | None,
    ) -> None:
        """retry_after 值正确传递到 Cloud 结果顶层字段"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.RATE_LIMIT,
            message="Rate limit exceeded",
            retry_after=retry_after_value,
            retryable=True,
        )

        cooldown_info = build_cooldown_info(failure_info=failure_info)

        result = build_cloud_error_result(
            goal="Test task",
            error=failure_info.message,
            failure_kind=CloudFailureKind.RATE_LIMIT.value,
            retry_after=retry_after_value,
            retryable=True,
            cooldown_info=cooldown_info,
        )

        # 断言顶层 retry_after 字段存在且值正确
        assert CloudResultFields.RETRY_AFTER in result
        assert result[CloudResultFields.RETRY_AFTER] == retry_after_value

        # 断言与 cooldown_info 一致
        assert (
            result[CloudResultFields.RETRY_AFTER]
            == result[CloudResultFields.COOLDOWN_INFO][CooldownInfoFields.RETRY_AFTER]
        )

    def test_rate_limit_all_stable_fields_present_regardless_of_retry_after(self) -> None:
        """无论 retry_after 值如何，所有稳定字段都必须存在"""
        for retry_after_value in [None, 0, 60]:
            failure_info = CloudFailureInfo(
                kind=CloudFailureKind.RATE_LIMIT,
                message="Rate limit exceeded",
                retry_after=retry_after_value,
                retryable=True,
            )

            cooldown_info = build_cooldown_info(failure_info=failure_info)

            # 无论 retry_after 值如何，所有稳定字段都必须存在
            missing_fields = COOLDOWN_INFO_MINIMUM_STABLE_FIELDS - set(cooldown_info.keys())
            assert not missing_fields, (
                f"retry_after={retry_after_value} 时 cooldown_info 缺少稳定字段: {missing_fields}"
            )

    def test_clamping_strategy_does_not_affect_field_existence(self) -> None:
        """夹逼策略（边界值）不影响契约字段存在性

        测试 retry_after 的边界值（极大值、极小值）是否影响字段存在性。
        """
        # 测试极端值
        extreme_values = [
            -1,  # 负值（异常输入）
            0,  # 零值边界
            1,  # 最小正值
            3600,  # 1小时
            86400,  # 1天
            999999,  # 极大值
        ]

        for retry_after_value in extreme_values:
            failure_info = CloudFailureInfo(
                kind=CloudFailureKind.RATE_LIMIT,
                message="Rate limit exceeded",
                retry_after=retry_after_value,
                retryable=True,
            )

            cooldown_info = build_cooldown_info(failure_info=failure_info)

            # 字段存在性不应受夹逼策略影响
            missing_fields = COOLDOWN_INFO_MINIMUM_STABLE_FIELDS - set(cooldown_info.keys())
            assert not missing_fields, f"retry_after={retry_after_value} 时契约字段缺失: {missing_fields}"

            # retry_after 字段值应正确传递（无夹逼）
            assert cooldown_info[CooldownInfoFields.RETRY_AFTER] == retry_after_value

    @pytest.mark.parametrize(
        "retry_after_value,expected_retryable",
        [
            (None, True),  # 无 retry_after 仍可重试
            (0, True),  # 立即可重试
            (60, True),  # 需等待后可重试
        ],
        ids=["no-wait", "immediate", "delayed"],
    )
    def test_retryable_independent_of_retry_after(
        self,
        retry_after_value: int | None,
        expected_retryable: bool,
    ) -> None:
        """retryable 字段独立于 retry_after 值"""
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.RATE_LIMIT,
            message="Rate limit exceeded",
            retry_after=retry_after_value,
            retryable=expected_retryable,
        )

        cooldown_info = build_cooldown_info(failure_info=failure_info)

        assert cooldown_info[CooldownInfoFields.RETRYABLE] == expected_retryable


# ============================================================
# cooldown_info 未知字段容忍测试
# ============================================================


class TestCooldownInfoUnknownFieldsTolerance:
    """验证 cooldown_info 处理逻辑对未知字段的容忍性

    测试场景：
    1. cooldown_info 包含稳定字段 + 额外未知字段
    2. 打印去重/输出选择逻辑只依赖稳定字段
    3. 不会因未知字段导致 KeyError/TypeError
    """

    @pytest.fixture
    def cooldown_info_with_unknown_fields(self) -> dict:
        """构造带未知字段的 cooldown_info

        保留所有稳定字段，同时添加多个未知/扩展字段。
        """

        return {
            # 稳定字段（COOLDOWN_INFO_REQUIRED_FIELDS）
            CooldownInfoFields.USER_MESSAGE: "测试消息：Cloud 不可用",
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.REASON: "未设置 CURSOR_API_KEY",
            CooldownInfoFields.RETRYABLE: True,
            CooldownInfoFields.RETRY_AFTER: 60,
            CooldownInfoFields.IN_COOLDOWN: True,
            CooldownInfoFields.REMAINING_SECONDS: 45.5,
            CooldownInfoFields.FAILURE_COUNT: 2,
            # 兼容字段
            CooldownInfoFields.FALLBACK_REASON: "未设置 CURSOR_API_KEY",
            CooldownInfoFields.ERROR_TYPE: "config_error",
            CooldownInfoFields.FAILURE_KIND: "no_key",
            # 扩展字段
            CooldownInfoFields.MESSAGE_LEVEL: "warning",
            # 未知字段（模拟未来扩展）
            "future_field_1": "future_value_1",
            "future_field_2": 12345,
            "future_field_3": {"nested": "object"},
            "future_field_4": ["list", "of", "items"],
            "future_field_5": None,
            "_internal_debug_info": "debug_data",
            "experimental_feature_flag": True,
        }

    def test_cooldown_info_with_unknown_fields_contains_stable_fields(
        self,
        cooldown_info_with_unknown_fields: dict,
    ) -> None:
        """验证带未知字段的 cooldown_info 仍包含所有稳定字段"""
        cooldown = cooldown_info_with_unknown_fields

        # 断言包含所有必需字段
        missing_fields = COOLDOWN_INFO_REQUIRED_FIELDS - set(cooldown.keys())
        assert not missing_fields, f"cooldown_info 缺少稳定字段: {missing_fields}"

    def test_get_stable_field_with_unknown_fields_no_error(
        self,
        cooldown_info_with_unknown_fields: dict,
    ) -> None:
        """验证使用 .get() 获取稳定字段不会因未知字段出错"""
        cooldown = cooldown_info_with_unknown_fields

        # 模拟打印逻辑中的字段访问方式
        # 这些是 run.py 和 scripts/run_iterate.py 中使用的访问模式
        user_message = cooldown.get(CooldownInfoFields.USER_MESSAGE)
        message_level = cooldown.get(CooldownInfoFields.MESSAGE_LEVEL, "info")

        # 断言不会 KeyError/TypeError
        assert user_message == "测试消息：Cloud 不可用"
        assert message_level == "warning"

    def test_message_level_decision_only_uses_stable_fields(
        self,
        cooldown_info_with_unknown_fields: dict,
    ) -> None:
        """验证消息级别决策只依赖稳定字段，忽略未知字段

        模拟 run.py 和 scripts/run_iterate.py 中的输出选择逻辑：
        - 读取 MESSAGE_LEVEL 决定使用 print_warning 或 print_info
        - 不应访问任何未知字段
        """
        cooldown = cooldown_info_with_unknown_fields

        # 模拟入口脚本中的判断逻辑
        # 参考: run.py 行 2770-2774, scripts/run_iterate.py 行 6088-6092
        message_level = cooldown.get(CooldownInfoFields.MESSAGE_LEVEL, "info")

        # 决策逻辑
        should_use_warning = message_level == "warning"

        # 断言决策只基于 MESSAGE_LEVEL 字段
        assert should_use_warning is True, "message_level='warning' 时应选择 print_warning"

        # 修改 message_level 验证决策变化
        cooldown_copy = cooldown_info_with_unknown_fields.copy()
        cooldown_copy[CooldownInfoFields.MESSAGE_LEVEL] = "info"
        should_use_warning_after = cooldown_copy.get(CooldownInfoFields.MESSAGE_LEVEL, "info") == "warning"
        assert should_use_warning_after is False, "message_level='info' 时应选择 print_info"

    def test_message_level_default_when_missing(self) -> None:
        """验证 MESSAGE_LEVEL 缺失时默认为 'info'"""
        # 构造不含 MESSAGE_LEVEL 的 cooldown_info
        cooldown = {
            CooldownInfoFields.USER_MESSAGE: "测试消息",
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.REASON: "原因",
            CooldownInfoFields.RETRYABLE: False,
            CooldownInfoFields.RETRY_AFTER: None,
            CooldownInfoFields.IN_COOLDOWN: False,
            CooldownInfoFields.REMAINING_SECONDS: None,
            CooldownInfoFields.FAILURE_COUNT: 1,
            # 添加未知字段
            "unknown_field": "some_value",
        }

        # 模拟入口脚本逻辑
        message_level = cooldown.get(CooldownInfoFields.MESSAGE_LEVEL, "info")

        assert message_level == "info", "MESSAGE_LEVEL 缺失时应默认为 'info'"

    def test_dedup_key_computation_with_unknown_fields(
        self,
        cooldown_info_with_unknown_fields: dict,
    ) -> None:
        """验证去重 key 计算只使用 USER_MESSAGE，忽略未知字段"""
        from core.execution_policy import compute_message_dedup_key

        cooldown = cooldown_info_with_unknown_fields

        # 模拟入口脚本中的去重逻辑
        # 参考: run.py 行 2766-2767, scripts/run_iterate.py 行 6085
        if cooldown.get(CooldownInfoFields.USER_MESSAGE):
            msg_key = compute_message_dedup_key(cooldown[CooldownInfoFields.USER_MESSAGE])
            # 断言 key 生成成功
            assert isinstance(msg_key, str)
            assert len(msg_key) > 0

    @pytest.mark.parametrize(
        "unknown_field_value",
        [
            {"deeply": {"nested": {"object": True}}},
            [1, 2, 3, {"nested_in_list": True}],
            lambda x: x,  # 不可哈希的对象
            object(),
        ],
        ids=["nested-dict", "nested-list", "lambda", "object"],
    )
    def test_unknown_field_exotic_values_no_impact(
        self,
        unknown_field_value,
    ) -> None:
        """验证未知字段的奇异值不影响稳定字段访问"""
        cooldown = {
            CooldownInfoFields.USER_MESSAGE: "测试消息",
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.REASON: "原因",
            CooldownInfoFields.RETRYABLE: False,
            CooldownInfoFields.RETRY_AFTER: None,
            CooldownInfoFields.IN_COOLDOWN: False,
            CooldownInfoFields.REMAINING_SECONDS: None,
            CooldownInfoFields.FAILURE_COUNT: 1,
            CooldownInfoFields.MESSAGE_LEVEL: "info",
            # 奇异值的未知字段
            "exotic_unknown_field": unknown_field_value,
        }

        # 模拟入口脚本逻辑（不应触发任何异常）
        user_message = cooldown.get(CooldownInfoFields.USER_MESSAGE)
        message_level = cooldown.get(CooldownInfoFields.MESSAGE_LEVEL, "info")

        assert user_message == "测试消息"
        assert message_level == "info"


# ============================================================
# prepare_cooldown_message 函数测试
# ============================================================


# ============================================================
# cooldown_info 字符串键访问检测测试
# ============================================================


class TestCooldownInfoNoStringKeyAccess:
    """cooldown_info 字符串键访问检测测试

    扫描代码库中与 cooldown_info 直接相关的字符串键访问模式：
    - cooldown_info["xxx"] 或 cooldown_info['xxx']
    - cooldown_info.get("xxx") 或 cooldown_info.get('xxx')
    - result["cooldown_info"] 等顶层字段访问

    对命中结果做 allowlist 过滤，非允许位置一律报错。

    注意：仅检测与 cooldown_info 直接相关的访问模式，
    不检测其他 dict 的 "reason"、"kind" 等通用字段访问。
    """

    # 允许列表：这些文件/路径允许使用字符串键访问
    # - core/output_contract.py: 构建器函数内部赋值
    # - 测试文件中的注释/docstring
    ALLOWLIST_PATTERNS: list[tuple[str, str]] = [
        # 文件路径, 允许原因
        ("core/output_contract.py", "构建器函数内部使用字符串键构建 dict"),
    ]

    # 扫描的目标目录
    SCAN_PATHS: list[str] = [
        "run.py",
        "scripts/",
        "tests/",
        "cursor/",
        "core/",
        "agents/",
        "coordinator/",
    ]

    # 检测的字符串键访问模式（与 cooldown_info 直接相关）
    # 模式设计：仅检测 cooldown_info.xxx 或 cooldown_info[xxx] 形式
    STRING_KEY_PATTERNS: list[tuple[str, str]] = [
        # (pattern, description)
        # cooldown_info 直接字符串键访问
        (r'cooldown_info\[["\']user_message["\']\]', 'cooldown_info["user_message"]'),
        (r'cooldown_info\.get\(["\']user_message["\']\)', 'cooldown_info.get("user_message")'),
        (r'cooldown_info\[["\']message_level["\']\]', 'cooldown_info["message_level"]'),
        (r'cooldown_info\.get\(["\']message_level["\']\)', 'cooldown_info.get("message_level")'),
        (r'cooldown_info\[["\']kind["\']\]', 'cooldown_info["kind"]'),
        (r'cooldown_info\.get\(["\']kind["\']\)', 'cooldown_info.get("kind")'),
        (r'cooldown_info\[["\']reason["\']\]', 'cooldown_info["reason"]'),
        (r'cooldown_info\.get\(["\']reason["\']\)', 'cooldown_info.get("reason")'),
        (r'cooldown_info\[["\']retryable["\']\]', 'cooldown_info["retryable"]'),
        (r'cooldown_info\.get\(["\']retryable["\']\)', 'cooldown_info.get("retryable")'),
        (r'cooldown_info\[["\']retry_after["\']\]', 'cooldown_info["retry_after"]'),
        (r'cooldown_info\.get\(["\']retry_after["\']\)', 'cooldown_info.get("retry_after")'),
        (r'cooldown_info\[["\']failure_kind["\']\]', 'cooldown_info["failure_kind"]'),
        (r'cooldown_info\.get\(["\']failure_kind["\']\)', 'cooldown_info.get("failure_kind")'),
        (r'cooldown_info\[["\']in_cooldown["\']\]', 'cooldown_info["in_cooldown"]'),
        (r'cooldown_info\.get\(["\']in_cooldown["\']\)', 'cooldown_info.get("in_cooldown")'),
        (r'cooldown_info\[["\']remaining_seconds["\']\]', 'cooldown_info["remaining_seconds"]'),
        (r'cooldown_info\.get\(["\']remaining_seconds["\']\)', 'cooldown_info.get("remaining_seconds")'),
        (r'cooldown_info\[["\']failure_count["\']\]', 'cooldown_info["failure_count"]'),
        (r'cooldown_info\.get\(["\']failure_count["\']\)', 'cooldown_info.get("failure_count")'),
        (r'cooldown_info\[["\']fallback_reason["\']\]', 'cooldown_info["fallback_reason"]'),
        (r'cooldown_info\.get\(["\']fallback_reason["\']\)', 'cooldown_info.get("fallback_reason")'),
        (r'cooldown_info\[["\']error_type["\']\]', 'cooldown_info["error_type"]'),
        (r'cooldown_info\.get\(["\']error_type["\']\)', 'cooldown_info.get("error_type")'),
        # 顶层结果的 cooldown_info 字符串键访问（构建器除外）
        (r'result\[["\']cooldown_info["\']\](?!\s*=)', 'result["cooldown_info"]（读取）'),
        (r'\.get\(["\']cooldown_info["\']\)', '.get("cooldown_info")'),
    ]

    # 允许的字符串键访问位置（文件:行号 或 文件:代码片段）
    ALLOWLIST_LOCATIONS: list[tuple[str, str]] = [
        # (文件路径, 允许的代码片段或行号描述)
        # core/output_contract.py 中的构建器赋值语句
        ("core/output_contract.py", 'result["cooldown_info"] = cooldown_info'),
        ("core/output_contract.py", 'result["failure_kind"]'),
        ("core/output_contract.py", 'result["retry_after"]'),
        ("core/output_contract.py", 'result["retryable"]'),
        # 测试文件中的 docstring/注释（非代码）
        ("tests/test_cursor_client.py", 'cooldown_info["user_message"] 结构正确'),
        ("tests/test_cursor_client.py", 'cooldown_info["user_message"] 存在且非空'),
    ]

    def _is_in_allowlist(self, filepath: str, line_content: str) -> bool:
        """检查命中位置是否在允许列表中"""
        import os

        # 获取相对路径
        if os.path.isabs(filepath):
            # 尝试获取相对路径
            try:
                from pathlib import Path

                workspace = Path(__file__).parent.parent
                filepath = str(Path(filepath).relative_to(workspace))
            except ValueError:
                pass

        # 检查文件级别允许
        for allowed_path, _ in self.ALLOWLIST_PATTERNS:
            if filepath.endswith(allowed_path) or allowed_path in filepath:
                return True

        # 检查具体位置允许
        for allowed_file, allowed_snippet in self.ALLOWLIST_LOCATIONS:
            if allowed_file in filepath or filepath.endswith(allowed_file):
                if allowed_snippet in line_content:
                    return True

        # 检查是否是注释或 docstring
        stripped = line_content.strip()
        if stripped.startswith("#"):
            return True
        if stripped.startswith('"""') or stripped.startswith("'''"):
            return True
        if stripped.startswith("-") and '["' in stripped:
            # Markdown 列表项中的示例（docstring 内）
            return True

        # 检查是否是测试类中的模式定义（元组定义）
        # 模式: (r'pattern', 'description')
        if stripped.startswith("(r'") or stripped.startswith('(r"'):
            return True

        # 检查是否是允许列表定义中的元组
        # 模式: ("file.py", 'cooldown_info["xxx"]')
        if stripped.startswith('("') or stripped.startswith("('"):
            return True

        return False

    def _scan_file_for_patterns(
        self,
        filepath: str,
        patterns: list[tuple[str, str]],
    ) -> list[tuple[int, str, str]]:
        """扫描文件中的字符串键访问模式

        Returns:
            list of (line_number, line_content, pattern_description) for violations
        """
        import re
        from pathlib import Path

        violations: list[tuple[int, str, str]] = []

        try:
            content = Path(filepath).read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            return violations

        lines = content.splitlines()
        for line_num, line in enumerate(lines, start=1):
            for pattern, desc in patterns:
                if re.search(pattern, line):
                    if not self._is_in_allowlist(filepath, line):
                        violations.append((line_num, line.strip(), desc))

        return violations

    def _get_workspace_root(self) -> str:
        """获取工作区根目录"""
        from pathlib import Path

        return str(Path(__file__).parent.parent)

    def _collect_python_files(self, path: str) -> list[str]:
        """收集指定路径下的所有 Python 文件"""
        from pathlib import Path

        workspace = Path(self._get_workspace_root())
        target = workspace / path

        if target.is_file():
            return [str(target)] if target.suffix == ".py" else []

        if target.is_dir():
            return [str(f) for f in target.rglob("*.py")]

        return []

    def test_no_string_key_access_in_codebase(self) -> None:
        """扫描代码库，确保没有使用字符串键访问 cooldown_info 相关字段

        这是一个静态代码检查测试，扫描以下路径：
        - run.py
        - scripts/
        - tests/
        - cursor/
        - core/
        - agents/
        - coordinator/

        对于非允许列表中的命中，报告错误并给出命中位置。
        """
        all_violations: list[tuple[str, int, str, str]] = []

        for scan_path in self.SCAN_PATHS:
            python_files = self._collect_python_files(scan_path)

            for filepath in python_files:
                violations = self._scan_file_for_patterns(
                    filepath,
                    self.STRING_KEY_PATTERNS,
                )
                for line_num, line_content, pattern_desc in violations:
                    all_violations.append((filepath, line_num, line_content, pattern_desc))

        if all_violations:
            # 格式化错误报告
            error_lines = [
                "发现以下字符串键访问模式（应使用 CooldownInfoFields 常量）：",
                "",
            ]
            for filepath, line_num, line_content, pattern_desc in all_violations:
                error_lines.append(f"  {filepath}:{line_num}")
                error_lines.append(f"    模式: {pattern_desc}")
                error_lines.append(f"    代码: {line_content}")
                error_lines.append("")

            error_lines.append("请使用 CooldownInfoFields.XXX 替代字符串键访问。")
            error_lines.append("例如: cooldown_info[CooldownInfoFields.USER_MESSAGE]")

            pytest.fail("\n".join(error_lines))

    def test_allowlist_patterns_are_valid(self) -> None:
        """验证允许列表中的文件确实存在"""
        from pathlib import Path

        workspace = Path(self._get_workspace_root())

        for allowed_path, reason in self.ALLOWLIST_PATTERNS:
            full_path = workspace / allowed_path
            assert full_path.exists(), f"允许列表中的文件不存在: {allowed_path}\n原因: {reason}"

    def test_string_key_patterns_are_valid_regex(self) -> None:
        """验证检测模式是有效的正则表达式"""
        import re

        for pattern, desc in self.STRING_KEY_PATTERNS:
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"无效的正则表达式: {pattern} ({desc})\n错误: {e}")


class TestPrepareCooldownMessage:
    """prepare_cooldown_message 函数测试

    验证 prepare_cooldown_message() 的行为：
    - cooldown_info=None 或缺少 user_message → 返回 None
    - message_level 缺失 → level 默认为 info
    - message_level="warning" → level 为 warning
    - message_level 为未知值 → 退化为 info
    - 同一 user_message → dedup_key 稳定一致
    """

    def test_cooldown_info_none_returns_none(self) -> None:
        """cooldown_info=None 时返回 None"""
        from core.output_contract import prepare_cooldown_message

        result = prepare_cooldown_message(None)

        assert result is None, "cooldown_info=None 时 prepare_cooldown_message() 应返回 None"

    def test_cooldown_info_empty_dict_returns_none(self) -> None:
        """cooldown_info 为空字典（缺少 user_message）时返回 None"""
        from core.output_contract import prepare_cooldown_message

        result = prepare_cooldown_message({})

        assert result is None, "cooldown_info 为空字典时应返回 None"

    def test_user_message_missing_returns_none(self) -> None:
        """cooldown_info 缺少 user_message 字段时返回 None"""
        from core.output_contract import prepare_cooldown_message

        cooldown_info = {
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.REASON: "测试原因",
            CooldownInfoFields.MESSAGE_LEVEL: "warning",
        }

        result = prepare_cooldown_message(cooldown_info)

        assert result is None, "缺少 user_message 字段时应返回 None"

    def test_user_message_none_returns_none(self) -> None:
        """user_message 为 None 时返回 None"""
        from core.output_contract import prepare_cooldown_message

        cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: None,
            CooldownInfoFields.KIND: "no_key",
        }

        result = prepare_cooldown_message(cooldown_info)

        assert result is None, "user_message=None 时应返回 None"

    def test_user_message_empty_string_returns_none(self) -> None:
        """user_message 为空字符串时返回 None"""
        from core.output_contract import prepare_cooldown_message

        cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "",
            CooldownInfoFields.KIND: "no_key",
        }

        result = prepare_cooldown_message(cooldown_info)

        assert result is None, "user_message 为空字符串时应返回 None"

    def test_message_level_missing_defaults_to_info(self) -> None:
        """message_level 缺失时 level 默认为 info"""
        from core.output_contract import prepare_cooldown_message

        cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "测试消息",
            CooldownInfoFields.KIND: "no_key",
            # 不包含 MESSAGE_LEVEL 字段
        }

        result = prepare_cooldown_message(cooldown_info)

        assert result is not None
        assert result.level == "info", "message_level 缺失时应默认为 'info'"

    def test_message_level_warning_returns_warning(self) -> None:
        """message_level="warning" 时 level 为 warning"""
        from core.output_contract import prepare_cooldown_message

        cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "警告消息",
            CooldownInfoFields.KIND: "rate_limit",
            CooldownInfoFields.MESSAGE_LEVEL: "warning",
        }

        result = prepare_cooldown_message(cooldown_info)

        assert result is not None
        assert result.level == "warning", "message_level='warning' 时 level 应为 'warning'"

    def test_message_level_info_returns_info(self) -> None:
        """message_level="info" 时 level 为 info"""
        from core.output_contract import prepare_cooldown_message

        cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "信息消息",
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.MESSAGE_LEVEL: "info",
        }

        result = prepare_cooldown_message(cooldown_info)

        assert result is not None
        assert result.level == "info", "message_level='info' 时 level 应为 'info'"

    @pytest.mark.parametrize(
        "unknown_level",
        ["error", "debug", "critical", "ERROR", "WARNING", "Info", "unknown", "", "0", "1"],
        ids=[
            "error",
            "debug",
            "critical",
            "ERROR-upper",
            "WARNING-upper",
            "Info-mixed",
            "unknown",
            "empty",
            "zero",
            "one",
        ],
    )
    def test_message_level_unknown_value_defaults_to_info(
        self,
        unknown_level: str,
    ) -> None:
        """message_level 为未知值时退化为 info"""
        from core.output_contract import prepare_cooldown_message

        cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "测试消息",
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.MESSAGE_LEVEL: unknown_level,
        }

        result = prepare_cooldown_message(cooldown_info)

        assert result is not None
        assert result.level == "info", f"message_level='{unknown_level}' 时应退化为 'info'，实际为 '{result.level}'"

    def test_dedup_key_stable_for_same_user_message(self) -> None:
        """同一 user_message 的 dedup_key 稳定一致"""
        from core.output_contract import prepare_cooldown_message

        user_message = "Cloud 不可用，已回退到本地 CLI"

        cooldown_info_1 = {
            CooldownInfoFields.USER_MESSAGE: user_message,
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.MESSAGE_LEVEL: "info",
        }

        cooldown_info_2 = {
            CooldownInfoFields.USER_MESSAGE: user_message,
            CooldownInfoFields.KIND: "rate_limit",  # 不同的 kind
            CooldownInfoFields.MESSAGE_LEVEL: "warning",  # 不同的 level
            CooldownInfoFields.RETRY_AFTER: 60,  # 额外字段
        }

        result_1 = prepare_cooldown_message(cooldown_info_1)
        result_2 = prepare_cooldown_message(cooldown_info_2)

        assert result_1 is not None
        assert result_2 is not None
        assert result_1.dedup_key == result_2.dedup_key, (
            "同一 user_message 的 dedup_key 应稳定一致，"
            f"但第一次为 '{result_1.dedup_key}'，第二次为 '{result_2.dedup_key}'"
        )

    def test_dedup_key_different_for_different_user_message(self) -> None:
        """不同 user_message 的 dedup_key 应不同"""
        from core.output_contract import prepare_cooldown_message

        cooldown_info_1 = {
            CooldownInfoFields.USER_MESSAGE: "消息 A",
            CooldownInfoFields.KIND: "no_key",
        }

        cooldown_info_2 = {
            CooldownInfoFields.USER_MESSAGE: "消息 B",
            CooldownInfoFields.KIND: "no_key",
        }

        result_1 = prepare_cooldown_message(cooldown_info_1)
        result_2 = prepare_cooldown_message(cooldown_info_2)

        assert result_1 is not None
        assert result_2 is not None
        assert result_1.dedup_key != result_2.dedup_key, "不同 user_message 的 dedup_key 应不同"

    def test_message_output_namedtuple_fields(self) -> None:
        """验证 MessageOutput NamedTuple 包含正确的字段"""
        from core.output_contract import MessageOutput, prepare_cooldown_message

        cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: "测试消息",
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.MESSAGE_LEVEL: "warning",
        }

        result = prepare_cooldown_message(cooldown_info)

        assert result is not None
        assert isinstance(result, MessageOutput)
        # 验证 NamedTuple 字段
        assert result.user_message == "测试消息"
        assert isinstance(result.dedup_key, str)
        assert len(result.dedup_key) > 0
        assert result.level == "warning"

    def test_dedup_key_consistent_across_multiple_calls(self) -> None:
        """验证多次调用 dedup_key 保持一致（幂等性）"""
        from core.output_contract import prepare_cooldown_message

        user_message = "一致性测试消息"
        cooldown_info = {
            CooldownInfoFields.USER_MESSAGE: user_message,
            CooldownInfoFields.KIND: "test",
        }

        # 多次调用
        results = [prepare_cooldown_message(cooldown_info) for _ in range(5)]

        # 验证所有结果一致
        assert all(r is not None for r in results)
        dedup_keys = [r.dedup_key for r in results]  # type: ignore
        assert len(set(dedup_keys)) == 1, f"多次调用的 dedup_key 应一致，但得到: {dedup_keys}"
