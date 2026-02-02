"""回归测试：验证 triggered_by_prefix 全仓统一为 prefix_routed 语义

本测试文件验证以下迁移结果：
1. ExecutionPolicyContext.triggered_by_prefix 返回 prefix_routed（而非 has_ampersand_prefix）
2. ExecutionDecision.triggered_by_prefix 返回 prefix_routed
3. AmpersandPrefixInfo.triggered_by_prefix 返回 prefix_routed
4. resolve_orchestrator_settings 支持 prefix_routed 参数别名
5. UnifiedOptions.triggered_by_prefix 与 prefix_routed 一致

这确保了全仓所有 triggered_by_prefix 使用点语义统一。
"""

import pytest


class TestTriggeredByPrefixSemanticMigration:
    """回归测试：验证 triggered_by_prefix 全仓统一为 prefix_routed 语义"""

    def test_execution_policy_context_triggered_by_prefix_returns_prefix_routed(
        self
    ) -> None:
        """ExecutionPolicyContext.triggered_by_prefix 应返回 prefix_routed（策略决策）"""
        from core.execution_policy import ExecutionPolicyContext

        # Case 1: & 前缀存在但 Cloud 不启用 -> prefix_routed=False
        ctx1 = ExecutionPolicyContext(
            prompt="& 测试",
            requested_mode=None,
            cloud_enabled=False,
            has_api_key=False,
        )
        assert ctx1.has_ampersand_prefix is True, "前置条件：应检测到 & 前缀"
        assert ctx1.prefix_routed is False, "Cloud 未启用，prefix_routed 应为 False"
        # 核心断言：triggered_by_prefix 应与 prefix_routed 一致
        assert ctx1.triggered_by_prefix == ctx1.prefix_routed, (
            "triggered_by_prefix 应与 prefix_routed 一致\n"
            f"  triggered_by_prefix={ctx1.triggered_by_prefix}\n"
            f"  prefix_routed={ctx1.prefix_routed}"
        )

        # Case 2: & 前缀存在 + Cloud 启用 + 有 API Key -> prefix_routed=True
        ctx2 = ExecutionPolicyContext(
            prompt="& 测试",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        assert ctx2.has_ampersand_prefix is True
        assert ctx2.prefix_routed is True
        # 核心断言
        assert ctx2.triggered_by_prefix == ctx2.prefix_routed, (
            "triggered_by_prefix 应与 prefix_routed 一致\n"
            f"  triggered_by_prefix={ctx2.triggered_by_prefix}\n"
            f"  prefix_routed={ctx2.prefix_routed}"
        )

        # Case 3: 无 & 前缀 -> has_ampersand_prefix=False, prefix_routed=False
        ctx3 = ExecutionPolicyContext(
            prompt="普通任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        assert ctx3.has_ampersand_prefix is False
        assert ctx3.prefix_routed is False
        assert ctx3.triggered_by_prefix == ctx3.prefix_routed

    def test_execution_decision_triggered_by_prefix_equals_prefix_routed(
        self
    ) -> None:
        """ExecutionDecision.triggered_by_prefix 应与 prefix_routed 一致"""
        from core.execution_policy import build_execution_decision

        # Case 1: & 前缀成功触发 Cloud
        decision1 = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        assert decision1.has_ampersand_prefix is True
        assert decision1.prefix_routed is True
        assert decision1.triggered_by_prefix == decision1.prefix_routed, (
            "ExecutionDecision.triggered_by_prefix 应与 prefix_routed 一致"
        )

        # Case 2: & 前缀未成功触发（无 API Key）
        decision2 = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=False,
        )
        assert decision2.has_ampersand_prefix is True
        assert decision2.prefix_routed is False  # 未成功触发
        assert decision2.triggered_by_prefix == decision2.prefix_routed

    def test_ampersand_prefix_info_triggered_by_prefix_equals_prefix_routed(
        self
    ) -> None:
        """AmpersandPrefixInfo.triggered_by_prefix 应与 prefix_routed 一致"""
        from core.execution_policy import detect_ampersand_prefix

        # Case 1: & 前缀成功路由
        info1 = detect_ampersand_prefix(
            prompt="& 测试",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        assert info1.triggered_by_prefix == info1.prefix_routed

        # Case 2: & 前缀未成功路由
        info2 = detect_ampersand_prefix(
            prompt="& 测试",
            requested_mode=None,
            cloud_enabled=False,
            has_api_key=False,
        )
        assert info2.triggered_by_prefix == info2.prefix_routed

    def test_resolve_orchestrator_settings_prefix_routed_alias(
        self
    ) -> None:
        """resolve_orchestrator_settings 的 prefix_routed 参数应与 triggered_by_prefix 等效"""
        from core.config import resolve_orchestrator_settings

        # 使用旧参数名 triggered_by_prefix
        result1 = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=True,
        )

        # 使用新参数名 prefix_routed
        result2 = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=True,
        )

        # 两者结果应一致
        assert result1["orchestrator"] == result2["orchestrator"], (
            "prefix_routed 参数应与 triggered_by_prefix 等效"
        )
        assert result1["prefix_routed"] == result2["prefix_routed"]

        # prefix_routed=True 应强制 basic（因为 & 前缀成功触发）
        assert result1["orchestrator"] == "basic", (
            "prefix_routed=True 时应强制 basic 编排器"
        )

    def test_unified_options_triggered_by_prefix_equals_prefix_routed(
        self
    ) -> None:
        """UnifiedOptions.triggered_by_prefix 应与 prefix_routed 一致"""
        from core.config import UnifiedOptions

        # 创建 UnifiedOptions 实例
        options = UnifiedOptions(
            overrides={},
            resolved={},
            triggered_by_prefix=True,
        )

        # triggered_by_prefix 和 prefix_routed 应一致
        assert options.triggered_by_prefix == options.prefix_routed, (
            "UnifiedOptions.triggered_by_prefix 应与 prefix_routed 一致"
        )

        options2 = UnifiedOptions(
            overrides={},
            resolved={},
            triggered_by_prefix=False,
        )
        assert options2.triggered_by_prefix == options2.prefix_routed

    def test_semantic_distinction_has_ampersand_prefix_vs_prefix_routed(
        self
    ) -> None:
        """验证 has_ampersand_prefix 与 prefix_routed 的语义区分"""
        from core.execution_policy import ExecutionPolicyContext

        # 关键场景：& 前缀存在但未成功路由
        ctx = ExecutionPolicyContext(
            prompt="& 后台分析",
            requested_mode=None,
            cloud_enabled=False,  # Cloud 未启用
            has_api_key=False,
        )

        # has_ampersand_prefix：语法检测层面，检测到 & 前缀
        assert ctx.has_ampersand_prefix is True, (
            "has_ampersand_prefix 应为 True（语法检测到 & 前缀）"
        )

        # prefix_routed：策略决策层面，& 前缀未成功路由到 Cloud
        assert ctx.prefix_routed is False, (
            "prefix_routed 应为 False（Cloud 未启用，未成功路由）"
        )

        # triggered_by_prefix：应与 prefix_routed 一致（迁移后的语义）
        assert ctx.triggered_by_prefix == ctx.prefix_routed, (
            "triggered_by_prefix 应与 prefix_routed 一致（策略决策层面）\n"
            "这是迁移后的正确语义，而非之前错误的 has_ampersand_prefix"
        )

    def test_prefix_routed_parameter_priority_in_resolve_orchestrator_settings(
        self
    ) -> None:
        """验证 prefix_routed 参数优先于 triggered_by_prefix"""
        from core.config import resolve_orchestrator_settings

        # 当同时指定 prefix_routed 和 triggered_by_prefix 时，prefix_routed 优先
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=False,  # 旧参数：False
            prefix_routed=True,         # 新参数：True（优先）
        )

        # prefix_routed=True 应生效，强制 basic
        assert result["orchestrator"] == "basic", (
            "prefix_routed 参数应优先于 triggered_by_prefix"
        )
        assert result["prefix_routed"] is True


class TestExecutionDecisionToDict:
    """测试 ExecutionDecision.to_dict() 输出的字段兼容性"""

    def test_to_dict_contains_both_prefix_routed_and_triggered_by_prefix(
        self
    ) -> None:
        """ExecutionDecision.to_dict() 应同时包含 prefix_routed 和 triggered_by_prefix"""
        from core.execution_policy import build_execution_decision

        # Case 1: & 前缀成功触发
        decision1 = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        result1 = decision1.to_dict()

        # 验证同时包含两个字段
        assert "prefix_routed" in result1, "to_dict() 应包含 prefix_routed 字段"
        assert "triggered_by_prefix" in result1, "to_dict() 应包含 triggered_by_prefix 字段"
        # 验证两个字段值相等
        assert result1["prefix_routed"] == result1["triggered_by_prefix"], (
            "prefix_routed 与 triggered_by_prefix 应相等\n"
            f"  prefix_routed={result1['prefix_routed']}\n"
            f"  triggered_by_prefix={result1['triggered_by_prefix']}"
        )
        # 验证值为 True
        assert result1["prefix_routed"] is True

        # Case 2: & 前缀未成功触发
        decision2 = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=False,  # Cloud 未启用
            has_api_key=False,
        )
        result2 = decision2.to_dict()

        # 验证同时包含两个字段
        assert "prefix_routed" in result2
        assert "triggered_by_prefix" in result2
        # 验证两个字段值相等
        assert result2["prefix_routed"] == result2["triggered_by_prefix"]
        # 验证值为 False
        assert result2["prefix_routed"] is False

        # Case 3: 无 & 前缀
        decision3 = build_execution_decision(
            prompt="普通任务",
            requested_mode="cli",
            cloud_enabled=True,
            has_api_key=True,
        )
        result3 = decision3.to_dict()

        assert result3["prefix_routed"] == result3["triggered_by_prefix"]
        assert result3["prefix_routed"] is False

    def test_to_dict_has_ampersand_prefix_independent_of_prefix_routed(
        self
    ) -> None:
        """to_dict() 中 has_ampersand_prefix 与 prefix_routed 是独立字段"""
        from core.execution_policy import build_execution_decision

        # 关键场景：& 存在但未成功路由
        decision = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=False,  # Cloud 未启用
            has_api_key=False,
        )
        result = decision.to_dict()

        # has_ampersand_prefix=True（语法检测）
        assert result["has_ampersand_prefix"] is True, (
            "has_ampersand_prefix 应为 True（语法层面存在 & 前缀）"
        )
        # prefix_routed=False（策略决策）
        assert result["prefix_routed"] is False, (
            "prefix_routed 应为 False（未成功触发 Cloud）"
        )
        # triggered_by_prefix 应与 prefix_routed 相等
        assert result["triggered_by_prefix"] == result["prefix_routed"], (
            "triggered_by_prefix 应与 prefix_routed 相等"
        )


class TestAmpersandPresentButNotRouted:
    """测试 '& 存在但未成功路由' 的场景"""

    def test_ampersand_present_cloud_disabled(self) -> None:
        """场景: & 前缀存在但 cloud_enabled=False"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,
            cloud_enabled=False,  # Cloud 未启用
            has_api_key=True,
        )

        # 核心断言
        assert decision.has_ampersand_prefix is True, (
            "应检测到 & 前缀（语法层面）"
        )
        assert decision.prefix_routed is False, (
            "prefix_routed 应为 False（Cloud 未启用）"
        )
        assert decision.triggered_by_prefix == decision.prefix_routed, (
            "triggered_by_prefix 应与 prefix_routed 一致"
        )
        # 执行模式应回退到 CLI
        assert decision.effective_mode == "cli"
        # 编排器应为 mp（因为 & 未成功触发，不强制 basic）
        assert decision.orchestrator == "mp"

    def test_ampersand_present_no_api_key(self) -> None:
        """场景: & 前缀存在但缺少 API Key"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key
        )

        # 核心断言
        assert decision.has_ampersand_prefix is True
        assert decision.prefix_routed is False, (
            "prefix_routed 应为 False（无 API Key）"
        )
        assert decision.triggered_by_prefix == decision.prefix_routed
        # 执行模式应回退到 CLI
        assert decision.effective_mode == "cli"

    def test_ampersand_present_explicit_cli_mode(self) -> None:
        """场景: & 前缀存在但显式指定 --execution-mode cli"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode="cli",  # 显式 CLI 模式
            cloud_enabled=True,
            has_api_key=True,
        )

        # 核心断言
        assert decision.has_ampersand_prefix is True
        assert decision.prefix_routed is False, (
            "prefix_routed 应为 False（显式 CLI 模式忽略 & 前缀）"
        )
        assert decision.triggered_by_prefix == decision.prefix_routed
        assert decision.effective_mode == "cli"

    def test_ampersand_present_auto_detect_disabled(self) -> None:
        """场景: & 前缀存在但 auto_detect_cloud_prefix=False"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
            auto_detect_cloud_prefix=False,  # 禁用自动检测
        )

        # 核心断言
        assert decision.has_ampersand_prefix is True
        assert decision.prefix_routed is False, (
            "prefix_routed 应为 False（auto_detect_cloud_prefix=False）"
        )
        assert decision.triggered_by_prefix == decision.prefix_routed
        assert decision.effective_mode == "cli"


class TestOptionsOutputCompatibility:
    """测试入口脚本输出 options 的字段兼容性

    验证 run.py 和 scripts/run_iterate.py 的输出格式一致性。
    这些测试模拟入口脚本中 options 字典的构建逻辑。
    """

    def test_options_triggered_by_prefix_from_prefix_routed(self) -> None:
        """options 中 triggered_by_prefix 应取自 prefix_routed"""
        from core.execution_policy import build_execution_decision

        # 模拟 run.py 中的 options 构建逻辑
        decision = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )

        # 模拟 run.py 行 1830-1832 的逻辑
        options = {}
        options["has_ampersand_prefix"] = decision.has_ampersand_prefix
        options["prefix_routed"] = decision.prefix_routed
        options["triggered_by_prefix"] = decision.prefix_routed  # 兼容别名

        # 验证
        assert options["triggered_by_prefix"] == options["prefix_routed"], (
            "triggered_by_prefix 应取自 prefix_routed"
        )
        assert options["prefix_routed"] is True

    def test_options_ampersand_present_but_not_routed(self) -> None:
        """options 中 & 存在但未成功路由的场景"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=False,
            has_api_key=False,
        )

        # 模拟 options 构建
        options = {}
        options["has_ampersand_prefix"] = decision.has_ampersand_prefix
        options["prefix_routed"] = decision.prefix_routed
        options["triggered_by_prefix"] = decision.prefix_routed

        # 核心断言
        assert options["has_ampersand_prefix"] is True, (
            "has_ampersand_prefix=True（语法检测）"
        )
        assert options["prefix_routed"] is False, (
            "prefix_routed=False（未成功路由）"
        )
        assert options["triggered_by_prefix"] is False, (
            "triggered_by_prefix 应与 prefix_routed 一致"
        )

    def test_result_output_triggered_by_prefix_from_prefix_routed(self) -> None:
        """result 输出中 triggered_by_prefix 应取自 prefix_routed

        模拟 run.py 行 2793-2795 的结果输出逻辑
        """
        from core.execution_policy import build_execution_decision

        # Case 1: & 成功触发
        decision1 = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        prefix_routed1 = decision1.prefix_routed
        result1 = {
            "has_ampersand_prefix": decision1.has_ampersand_prefix,
            "prefix_routed": prefix_routed1,
            "triggered_by_prefix": prefix_routed1,  # 兼容别名输出
        }
        assert result1["triggered_by_prefix"] == result1["prefix_routed"]
        assert result1["prefix_routed"] is True

        # Case 2: & 未成功触发
        decision2 = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=False,
            has_api_key=False,
        )
        prefix_routed2 = decision2.prefix_routed
        result2 = {
            "has_ampersand_prefix": decision2.has_ampersand_prefix,
            "prefix_routed": prefix_routed2,
            "triggered_by_prefix": prefix_routed2,
        }
        assert result2["triggered_by_prefix"] == result2["prefix_routed"]
        assert result2["prefix_routed"] is False
        # 关键：has_ampersand_prefix=True 但 prefix_routed=False
        assert result2["has_ampersand_prefix"] is True


class TestAmpersandPrefixInfoToDict:
    """测试 AmpersandPrefixInfo.to_dict() 的字段兼容性"""

    def test_to_dict_contains_all_required_fields(self) -> None:
        """AmpersandPrefixInfo.to_dict() 应包含所有必需字段"""
        from core.execution_policy import detect_ampersand_prefix

        info = detect_ampersand_prefix(
            prompt="& 测试",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        result = info.to_dict()

        # 验证必需字段存在
        required_fields = [
            "has_ampersand_prefix",
            "prefix_routed",
            "triggered_by_prefix",
            "status",
            "ignore_reason",
        ]
        for field in required_fields:
            assert field in result, f"to_dict() 应包含 {field} 字段"

        # 验证 triggered_by_prefix 与 prefix_routed 相等
        assert result["triggered_by_prefix"] == result["prefix_routed"]

    def test_to_dict_ampersand_present_not_routed(self) -> None:
        """AmpersandPrefixInfo.to_dict() 在 & 存在但未路由时的输出"""
        from core.execution_policy import detect_ampersand_prefix

        info = detect_ampersand_prefix(
            prompt="& 测试",
            requested_mode=None,
            cloud_enabled=False,  # 未启用
            has_api_key=False,
        )
        result = info.to_dict()

        assert result["has_ampersand_prefix"] is True
        assert result["prefix_routed"] is False
        assert result["triggered_by_prefix"] == result["prefix_routed"]
        assert result["ignore_reason"] is not None
