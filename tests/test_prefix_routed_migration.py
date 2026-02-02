"""回归测试：验证 triggered_by_prefix 全仓统一为 prefix_routed 语义

本测试文件验证以下迁移结果：
1. ExecutionPolicyContext.triggered_by_prefix 返回 prefix_routed（而非 has_ampersand_prefix）
2. ExecutionDecision.triggered_by_prefix 返回 prefix_routed
3. AmpersandPrefixInfo.triggered_by_prefix 返回 prefix_routed
4. resolve_orchestrator_settings 支持 prefix_routed 参数别名
5. UnifiedOptions.triggered_by_prefix 与 prefix_routed 一致

这确保了全仓所有 triggered_by_prefix 使用点语义统一。

================================================================================
迁移状态：✅ 完成
================================================================================

静态检查结果（由 TestResolveEffectiveExecutionModeNoTriggeredByPrefix 验证）：
- resolve_effective_execution_mode() 调用点：全部使用 has_ampersand_prefix 参数
- resolve_effective_execution_mode_full() 调用点：全部使用 has_ampersand_prefix 参数
- 源代码目录（core, cursor, agents, coordinator, scripts）：无 triggered_by_prefix= 调用

下一步计划（破坏性变更版本）：
1. 从 resolve_effective_execution_mode() 签名中移除 triggered_by_prefix 参数
2. 从 resolve_effective_execution_mode_full() 签名中移除 triggered_by_prefix 参数
3. 删除 test_deprecation_warning_triggered 和 test_deprecation_warning_full_triggered 测试
4. 更新 AGENTS.md 文档中的相关说明
================================================================================
"""

import pytest


class TestTriggeredByPrefixSemanticMigration:
    """回归测试：验证 triggered_by_prefix 全仓统一为 prefix_routed 语义"""

    def test_execution_policy_context_triggered_by_prefix_returns_prefix_routed(self) -> None:
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

    def test_execution_decision_triggered_by_prefix_equals_prefix_routed(self) -> None:
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

    def test_ampersand_prefix_info_triggered_by_prefix_equals_prefix_routed(self) -> None:
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

    def test_resolve_orchestrator_settings_prefix_routed_alias(self) -> None:
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
        assert result1["orchestrator"] == result2["orchestrator"], "prefix_routed 参数应与 triggered_by_prefix 等效"
        assert result1["prefix_routed"] == result2["prefix_routed"]

        # prefix_routed=True 应强制 basic（因为 & 前缀成功触发）
        assert result1["orchestrator"] == "basic", "prefix_routed=True 时应强制 basic 编排器"

    def test_unified_options_triggered_by_prefix_equals_prefix_routed(self) -> None:
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

    def test_semantic_distinction_has_ampersand_prefix_vs_prefix_routed(self) -> None:
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
        assert ctx.has_ampersand_prefix is True, "has_ampersand_prefix 应为 True（语法检测到 & 前缀）"

        # prefix_routed：策略决策层面，& 前缀未成功路由到 Cloud
        assert ctx.prefix_routed is False, "prefix_routed 应为 False（Cloud 未启用，未成功路由）"

        # triggered_by_prefix：应与 prefix_routed 一致（迁移后的语义）
        assert ctx.triggered_by_prefix == ctx.prefix_routed, (
            "triggered_by_prefix 应与 prefix_routed 一致（策略决策层面）\n"
            "这是迁移后的正确语义，而非之前错误的 has_ampersand_prefix"
        )

    def test_prefix_routed_parameter_priority_in_resolve_orchestrator_settings(self) -> None:
        """验证 prefix_routed 参数优先于 triggered_by_prefix"""
        from core.config import resolve_orchestrator_settings

        # 当同时指定 prefix_routed 和 triggered_by_prefix 时，prefix_routed 优先
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=False,  # 旧参数：False
            prefix_routed=True,  # 新参数：True（优先）
        )

        # prefix_routed=True 应生效，强制 basic
        assert result["orchestrator"] == "basic", "prefix_routed 参数应优先于 triggered_by_prefix"
        assert result["prefix_routed"] is True


class TestExecutionDecisionToDict:
    """测试 ExecutionDecision.to_dict() 输出的字段兼容性"""

    def test_to_dict_contains_both_prefix_routed_and_triggered_by_prefix(self) -> None:
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

    def test_to_dict_has_ampersand_prefix_independent_of_prefix_routed(self) -> None:
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
        assert result["has_ampersand_prefix"] is True, "has_ampersand_prefix 应为 True（语法层面存在 & 前缀）"
        # prefix_routed=False（策略决策）
        assert result["prefix_routed"] is False, "prefix_routed 应为 False（未成功触发 Cloud）"
        # triggered_by_prefix 应与 prefix_routed 相等
        assert result["triggered_by_prefix"] == result["prefix_routed"], "triggered_by_prefix 应与 prefix_routed 相等"


class TestAmpersandPresentButNotRouted:
    """测试 '& 存在但未成功路由' 的场景

    规则说明：
    - 当 has_ampersand_prefix=True 但 prefix_routed=False 时，默认 orchestrator=basic
    - 如需 mp 编排器，必须显式 `--execution-mode cli` 或禁用 auto_detect_cloud_prefix
    - 这确保 & 前缀表达的 Cloud 意图被尊重，即使路由条件未满足
    """

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
        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀（语法层面）"
        assert decision.prefix_routed is False, "prefix_routed 应为 False（Cloud 未启用）"
        assert decision.triggered_by_prefix == decision.prefix_routed, "triggered_by_prefix 应与 prefix_routed 一致"
        # 执行模式应回退到 CLI
        assert decision.effective_mode == "cli"
        # 编排器应为 basic（& 前缀表达 Cloud 意图，即使未成功触发也使用 basic）
        # 注意：如需 mp，必须显式 --execution-mode cli 或禁用 auto_detect
        assert decision.orchestrator == "basic"

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
        assert decision.prefix_routed is False, "prefix_routed 应为 False（无 API Key）"
        assert decision.triggered_by_prefix == decision.prefix_routed
        # 执行模式应回退到 CLI
        assert decision.effective_mode == "cli"
        # 编排器应为 basic（& 前缀表达 Cloud 意图，即使未成功触发也使用 basic）
        # 注意：如需 mp，必须显式 --execution-mode cli 或禁用 auto_detect
        assert decision.orchestrator == "basic"

    def test_ampersand_present_explicit_cli_mode(self) -> None:
        """场景: & 前缀存在但显式指定 --execution-mode cli

        显式 --execution-mode cli 会覆盖 & 前缀的 Cloud 意图。
        此时 prefix_routed=False，编排器为 mp（用户明确请求 CLI 模式）。
        这是 "如需 mp，必须显式 --execution-mode cli" 规则的验证。
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode="cli",  # 显式 CLI 模式
            cloud_enabled=True,
            has_api_key=True,
        )

        # 核心断言
        assert decision.has_ampersand_prefix is True
        assert decision.prefix_routed is False, "prefix_routed 应为 False（显式 CLI 模式忽略 & 前缀）"
        assert decision.triggered_by_prefix == decision.prefix_routed
        assert decision.effective_mode == "cli"
        # 显式 --execution-mode cli 时编排器为 mp（用户明确请求）
        # 注意：如需 mp，必须显式 --execution-mode cli 或禁用 auto_detect
        assert decision.orchestrator == "mp"

    def test_ampersand_present_auto_detect_disabled(self) -> None:
        """场景: & 前缀存在但 auto_detect_cloud_prefix=False

        禁用 auto_detect 会覆盖 & 前缀的 Cloud 意图。
        此时 prefix_routed=False，编排器为 mp（用户明确禁用前缀检测）。
        这是 "如需 mp，必须禁用 auto_detect" 规则的验证。
        """
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
        assert decision.prefix_routed is False, "prefix_routed 应为 False（auto_detect_cloud_prefix=False）"
        assert decision.triggered_by_prefix == decision.prefix_routed
        assert decision.effective_mode == "cli"
        # 禁用 auto_detect 时编排器为 mp（用户明确禁用前缀检测）
        # 注意：如需 mp，必须显式 --execution-mode cli 或禁用 auto_detect
        assert decision.orchestrator == "mp"


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
        assert options["triggered_by_prefix"] == options["prefix_routed"], "triggered_by_prefix 应取自 prefix_routed"
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
        assert options["has_ampersand_prefix"] is True, "has_ampersand_prefix=True（语法检测）"
        assert options["prefix_routed"] is False, "prefix_routed=False（未成功路由）"
        assert options["triggered_by_prefix"] is False, "triggered_by_prefix 应与 prefix_routed 一致"

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


class TestResolveEffectiveExecutionModeNoTriggeredByPrefix:
    """静态检查：确保仓库内没有 resolve_effective_execution_mode(triggered_by_prefix=...) 调用点

    这是迁移完成的保证测试，确保新代码不再使用已弃用的参数。
    triggered_by_prefix 参数是 has_ampersand_prefix 的旧别名，严禁与 prefix_routed 混用。
    """

    def test_no_resolve_effective_execution_mode_triggered_by_prefix_calls(
        self,
    ) -> None:
        """确保仓库内没有 resolve_effective_execution_mode(triggered_by_prefix=...) 的调用

        此测试扫描仓库内所有 Python 文件，确保没有使用已弃用的 triggered_by_prefix
        参数调用 resolve_effective_execution_mode 或 resolve_effective_execution_mode_full。

        预期结果：调用点数量应为 0
        """
        import re
        from pathlib import Path

        project_root = Path(__file__).parent.parent

        # 需要扫描的源代码目录（排除测试文件和函数定义本身）
        source_dirs = ["core", "cursor", "agents", "coordinator", "scripts"]

        # 匹配模式：resolve_effective_execution_mode[_full](xxx triggered_by_prefix=xxx)
        # 使用非贪婪匹配和多行模式
        call_pattern = re.compile(
            r"resolve_effective_execution_mode(?:_full)?\s*\([^)]*"
            r"triggered_by_prefix\s*=",
            re.MULTILINE,
        )

        violations: list[str] = []

        for source_dir in source_dirs:
            dir_path = project_root / source_dir
            if not dir_path.exists():
                continue

            for py_file in dir_path.rglob("*.py"):
                # 跳过测试文件
                if "test" in py_file.name.lower():
                    continue

                try:
                    content = py_file.read_text(encoding="utf-8")
                except Exception:
                    continue

                # 搜索匹配
                for match in call_pattern.finditer(content):
                    # 获取匹配位置的行号
                    line_start = content.count("\n", 0, match.start()) + 1
                    # 获取匹配的代码片段（截取前后文）
                    snippet = content[match.start() : match.end() + 20].strip()
                    if len(snippet) > 80:
                        snippet = snippet[:80] + "..."

                    relative_path = py_file.relative_to(project_root)
                    violations.append(f"{relative_path}:{line_start}: {snippet}")

        # 断言：不应存在任何调用点
        if violations:
            violation_list = "\n  ".join(violations)
            pytest.fail(
                f"发现 {len(violations)} 处 resolve_effective_execution_mode(triggered_by_prefix=...) 调用，"
                f"这违反了迁移规范。triggered_by_prefix 参数已弃用，请使用 has_ampersand_prefix。\n\n"
                f"违规位置:\n  {violation_list}\n\n"
                f"迁移指南:\n"
                f"  - triggered_by_prefix 是 has_ampersand_prefix 的旧别名\n"
                f"  - 严禁将其与 prefix_routed 混用\n"
                f"  - 请将调用改为: resolve_effective_execution_mode(..., has_ampersand_prefix=...)"
            )

    def test_deprecation_warning_triggered(self) -> None:
        """验证使用 triggered_by_prefix 参数时会触发 DeprecationWarning"""
        import warnings

        from core.execution_policy import resolve_effective_execution_mode

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # 使用已弃用的 triggered_by_prefix 参数
            resolve_effective_execution_mode(
                requested_mode=None,
                triggered_by_prefix=True,
                cloud_enabled=True,
                has_api_key=True,
            )

            # 验证触发了 DeprecationWarning
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning) and "triggered_by_prefix" in str(warning.message)
            ]
            assert len(deprecation_warnings) >= 1, "使用 triggered_by_prefix 参数应触发 DeprecationWarning"

    def test_deprecation_warning_full_triggered(self) -> None:
        """验证 resolve_effective_execution_mode_full 使用 triggered_by_prefix 时触发警告"""
        import warnings

        from core.execution_policy import resolve_effective_execution_mode_full

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # 使用已弃用的 triggered_by_prefix 参数
            resolve_effective_execution_mode_full(
                requested_mode=None,
                triggered_by_prefix=True,
                cloud_enabled=True,
                has_api_key=True,
            )

            # 验证触发了 DeprecationWarning
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning) and "triggered_by_prefix" in str(warning.message)
            ]
            assert len(deprecation_warnings) >= 1, "使用 triggered_by_prefix 参数应触发 DeprecationWarning"


class TestTriggeredByPrefixUsageRestriction:
    """静态检查：禁止在非兼容输出路径使用 triggered_by_prefix

    白名单场景（允许使用）：
    1. tests/ 目录下的测试文件
    2. to_dict() 方法实现中（输出兼容）
    3. 作为函数/方法参数定义（兼容别名参数）
    4. 注释/文档字符串中的说明
    5. 字典键赋值用于输出兼容（如 options["triggered_by_prefix"] = ...）
    6. @property 定义的兼容别名属性

    禁止场景：
    - 在业务逻辑 if/elif 条件分支中读取 triggered_by_prefix
    - 例如: if obj.triggered_by_prefix:  # 应改为 if obj.prefix_routed:
    """

    def test_no_triggered_by_prefix_in_business_logic_branches(self) -> None:
        """检查非测试代码中不存在业务逻辑分支使用 triggered_by_prefix

        扫描源代码文件，检查 triggered_by_prefix 的使用是否符合白名单规则。
        """
        import re
        from pathlib import Path

        # 项目根目录
        project_root = Path(__file__).parent.parent

        # 需要扫描的源代码目录
        source_dirs = ["core", "cursor", "agents", "coordinator", "scripts"]

        # 白名单模式（正则表达式）
        whitelist_patterns = [
            # 1. to_dict() 或字典输出中的赋值
            r'^\s*["\']triggered_by_prefix["\']',  # 字典键
            r'triggered_by_prefix["\']?\s*[:=]',  # 赋值或字典键值对
            # 2. 函数/方法参数定义
            r"def\s+\w+\([^)]*triggered_by_prefix",
            r"triggered_by_prefix\s*:\s*(bool|Optional)",
            r"triggered_by_prefix\s*=\s*(True|False|None)",
            # 3. 注释或文档字符串
            r"^\s*#.*triggered_by_prefix",
            r'^\s*["\'].*triggered_by_prefix.*["\'],?\s*$',
            # 4. @property 定义
            r"def triggered_by_prefix\(",
            # 5. 返回语句（用于属性返回）
            r"return\s+self\.(prefix_routed|triggered_by_prefix)",
            # 6. logger/日志输出
            r"logger\.(debug|info|warning|error).*triggered_by_prefix",
            # 7. getattr 调用（通常用于兼容处理）
            r'getattr\([^,]+,\s*["\']triggered_by_prefix["\']',
            # 8. 属性访问赋值给变量（用于输出）
            r"\w+\s*=\s*\w+\.triggered_by_prefix\s*#\s*兼容",
            # 9. 方法调用传参（用于兼容别名）
            r"\(\s*triggered_by_prefix\s*=",
        ]

        # 禁止模式（业务逻辑分支）
        forbidden_patterns = [
            # if/elif 条件分支中读取
            r"^\s*if\s+.*\.triggered_by_prefix[^=]",
            r"^\s*elif\s+.*\.triggered_by_prefix[^=]",
            r"^\s*if\s+triggered_by_prefix[^=\s]",
            r"^\s*elif\s+triggered_by_prefix[^=\s]",
            # 布尔表达式中使用（非赋值）
            r"\sand\s+\w+\.triggered_by_prefix\b",
            r"\sor\s+\w+\.triggered_by_prefix\b",
            r"\snot\s+\w+\.triggered_by_prefix\b",
        ]

        violations = []

        for source_dir in source_dirs:
            dir_path = project_root / source_dir
            if not dir_path.exists():
                continue

            for py_file in dir_path.rglob("*.py"):
                # 跳过测试文件
                if "test" in py_file.name.lower():
                    continue

                try:
                    content = py_file.read_text(encoding="utf-8")
                except Exception:
                    continue

                for line_num, line in enumerate(content.splitlines(), 1):
                    # 检查是否包含 triggered_by_prefix
                    if "triggered_by_prefix" not in line:
                        continue

                    # 跳过空行和纯注释行
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue

                    # 检查是否匹配白名单模式
                    is_whitelisted = any(re.search(pattern, line) for pattern in whitelist_patterns)

                    # 检查是否匹配禁止模式
                    is_forbidden = any(re.search(pattern, line) for pattern in forbidden_patterns)

                    # 如果匹配禁止模式且不在白名单中，记录违规
                    if is_forbidden and not is_whitelisted:
                        relative_path = py_file.relative_to(project_root)
                        violations.append(f"{relative_path}:{line_num}: {stripped}")

        # 断言：不应存在违规使用
        if violations:
            violation_list = "\n  ".join(violations)
            pytest.fail(
                f"发现 {len(violations)} 处业务逻辑分支中使用 triggered_by_prefix，"
                f"应改为 prefix_routed:\n  {violation_list}\n\n"
                "说明：triggered_by_prefix 仅作为输出兼容别名，"
                "内部条件分支请使用 prefix_routed"
            )

    def test_triggered_by_prefix_always_equals_prefix_routed_in_dataclasses(self) -> None:
        """验证所有数据类中 triggered_by_prefix 属性返回 prefix_routed"""
        from core.execution_policy import (
            AmpersandPrefixInfo,
            AmpersandPrefixStatus,
            ExecutionDecision,
            ExecutionPolicyContext,
        )

        # 测试 AmpersandPrefixStatus
        for status in AmpersandPrefixStatus:
            assert status.triggered_by_prefix == status.prefix_routed, (
                f"AmpersandPrefixStatus.{status.name}: "
                f"triggered_by_prefix ({status.triggered_by_prefix}) != "
                f"prefix_routed ({status.prefix_routed})"
            )

        # 测试 AmpersandPrefixInfo
        for prefix_routed in [True, False]:
            info = AmpersandPrefixInfo(
                has_ampersand_prefix=True,
                prefix_routed=prefix_routed,
            )
            assert info.triggered_by_prefix == info.prefix_routed

        # 测试 ExecutionDecision
        for prefix_routed in [True, False]:
            decision = ExecutionDecision(
                effective_mode="cli",
                orchestrator="mp",
                prefix_routed=prefix_routed,
            )
            assert decision.triggered_by_prefix == decision.prefix_routed

        # 测试 ExecutionPolicyContext
        test_cases = [
            # (prompt, cloud_enabled, has_api_key, expected_prefix_routed)
            ("& 任务", True, True, True),
            ("& 任务", False, False, False),
            ("普通任务", True, True, False),
        ]
        for prompt, cloud_enabled, has_api_key, expected in test_cases:
            ctx = ExecutionPolicyContext(
                prompt=prompt,
                cloud_enabled=cloud_enabled,
                has_api_key=has_api_key,
            )
            assert ctx.triggered_by_prefix == ctx.prefix_routed, (
                "ExecutionPolicyContext: triggered_by_prefix != prefix_routed"
            )
            assert ctx.prefix_routed == expected
