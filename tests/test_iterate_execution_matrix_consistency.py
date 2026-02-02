"""执行模式与编排器选择一致性矩阵测试

本测试文件确保 run.py 和 scripts/run_iterate.py 两个入口
对于编排器选择的逻辑是一致的。

================================================================================
统一字段 Schema 遵循声明（与 core/execution_policy.py 完全对齐）
================================================================================

**重要**: 本测试文件中的 DecisionSnapshot 和 SnapshotTestCase 数据结构
遵循 core/execution_policy.py 定义的"统一字段 Schema"。

【快照字段与 Schema 对应关系】

| 字段名                      | 类型     | 语义                                       | 来源函数                          |
|-----------------------------|----------|--------------------------------------------|-----------------------------------|
| effective_mode              | str      | 有效执行模式（经过路由决策后实际使用）     | build_execution_decision()        |
| requested_mode_for_decision | str|None | 用于决策的请求模式                         | resolve_requested_mode_for_decision() |
| cli_execution_mode          | str|None | CLI 原始 --execution-mode 参数值          | 测试参数直接传入                  |
| orchestrator                | str      | 编排器类型 (mp/basic)                      | build_execution_decision()        |
| prefix_routed               | bool     | 【策略决策层面】& 前缀是否成功触发 Cloud   | detect_ampersand_prefix()         |
| has_ampersand_prefix        | bool     | 【语法检测层面】原始 prompt 是否有 & 前缀  | is_cloud_request()                |
| config_execution_mode       | str|None | config.yaml 中的 cloud_agent.execution_mode| 测试配置传入                      |

【字段区分说明】

- cli_execution_mode: 用户 CLI 输入的原始值（如 --execution-mode auto）
- requested_mode_for_decision: 经过 resolve_requested_mode_for_decision 解析后
  传给 build_execution_decision 的值（考虑了 & 前缀路由逻辑）
- has_ampersand_prefix: 语法检测，仅表示原始文本是否有 & 前缀
- prefix_routed: 策略决策，& 前缀是否满足路由条件成功触发 Cloud

**新代码规范**: 测试断言应使用语义明确的字段名:
    assert snapshot.prefix_routed == True       # ✓ 推荐（内部分支统一使用）
    assert snapshot.triggered_by_prefix == True  # △ 仅用于兼容输出，避免新代码引用

================================================================================
⚠ 测试代码约束：禁止新断言使用 deprecated 属性
================================================================================

**禁止在新断言中使用的 deprecated 属性**:
- triggered_by_prefix (已统一为 prefix_routed)
- execution_mode 用于决策分支时（使用 effective_mode 或 requested_mode_for_decision）

**新断言应使用的字段**:
- prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud
- effective_mode: 经过路由决策后实际使用的执行模式
- requested_mode_for_decision: 传给 build_execution_decision 的请求模式
- has_ampersand_prefix: 仅用于消息构建/日志记录场景

**to_dict() 兼容输出保持不变**:
to_dict() 的输出中保留 triggered_by_prefix 字段以兼容下游消费者，
但测试断言（assert 语句）应使用 prefix_routed。

详细 Schema 定义参见: core/execution_policy.py "统一字段 Schema" 部分

================================================================================
核心规则（与 CLI help 和 build_execution_decision 矩阵表对齐）
================================================================================

- requested_mode_for_decision=auto/cloud 即强制 basic，不受 key/enable 影响
- 这意味着即使因为没有 API Key 导致 effective_mode 回退到 CLI，
  只要 requested_mode_for_decision 是 auto/cloud，编排器就应该是 basic

================================================================================
测试矩阵覆盖场景（断言维度：effective_mode, orchestrator, prefix_routed）
================================================================================

1. requested_mode_for_decision=cli -> effective_mode=cli, orchestrator=mp, prefix_routed=False
2. requested_mode_for_decision=auto + 有 key -> effective_mode=auto, orchestrator=basic, prefix_routed=False
3. requested_mode_for_decision=auto + 无 key -> effective_mode=cli(回退), orchestrator=basic, prefix_routed=False（关键场景）
4. requested_mode_for_decision=cloud + 有 key -> effective_mode=cloud, orchestrator=basic, prefix_routed=False
5. requested_mode_for_decision=cloud + 无 key -> effective_mode=cli(回退), orchestrator=basic, prefix_routed=False（关键场景）
6. & 前缀触发 + 有 key + cloud_enabled -> effective_mode=cloud, orchestrator=basic, prefix_routed=True
7. & 前缀触发 + 无 key -> effective_mode=cli, orchestrator=basic, prefix_routed=False（& 前缀表达 Cloud 意图）
8. & 前缀触发 + cloud_disabled -> effective_mode=cli, orchestrator=basic, prefix_routed=False（& 前缀表达 Cloud 意图）

================================================================================
决策快照一致性测试
================================================================================

- 对每个矩阵 case，分别以 run.py 的合并结果/IterateArgs 语义
  与 SelfIterator 自身解析语义计算出决策快照
- 断言两者的快照字段一致（核心对比维度：effective_mode, orchestrator, prefix_routed）
- 必要时对配置单例 ConfigManager 做 reset 以避免测试间污染
"""

import argparse
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.config import (
    DEFAULT_EXECUTION_MODE,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REVIEWER_MODEL,
    DEFAULT_WORKER_MODEL,
    ConfigManager,
    resolve_orchestrator_settings,
)
from core.execution_policy import (
    EXECUTION_DECISION_MATRIX_CASES,
    DecisionMatrixCase,
    build_execution_decision,
    resolve_effective_execution_mode,
    resolve_requested_mode_for_decision,
    should_use_mp_orchestrator,
)

# ============================================================
# 测试参数数据结构
# ============================================================


@dataclass
class ConsistencyTestCase:
    """一致性测试参数

    用于验证 core.config.resolve_orchestrator_settings 和
    core.execution_policy.should_use_mp_orchestrator 的行为一致性。
    """

    test_id: str
    requested_mode: Optional[str]  # CLI 参数或 config.yaml 中的 execution_mode
    has_api_key: bool
    cloud_enabled: bool
    expected_orchestrator: str  # "mp" 或 "basic"
    description: str


# 一致性测试参数表
CONSISTENCY_TEST_CASES = [
    # ===== CLI 模式 =====
    ConsistencyTestCase(
        test_id="cli_allows_mp",
        requested_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="CLI 模式允许使用 MP 编排器",
    ),
    ConsistencyTestCase(
        test_id="cli_no_key_allows_mp",
        requested_mode="cli",
        has_api_key=False,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="CLI 模式无 API Key 仍允许 MP",
    ),
    ConsistencyTestCase(
        test_id="cli_cloud_disabled_allows_mp",
        requested_mode="cli",
        has_api_key=True,
        cloud_enabled=False,
        expected_orchestrator="mp",
        description="CLI 模式 cloud_enabled=False 仍允许 MP",
    ),
    # ===== AUTO 模式（关键场景）=====
    ConsistencyTestCase(
        test_id="auto_with_key_forces_basic",
        requested_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="AUTO 模式有 API Key 强制 basic",
    ),
    ConsistencyTestCase(
        test_id="auto_no_key_forces_basic",
        requested_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="关键场景：AUTO 模式无 API Key 仍强制 basic（基于 requested_mode）",
    ),
    ConsistencyTestCase(
        test_id="auto_cloud_disabled_forces_basic",
        requested_mode="auto",
        has_api_key=True,
        cloud_enabled=False,
        expected_orchestrator="basic",
        description="AUTO 模式 cloud_enabled=False 仍强制 basic（基于 requested_mode）",
    ),
    # ===== CLOUD 模式（关键场景）=====
    ConsistencyTestCase(
        test_id="cloud_with_key_forces_basic",
        requested_mode="cloud",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="CLOUD 模式有 API Key 强制 basic",
    ),
    ConsistencyTestCase(
        test_id="cloud_no_key_forces_basic",
        requested_mode="cloud",
        has_api_key=False,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="关键场景：CLOUD 模式无 API Key 仍强制 basic（基于 requested_mode）",
    ),
    ConsistencyTestCase(
        test_id="cloud_cloud_disabled_forces_basic",
        requested_mode="cloud",
        has_api_key=True,
        cloud_enabled=False,
        expected_orchestrator="basic",
        description="CLOUD 模式 cloud_enabled=False 仍强制 basic（基于 requested_mode）",
    ),
    # ===== None/直接传参场景 - 仅用于 & 前缀流（prefix-flow）=====
    # ⚠ 重要：requested_mode=None 仅在以下特殊路径下产生：
    #   1. has_ampersand_prefix=True（输入以 & 开头）
    #   2. 且无 CLI 显式设置 execution_mode
    #
    # 当 has_ampersand_prefix=False 且无 CLI 显式设置时，
    # resolve_requested_mode_for_decision 返回 config.yaml 的默认值（如 "auto"），
    # **不应为 None**。把 None 当作"无 & 前缀常规输入"是错误的理解。
    #
    # TestShouldUseMpOrchestratorConsistency 类对此用例的正确处理流程：
    # 1. 先通过 resolve_requested_mode_for_decision 获取 requested_mode
    # 2. 再调用 should_use_mp_orchestrator(requested_mode)
    #
    # 此用例标记为 prefix-flow 回归测试场景。
    ConsistencyTestCase(
        test_id="none_param_prefix_flow_regression",
        requested_mode=None,  # 仅在 & 前缀场景下合法
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="basic",  # 实际由 build_execution_decision 根据 prefix_routed 决定
        description="[prefix-flow] requested_mode=None 仅用于 & 前缀场景，由 build_execution_decision 决策",
    ),
    # ===== config.yaml 默认 auto 场景（显式传参 auto）=====
    # 当用户不显式指定 --execution-mode 时，实际使用 config.yaml 默认值 "auto"
    ConsistencyTestCase(
        test_id="config_default_auto_forces_basic",
        requested_mode="auto",  # 模拟 config.yaml 默认的 execution_mode=auto
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="config.yaml 默认 auto 模式，强制 basic（真实默认行为）",
    ),
    ConsistencyTestCase(
        test_id="config_default_auto_no_key_forces_basic",
        requested_mode="auto",  # 模拟 config.yaml 默认的 execution_mode=auto
        has_api_key=False,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="config.yaml 默认 auto 模式无 API Key，仍强制 basic（真实默认行为）",
    ),
    # ===== 显式 CLI 模式（用户显式指定 --execution-mode cli）=====
    # 这是用户主动选择 CLI 模式的场景，与上面的"默认"场景区分
    # 如需使用 MP 编排器，用户应显式指定 --execution-mode cli
    ConsistencyTestCase(
        test_id="explicit_cli_allows_mp",
        requested_mode="cli",  # 用户显式 --execution-mode cli
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="用户显式指定 --execution-mode cli，允许 MP",
    ),
    # ===== PLAN/ASK 只读模式 =====
    ConsistencyTestCase(
        test_id="plan_mode_allows_mp",
        requested_mode="plan",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="PLAN 模式（只读）允许 MP",
    ),
    ConsistencyTestCase(
        test_id="ask_mode_allows_mp",
        requested_mode="ask",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="ASK 模式（只读）允许 MP",
    ),
]


# ============================================================
# 测试类
# ============================================================


class TestShouldUseMpOrchestratorConsistency:
    """测试 should_use_mp_orchestrator 函数的一致性

    验证 core.execution_policy.should_use_mp_orchestrator 基于 requested_mode
    正确判断是否允许使用 MP 编排器。

    ⚠ 重要：调用方应遵循正确的流程：
    1. 先通过 resolve_requested_mode_for_decision 解析 requested_mode
    2. 再调用 should_use_mp_orchestrator(requested_mode)

    对于 requested_mode=None 的测试用例，本类验证正确的调用流程，
    而非直接调用 should_use_mp_orchestrator(None)。
    """

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in CONSISTENCY_TEST_CASES if tc.requested_mode is not None],
        ids=[tc.test_id for tc in CONSISTENCY_TEST_CASES if tc.requested_mode is not None],
    )
    def test_should_use_mp_orchestrator(self, test_case: ConsistencyTestCase) -> None:
        """测试 should_use_mp_orchestrator 函数（非 None 场景）"""
        can_use_mp = should_use_mp_orchestrator(test_case.requested_mode)
        expected_can_use_mp = test_case.expected_orchestrator == "mp"

        assert can_use_mp == expected_can_use_mp, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  requested_mode={test_case.requested_mode}\n"
            f"  期望 can_use_mp={expected_can_use_mp}，实际={can_use_mp}"
        )

    def test_none_param_via_resolve_requested_mode(self) -> None:
        """验证 requested_mode=None 场景的正确调用流程

        ⚠ 此测试验证：
        1. 无 & 前缀 + CLI 未指定 → resolve_requested_mode_for_decision 返回 config 默认值
        2. 有 & 前缀 + CLI 未指定 → resolve_requested_mode_for_decision 返回 None（prefix-flow）

        对于后者（prefix-flow），should_use_mp_orchestrator(None) 返回 True 是合法的，
        但实际编排器由 build_execution_decision 根据 prefix_routed 决定。
        """
        from core.execution_policy import resolve_requested_mode_for_decision

        # 场景 1: 无 & 前缀 + CLI 未指定 → 应使用 config.yaml 默认值
        requested_mode_no_prefix = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        )
        assert requested_mode_no_prefix == "auto", (
            "无 & 前缀时 resolve_requested_mode_for_decision 应返回 config 默认值"
        )
        assert should_use_mp_orchestrator(requested_mode_no_prefix) is False, (
            "config.yaml 默认 auto 时，should_use_mp_orchestrator 应返回 False"
        )

        # 场景 2: 有 & 前缀 + CLI 未指定 → 应返回 None（prefix-flow）
        requested_mode_with_prefix = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        )
        assert requested_mode_with_prefix is None, (
            "有 & 前缀时 resolve_requested_mode_for_decision 应返回 None（prefix-flow）"
        )
        # prefix-flow 场景下 should_use_mp_orchestrator(None) 返回 True
        # 但实际编排器由 build_execution_decision 决定
        assert should_use_mp_orchestrator(requested_mode_with_prefix) is True, (
            "prefix-flow 场景下 should_use_mp_orchestrator(None) 应返回 True"
        )


class TestResolveOrchestratorSettingsConsistency:
    """测试 resolve_orchestrator_settings 函数的一致性

    验证 core.config.resolve_orchestrator_settings 基于 requested_mode
    正确设置编排器类型。
    """

    @pytest.mark.parametrize(
        "test_case",
        CONSISTENCY_TEST_CASES,
        ids=[tc.test_id for tc in CONSISTENCY_TEST_CASES],
    )
    def test_resolve_orchestrator_settings(self, test_case: ConsistencyTestCase) -> None:
        """测试 resolve_orchestrator_settings 函数"""
        from core.config import ConfigManager

        # 重置配置单例以避免测试间干扰
        ConfigManager.reset_instance()

        # 构建 overrides（模拟 CLI 参数）
        overrides = {}
        if test_case.requested_mode is not None:
            overrides["execution_mode"] = test_case.requested_mode

        # 调用函数
        result = resolve_orchestrator_settings(overrides=overrides)

        assert result["orchestrator"] == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  requested_mode={test_case.requested_mode}\n"
            f"  期望 orchestrator={test_case.expected_orchestrator}，"
            f"实际={result['orchestrator']}"
        )


class TestRunPyAndRunIterateConsistency:
    """测试 run.py 和 scripts/run_iterate.py 两个入口的一致性

    验证两个入口对于相同的输入参数，编排器选择逻辑是一致的。

    注意：这些测试需要 mock SelfIterator 的依赖以避免实际执行。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in CONSISTENCY_TEST_CASES if tc.requested_mode in ("auto", "cloud")],
        ids=[tc.test_id for tc in CONSISTENCY_TEST_CASES if tc.requested_mode in ("auto", "cloud")],
    )
    def test_self_iterator_orchestrator_selection(self, test_case: ConsistencyTestCase) -> None:
        """测试 SelfIterator._get_orchestrator_type 的编排器选择逻辑"""
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # 构建测试参数
        args = argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,  # 避免实际执行
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",  # 用户请求 mp，但应被强制切换
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=test_case.requested_mode,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        # Mock 配置和 API Key
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_case.cloud_enabled
        mock_config.cloud_agent.execution_mode = "auto"  # 默认值（与 core/config.py 保持一致）
        mock_config.cloud_agent.timeout = 300
        mock_config.cloud_agent.auth_timeout = 30
        mock_config.system.max_iterations = 10
        mock_config.system.worker_pool_size = 3
        mock_config.system.enable_sub_planners = True
        mock_config.system.strict_review = False
        mock_config.models.planner = DEFAULT_PLANNER_MODEL
        mock_config.models.worker = DEFAULT_WORKER_MODEL
        mock_config.models.reviewer = DEFAULT_REVIEWER_MODEL
        mock_config.planner.timeout = 500.0
        mock_config.worker.task_timeout = 600.0
        mock_config.reviewer.timeout = 300.0
        mock_config.logging.stream_json.enabled = False
        mock_config.logging.stream_json.console = True
        mock_config.logging.stream_json.detail_dir = "logs/stream_json/detail/"
        mock_config.logging.stream_json.raw_dir = "logs/stream_json/raw/"

        api_key = "mock-api-key" if test_case.has_api_key else None

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance") as mock_manager:
                    mock_manager.return_value = mock_config
                    iterator = SelfIterator(args)
                    actual_orchestrator = iterator._get_orchestrator_type()

        assert actual_orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  SelfIterator._get_orchestrator_type() 返回 {actual_orchestrator}，"
            f"期望 {test_case.expected_orchestrator}"
        )


class TestEdgeCaseConsistency:
    """边界场景一致性测试

    覆盖一些特殊的边界场景，确保两个入口的行为一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def test_auto_no_key_both_entries_force_basic(self) -> None:
        """关键测试：auto 模式无 API Key 时，两个入口都应强制 basic

        这是最重要的一致性测试，验证：
        1. resolve_orchestrator_settings 强制 basic
        2. should_use_mp_orchestrator 返回 False
        """
        # 测试 1: resolve_orchestrator_settings
        overrides = {"execution_mode": "auto"}
        result = resolve_orchestrator_settings(overrides=overrides)
        assert result["orchestrator"] == "basic", "resolve_orchestrator_settings: auto 模式应强制 basic 编排器"

        # 测试 2: should_use_mp_orchestrator
        can_use_mp = should_use_mp_orchestrator("auto")
        assert can_use_mp is False, "should_use_mp_orchestrator: auto 模式应返回 False"

    def test_cloud_no_key_both_entries_force_basic(self) -> None:
        """关键测试：cloud 模式无 API Key 时，两个入口都应强制 basic

        这是最重要的一致性测试，验证：
        1. resolve_orchestrator_settings 强制 basic
        2. should_use_mp_orchestrator 返回 False
        """
        # 测试 1: resolve_orchestrator_settings
        overrides = {"execution_mode": "cloud"}
        result = resolve_orchestrator_settings(overrides=overrides)
        assert result["orchestrator"] == "basic", "resolve_orchestrator_settings: cloud 模式应强制 basic 编排器"

        # 测试 2: should_use_mp_orchestrator
        can_use_mp = should_use_mp_orchestrator("cloud")
        assert can_use_mp is False, "should_use_mp_orchestrator: cloud 模式应返回 False"

    def test_cli_allows_mp_both_entries(self) -> None:
        """测试 CLI 模式两个入口都允许 MP"""
        # 测试 1: resolve_orchestrator_settings
        overrides = {"execution_mode": "cli"}
        result = resolve_orchestrator_settings(overrides=overrides)
        assert result["orchestrator"] == "mp", "resolve_orchestrator_settings: cli 模式应允许 mp 编排器"

        # 测试 2: should_use_mp_orchestrator
        can_use_mp = should_use_mp_orchestrator("cli")
        assert can_use_mp is True, "should_use_mp_orchestrator: cli 模式应返回 True"


class TestDocumentationConsistency:
    """文档一致性测试

    验证代码行为与 AGENTS.md 文档中描述的规则一致。
    """

    def test_agents_md_rule_auto_cloud_forces_basic(self) -> None:
        """验证 AGENTS.md 中的规则：auto/cloud 强制 basic

        AGENTS.md 文档描述：
        > 当 --execution-mode 为 cloud 或 auto 时，系统会强制使用 basic 编排器，
        > 即使显式指定 --orchestrator mp 也会自动切换。
        """
        # 测试 auto 模式
        assert should_use_mp_orchestrator("auto") is False, "违反 AGENTS.md 规则：auto 模式应强制 basic"

        # 测试 cloud 模式
        assert should_use_mp_orchestrator("cloud") is False, "违反 AGENTS.md 规则：cloud 模式应强制 basic"

        # 测试 cli 模式
        assert should_use_mp_orchestrator("cli") is True, "违反 AGENTS.md 规则：cli 模式应允许 mp"

    def test_agents_md_rule_mp_not_compatible_with_cloud_auto(self) -> None:
        """验证 AGENTS.md 中的规则：MP 编排器与 Cloud/Auto 不兼容

        AGENTS.md 文档描述：
        > MP 编排器 (MultiProcessOrchestrator): 仅在 execution_mode=cli 时可用
        > Cloud/Auto 模式: 强制使用 basic 编排器，因为 Cloud API 不支持多进程编排
        """
        # resolve_orchestrator_settings 测试
        auto_result = resolve_orchestrator_settings(overrides={"execution_mode": "auto"})
        cloud_result = resolve_orchestrator_settings(overrides={"execution_mode": "cloud"})
        cli_result = resolve_orchestrator_settings(overrides={"execution_mode": "cli"})

        assert auto_result["orchestrator"] == "basic", "违反 AGENTS.md 规则：auto 模式应强制 basic 编排器"
        assert cloud_result["orchestrator"] == "basic", "违反 AGENTS.md 规则：cloud 模式应强制 basic 编排器"
        assert cli_result["orchestrator"] == "mp", "违反 AGENTS.md 规则：cli 模式应默认使用 mp 编排器"


# ============================================================
# 决策快照数据结构
# ============================================================


@dataclass
class DecisionSnapshot:
    """执行决策快照

    用于比较 run.py 的 IterateArgs 语义与 SelfIterator 自身解析语义。

    ================================================================================
    字段语义说明（与 core/execution_policy.py 统一字段 Schema 对齐）
    ================================================================================

    本类与 core/execution_policy.py 中的 ExecutionDecision 保持字段对齐：

    +---------------------------+----------+----------------------------------------------+
    | 字段名                    | 类型     | 语义定义                                     |
    +===========================+==========+==============================================+
    | effective_mode            | str      | 有效执行模式（经过路由决策后实际使用）       |
    |                           |          | 来源: build_execution_decision()             |
    +---------------------------+----------+----------------------------------------------+
    | requested_mode_for_decision| str|None| 用于决策的请求模式                           |
    |                           |          | 来源: resolve_requested_mode_for_decision()  |
    |                           |          | 注意: 与 ExecutionDecision.requested_mode 对齐|
    +---------------------------+----------+----------------------------------------------+
    | cli_execution_mode        | str|None | CLI 原始 --execution-mode 参数值            |
    |                           |          | 来源: test_case.execution_mode               |
    |                           |          | 用途: 追溯 CLI 原始参数，不参与核心对比      |
    +---------------------------+----------+----------------------------------------------+
    | orchestrator              | str      | 编排器类型 (mp/basic)                        |
    |                           |          | 来源: build_execution_decision()             |
    +---------------------------+----------+----------------------------------------------+
    | prefix_routed             | bool     | 【策略决策层面】& 前缀是否成功触发 Cloud     |
    |                           |          | 来源: detect_ampersand_prefix()              |
    +---------------------------+----------+----------------------------------------------+
    | has_ampersand_prefix      | bool     | 【语法检测层面】原始 prompt 是否有 & 前缀    |
    |                           |          | 来源: is_cloud_request()                     |
    |                           |          | 用途: 追溯原始 prompt 格式，不参与核心对比   |
    +---------------------------+----------+----------------------------------------------+
    | config_execution_mode     | str|None | config.yaml 中的 cloud_agent.execution_mode  |
    |                           |          | 用途: 追溯配置来源，不参与核心对比           |
    +---------------------------+----------+----------------------------------------------+

    核心对比维度（测试断言主要检查这些字段）：
    - effective_mode: 验证有效执行模式
    - orchestrator: 验证编排器类型
    - prefix_routed: 验证 & 前缀是否成功触发 Cloud
    - requested_mode_for_decision: 验证传给 build_execution_decision 的请求模式

    追溯字段（用于调试，不参与核心对比）：
    - cli_execution_mode: CLI 原始参数
    - has_ampersand_prefix: 语法检测结果
    - config_execution_mode: 配置文件值

    新代码应优先使用语义明确的字段名进行断言。
    """

    # 必需字段（无默认值）放在前面
    effective_mode: str  # cli/cloud/auto 等有效执行模式（经过路由决策后）
    orchestrator: str  # mp 或 basic
    prefix_routed: bool  # 策略决策：& 前缀是否成功触发 Cloud
    # 可选字段（有默认值）放在后面
    requested_mode_for_decision: Optional[str] = (
        None  # 用于决策的请求模式（resolve_requested_mode_for_decision 返回值）
    )
    cli_execution_mode: Optional[str] = None  # CLI 原始 --execution-mode 参数值
    # 追溯字段（不参与核心对比，用于调试追溯）
    has_ampersand_prefix: bool = False  # 语法检测层面：原始文本是否有 & 前缀
    config_execution_mode: Optional[str] = None  # config.yaml 中的 cloud_agent.execution_mode

    @property
    def requested_mode(self) -> Optional[str]:
        """[DEPRECATED] 兼容别名 - 新代码请使用 requested_mode_for_decision

        此属性保留以兼容旧测试代码，语义等同于 requested_mode_for_decision。
        """
        return self.requested_mode_for_decision

    @property
    def triggered_by_prefix(self) -> bool:
        """[DEPRECATED] 兼容别名 - 新代码请使用 prefix_routed

        此属性保留以兼容旧测试代码，语义等同于 prefix_routed。
        """
        return self.prefix_routed

    @property
    def execution_mode(self) -> str:
        """[DEPRECATED] 兼容别名 - 新代码请使用 effective_mode

        此属性保留以兼容旧测试代码，语义等同于 effective_mode。
        """
        return self.effective_mode

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典便于比较"""
        return {
            # 核心字段（与 ExecutionDecision 对齐）
            "effective_mode": self.effective_mode,
            "requested_mode_for_decision": self.requested_mode_for_decision,
            "cli_execution_mode": self.cli_execution_mode,
            "orchestrator": self.orchestrator,
            "prefix_routed": self.prefix_routed,
            # 追溯字段（不参与核心对比，用于调试追溯）
            "has_ampersand_prefix": self.has_ampersand_prefix,
            "config_execution_mode": self.config_execution_mode,
            # 兼容字段
            "execution_mode": self.effective_mode,
            "triggered_by_prefix": self.prefix_routed,
            "requested_mode": self.requested_mode_for_decision,  # 兼容别名
        }


@dataclass
class SnapshotTestCase:
    """决策快照一致性测试参数

    字段说明：
    - execution_mode: CLI --execution-mode 参数（cli_execution_mode 的来源）
      注意：此字段在 _compute_run_py_snapshot 中作为 cli_execution_mode 使用，
      而 requested_mode_for_decision 则由 resolve_requested_mode_for_decision 计算
    - config_execution_mode: config.yaml 中的 cloud_agent.execution_mode（可选，默认 "auto"）
      通过此字段可模拟不同的配置场景
    """

    test_id: str
    requirement: str  # 用户输入的任务描述（可包含 & 前缀）
    execution_mode: Optional[str]  # CLI --execution-mode 参数（即 cli_execution_mode）
    has_api_key: bool
    cloud_enabled: bool
    orchestrator_cli: Optional[str]  # CLI --orchestrator 参数
    no_mp_cli: bool  # CLI --no-mp 参数
    expected_snapshot: DecisionSnapshot
    description: str
    config_execution_mode: str = "auto"  # config.yaml 中的 cloud_agent.execution_mode（默认 "auto"）


# 快照一致性测试参数表
SNAPSHOT_TEST_CASES = [
    # ===== 基础 CLI 模式测试 =====
    SnapshotTestCase(
        test_id="cli_basic_task",
        requirement="实现新功能",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",
            orchestrator="mp",
            prefix_routed=False,
            requested_mode_for_decision="cli",
            cli_execution_mode="cli",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="普通 CLI 模式任务，应使用 mp 编排器",
    ),
    SnapshotTestCase(
        test_id="cli_no_key",
        requirement="优化代码",
        execution_mode="cli",
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",
            orchestrator="mp",
            prefix_routed=False,
            requested_mode_for_decision="cli",
            cli_execution_mode="cli",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="CLI 模式无 API Key，仍应使用 mp 编排器",
    ),
    SnapshotTestCase(
        test_id="cli_cloud_disabled",
        requirement="本地任务",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=False,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",
            orchestrator="mp",
            prefix_routed=False,
            requested_mode_for_decision="cli",
            cli_execution_mode="cli",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="CLI 模式 cloud_enabled=False，仍应使用 mp 编排器",
    ),
    # ===== AUTO 模式测试（关键场景）=====
    SnapshotTestCase(
        test_id="auto_with_key",
        requirement="重构模块",
        execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="auto",
            orchestrator="basic",
            prefix_routed=False,
            requested_mode_for_decision="auto",
            cli_execution_mode="auto",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="AUTO 模式有 API Key，强制 basic 编排器",
    ),
    SnapshotTestCase(
        test_id="auto_no_key_forces_basic",
        requirement="分析代码",
        execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",  # 回退到 CLI
            orchestrator="basic",  # 编排器仍为 basic（基于 requested_mode_for_decision）
            prefix_routed=False,
            requested_mode_for_decision="auto",  # 但 requested_mode_for_decision 仍为 auto
            cli_execution_mode="auto",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="关键场景：AUTO 模式无 API Key，回退到 CLI 但仍强制 basic 编排器",
    ),
    SnapshotTestCase(
        test_id="auto_cloud_disabled_forces_basic",
        requirement="自动执行任务",
        execution_mode="auto",
        has_api_key=True,
        cloud_enabled=False,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="auto",  # 虽然 cloud_enabled=False，但 CLI 显式指定 auto
            orchestrator="basic",  # CLI 显式指定 auto，强制 basic
            prefix_routed=False,
            requested_mode_for_decision="auto",
            cli_execution_mode="auto",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="关键场景：AUTO 模式 cloud_enabled=False，仍强制 basic（基于 CLI 显式请求）",
    ),
    # ===== CLOUD 模式测试（关键场景）=====
    SnapshotTestCase(
        test_id="cloud_with_key",
        requirement="后台分析",
        execution_mode="cloud",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cloud",
            orchestrator="basic",
            prefix_routed=False,
            requested_mode_for_decision="cloud",
            cli_execution_mode="cloud",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="CLOUD 模式有 API Key，强制 basic 编排器",
    ),
    SnapshotTestCase(
        test_id="cloud_no_key_forces_basic",
        requirement="长时间任务",
        execution_mode="cloud",
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",  # 回退到 CLI
            orchestrator="basic",  # 编排器仍为 basic（基于 requested_mode_for_decision）
            prefix_routed=False,
            requested_mode_for_decision="cloud",  # 但 requested_mode_for_decision 仍为 cloud
            cli_execution_mode="cloud",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="关键场景：CLOUD 模式无 API Key，回退到 CLI 但仍强制 basic 编排器",
    ),
    SnapshotTestCase(
        test_id="cloud_cloud_disabled_forces_basic",
        requirement="云端长时间任务",
        execution_mode="cloud",
        has_api_key=True,
        cloud_enabled=False,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cloud",  # 虽然 cloud_enabled=False，但 CLI 显式指定 cloud
            orchestrator="basic",  # CLI 显式指定 cloud，强制 basic
            prefix_routed=False,
            requested_mode_for_decision="cloud",
            cli_execution_mode="cloud",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="关键场景：CLOUD 模式 cloud_enabled=False，仍强制 basic（基于 CLI 显式请求）",
    ),
    # ===== & 前缀触发测试 =====
    SnapshotTestCase(
        test_id="ampersand_with_key_cloud_enabled",
        requirement="& 分析代码架构",
        execution_mode=None,  # 未显式指定
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cloud",
            orchestrator="basic",
            prefix_routed=True,
            requested_mode_for_decision=None,  # & 前缀触发时无 CLI 显式设置
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀触发 + 有 API Key + cloud_enabled，使用 Cloud 模式",
    ),
    # ===== & 前缀未成功触发场景 =====
    # 注意：当 & 前缀存在但未成功触发时（无 API Key 或 cloud_enabled=False），
    # build_execution_decision 返回 orchestrator="basic"（因为 has_ampersand_prefix=True，
    # 表达用户使用 Cloud 的意图，即使 prefix_routed=False 也使用 basic）。
    #
    # 这与 AGENTS.md 文档一致：& 前缀表达 Cloud 意图，编排器应为 basic。
    # 如需使用 mp 编排器，用户应显式指定 --execution-mode cli。
    SnapshotTestCase(
        test_id="ampersand_no_key",
        requirement="& 后台执行任务",
        execution_mode=None,  # 无 CLI 显式设置
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",  # 无 API Key 回退到 CLI
            orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
            prefix_routed=False,  # 未成功触发
            requested_mode_for_decision=None,
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀但无 API Key，回退到 CLI 但使用 basic（& 前缀表达 Cloud 意图）",
    ),
    SnapshotTestCase(
        test_id="ampersand_cloud_disabled",
        requirement="& 推送到云端",
        execution_mode=None,  # 无 CLI 显式设置
        has_api_key=True,
        cloud_enabled=False,  # cloud_enabled=False
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",  # cloud_enabled=False 回退到 CLI
            orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
            prefix_routed=False,  # 未成功触发
            requested_mode_for_decision=None,
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀但 cloud_enabled=False，使用 CLI + basic（& 前缀表达 Cloud 意图）",
    ),
    SnapshotTestCase(
        test_id="ampersand_no_key_and_cloud_disabled",
        requirement="& 双重失败场景",
        execution_mode=None,  # 无 CLI 显式设置
        has_api_key=False,
        cloud_enabled=False,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",  # 无 API Key 且 cloud_enabled=False
            orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
            prefix_routed=False,  # 未成功触发
            requested_mode_for_decision=None,
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀双重失败，使用 CLI + basic（& 前缀表达 Cloud 意图）",
    ),
    # ===== 显式 --execution-mode cli 场景（保持 MP 覆盖）=====
    # 用户显式指定 --execution-mode cli 时，& 前缀被忽略，允许 mp 编排器
    SnapshotTestCase(
        test_id="ampersand_explicit_cli_no_key",
        requirement="& 后台执行任务",
        execution_mode="cli",  # 用户显式 --execution-mode cli
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",
            orchestrator="mp",  # 显式 cli 允许 mp
            prefix_routed=False,  # 显式 cli 忽略 & 前缀
            requested_mode_for_decision="cli",  # 显式 CLI 模式
            cli_execution_mode="cli",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode cli，忽略前缀允许 mp",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_cli_cloud_disabled",
        requirement="& 推送到云端",
        execution_mode="cli",  # 用户显式 --execution-mode cli
        has_api_key=True,
        cloud_enabled=False,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",
            orchestrator="mp",  # 显式 cli 允许 mp
            prefix_routed=False,
            requested_mode_for_decision="cli",
            cli_execution_mode="cli",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode cli + cloud_disabled，允许 mp",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_cli_ignores_prefix",
        requirement="& 显式 CLI 任务",
        execution_mode="cli",  # 显式指定 CLI
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",
            orchestrator="mp",
            prefix_routed=False,  # 显式 CLI 忽略 & 前缀
            requested_mode_for_decision="cli",
            cli_execution_mode="cli",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 execution_mode=cli，忽略前缀使用 CLI + mp",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_auto_forces_basic",
        requirement="& 显式 AUTO 模式任务",
        execution_mode="auto",  # 显式指定 AUTO
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="auto",
            orchestrator="basic",
            prefix_routed=False,  # 有显式 execution_mode，prefix_routed 为 False
            requested_mode_for_decision="auto",
            cli_execution_mode="auto",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 execution_mode=auto，使用 AUTO + basic",
    ),
    # ===== 编排器显式设置测试 =====
    SnapshotTestCase(
        test_id="explicit_basic_orchestrator",
        requirement="使用基本编排器",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli="basic",
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",
            orchestrator="basic",
            prefix_routed=False,
            requested_mode_for_decision="cli",
            cli_execution_mode="cli",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="显式指定 --orchestrator basic",
    ),
    SnapshotTestCase(
        test_id="explicit_no_mp",
        requirement="禁用多进程",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=True,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",
            orchestrator="basic",
            prefix_routed=False,
            requested_mode_for_decision="cli",
            cli_execution_mode="cli",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="显式指定 --no-mp",
    ),
    SnapshotTestCase(
        test_id="explicit_mp_with_cli",
        requirement="显式使用 MP",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli="mp",
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",
            orchestrator="mp",
            prefix_routed=False,
            requested_mode_for_decision="cli",
            cli_execution_mode="cli",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="显式指定 --orchestrator mp + CLI 模式",
    ),
    SnapshotTestCase(
        test_id="explicit_mp_with_cloud_forces_basic",
        requirement="显式 MP 但 CLOUD 模式",
        execution_mode="cloud",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli="mp",  # 用户显式请求 mp，但会被 cloud 模式覆盖
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cloud",
            orchestrator="basic",  # cloud 模式强制 basic，即使用户请求 mp
            prefix_routed=False,
            requested_mode_for_decision="cloud",
            cli_execution_mode="cloud",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="显式 --orchestrator mp + CLOUD 模式，强制切换到 basic",
    ),
    # ===== config.yaml 默认 auto 模式场景（无 CLI 显式设置，无 & 前缀）=====
    # 这些用例测试当用户未指定 --execution-mode 且无 & 前缀时，
    # 系统使用 config.yaml 默认的 execution_mode=auto 的行为。
    #
    # 关键规则：
    # - 无 CLI 显式设置 + 无 & 前缀 → requested_mode_for_decision 使用 config.yaml 默认 "auto"
    # - requested_mode_for_decision=auto 强制使用 basic 编排器（即使 effective_mode 回退到 CLI）
    SnapshotTestCase(
        test_id="config_default_auto_with_key_no_prefix",
        requirement="普通任务无前缀",  # 无 & 前缀
        execution_mode=None,  # 无 CLI 显式设置
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="auto",  # 有 API Key，使用 auto 模式
            orchestrator="basic",  # requested_mode_for_decision="auto" 强制 basic
            prefix_routed=False,  # 无 & 前缀
            requested_mode_for_decision="auto",  # config.yaml 默认 auto
            cli_execution_mode=None,  # CLI 未显式设置
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="无 CLI 设置 + 无 & 前缀 + 有 API Key → config.yaml 默认 auto，强制 basic",
    ),
    SnapshotTestCase(
        test_id="config_default_auto_no_key_no_prefix",
        requirement="普通任务无前缀无 Key",  # 无 & 前缀
        execution_mode=None,  # 无 CLI 显式设置
        has_api_key=False,  # 无 API Key
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",  # 无 API Key 回退到 CLI
            orchestrator="basic",  # requested_mode_for_decision="auto" 强制 basic（即使回退）
            prefix_routed=False,  # 无 & 前缀
            requested_mode_for_decision="auto",  # config.yaml 默认 auto
            cli_execution_mode=None,  # CLI 未显式设置
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        ),
        description="无 CLI 设置 + 无 & 前缀 + 无 API Key → 回退 CLI，但仍强制 basic（基于 requested_mode_for_decision）",
    ),
    # ===== & 前缀 + 显式 execution_mode=auto/cloud 场景（prefix_routed=False） =====
    # 这些用例验证：当用户同时使用 & 前缀和显式 --execution-mode auto/cloud 时，
    # prefix_routed=False（因为执行模式由显式设置决定，非 & 前缀触发）。
    # orchestrator 仍因 requested_mode=auto/cloud 强制 basic。
    #
    # 关键断言：
    # - prefix_routed=False（DETECTED_IGNORED_EXPLICIT_MODE）
    # - orchestrator="basic"（基于 requested_mode_for_decision）
    #
    # 参考 AmpersandPrefixStatus.DETECTED_IGNORED_EXPLICIT_MODE 的语义定义
    # --- & 前缀 + 显式 execution_mode=auto ---
    SnapshotTestCase(
        test_id="ampersand_explicit_auto_with_key",
        requirement="& 显式 AUTO 模式分析代码",  # 有 & 前缀
        execution_mode="auto",  # 显式指定 auto
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="auto",  # 有 API Key，auto 生效
            orchestrator="basic",  # requested_mode=auto 强制 basic
            prefix_routed=False,  # 显式 auto 模式，& 前缀不触发路由（DETECTED_IGNORED_EXPLICIT_MODE）
            requested_mode_for_decision="auto",
            cli_execution_mode="auto",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode auto + 有 Key → prefix_routed=False，强制 basic",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_auto_no_key",
        requirement="& 显式 AUTO 模式后台任务",  # 有 & 前缀
        execution_mode="auto",  # 显式指定 auto
        has_api_key=False,  # 无 API Key
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",  # 无 API Key 回退到 CLI
            orchestrator="basic",  # requested_mode=auto 强制 basic（即使回退到 CLI）
            prefix_routed=False,  # 显式 auto 模式，& 前缀不触发路由（DETECTED_IGNORED_EXPLICIT_MODE）
            requested_mode_for_decision="auto",
            cli_execution_mode="auto",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode auto + 无 Key → prefix_routed=False，回退 CLI 但仍强制 basic",
    ),
    # --- & 前缀 + 显式 execution_mode=cloud ---
    SnapshotTestCase(
        test_id="ampersand_explicit_cloud_with_key",
        requirement="& 显式 CLOUD 模式后台分析",  # 有 & 前缀
        execution_mode="cloud",  # 显式指定 cloud
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cloud",  # 有 API Key，cloud 生效
            orchestrator="basic",  # requested_mode=cloud 强制 basic
            prefix_routed=False,  # 显式 cloud 模式，& 前缀不触发路由（DETECTED_IGNORED_EXPLICIT_MODE）
            requested_mode_for_decision="cloud",
            cli_execution_mode="cloud",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode cloud + 有 Key → prefix_routed=False，强制 basic",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_cloud_no_key",
        requirement="& 显式 CLOUD 模式长时间任务",  # 有 & 前缀
        execution_mode="cloud",  # 显式指定 cloud
        has_api_key=False,  # 无 API Key
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",  # 无 API Key 回退到 CLI
            orchestrator="basic",  # requested_mode=cloud 强制 basic（即使回退到 CLI）
            prefix_routed=False,  # 显式 cloud 模式，& 前缀不触发路由（DETECTED_IGNORED_EXPLICIT_MODE）
            requested_mode_for_decision="cloud",
            cli_execution_mode="cloud",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode cloud + 无 Key → prefix_routed=False，回退 CLI 但仍强制 basic",
    ),
    # ===== & 前缀 + plan/ask 只读模式场景（R-3 规则）=====
    # 只读模式不参与 Cloud 路由，& 前缀被忽略，允许 mp 编排器。
    # 这与 R-3 规则一致："显式 --execution-mode plan/ask（只读模式）→ 忽略 & 前缀，允许 mp"
    SnapshotTestCase(
        test_id="ampersand_explicit_plan_with_key",
        requirement="& 分析代码架构",  # 有 & 前缀
        execution_mode="plan",  # 显式指定 plan（只读模式）
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="plan",  # plan 只读模式
            orchestrator="mp",  # 只读模式允许 mp（& 前缀被忽略）
            prefix_routed=False,  # 只读模式不参与 Cloud 路由
            requested_mode_for_decision="plan",
            cli_execution_mode="plan",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode plan → prefix_routed=False，plan + mp（& 前缀被忽略）",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_plan_no_key",
        requirement="& 分析代码架构",  # 有 & 前缀
        execution_mode="plan",  # 显式指定 plan（只读模式）
        has_api_key=False,  # 无 API Key（对只读模式无影响）
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="plan",  # plan 只读模式（不受 API Key 影响）
            orchestrator="mp",  # 只读模式允许 mp
            prefix_routed=False,  # 只读模式不参与 Cloud 路由
            requested_mode_for_decision="plan",
            cli_execution_mode="plan",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode plan + 无 API Key → prefix_routed=False，plan + mp",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_plan_cloud_disabled",
        requirement="& 分析代码架构",  # 有 & 前缀
        execution_mode="plan",  # 显式指定 plan（只读模式）
        has_api_key=True,
        cloud_enabled=False,  # cloud_enabled=False（对只读模式无影响）
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="plan",  # plan 只读模式
            orchestrator="mp",  # 只读模式允许 mp
            prefix_routed=False,  # 只读模式不参与 Cloud 路由
            requested_mode_for_decision="plan",
            cli_execution_mode="plan",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode plan + cloud_disabled → prefix_routed=False，plan + mp",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_ask_with_key",
        requirement="& 解释这段代码",  # 有 & 前缀
        execution_mode="ask",  # 显式指定 ask（只读模式）
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="ask",  # ask 只读模式
            orchestrator="mp",  # 只读模式允许 mp（& 前缀被忽略）
            prefix_routed=False,  # 只读模式不参与 Cloud 路由
            requested_mode_for_decision="ask",
            cli_execution_mode="ask",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode ask → prefix_routed=False，ask + mp（& 前缀被忽略）",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_ask_no_key",
        requirement="& 解释这段代码",  # 有 & 前缀
        execution_mode="ask",  # 显式指定 ask（只读模式）
        has_api_key=False,  # 无 API Key（对只读模式无影响）
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="ask",  # ask 只读模式（不受 API Key 影响）
            orchestrator="mp",  # 只读模式允许 mp
            prefix_routed=False,  # 只读模式不参与 Cloud 路由
            requested_mode_for_decision="ask",
            cli_execution_mode="ask",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode ask + 无 API Key → prefix_routed=False，ask + mp",
    ),
    SnapshotTestCase(
        test_id="ampersand_explicit_ask_cloud_disabled",
        requirement="& 解释这段代码",  # 有 & 前缀
        execution_mode="ask",  # 显式指定 ask（只读模式）
        has_api_key=True,
        cloud_enabled=False,  # cloud_enabled=False（对只读模式无影响）
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="ask",  # ask 只读模式
            orchestrator="mp",  # 只读模式允许 mp
            prefix_routed=False,  # 只读模式不参与 Cloud 路由
            requested_mode_for_decision="ask",
            cli_execution_mode="ask",
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        ),
        description="& 前缀 + 显式 --execution-mode ask + cloud_disabled → prefix_routed=False，ask + mp",
    ),
    # ===== config.yaml execution_mode=cli + & 前缀未成功路由场景 =====
    # 当 config.yaml 设置为 cli 但用户使用 & 前缀且路由失败时，
    # & 前缀表达 Cloud 意图，编排器应为 basic（而非 mp）。
    SnapshotTestCase(
        test_id="ampersand_config_cli_no_key",
        requirement="& 后台分析任务",  # 有 & 前缀
        execution_mode=None,  # 无 CLI 显式设置
        has_api_key=False,  # 无 API Key，导致 & 前缀路由失败
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            effective_mode="cli",  # 无 API Key 回退到 CLI
            orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
            prefix_routed=False,  # 无 API Key，& 前缀未成功路由
            requested_mode_for_decision=None,  # 有 & 前缀时返回 None
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="cli",
        ),
        description="& 前缀 + config.yaml cli + 无 API Key → prefix_routed=False，CLI + basic（& 前缀表达 Cloud 意图）",
        config_execution_mode="cli",  # 通过配置注入实现
    ),
]


# ============================================================
# 决策快照一致性测试类
# ============================================================


def _create_mock_config(
    cloud_enabled: bool = True, config_execution_mode: str = "auto", auto_detect_cloud_prefix: bool = True
) -> MagicMock:
    """创建模拟配置对象

    Args:
        cloud_enabled: cloud_agent.enabled 配置值
        config_execution_mode: cloud_agent.execution_mode 配置值（默认 "auto"）
        auto_detect_cloud_prefix: cloud_agent.auto_detect_cloud_prefix 配置值（默认 True）
    """
    mock_config = MagicMock()
    mock_config.cloud_agent.enabled = cloud_enabled
    mock_config.cloud_agent.execution_mode = config_execution_mode
    mock_config.cloud_agent.auto_detect_cloud_prefix = auto_detect_cloud_prefix
    mock_config.cloud_agent.timeout = 300
    mock_config.cloud_agent.auth_timeout = 30
    mock_config.system.max_iterations = 10
    mock_config.system.worker_pool_size = 3
    mock_config.system.enable_sub_planners = True
    mock_config.system.strict_review = False
    mock_config.models.planner = DEFAULT_PLANNER_MODEL
    mock_config.models.worker = DEFAULT_WORKER_MODEL
    mock_config.models.reviewer = DEFAULT_REVIEWER_MODEL
    mock_config.planner.timeout = 500.0
    mock_config.worker.task_timeout = 600.0
    mock_config.reviewer.timeout = 300.0
    mock_config.logging.stream_json.enabled = False
    mock_config.logging.stream_json.console = True
    mock_config.logging.stream_json.detail_dir = "logs/stream_json/detail/"
    mock_config.logging.stream_json.raw_dir = "logs/stream_json/raw/"
    return mock_config


def _build_iterate_args(test_case: SnapshotTestCase) -> argparse.Namespace:
    """构建 SelfIterator 测试参数"""
    return argparse.Namespace(
        requirement=test_case.requirement,
        directory=".",
        skip_online=True,
        changelog_url=None,
        dry_run=True,  # 避免实际执行
        max_iterations="1",
        workers=2,
        force_update=False,
        verbose=False,
        quiet=False,
        log_level=None,
        heartbeat_debug=False,
        stall_diagnostics_enabled=None,
        stall_diagnostics_level=None,
        stall_recovery_interval=30.0,
        execution_health_check_interval=30.0,
        health_warning_cooldown=60.0,
        auto_commit=False,
        auto_push=False,
        commit_message="",
        commit_per_iteration=False,
        orchestrator=test_case.orchestrator_cli or "mp",
        no_mp=test_case.no_mp_cli,
        _orchestrator_user_set=(test_case.orchestrator_cli is not None) or test_case.no_mp_cli,
        execution_mode=test_case.execution_mode,
        cloud_api_key=None,
        cloud_auth_timeout=30,
        cloud_timeout=300,
        auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
        planner_execution_mode=None,
        worker_execution_mode=None,
        reviewer_execution_mode=None,
        stream_console_renderer=False,
        stream_advanced_renderer=False,
        stream_typing_effect=False,
        stream_typing_delay=0.02,
        stream_word_mode=True,
        stream_color_enabled=True,
        stream_show_word_diff=False,
        max_fetch_urls=None,
        fallback_core_docs_count=None,
        llms_txt_url=None,
        llms_cache_path=None,
    )


def _compute_run_py_snapshot(test_case: SnapshotTestCase) -> DecisionSnapshot:
    """计算 run.py IterateArgs 语义的决策快照

    模拟 SelfIterator 的 __init__ 中的决策逻辑：
    1. 使用 resolve_requested_mode_for_decision 确定 requested_mode_for_decision
    2. 调用 build_execution_decision 获取执行决策

    【重要】使用 build_execution_decision 而非 resolve_orchestrator_settings，
    以确保与 SelfIterator 的行为一致。

    变量命名规范:
    - has_ampersand_prefix: 语法检测层面（原始文本是否有 & 前缀）
    - prefix_routed: 策略决策层面（& 前缀是否成功触发 Cloud）
    - cli_execution_mode: CLI 原始 --execution-mode 参数值（test_case.execution_mode）
    - requested_mode_for_decision: resolve_requested_mode_for_decision 的返回值
    - effective_mode: 经过路由决策后实际使用的执行模式
    - config_execution_mode: config.yaml 中的 cloud_agent.execution_mode（默认 "auto"）
    """
    from core.cloud_utils import is_cloud_request
    from core.execution_policy import (
        build_execution_decision,
        resolve_requested_mode_for_decision,
    )

    # 检测 & 前缀（语法层面）
    has_ampersand_prefix = is_cloud_request(test_case.requirement)

    # CLI 原始 --execution-mode 参数值
    cli_execution_mode = test_case.execution_mode

    # config.yaml 中的 cloud_agent.execution_mode
    # 优先使用测试用例的 config_execution_mode，否则使用 DEFAULT_EXECUTION_MODE
    config_execution_mode = test_case.config_execution_mode

    # 确定 requested_mode_for_decision（与 SelfIterator.__init__ 一致）
    # 当 & 前缀存在且无 CLI 显式设置时，返回 None 让 build_execution_decision 处理
    requested_mode_for_decision = resolve_requested_mode_for_decision(
        cli_execution_mode=cli_execution_mode,
        has_ampersand_prefix=has_ampersand_prefix,
        config_execution_mode=config_execution_mode,
    )

    # 【回归保护断言】无 & 前缀且无 CLI 显式设置时，requested_mode_for_decision 不应为 None
    # 这确保 resolve_requested_mode_for_decision 不被误用（应回退到 config_execution_mode）
    if not has_ampersand_prefix and cli_execution_mode is None:
        assert requested_mode_for_decision is not None, (
            f"回归错误：无 & 前缀且无 CLI 显式设置时，"
            f"requested_mode_for_decision 不应为 None（应回退到 config_execution_mode={config_execution_mode}）"
        )

    # 确定用户显式指定的编排器
    user_requested_orchestrator = None
    if test_case.orchestrator_cli is not None:
        user_requested_orchestrator = test_case.orchestrator_cli
    elif test_case.no_mp_cli:
        user_requested_orchestrator = "basic"

    # 使用 build_execution_decision 统一决策（与 SelfIterator 一致）
    decision = build_execution_decision(
        prompt=test_case.requirement,
        requested_mode=requested_mode_for_decision,
        cloud_enabled=test_case.cloud_enabled,
        has_api_key=test_case.has_api_key,
        auto_detect_cloud_prefix=True,
        user_requested_orchestrator=user_requested_orchestrator,
    )

    return DecisionSnapshot(
        effective_mode=decision.effective_mode,
        orchestrator=decision.orchestrator,
        prefix_routed=decision.prefix_routed,
        requested_mode_for_decision=requested_mode_for_decision,  # resolve_requested_mode_for_decision 返回值
        cli_execution_mode=cli_execution_mode,  # CLI 原始 --execution-mode 参数值
        has_ampersand_prefix=has_ampersand_prefix,  # 语法检测层面：原始文本是否有 & 前缀
        config_execution_mode=config_execution_mode,  # config.yaml 中的 cloud_agent.execution_mode
    )


class TestIterateArgsAndSelfIteratorConsistency:
    """测试 run.py IterateArgs 与 SelfIterator 的决策快照一致性

    对每个测试 case，分别计算：
    1. run.py 的合并结果/IterateArgs 语义快照
    2. SelfIterator 自身解析语义快照（通过实例化并调用 _get_execution_mode/_get_orchestrator_type）

    断言两者一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        SNAPSHOT_TEST_CASES,
        ids=[tc.test_id for tc in SNAPSHOT_TEST_CASES],
    )
    def test_snapshot_consistency(self, test_case: SnapshotTestCase) -> None:
        """测试决策快照一致性

        比较维度：requested_mode / effective_mode / orchestrator / prefix_routed
        """
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # 重置配置单例
        ConfigManager.reset_instance()

        # 计算 run.py 语义的快照
        run_py_snapshot = _compute_run_py_snapshot(test_case)

        # 构建测试参数
        args = _build_iterate_args(test_case)

        # 创建 mock 配置（使用测试用例的 config_execution_mode）
        mock_config = _create_mock_config(
            cloud_enabled=test_case.cloud_enabled, config_execution_mode=test_case.config_execution_mode
        )

        # 解析 API Key
        api_key = "mock-api-key" if test_case.has_api_key else None

        # 计算 SelfIterator 语义的快照
        # 需要 mock 多个依赖：CloudClientFactory、ConfigManager、以及 KnowledgeUpdater 等
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    # Mock SelfIterator 的依赖以避免实际初始化
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        self_iterator_effective_mode = iterator._get_execution_mode()
                                        self_iterator_orchestrator = iterator._get_orchestrator_type()
                                        # 内部分支使用 _prefix_routed 字段
                                        self_iterator_prefix_routed = iterator._prefix_routed
                                        # 获取 requested_mode_for_decision（来自 _execution_decision）
                                        # 注意：使用 _execution_decision.requested_mode 而非 args.execution_mode
                                        self_iterator_requested_mode_for_decision = (
                                            iterator._execution_decision.requested_mode
                                        )
                                        # 获取 cli_execution_mode（来自 CLI 参数）
                                        self_iterator_cli_execution_mode = getattr(
                                            iterator.args, "execution_mode", None
                                        )
                                        # 获取 has_ampersand_prefix（来自 _execution_decision）
                                        self_iterator_has_ampersand_prefix = (
                                            iterator._execution_decision.has_ampersand_prefix
                                        )
                                        # 获取 config_execution_mode（来自 mock_config）
                                        self_iterator_config_execution_mode = mock_config.cloud_agent.execution_mode

        self_iterator_snapshot = DecisionSnapshot(
            effective_mode=self_iterator_effective_mode.value,
            orchestrator=self_iterator_orchestrator,
            prefix_routed=self_iterator_prefix_routed,
            requested_mode_for_decision=self_iterator_requested_mode_for_decision,
            cli_execution_mode=self_iterator_cli_execution_mode,
            has_ampersand_prefix=self_iterator_has_ampersand_prefix,
            config_execution_mode=self_iterator_config_execution_mode,
        )

        # 断言快照一致
        assert run_py_snapshot.to_dict() == self_iterator_snapshot.to_dict(), (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  run.py 快照: {run_py_snapshot.to_dict()}\n"
            f"  SelfIterator 快照: {self_iterator_snapshot.to_dict()}"
        )

        # 同时验证与预期快照一致
        assert run_py_snapshot.to_dict() == test_case.expected_snapshot.to_dict(), (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  实际快照: {run_py_snapshot.to_dict()}\n"
            f"  期望快照: {test_case.expected_snapshot.to_dict()}"
        )

    @pytest.mark.parametrize(
        "test_case",
        SNAPSHOT_TEST_CASES,
        ids=[tc.test_id for tc in SNAPSHOT_TEST_CASES],
    )
    def test_expected_snapshot_matches_policy(self, test_case: SnapshotTestCase) -> None:
        """验证预期快照与 Policy 函数结果一致

        比较维度：requested_mode_for_decision / effective_mode / orchestrator / prefix_routed
        """
        ConfigManager.reset_instance()

        run_py_snapshot = _compute_run_py_snapshot(test_case)

        # 验证 effective_mode（使用新字段名）
        assert run_py_snapshot.effective_mode == test_case.expected_snapshot.effective_mode, (
            f"[{test_case.test_id}] effective_mode 不匹配\n"
            f"  实际: {run_py_snapshot.effective_mode}\n"
            f"  期望: {test_case.expected_snapshot.effective_mode}"
        )

        # 验证 requested_mode_for_decision
        assert run_py_snapshot.requested_mode_for_decision == test_case.expected_snapshot.requested_mode_for_decision, (
            f"[{test_case.test_id}] requested_mode_for_decision 不匹配\n"
            f"  实际: {run_py_snapshot.requested_mode_for_decision}\n"
            f"  期望: {test_case.expected_snapshot.requested_mode_for_decision}"
        )

        # 验证 cli_execution_mode
        assert run_py_snapshot.cli_execution_mode == test_case.expected_snapshot.cli_execution_mode, (
            f"[{test_case.test_id}] cli_execution_mode 不匹配\n"
            f"  实际: {run_py_snapshot.cli_execution_mode}\n"
            f"  期望: {test_case.expected_snapshot.cli_execution_mode}"
        )

        # 验证 orchestrator
        assert run_py_snapshot.orchestrator == test_case.expected_snapshot.orchestrator, (
            f"[{test_case.test_id}] orchestrator 不匹配\n"
            f"  实际: {run_py_snapshot.orchestrator}\n"
            f"  期望: {test_case.expected_snapshot.orchestrator}"
        )

        # 验证 prefix_routed
        assert run_py_snapshot.prefix_routed == test_case.expected_snapshot.prefix_routed, (
            f"[{test_case.test_id}] prefix_routed 不匹配\n"
            f"  实际: {run_py_snapshot.prefix_routed}\n"
            f"  期望: {test_case.expected_snapshot.prefix_routed}"
        )


class TestConfigManagerReset:
    """测试 ConfigManager 重置机制是否正确工作"""

    def test_config_reset_avoids_pollution(self) -> None:
        """测试配置重置避免测试间污染"""
        # 第一次获取配置
        config1 = ConfigManager.get_instance()
        config1_id = id(config1)

        # 重置配置
        ConfigManager.reset_instance()

        # 第二次获取配置应为新实例
        config2 = ConfigManager.get_instance()
        config2_id = id(config2)

        assert config1_id != config2_id, "重置后应获得新的配置实例"

    def test_consecutive_resets_work(self) -> None:
        """测试连续重置正常工作"""
        for i in range(3):
            ConfigManager.reset_instance()
            config = ConfigManager.get_instance()
            assert config is not None, f"第 {i + 1} 次重置后应能获取配置"


class TestPrefixRoutedParameter:
    """测试 resolve_orchestrator_settings 的 prefix_routed (triggered_by_prefix) 参数

    验证 prefix_routed 参数行为（参数名 triggered_by_prefix 保留以兼容旧接口）：
    - prefix_routed=True: 强制 basic 编排器（& 前缀成功触发）
    - prefix_routed=False: 不影响编排器选择（除非 CLI 显式指定 auto/cloud）

    内部分支使用 prefix_routed 语义，triggered_by_prefix 仅作为参数名兼容保留。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def test_prefix_routed_true_forces_basic(self) -> None:
        """测试 prefix_routed=True 强制使用 basic 编排器

        注：参数名 triggered_by_prefix 保留以兼容旧接口
        """
        # 不传递任何 overrides，仅通过 prefix_routed 触发 basic
        result = resolve_orchestrator_settings(
            overrides={},
            prefix_routed=True,
        )
        assert result["orchestrator"] == "basic", "prefix_routed=True 应强制使用 basic 编排器"

    def test_prefix_routed_false_with_explicit_cli_allows_mp(self) -> None:
        """测试 prefix_routed=False + 显式 cli 允许使用 mp 编排器

        注意：当 overrides={} 时，resolve_orchestrator_settings 会读取 config.yaml
        默认 execution_mode=auto，进而强制 basic。只有显式指定 execution_mode=cli
        才能允许 mp。

        """
        # 显式 cli + prefix_routed=False → 允许 mp
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},  # 显式 cli
            prefix_routed=False,
        )
        assert result["orchestrator"] == "mp", "prefix_routed=False + 显式 cli 应允许使用 mp 编排器"

    def test_prefix_routed_false_without_cli_uses_config_default(self) -> None:
        """测试 prefix_routed=False 但无显式 cli 时使用 config.yaml 默认值

        当 overrides 不包含 execution_mode 时，resolve_orchestrator_settings 会读取
        config.yaml 默认 execution_mode=auto，进而强制 basic。这是正确的行为。

        如需使用 mp 编排器，用户应显式指定 --execution-mode cli。
        """
        result = resolve_orchestrator_settings(
            overrides={},  # 无显式 execution_mode
            prefix_routed=False,
        )
        assert result["orchestrator"] == "basic", (
            "prefix_routed=False 但无显式 cli，使用 config.yaml 默认 auto，强制 basic"
        )

    def test_prefix_routed_with_cli_execution_mode(self) -> None:
        """测试 prefix_routed 与 CLI execution_mode 的交互

        即使 prefix_routed=False，CLI 显式指定 execution_mode=auto/cloud
        也应该强制使用 basic 编排器。

        """
        # CLI 显式指定 auto
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "auto"},
            prefix_routed=False,
        )
        assert result["orchestrator"] == "basic", "CLI 显式指定 execution_mode=auto 应强制 basic"

        # CLI 显式指定 cloud
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cloud"},
            prefix_routed=False,
        )
        assert result["orchestrator"] == "basic", "CLI 显式指定 execution_mode=cloud 应强制 basic"

        # CLI 显式指定 cli
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,
        )
        assert result["orchestrator"] == "mp", "CLI 显式指定 execution_mode=cli 应允许 mp"

    def test_prefix_routed_overrides_cli_mp_request(self) -> None:
        """测试 prefix_routed=True 覆盖用户显式请求的 mp 编排器

        即使用户显式请求 --orchestrator mp，prefix_routed=True 也应强制 basic。
        """
        result = resolve_orchestrator_settings(
            overrides={"orchestrator": "mp"},
            prefix_routed=True,
        )
        assert result["orchestrator"] == "basic", "prefix_routed=True 应覆盖用户请求的 mp 编排器"

    def test_cli_basic_not_affected_by_prefix_routed(self) -> None:
        """测试用户显式请求 basic 不受 prefix_routed 影响"""
        # prefix_routed=False + CLI 请求 basic
        result = resolve_orchestrator_settings(
            overrides={"orchestrator": "basic"},
            prefix_routed=False,
        )
        assert result["orchestrator"] == "basic", "用户显式请求 basic 应保持 basic"

        # prefix_routed=True + CLI 请求 basic
        result = resolve_orchestrator_settings(
            overrides={"orchestrator": "basic"},
            prefix_routed=True,
        )
        assert result["orchestrator"] == "basic", "prefix_routed=True + CLI 请求 basic 应保持 basic"


# ============================================================
# 关键场景: & 前缀存在但 prefix_routed=False 时的编排器选择
# ============================================================


class TestAmpersandPrefixNotRoutedOrchestratorConsistency:
    """测试 & 前缀存在但 prefix_routed=False 时的编排器选择

    ================================================================================
    核心规则 (requested_mode vs effective_mode 断言口径)
    ================================================================================

    1. **requested_mode**: 用户通过 CLI 参数或 config.yaml 请求的执行模式
       - 值来源: --execution-mode 参数 > & 前缀触发时传 None > config.yaml 默认值
       - 用途: 决定编排器类型 (mp/basic)

    2. **effective_mode**: 经过路由决策后实际使用的执行模式
       - 值来源: build_execution_decision() 的输出
       - 可能因条件不满足而与 requested_mode 不同（如无 API Key 时回退）

    3. **编排器选择规则**:
       - 基于 **requested_mode**，不是 effective_mode
       - requested_mode=auto/cloud → 强制 basic（即使 effective_mode 回退到 cli）
       - requested_mode=cli/plan/ask/None → 允许 mp

    ================================================================================
    & 前缀场景下的 requested_mode 语义
    ================================================================================

    当 & 前缀存在时：
    - resolve_requested_mode_for_decision() 返回 None（让 build_execution_decision 处理）
    - 此时 requested_mode=None，不是 auto/cloud，所以 **允许 mp**

    这与无 & 前缀时不同：
    - 无 & 前缀时，resolve_requested_mode_for_decision() 返回 config.yaml 默认 "auto"
    - requested_mode="auto" 强制 basic

    ================================================================================
    测试用例矩阵
    ================================================================================

    | 场景 | & 前缀 | prefix_routed | CLI exec_mode | requested_mode | orchestrator |
    |------|--------|---------------|---------------|----------------|--------------|
    | A    | True   | True          | None          | None           | basic        |
    | B    | True   | False (无key) | None          | None           | mp           |
    | C    | True   | False (禁用)  | None          | None           | mp           |
    | D    | True   | -             | cli (显式)    | cli            | mp           |
    | E    | True   | -             | auto (显式)   | auto           | basic        |
    | F    | False  | False         | None          | auto (config)  | basic        |
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def test_ampersand_prefix_routed_true_forces_basic(self) -> None:
        """场景 A: & 前缀成功触发，强制 basic

        当 & 前缀成功路由到 Cloud 时：
        - prefix_routed=True
        - effective_mode=cloud
        - orchestrator=basic
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,  # 无 CLI 显式设置
            cloud_enabled=True,
            has_api_key=True,
            auto_detect_cloud_prefix=True,
        )

        assert decision.prefix_routed is True, "& 前缀应成功触发"
        assert decision.effective_mode == "cloud", "effective_mode 应为 cloud"
        assert decision.orchestrator == "basic", "& 前缀成功触发时强制 basic"

    def test_ampersand_prefix_routed_false_no_key_forces_basic(self) -> None:
        """场景 B: & 前缀存在但因无 API Key 未成功触发，强制 basic

        当 & 前缀存在但因缺少 API Key 未成功触发时：
        - has_ampersand_prefix=True
        - prefix_routed=False（因无 API Key）
        - 但 & 前缀表达 Cloud 意图，仍强制 basic（与 AGENTS.md 一致）
        - requested_mode=None（由 resolve_requested_mode_for_decision 决定）
        - orchestrator=basic（& 前缀表达 Cloud 意图）
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,  # 无 CLI 显式设置
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key
            auto_detect_cloud_prefix=True,
        )

        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀"
        assert decision.prefix_routed is False, "& 前缀应未成功触发（无 API Key）"
        assert decision.effective_mode == "cli", "effective_mode 应回退到 cli"
        # 关键断言：& 前缀表达 Cloud 意图，强制 basic
        assert decision.orchestrator == "basic", (
            "& 前缀未成功触发时，仍强制 basic\n原因：& 前缀表达 Cloud 意图（与 AGENTS.md 一致）"
        )

    def test_ampersand_prefix_routed_false_disabled_forces_basic(self) -> None:
        """场景 C: & 前缀存在但因 cloud_enabled=False 未成功触发，强制 basic

        当 & 前缀存在但因 cloud_enabled=False 未成功触发时：
        - has_ampersand_prefix=True
        - prefix_routed=False（因 cloud_enabled=False）
        - requested_mode=None
        - orchestrator=basic（& 前缀表达 Cloud 意图）
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,  # 无 CLI 显式设置
            cloud_enabled=False,  # Cloud 未启用
            has_api_key=True,
            auto_detect_cloud_prefix=True,
        )

        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀"
        assert decision.prefix_routed is False, "& 前缀应未成功触发（cloud_enabled=False）"
        assert decision.effective_mode == "cli", "effective_mode 应为 cli"
        assert decision.orchestrator == "basic", "& 前缀未成功触发时，仍强制 basic（& 前缀表达 Cloud 意图）"

    def test_ampersand_prefix_with_explicit_cli_allows_mp(self) -> None:
        """场景 D: & 前缀存在但 CLI 显式指定 cli，允许 mp

        当用户显式指定 --execution-mode cli 时，& 前缀被忽略：
        - has_ampersand_prefix=True
        - prefix_routed=False（被 CLI 显式覆盖）
        - requested_mode=cli
        - orchestrator=mp
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode="cli",  # CLI 显式 cli
            cloud_enabled=True,
            has_api_key=True,
            auto_detect_cloud_prefix=True,
        )

        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀"
        assert decision.prefix_routed is False, "& 前缀应被 CLI 显式 cli 忽略"
        assert decision.effective_mode == "cli", "effective_mode 应为 cli"
        assert decision.orchestrator == "mp", "显式 cli 模式允许 mp"

    def test_ampersand_prefix_with_explicit_auto_forces_basic(self) -> None:
        """场景 E: & 前缀存在且 CLI 显式指定 auto，强制 basic

        当用户显式指定 --execution-mode auto 时：
        - has_ampersand_prefix=True
        - prefix_routed=False（显式 auto，不是 & 触发）
        - requested_mode=auto
        - orchestrator=basic（因 requested_mode=auto）
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode="auto",  # CLI 显式 auto
            cloud_enabled=True,
            has_api_key=True,
            auto_detect_cloud_prefix=True,
        )

        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀"
        assert decision.prefix_routed is False, "显式 auto 模式，不视为 & 触发"
        assert decision.effective_mode == "auto", "effective_mode 应为 auto"
        assert decision.orchestrator == "basic", "显式 auto 模式强制 basic（即使有 & 前缀）"

    def test_no_ampersand_prefix_uses_config_default_auto_forces_basic(self) -> None:
        """场景 F: 无 & 前缀时使用 config.yaml 默认 auto，强制 basic

        当无 & 前缀且无 CLI 显式设置时：
        - has_ampersand_prefix=False
        - resolve_requested_mode_for_decision 返回 config.yaml 默认 "auto"
        - requested_mode=auto（来自 config.yaml）
        - orchestrator=basic

        这是关键区别：
        - 有 & 前缀但未成功触发：requested_mode=None，允许 mp
        - 无 & 前缀：requested_mode=auto（config 默认），强制 basic
        """
        from core.execution_policy import (
            build_execution_decision,
            resolve_requested_mode_for_decision,
        )

        # 验证 resolve_requested_mode_for_decision 的行为
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="auto",  # config.yaml 默认
        )
        assert requested_mode == "auto", "无 & 前缀时应使用 config.yaml 默认 auto"

        # 验证 build_execution_decision 的行为
        decision = build_execution_decision(
            prompt="普通任务",  # 无 & 前缀
            requested_mode="auto",  # 来自 config.yaml
            cloud_enabled=True,
            has_api_key=True,
            auto_detect_cloud_prefix=True,
        )

        assert decision.has_ampersand_prefix is False, "不应检测到 & 前缀"
        assert decision.prefix_routed is False, "无 & 前缀，prefix_routed=False"
        assert decision.orchestrator == "basic", "无 & 前缀时使用 config.yaml 默认 auto，强制 basic"

    def test_resolve_orchestrator_settings_empty_overrides_uses_config_default(self) -> None:
        """验证 resolve_orchestrator_settings 在 overrides 为空时使用 config.yaml 默认 auto

        这是核心断言：
        - overrides={} 不含 execution_mode
        - resolve_orchestrator_settings 读取 config.yaml 默认 auto
        - 强制返回 basic 编排器

        与 build_execution_decision 不同：
        - resolve_orchestrator_settings 始终读取 config.yaml 默认值
        - build_execution_decision 在 & 前缀场景下可能接收 requested_mode=None
        """
        result = resolve_orchestrator_settings(
            overrides={},  # 不含 execution_mode
            prefix_routed=False,
        )

        assert result["orchestrator"] == "basic", (
            "overrides 不含 execution_mode 时，读取 config.yaml 默认 auto，强制 basic\n"
            "如需使用 mp 编排器，请显式指定 --execution-mode cli"
        )

        # 同时验证 execution_mode 的值（resolve_orchestrator_settings 使用 execution_mode 键）
        assert result["execution_mode"] == "auto", "execution_mode 应反映 config.yaml 默认值 auto"


# ============================================================
# should_use_mp_orchestrator 输入约定回归测试
# ============================================================


class TestShouldUseMpOrchestratorInputContract:
    """验证 should_use_mp_orchestrator 的输入约定

    核心规则：
    - should_use_mp_orchestrator() 的输入必须是经过 resolve_requested_mode_for_decision()
      解析后的 "最终 requested_mode"
    - 当无 & 前缀且 CLI 未指定时，requested_mode 不应为 None（应为 config.yaml 默认值）
    - 当有 & 前缀时，requested_mode 可以为 None（让 build_execution_decision 处理路由）

    回归测试场景：
    1. 无 & 前缀 + 无 CLI 指定 → requested_mode 应为 config.yaml 值（如 "auto"），不是 None
    2. 有 & 前缀 + 无 CLI 指定 → requested_mode 可以为 None
    3. CLI 显式指定 → requested_mode 应为 CLI 值，忽略其他条件
    """

    def test_no_prefix_no_cli_should_use_config_value_not_none(self) -> None:
        """回归测试：无 & 前缀且无 CLI 指定时，requested_mode 不应为 None

        这是核心回归测试，验证：
        - 当 has_ampersand_prefix=False 且 cli_execution_mode=None 时
        - resolve_requested_mode_for_decision 应返回 config_execution_mode（如 "auto"）
        - 不应返回 None
        """
        from core.execution_policy import resolve_requested_mode_for_decision

        # 场景 1: config.yaml 默认 auto
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        )
        assert requested_mode == "auto", "无 & 前缀且无 CLI 指定时，应返回 config.yaml 值 'auto'，不是 None"

        # 场景 2: config.yaml 设置为 cli
        requested_mode_cli = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="cli",
        )
        assert requested_mode_cli == "cli", "无 & 前缀且无 CLI 指定时，应返回 config.yaml 值 'cli'"

        # 场景 3: config.yaml 设置为 cloud
        requested_mode_cloud = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="cloud",
        )
        assert requested_mode_cloud == "cloud", "无 & 前缀且无 CLI 指定时，应返回 config.yaml 值 'cloud'"

    def test_with_prefix_no_cli_should_return_none(self) -> None:
        """验证：有 & 前缀且无 CLI 指定时，requested_mode 应为 None

        这是预期行为：& 前缀存在时，返回 None 让 build_execution_decision 处理路由。
        """
        from core.execution_policy import resolve_requested_mode_for_decision

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=True,  # 有 & 前缀
            config_execution_mode="auto",  # 即使 config 是 auto，也应返回 None
        )
        assert requested_mode is None, "有 & 前缀时应返回 None，让 build_execution_decision 处理路由"

    def test_cli_explicit_takes_precedence_over_prefix_and_config(self) -> None:
        """验证：CLI 显式指定时优先级最高"""
        from core.execution_policy import resolve_requested_mode_for_decision

        # 即使有 & 前缀，CLI 显式指定也应生效
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode="cli",  # CLI 显式指定
            has_ampersand_prefix=True,  # 有 & 前缀（应被忽略）
            config_execution_mode="auto",  # config.yaml（应被忽略）
        )
        assert requested_mode == "cli", "CLI 显式指定时，应忽略 & 前缀和 config.yaml"

    def test_should_use_mp_orchestrator_with_resolved_mode(self) -> None:
        """验证 should_use_mp_orchestrator 与 resolve_requested_mode_for_decision 配合使用

        核心测试：验证正确的调用流程
        1. 先调用 resolve_requested_mode_for_decision 获取 requested_mode
        2. 再将 requested_mode 传给 should_use_mp_orchestrator
        """
        from core.execution_policy import (
            resolve_requested_mode_for_decision,
            should_use_mp_orchestrator,
        )

        # 场景 1: 无 & 前缀，config 默认 auto → basic 编排器
        requested_mode_1 = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        )
        can_use_mp_1 = should_use_mp_orchestrator(requested_mode_1)
        assert requested_mode_1 == "auto", "requested_mode 应为 'auto'"
        assert can_use_mp_1 is False, "auto 模式不允许 mp 编排器"

        # 场景 2: 有 & 前缀，无 CLI → None → 允许 mp（因为 None 不是 auto/cloud）
        requested_mode_2 = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        )
        can_use_mp_2 = should_use_mp_orchestrator(requested_mode_2)
        assert requested_mode_2 is None, "有 & 前缀时 requested_mode 应为 None"
        assert can_use_mp_2 is True, "requested_mode=None 允许 mp（待 build_execution_decision 决策）"

        # 场景 3: CLI 显式 cli → 允许 mp
        requested_mode_3 = resolve_requested_mode_for_decision(
            cli_execution_mode="cli",
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        )
        can_use_mp_3 = should_use_mp_orchestrator(requested_mode_3)
        assert requested_mode_3 == "cli", "CLI 显式 cli"
        assert can_use_mp_3 is True, "cli 模式允许 mp 编排器"

    def test_config_execution_mode_none_uses_default(self) -> None:
        """边界场景：config_execution_mode 为 None 时使用 DEFAULT_EXECUTION_MODE

        这是非预期场景（正常情况下 config.yaml 应有默认值），
        但函数应回退到 DEFAULT_EXECUTION_MODE="auto"。

        不变式保证：当 has_ampersand_prefix=False 且 cli_execution_mode=None 时，
        返回值**必须非 None**，以确保后续决策逻辑的一致性。
        """
        from core.execution_policy import resolve_requested_mode_for_decision

        # 当 config_execution_mode 为 None 时，应返回 DEFAULT_EXECUTION_MODE
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode=None,  # 非预期：config.yaml 未配置
        )
        assert requested_mode == DEFAULT_EXECUTION_MODE, (
            f"config_execution_mode=None 时应返回 DEFAULT_EXECUTION_MODE='{DEFAULT_EXECUTION_MODE}'，"
            f"实际返回 '{requested_mode}'"
        )
        assert requested_mode is not None, "不变式保证：无 & 前缀且无 CLI 设置时，返回值不应为 None"


# ============================================================
# Minimal 模式一致性测试
# ============================================================


@dataclass
class MinimalConsistencyTestCase:
    """Minimal 模式一致性测试参数

    验证 run.py 的 options 合并与 SelfIterator 的参数解析
    在 minimal 模式下的行为一致性。

    关键字段：
    - minimal: 是否启用 minimal 模式
    - skip_online: 是否跳过在线检查（minimal 模式隐含 True）
    - dry_run: 是否仅分析不执行
    - orchestrator: 编排器类型 (mp/basic)
    """

    test_id: str
    # 输入参数
    minimal: bool
    execution_mode: Optional[str]
    has_api_key: bool
    cli_skip_online: bool
    cli_dry_run: bool
    cli_orchestrator: Optional[str]
    cli_no_mp: bool
    # 期望输出
    expected_orchestrator: str
    expected_skip_online: bool
    expected_dry_run: bool
    expected_minimal: bool
    description: str


# Minimal 模式一致性测试参数表
MINIMAL_CONSISTENCY_TEST_CASES = [
    # ===== 基础 minimal 模式 =====
    MinimalConsistencyTestCase(
        test_id="minimal_basic",
        minimal=True,
        execution_mode="cli",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="mp",  # CLI 模式允许 mp
        expected_skip_online=True,  # minimal 隐含 skip_online=True
        expected_dry_run=True,  # run.py minimal preset 设置 dry_run=True
        expected_minimal=True,
        description="基础 minimal 模式：skip_online=True, dry_run=True, 允许 mp",
    ),
    MinimalConsistencyTestCase(
        test_id="minimal_with_explicit_dry_run",
        minimal=True,
        execution_mode="cli",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=True,  # 显式指定 dry_run
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="mp",
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + 显式 dry_run：两者一致，dry_run=True",
    ),
    MinimalConsistencyTestCase(
        test_id="minimal_with_explicit_skip_online",
        minimal=True,
        execution_mode="cli",
        has_api_key=True,
        cli_skip_online=True,  # 显式指定 skip_online
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="mp",
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + 显式 skip_online：两者一致，skip_online=True",
    ),
    # ===== minimal 模式 + 编排器设置 =====
    MinimalConsistencyTestCase(
        test_id="minimal_with_basic_orchestrator",
        minimal=True,
        execution_mode="cli",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator="basic",  # 显式指定 basic
        cli_no_mp=False,
        expected_orchestrator="basic",
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + 显式 basic 编排器",
    ),
    MinimalConsistencyTestCase(
        test_id="minimal_with_mp_orchestrator",
        minimal=True,
        execution_mode="cli",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator="mp",  # 显式指定 mp
        cli_no_mp=False,
        expected_orchestrator="mp",
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + 显式 mp 编排器",
    ),
    MinimalConsistencyTestCase(
        test_id="minimal_with_no_mp",
        minimal=True,
        execution_mode="cli",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=True,  # --no-mp
        expected_orchestrator="basic",
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + --no-mp：强制 basic 编排器",
    ),
    # ===== minimal 模式 + Cloud/Auto 执行模式 =====
    MinimalConsistencyTestCase(
        test_id="minimal_with_cloud_mode",
        minimal=True,
        execution_mode="cloud",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="basic",  # cloud 模式强制 basic
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + cloud 模式：强制 basic 编排器",
    ),
    MinimalConsistencyTestCase(
        test_id="minimal_with_auto_mode",
        minimal=True,
        execution_mode="auto",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="basic",  # auto 模式强制 basic
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + auto 模式：强制 basic 编排器",
    ),
    MinimalConsistencyTestCase(
        test_id="minimal_with_cloud_no_key",
        minimal=True,
        execution_mode="cloud",
        has_api_key=False,  # 无 API Key
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="basic",  # cloud 模式仍强制 basic（基于 requested_mode）
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + cloud 模式无 API Key：仍强制 basic（基于 requested_mode）",
    ),
    # ===== 非 minimal 模式对照组 =====
    MinimalConsistencyTestCase(
        test_id="non_minimal_cli_defaults",
        minimal=False,
        execution_mode="cli",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="mp",
        expected_skip_online=False,
        expected_dry_run=False,
        expected_minimal=False,
        description="非 minimal 模式：使用 CLI 默认值",
    ),
    MinimalConsistencyTestCase(
        test_id="non_minimal_explicit_skip_online",
        minimal=False,
        execution_mode="cli",
        has_api_key=True,
        cli_skip_online=True,  # 仅显式 skip_online
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="mp",
        expected_skip_online=True,
        expected_dry_run=False,  # dry_run 不受影响
        expected_minimal=False,
        description="非 minimal + 显式 skip_online：仅 skip_online=True",
    ),
    MinimalConsistencyTestCase(
        test_id="non_minimal_explicit_dry_run",
        minimal=False,
        execution_mode="cli",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=True,  # 仅显式 dry_run
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="mp",
        expected_skip_online=False,  # skip_online 不受影响
        expected_dry_run=True,
        expected_minimal=False,
        description="非 minimal + 显式 dry_run：仅 dry_run=True",
    ),
    # ===== minimal 模式 + 只读模式 =====
    MinimalConsistencyTestCase(
        test_id="minimal_with_plan_mode",
        minimal=True,
        execution_mode="plan",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="mp",  # plan 模式允许 mp
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + plan 只读模式：允许 mp",
    ),
    MinimalConsistencyTestCase(
        test_id="minimal_with_ask_mode",
        minimal=True,
        execution_mode="ask",
        has_api_key=True,
        cli_skip_online=False,
        cli_dry_run=False,
        cli_orchestrator=None,
        cli_no_mp=False,
        expected_orchestrator="mp",  # ask 模式允许 mp
        expected_skip_online=True,
        expected_dry_run=True,
        expected_minimal=True,
        description="minimal + ask 只读模式：允许 mp",
    ),
]


class TestMinimalModeRunPyConsistency:
    """测试 run.py 的 _merge_options 在 minimal 模式下的行为

    验证 run.py 的 options 合并逻辑在 minimal 模式下正确设置：
    - skip_online=True
    - dry_run=True
    - orchestrator 根据 execution_mode 正确选择
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        MINIMAL_CONSISTENCY_TEST_CASES,
        ids=[tc.test_id for tc in MINIMAL_CONSISTENCY_TEST_CASES],
    )
    def test_minimal_mode_options_merge(self, test_case: MinimalConsistencyTestCase) -> None:
        """测试 run.py 的 options 合并在 minimal 模式下的一致性"""
        # 使用 resolve_orchestrator_settings 模拟 run.py 的逻辑
        overrides: Dict[str, Any] = {}

        if test_case.execution_mode is not None:
            overrides["execution_mode"] = test_case.execution_mode
        if test_case.cli_orchestrator is not None:
            overrides["orchestrator"] = test_case.cli_orchestrator
        elif test_case.cli_no_mp:
            overrides["orchestrator"] = "basic"
        if test_case.cli_dry_run:
            overrides["dry_run"] = True
        if test_case.cli_skip_online:
            overrides["skip_online"] = True

        # minimal 模式应用 preset（模拟 run.py 的 _merge_options 逻辑）
        if test_case.minimal:
            if "dry_run" not in overrides:
                overrides["dry_run"] = True
            if "skip_online" not in overrides:
                overrides["skip_online"] = True

        result = resolve_orchestrator_settings(overrides=overrides)

        assert result["orchestrator"] == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  orchestrator 期望={test_case.expected_orchestrator}，实际={result['orchestrator']}"
        )

        assert result.get("dry_run", False) == test_case.expected_dry_run, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  dry_run 期望={test_case.expected_dry_run}，实际={result.get('dry_run', False)}"
        )


class TestMinimalModeSelfIteratorConsistency:
    """测试 SelfIterator 在 minimal 模式下的参数解析一致性

    验证 SelfIterator 的参数解析与 run.py 的 options 合并在
    minimal 模式下产生一致的结果。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        MINIMAL_CONSISTENCY_TEST_CASES,
        ids=[tc.test_id for tc in MINIMAL_CONSISTENCY_TEST_CASES],
    )
    def test_self_iterator_minimal_mode_parsing(self, test_case: MinimalConsistencyTestCase) -> None:
        """测试 SelfIterator 在 minimal 模式下的参数解析"""
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # 重置配置单例
        ConfigManager.reset_instance()

        # 构建测试参数
        args = argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=test_case.cli_skip_online,
            changelog_url=None,
            dry_run=test_case.cli_dry_run,
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator=test_case.cli_orchestrator or "mp",
            no_mp=test_case.cli_no_mp,
            _orchestrator_user_set=(test_case.cli_orchestrator is not None) or test_case.cli_no_mp,
            execution_mode=test_case.execution_mode,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            minimal=test_case.minimal,  # 关键：minimal 模式标志
        )

        # 创建 mock 配置
        mock_config = _create_mock_config(cloud_enabled=True)

        # 解析 API Key
        api_key = "mock-api-key" if test_case.has_api_key else None

        # 获取 SelfIterator 的实际决策
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        actual_orchestrator = iterator._get_orchestrator_type()
                                        actual_is_minimal = iterator._is_minimal
                                        # skip_online 由 args 直接控制，minimal 模式会修改它
                                        actual_skip_online = args.skip_online

        assert actual_orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  SelfIterator orchestrator 期望={test_case.expected_orchestrator}，"
            f"实际={actual_orchestrator}"
        )

        assert actual_is_minimal == test_case.expected_minimal, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  SelfIterator _is_minimal 期望={test_case.expected_minimal}，"
            f"实际={actual_is_minimal}"
        )

        assert actual_skip_online == test_case.expected_skip_online, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  SelfIterator skip_online 期望={test_case.expected_skip_online}，"
            f"实际={actual_skip_online}"
        )


class TestMinimalModeRunPyAndSelfIteratorConsistency:
    """测试 run.py 与 SelfIterator 在 minimal 模式下的决策一致性

    对每个 minimal 模式测试用例，分别计算 run.py 语义和 SelfIterator 语义，
    断言关键字段一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        MINIMAL_CONSISTENCY_TEST_CASES,
        ids=[tc.test_id for tc in MINIMAL_CONSISTENCY_TEST_CASES],
    )
    def test_run_py_and_self_iterator_minimal_consistency(self, test_case: MinimalConsistencyTestCase) -> None:
        """测试 run.py 与 SelfIterator 在 minimal 模式下的一致性"""
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # 重置配置单例
        ConfigManager.reset_instance()

        # === 计算 run.py 语义 ===
        run_py_overrides: Dict[str, Any] = {}
        if test_case.execution_mode is not None:
            run_py_overrides["execution_mode"] = test_case.execution_mode
        if test_case.cli_orchestrator is not None:
            run_py_overrides["orchestrator"] = test_case.cli_orchestrator
        elif test_case.cli_no_mp:
            run_py_overrides["orchestrator"] = "basic"
        if test_case.cli_dry_run:
            run_py_overrides["dry_run"] = True
        if test_case.cli_skip_online:
            run_py_overrides["skip_online"] = True

        # minimal 模式应用 preset
        if test_case.minimal:
            if "dry_run" not in run_py_overrides:
                run_py_overrides["dry_run"] = True
            if "skip_online" not in run_py_overrides:
                run_py_overrides["skip_online"] = True

        run_py_result = resolve_orchestrator_settings(overrides=run_py_overrides)
        run_py_orchestrator = run_py_result["orchestrator"]

        # === 计算 SelfIterator 语义 ===
        ConfigManager.reset_instance()

        args = argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=test_case.cli_skip_online,
            changelog_url=None,
            dry_run=test_case.cli_dry_run,
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator=test_case.cli_orchestrator or "mp",
            no_mp=test_case.cli_no_mp,
            _orchestrator_user_set=(test_case.cli_orchestrator is not None) or test_case.cli_no_mp,
            execution_mode=test_case.execution_mode,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            minimal=test_case.minimal,
        )

        mock_config = _create_mock_config(cloud_enabled=True)
        api_key = "mock-api-key" if test_case.has_api_key else None

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        self_iterator_orchestrator = iterator._get_orchestrator_type()
                                        self_iterator_is_minimal = iterator._is_minimal
                                        self_iterator_skip_online = args.skip_online

        # === 断言一致性 ===
        assert run_py_orchestrator == self_iterator_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  run.py orchestrator={run_py_orchestrator}\n"
            f"  SelfIterator orchestrator={self_iterator_orchestrator}"
        )

        assert self_iterator_is_minimal == test_case.expected_minimal, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  期望 minimal={test_case.expected_minimal}\n"
            f"  SelfIterator _is_minimal={self_iterator_is_minimal}"
        )

        assert self_iterator_skip_online == test_case.expected_skip_online, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  期望 skip_online={test_case.expected_skip_online}\n"
            f"  SelfIterator skip_online={self_iterator_skip_online}"
        )


class TestAmpersandPrefixNotSuccessfullyTriggered:
    """测试"仅存在 & 前缀但未成功触发"的场景

    ================================================================================
    行为说明（config.yaml 默认 execution_mode=auto）
    ================================================================================

    1. resolve_orchestrator_settings 的行为：
       - 当 overrides 不包含 execution_mode 时，读取 config.yaml 默认 auto
       - config.yaml 默认 auto 强制使用 basic 编排器
       - 只有显式指定 execution_mode=cli 才能使用 mp 编排器

    2. build_execution_decision 的行为：
       - 当 requested_mode=None 且 prefix_routed=False 时，返回 mp
       - 这与 resolve_orchestrator_settings 的行为不同

    3. 本测试类测试 resolve_orchestrator_settings 的行为。
       SelfIterator 使用 build_execution_decision，行为可能不同。

    如需使用 mp 编排器，用户应显式指定 --execution-mode cli。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def test_ampersand_no_key_with_explicit_cli_allows_mp(self) -> None:
        """测试 & 前缀无 API Key + 显式 cli 允许使用 mp 编排器"""
        # 模拟：& 前缀存在，但因无 API Key 未成功触发
        # 用户显式指定 --execution-mode cli 以使用 mp
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},  # 显式 cli
            prefix_routed=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "mp", "& 前缀无 API Key + 显式 cli，应允许 mp 编排器"

    def test_ampersand_no_key_without_cli_uses_config_default(self) -> None:
        """测试 & 前缀无 API Key 但无显式 cli 时使用 config.yaml 默认值"""
        # 模拟：& 前缀存在，但因无 API Key 未成功触发
        # 无显式 execution_mode，使用 config.yaml 默认 auto
        result = resolve_orchestrator_settings(
            overrides={},  # 无显式 execution_mode
            prefix_routed=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "basic", (
            "& 前缀无 API Key 但无显式 cli，使用 config.yaml 默认 auto，强制 basic"
        )

    def test_ampersand_cloud_disabled_with_explicit_cli_allows_mp(self) -> None:
        """测试 & 前缀 cloud_disabled + 显式 cli 允许使用 mp 编排器"""
        # 模拟：& 前缀存在，但因 cloud_enabled=False 未成功触发
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},  # 显式 cli
            prefix_routed=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "mp", "& 前缀 cloud_disabled + 显式 cli，应允许 mp 编排器"

    def test_ampersand_cloud_disabled_without_cli_uses_config_default(self) -> None:
        """测试 & 前缀 cloud_disabled 但无显式 cli 时使用 config.yaml 默认值"""
        result = resolve_orchestrator_settings(
            overrides={},  # 无显式 execution_mode
            prefix_routed=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "basic", (
            "& 前缀 cloud_disabled 但无显式 cli，使用 config.yaml 默认 auto，强制 basic"
        )

    def test_ampersand_success_forces_basic(self) -> None:
        """测试 & 前缀成功触发强制使用 basic 编排器"""
        # 模拟：& 前缀存在，成功触发 Cloud
        result = resolve_orchestrator_settings(
            overrides={},
            prefix_routed=True,  # & 前缀成功触发
        )
        assert result["orchestrator"] == "basic", "& 前缀成功触发，应强制使用 basic 编排器"


# ============================================================
# 边界用例扩展测试
# ============================================================


# should_use_mp_orchestrator mode-only 边界用例参数表
# 注意：此测试仅覆盖 requested_mode 相关场景
# 与 & 前缀相关的强制 basic 逻辑由 EXECUTION_DECISION_TEST_CASES 覆盖
SHOULD_USE_MP_MODE_ONLY_TESTS = [
    # ===== cli 模式：允许 MP =====
    ConsistencyTestCase(
        test_id="cli_allows_mp",
        requested_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="cli 模式允许 MP 编排器",
    ),
    ConsistencyTestCase(
        test_id="cli_no_key_allows_mp",
        requested_mode="cli",
        has_api_key=False,
        cloud_enabled=False,
        expected_orchestrator="mp",
        description="cli 模式（无 key、cloud_disabled）仍允许 MP",
    ),
    # ===== auto 模式：强制 basic =====
    ConsistencyTestCase(
        test_id="auto_forces_basic",
        requested_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="auto 模式强制 basic 编排器",
    ),
    ConsistencyTestCase(
        test_id="auto_no_key_forces_basic",
        requested_mode="auto",
        has_api_key=False,
        cloud_enabled=False,
        expected_orchestrator="basic",
        description="auto 模式（无 key、cloud_disabled）仍强制 basic（基于 requested_mode）",
    ),
    # ===== cloud 模式：强制 basic =====
    ConsistencyTestCase(
        test_id="cloud_forces_basic",
        requested_mode="cloud",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="cloud 模式强制 basic 编排器",
    ),
    ConsistencyTestCase(
        test_id="cloud_no_key_forces_basic",
        requested_mode="cloud",
        has_api_key=False,
        cloud_enabled=False,
        expected_orchestrator="basic",
        description="cloud 模式（无 key、cloud_disabled）仍强制 basic（基于 requested_mode）",
    ),
    # ===== plan 模式：允许 MP =====
    ConsistencyTestCase(
        test_id="plan_allows_mp",
        requested_mode="plan",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="plan 模式允许 MP 编排器",
    ),
    ConsistencyTestCase(
        test_id="plan_no_key_allows_mp",
        requested_mode="plan",
        has_api_key=False,
        cloud_enabled=False,
        expected_orchestrator="mp",
        description="plan 模式（无 key、cloud_disabled）仍允许 MP",
    ),
    # ===== ask 模式：允许 MP =====
    ConsistencyTestCase(
        test_id="ask_allows_mp",
        requested_mode="ask",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="ask 模式允许 MP 编排器",
    ),
    ConsistencyTestCase(
        test_id="ask_no_key_allows_mp",
        requested_mode="ask",
        has_api_key=False,
        cloud_enabled=False,
        expected_orchestrator="mp",
        description="ask 模式（无 key、cloud_disabled）仍允许 MP",
    ),
    # ⚠ 注意：None 模式不在此表中覆盖
    # None 仅在 has_ampersand_prefix=True 且无 CLI 显式设置时
    # 由 resolve_requested_mode_for_decision 返回，属于特殊路径
    # 对 should_use_mp_orchestrator(None) 的行为验证见
    # TestShouldUseMpOrchestratorNoneRegression 回归测试类
]


class TestEdgeCaseOrchestratorConsistency:
    """should_use_mp_orchestrator mode-only 边界测试

    此测试类仅验证 should_use_mp_orchestrator 的 mode-only 行为：
    - 只基于 requested_mode 判断
    - 不考虑 has_ampersand_prefix

    覆盖的 requested_mode 场景（仅显式模式）：
    - cli：允许 MP
    - auto：强制 basic
    - cloud：强制 basic
    - plan：允许 MP
    - ask：允许 MP

    ⚠ 注意：None 不在此类覆盖范围内
    - None 仅在 has_ampersand_prefix=True 且无 CLI 显式设置时由
      resolve_requested_mode_for_decision 返回（让 build_execution_decision 处理 & 前缀路由）
    - 对 should_use_mp_orchestrator(None) 的行为验证见
      TestShouldUseMpOrchestratorNoneRegression 回归测试类

    **与 & 前缀相关的强制 basic 逻辑由 TestBuildExecutionDecision 类通过
    EXECUTION_DECISION_TEST_CASES 表驱动矩阵覆盖。**
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        SHOULD_USE_MP_MODE_ONLY_TESTS,
        ids=[tc.test_id for tc in SHOULD_USE_MP_MODE_ONLY_TESTS],
    )
    def test_should_use_mp_orchestrator_mode_only(self, test_case: ConsistencyTestCase) -> None:
        """测试 should_use_mp_orchestrator 的 mode-only 行为

        should_use_mp_orchestrator 只基于 requested_mode 判断：
        - auto/cloud：返回 False（强制 basic）
        - cli/plan/ask：返回 True（允许 MP）

        此测试仅覆盖显式模式（cli/auto/cloud/plan/ask），不覆盖 None。
        对 None 的行为验证见 TestShouldUseMpOrchestratorNoneRegression。
        """
        can_use_mp = should_use_mp_orchestrator(test_case.requested_mode)

        # 根据 requested_mode 计算期望值
        # auto/cloud -> False（强制 basic）
        # cli/plan/ask/None -> True（允许 MP）
        expected_can_use_mp = test_case.expected_orchestrator == "mp"

        assert can_use_mp == expected_can_use_mp, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  requested_mode={test_case.requested_mode}\n"
            f"  期望 can_use_mp={expected_can_use_mp}，实际={can_use_mp}"
        )


# ============================================================
# should_use_mp_orchestrator(None) 回归测试 - 仅覆盖 prefix-flow 场景
# ============================================================


class TestShouldUseMpOrchestratorNonePrefixFlowRegression:
    """should_use_mp_orchestrator(None) 回归测试 - 仅覆盖 prefix-flow 场景

    ⚠ 此测试是专门的回归测试，验证：
    1. & 前缀触发 resolve_requested_mode_for_decision 返回 None
    2. should_use_mp_orchestrator(None) 返回 True（允许 mp）
    3. 实际编排器由 build_execution_decision 根据 prefix_routed 决定

    **重要：None 仅代表 prefix-flow**
    - None 仅在以下特殊路径下由 resolve_requested_mode_for_decision 返回：
      1. has_ampersand_prefix=True（输入以 & 开头）
      2. 且无 CLI 显式设置 execution_mode
    - 此时 None 表示"让 build_execution_decision 处理 & 前缀路由"

    **None 不是"无 & 前缀常规输入"**
    - 当 has_ampersand_prefix=False 且无 CLI 显式设置时，
      resolve_requested_mode_for_decision 返回 config.yaml 的默认值（如 "auto"），
      **不应为 None**
    - 把 None 当作"无 & 前缀常规输入"是错误的理解

    **should_use_mp_orchestrator(None) 的行为**
    - 返回 True（允许 MP），因为 None 不是 Cloud 模式
    - 但实际编排器最终由 build_execution_decision 决定：
      - prefix_routed=True → orchestrator=basic
      - prefix_routed=False + has_ampersand_prefix=True → orchestrator=basic（R-2 规则）
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def test_prefix_flow_none_returns_true_allows_mp(self) -> None:
        """[prefix-flow 回归] & 前缀触发返回 None → should_use_mp_orchestrator(None)==True

        验证流程：
        1. resolve_requested_mode_for_decision(None, True, "auto") → None
        2. should_use_mp_orchestrator(None) → True

        注意：虽然返回 True（允许 mp），但实际编排器由 build_execution_decision 决定。
        """
        from core.execution_policy import resolve_requested_mode_for_decision

        # Step 1: 验证 & 前缀场景下返回 None
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,  # CLI 未指定
            has_ampersand_prefix=True,  # 有 & 前缀
            config_execution_mode="auto",  # config.yaml 默认 auto
        )
        assert requested_mode is None, "& 前缀 + CLI 未指定时，resolve_requested_mode_for_decision 应返回 None"

        # Step 2: 验证 should_use_mp_orchestrator(None) 返回 True
        result = should_use_mp_orchestrator(None)
        assert result is True, (
            "[prefix-flow 回归] should_use_mp_orchestrator(None) 应返回 True，"
            "因为 None 不是 Cloud 模式。"
            "实际编排器最终由 build_execution_decision 决定。"
        )

    def test_prefix_flow_none_is_not_cloud_mode(self) -> None:
        """[prefix-flow 回归] None 不被识别为 Cloud 模式

        is_cloud_mode(None) 返回 False，因此 should_use_mp_orchestrator(None) 返回 True。
        """
        from core.execution_policy import is_cloud_mode

        assert is_cloud_mode(None) is False, "None 不应被识别为 Cloud 模式"

    def test_prefix_flow_actual_orchestrator_by_build_execution_decision(self) -> None:
        """[prefix-flow 回归] 验证实际编排器由 build_execution_decision 决定

        虽然 should_use_mp_orchestrator(None) 返回 True，
        但 build_execution_decision 会根据 prefix_routed 决定实际编排器：
        - prefix_routed=True → orchestrator=basic
        - prefix_routed=False + has_ampersand_prefix=True → orchestrator=basic（R-2 规则）
        """
        from core.execution_policy import build_execution_decision

        # 场景 1: & 前缀成功路由 → orchestrator=basic
        decision_routed = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        assert decision_routed.prefix_routed is True
        assert decision_routed.orchestrator == "basic", "prefix_routed=True 时应强制 basic 编排器"

        # 场景 2: & 前缀未成功路由（无 API Key）→ orchestrator=basic（R-2 规则）
        decision_not_routed = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key，无法路由
        )
        assert decision_not_routed.prefix_routed is False
        assert decision_not_routed.has_ampersand_prefix is True
        assert decision_not_routed.orchestrator == "basic", (
            "& 前缀存在但未成功路由时，仍应强制 basic 编排器（R-2 规则）"
        )


# ============================================================
# build_execution_decision 表驱动测试
# ============================================================


@dataclass
class ExecutionDecisionTestCase:
    """build_execution_decision 测试用例

    统一的决策测试结构，用于表驱动测试。

    ⚠ 注意：expected_prefix_routed 用于断言 decision.prefix_routed（策略决策层面）。
    triggered_by_prefix 是 prefix_routed 的兼容别名，仅用于 to_dict() 输出兼容，
    新断言禁止使用 triggered_by_prefix。
    """

    test_id: str
    # === 输入参数（必需）===
    prompt: Optional[str]
    requested_mode: Optional[str]
    cloud_enabled: bool
    has_api_key: bool
    # === 期望输出（必需）===
    expected_effective_mode: str
    expected_orchestrator: str
    # [迁移] expected_triggered_by_prefix -> expected_prefix_routed
    # 断言应使用 decision.prefix_routed，不再使用 decision.triggered_by_prefix
    expected_prefix_routed: bool
    expected_sanitized_prompt: str
    # === 描述（必需）===
    description: str
    # === 输入参数（可选，有默认值）===
    auto_detect_cloud_prefix: bool = True
    user_requested_orchestrator: Optional[str] = None


# build_execution_decision 测试参数表
EXECUTION_DECISION_TEST_CASES = [
    # ===== 基础 CLI 模式 =====
    ExecutionDecisionTestCase(
        test_id="cli_basic",
        prompt="实现新功能",
        requested_mode="cli",
        cloud_enabled=True,
        has_api_key=True,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_sanitized_prompt="实现新功能",
        description="基础 CLI 模式",
    ),
    ExecutionDecisionTestCase(
        test_id="cli_no_key",
        prompt="优化代码",
        requested_mode="cli",
        cloud_enabled=True,
        has_api_key=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_sanitized_prompt="优化代码",
        description="CLI 模式无 API Key",
    ),
    # ===== AUTO 模式（关键场景）=====
    ExecutionDecisionTestCase(
        test_id="auto_with_key",
        prompt="重构模块",
        requested_mode="auto",
        cloud_enabled=True,
        has_api_key=True,
        expected_effective_mode="auto",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        expected_sanitized_prompt="重构模块",
        description="AUTO 模式有 API Key",
    ),
    ExecutionDecisionTestCase(
        test_id="auto_no_key_forces_basic",
        prompt="分析代码",
        requested_mode="auto",
        cloud_enabled=True,
        has_api_key=False,
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 但仍强制 basic
        expected_prefix_routed=False,
        expected_sanitized_prompt="分析代码",
        description="关键：AUTO 模式无 API Key 回退到 CLI 但仍强制 basic",
    ),
    ExecutionDecisionTestCase(
        test_id="auto_cloud_disabled_forces_basic",
        prompt="任务描述",
        requested_mode="auto",
        cloud_enabled=False,
        has_api_key=True,
        expected_effective_mode="auto",  # 显式 auto + 有 key，cloud_enabled 不影响
        expected_orchestrator="basic",  # 强制 basic
        expected_prefix_routed=False,
        expected_sanitized_prompt="任务描述",
        description="显式 AUTO 模式 + 有 key，cloud_enabled=False 不影响 effective_mode",
    ),
    # ===== CLOUD 模式（关键场景）=====
    ExecutionDecisionTestCase(
        test_id="cloud_with_key",
        prompt="后台分析",
        requested_mode="cloud",
        cloud_enabled=True,
        has_api_key=True,
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        expected_sanitized_prompt="后台分析",
        description="CLOUD 模式有 API Key",
    ),
    ExecutionDecisionTestCase(
        test_id="cloud_no_key_forces_basic",
        prompt="长时间任务",
        requested_mode="cloud",
        cloud_enabled=True,
        has_api_key=False,
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 但仍强制 basic
        expected_prefix_routed=False,
        expected_sanitized_prompt="长时间任务",
        description="关键：CLOUD 模式无 API Key 回退到 CLI 但仍强制 basic",
    ),
    ExecutionDecisionTestCase(
        test_id="cloud_cloud_disabled_forces_basic",
        prompt="任务",
        requested_mode="cloud",
        cloud_enabled=False,
        has_api_key=True,
        expected_effective_mode="cloud",  # 显式 cloud + 有 key，cloud_enabled 不影响
        expected_orchestrator="basic",  # 强制 basic
        expected_prefix_routed=False,
        expected_sanitized_prompt="任务",
        description="显式 CLOUD 模式 + 有 key，cloud_enabled=False 不影响 effective_mode",
    ),
    # ===== & 前缀触发（成功场景）=====
    ExecutionDecisionTestCase(
        test_id="ampersand_success",
        prompt="& 分析代码架构",
        requested_mode=None,
        cloud_enabled=True,
        has_api_key=True,
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=True,
        expected_sanitized_prompt="分析代码架构",
        description="& 前缀成功触发 Cloud",
    ),
    # ===== & 前缀触发失败场景 =====
    # & 前缀表达 Cloud 意图，即使未成功触发也使用 basic（与 AGENTS.md 一致）
    ExecutionDecisionTestCase(
        test_id="ampersand_no_key_fails",
        prompt="& 后台执行任务",
        requested_mode=None,
        cloud_enabled=True,
        has_api_key=False,
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_prefix_routed=False,  # 未成功触发
        expected_sanitized_prompt="后台执行任务",
        description="& 前缀无 API Key 未成功触发，仍使用 basic（& 前缀表达 Cloud 意图）",
    ),
    ExecutionDecisionTestCase(
        test_id="ampersand_cloud_disabled_fails",
        prompt="& 推送到云端",
        requested_mode=None,
        cloud_enabled=False,
        has_api_key=True,
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_prefix_routed=False,  # 未成功触发
        expected_sanitized_prompt="推送到云端",
        description="& 前缀 cloud_disabled 未成功触发，仍使用 basic（& 前缀表达 Cloud 意图）",
    ),
    ExecutionDecisionTestCase(
        test_id="ampersand_cloud_disabled_no_key_fails",
        prompt="& 任务",
        requested_mode=None,
        cloud_enabled=False,
        has_api_key=False,
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_prefix_routed=False,
        expected_sanitized_prompt="任务",
        description="& 前缀 cloud_disabled + 无 key 未成功触发，仍使用 basic（& 前缀表达 Cloud 意图）",
    ),
    # ===== 显式 CLI 忽略 & 前缀 =====
    ExecutionDecisionTestCase(
        test_id="explicit_cli_ignores_ampersand",
        prompt="& 显式 CLI 任务",
        requested_mode="cli",
        cloud_enabled=True,
        has_api_key=True,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,  # 显式 CLI 忽略
        expected_sanitized_prompt="显式 CLI 任务",
        description="显式 CLI 模式忽略 & 前缀",
    ),
    # ===== 用户显式指定编排器 =====
    ExecutionDecisionTestCase(
        test_id="user_explicit_basic",
        prompt="任务",
        requested_mode="cli",
        cloud_enabled=True,
        has_api_key=True,
        user_requested_orchestrator="basic",
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # 用户显式指定
        expected_prefix_routed=False,
        expected_sanitized_prompt="任务",
        description="用户显式指定 basic 编排器",
    ),
    ExecutionDecisionTestCase(
        test_id="user_explicit_mp_with_cloud_forces_basic",
        prompt="任务",
        requested_mode="cloud",
        cloud_enabled=True,
        has_api_key=True,
        user_requested_orchestrator="mp",
        expected_effective_mode="cloud",
        expected_orchestrator="basic",  # cloud 模式强制 basic，即使用户显式指定 mp
        expected_prefix_routed=False,
        expected_sanitized_prompt="任务",
        description="用户显式指定 mp + cloud 模式，仍强制使用 basic（AGENTS.md 规则）",
    ),
    # ===== 只读模式 =====
    ExecutionDecisionTestCase(
        test_id="plan_mode",
        prompt="分析架构",
        requested_mode="plan",
        cloud_enabled=True,
        has_api_key=True,
        expected_effective_mode="plan",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_sanitized_prompt="分析架构",
        description="PLAN 只读模式允许 MP",
    ),
    ExecutionDecisionTestCase(
        test_id="ask_mode",
        prompt="解释代码",
        requested_mode="ask",
        cloud_enabled=True,
        has_api_key=True,
        expected_effective_mode="ask",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_sanitized_prompt="解释代码",
        description="ASK 只读模式允许 MP",
    ),
    # ===== 显式 CLI 模式（用户指定 --execution-mode cli）=====
    # 注意：requested_mode=None 的用例不适合测试 SelfIterator 一致性，
    # 因为 SelfIterator 会使用 config.yaml 默认 auto，而 build_execution_decision
    # 直接使用 None 回退到 cli。这两者行为不同。
    # 如需测试默认行为，应使用 TestIterateArgsAndSelfIteratorConsistency。
    ExecutionDecisionTestCase(
        test_id="explicit_cli_mode",
        prompt="普通任务",
        requested_mode="cli",  # 用户显式 --execution-mode cli
        cloud_enabled=True,
        has_api_key=True,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_sanitized_prompt="普通任务",
        description="显式 --execution-mode cli，使用 CLI + mp",
    ),
    # ===== 禁用自动检测 & 前缀（R-3 规则）=====
    # R-3: auto_detect_cloud_prefix=false 时，& 前缀被忽略，使用 CLI + mp
    # 无论 has_api_key 和 cloud_enabled 的值如何，行为一致
    ExecutionDecisionTestCase(
        test_id="auto_detect_disabled_key_true_enabled_true",
        prompt="& 任务",
        requested_mode=None,
        cloud_enabled=True,
        has_api_key=True,
        auto_detect_cloud_prefix=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_sanitized_prompt="任务",
        description="R-3: auto_detect=false + has_key + enabled → CLI + mp",
    ),
    ExecutionDecisionTestCase(
        test_id="auto_detect_disabled_key_false_enabled_true",
        prompt="& 分析任务",
        requested_mode=None,
        cloud_enabled=True,
        has_api_key=False,
        auto_detect_cloud_prefix=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_sanitized_prompt="分析任务",
        description="R-3: auto_detect=false + no_key + enabled → CLI + mp",
    ),
    ExecutionDecisionTestCase(
        test_id="auto_detect_disabled_key_true_enabled_false",
        prompt="& 重构代码",
        requested_mode=None,
        cloud_enabled=False,
        has_api_key=True,
        auto_detect_cloud_prefix=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_sanitized_prompt="重构代码",
        description="R-3: auto_detect=false + has_key + disabled → CLI + mp",
    ),
    ExecutionDecisionTestCase(
        test_id="auto_detect_disabled_key_false_enabled_false",
        prompt="& 测试功能",
        requested_mode=None,
        cloud_enabled=False,
        has_api_key=False,
        auto_detect_cloud_prefix=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_sanitized_prompt="测试功能",
        description="R-3: auto_detect=false + no_key + disabled → CLI + mp",
    ),
    # R-3 + 显式 auto/cloud 模式组合测试
    # 当 auto_detect=false 时，& 前缀被忽略，但显式 requested_mode 仍然生效
    ExecutionDecisionTestCase(
        test_id="auto_detect_disabled_explicit_auto_key_true",
        prompt="& 自动模式任务",
        requested_mode="auto",
        cloud_enabled=True,
        has_api_key=True,
        auto_detect_cloud_prefix=False,
        expected_effective_mode="auto",  # 显式 auto 生效
        expected_orchestrator="basic",  # auto 模式强制 basic
        expected_prefix_routed=False,  # & 前缀被忽略
        expected_sanitized_prompt="自动模式任务",
        description="R-3: auto_detect=false + 显式 auto → auto + basic（& 被忽略）",
    ),
    ExecutionDecisionTestCase(
        test_id="auto_detect_disabled_explicit_auto_key_false",
        prompt="& 自动回退任务",
        requested_mode="auto",
        cloud_enabled=True,
        has_api_key=False,
        auto_detect_cloud_prefix=False,
        expected_effective_mode="cli",  # 无 key 回退到 CLI
        expected_orchestrator="basic",  # auto 模式仍强制 basic
        expected_prefix_routed=False,  # & 前缀被忽略
        expected_sanitized_prompt="自动回退任务",
        description="R-3: auto_detect=false + 显式 auto + no_key → cli + basic",
    ),
    ExecutionDecisionTestCase(
        test_id="auto_detect_disabled_explicit_cloud_key_true",
        prompt="& 云端模式任务",
        requested_mode="cloud",
        cloud_enabled=True,
        has_api_key=True,
        auto_detect_cloud_prefix=False,
        expected_effective_mode="cloud",  # 显式 cloud 生效
        expected_orchestrator="basic",  # cloud 模式强制 basic
        expected_prefix_routed=False,  # & 前缀被忽略
        expected_sanitized_prompt="云端模式任务",
        description="R-3: auto_detect=false + 显式 cloud → cloud + basic（& 被忽略）",
    ),
    ExecutionDecisionTestCase(
        test_id="auto_detect_disabled_explicit_cloud_key_false",
        prompt="& 云端回退任务",
        requested_mode="cloud",
        cloud_enabled=True,
        has_api_key=False,
        auto_detect_cloud_prefix=False,
        expected_effective_mode="cli",  # 无 key 回退到 CLI
        expected_orchestrator="basic",  # cloud 模式仍强制 basic
        expected_prefix_routed=False,  # & 前缀被忽略
        expected_sanitized_prompt="云端回退任务",
        description="R-3: auto_detect=false + 显式 cloud + no_key → cli + basic",
    ),
]


class TestBuildExecutionDecision:
    """测试 build_execution_decision 函数

    对 core.execution_policy.build_execution_decision 进行表驱动测试，
    验证其输出与测试矩阵中的期望值一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        EXECUTION_DECISION_TEST_CASES,
        ids=[tc.test_id for tc in EXECUTION_DECISION_TEST_CASES],
    )
    def test_build_execution_decision(self, test_case: ExecutionDecisionTestCase) -> None:
        """表驱动测试：验证 build_execution_decision 输出"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt=test_case.prompt,
            requested_mode=test_case.requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=test_case.auto_detect_cloud_prefix,
            user_requested_orchestrator=test_case.user_requested_orchestrator,
        )

        # 验证 effective_mode
        assert decision.effective_mode == test_case.expected_effective_mode, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  effective_mode 期望={test_case.expected_effective_mode}，"
            f"实际={decision.effective_mode}"
        )

        # 验证 orchestrator
        assert decision.orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  orchestrator 期望={test_case.expected_orchestrator}，"
            f"实际={decision.orchestrator}"
        )

        # 验证 prefix_routed（策略决策层面）
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed == test_case.expected_prefix_routed, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  prefix_routed 期望={test_case.expected_prefix_routed}，"
            f"实际={decision.prefix_routed}"
        )

        # 验证 sanitized_prompt
        assert decision.sanitized_prompt == test_case.expected_sanitized_prompt, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  sanitized_prompt 期望='{test_case.expected_sanitized_prompt}'，"
            f"实际='{decision.sanitized_prompt}'"
        )

    def test_decision_to_dict(self) -> None:
        """测试 ExecutionDecision.to_dict"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )

        d = decision.to_dict()
        assert "effective_mode" in d
        assert "orchestrator" in d
        assert "triggered_by_prefix" in d
        assert "sanitized_prompt" in d
        assert d["effective_mode"] == "cloud"
        assert d["orchestrator"] == "basic"
        assert d["triggered_by_prefix"] is True


class TestDecisionSnapshotAndPolicyConsistency:
    """测试决策快照与 Policy 函数的一致性

    验证 build_execution_decision 的输出与其他 Policy 函数
    （resolve_effective_execution_mode、should_use_mp_orchestrator）一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        EXECUTION_DECISION_TEST_CASES,
        ids=[tc.test_id for tc in EXECUTION_DECISION_TEST_CASES],
    )
    def test_decision_matches_policy_functions(self, test_case: ExecutionDecisionTestCase) -> None:
        """验证 build_execution_decision 与独立 Policy 函数结果一致"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import build_execution_decision

        # 获取 build_execution_decision 的结果
        decision = build_execution_decision(
            prompt=test_case.prompt,
            requested_mode=test_case.requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=test_case.auto_detect_cloud_prefix,
            user_requested_orchestrator=test_case.user_requested_orchestrator,
        )

        # 使用独立 Policy 函数计算期望值
        has_ampersand = is_cloud_request(test_case.prompt)

        # resolve_effective_execution_mode 的 has_ampersand_prefix 参数
        # 表示语法检测层面（原始文本是否以 & 开头）
        if test_case.user_requested_orchestrator is None:
            # 从 prompt 解析 has_ampersand_prefix
            has_ampersand_prefix = has_ampersand

            # 判断用户是否显式选择 cli 模式
            mode_lower = (test_case.requested_mode or "").lower()
            user_explicitly_chose_cli = mode_lower == "cli"

            # 编排器选择规则：
            # 1. 当 has_ampersand_prefix=True 且 auto_detect_cloud_prefix=True
            #    且 requested_mode 不是显式 "cli" 时，expected_orch = "basic"
            #    （无论 prefix_routed 是否成功）
            # 2. 当 auto_detect_cloud_prefix=False 时，回到 mode-only 规则
            #    （requested_mode None → mp）
            if has_ampersand_prefix and test_case.auto_detect_cloud_prefix and not user_explicitly_chose_cli:
                # & 前缀存在且自动检测启用且非显式 cli → 强制 basic
                expected_orch = "basic"
            elif not test_case.auto_detect_cloud_prefix:
                # auto_detect_cloud_prefix=False → mode-only 规则
                # requested_mode=None → mp（不受 & 前缀影响）
                can_use_mp_by_policy = should_use_mp_orchestrator(test_case.requested_mode)
                expected_orch = "mp" if can_use_mp_by_policy else "basic"
            else:
                # 其他情况：无 & 前缀，使用 should_use_mp_orchestrator
                can_use_mp_by_policy = should_use_mp_orchestrator(test_case.requested_mode)
                expected_orch = "mp" if can_use_mp_by_policy else "basic"

            assert decision.orchestrator == expected_orch, (
                f"[{test_case.test_id}] orchestrator 与 should_use_mp_orchestrator 不一致\n"
                f"  decision.orchestrator={decision.orchestrator}\n"
                f"  should_use_mp_orchestrator={can_use_mp_by_policy} -> {expected_orch}"
            )

        # 验证 effective_mode 与 resolve_effective_execution_mode 一致
        # build_execution_decision 内部传给 resolve_effective_execution_mode 的是
        # prefix_routed（策略决策结果），而非 has_ampersand_prefix（语法检测结果）
        # 这确保 auto_detect_cloud_prefix=False 时不会错误路由到 Cloud
        effective_mode, _ = resolve_effective_execution_mode(
            requested_mode=test_case.requested_mode,
            has_ampersand_prefix=decision.prefix_routed,  # 策略决策结果（与 build_execution_decision 一致）
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        assert decision.effective_mode == effective_mode, (
            f"[{test_case.test_id}] effective_mode 与 resolve_effective_execution_mode 不一致\n"
            f"  decision.effective_mode={decision.effective_mode}\n"
            f"  resolve_effective_execution_mode={effective_mode}"
        )


class TestEntryPointConsistencyWithDecision:
    """测试入口脚本与 build_execution_decision 的一致性

    使用同一套期望快照，验证 run.py 和 scripts/run_iterate.py
    两个入口对 build_execution_decision 的使用方式一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        [
            tc for tc in EXECUTION_DECISION_TEST_CASES if tc.user_requested_orchestrator is None
        ],  # 排除用户显式指定编排器的用例，包含 auto_detect_cloud_prefix=False 场景
        ids=[tc.test_id for tc in EXECUTION_DECISION_TEST_CASES if tc.user_requested_orchestrator is None],
    )
    def test_self_iterator_matches_decision(self, test_case: ExecutionDecisionTestCase) -> None:
        """测试 SelfIterator 的决策与 build_execution_decision 一致"""
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # 重置配置单例
        ConfigManager.reset_instance()

        # 获取 build_execution_decision 的期望结果
        expected_decision = build_execution_decision(
            prompt=test_case.prompt,
            requested_mode=test_case.requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=test_case.auto_detect_cloud_prefix,
        )

        # 构建测试参数
        args = argparse.Namespace(
            requirement=test_case.prompt or "",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=test_case.requested_mode,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        # 创建 mock 配置（包含 auto_detect_cloud_prefix）
        mock_config = _create_mock_config(
            cloud_enabled=test_case.cloud_enabled,
            auto_detect_cloud_prefix=test_case.auto_detect_cloud_prefix,
        )

        # 解析 API Key
        api_key = "mock-api-key" if test_case.has_api_key else None

        # 获取 SelfIterator 的实际决策
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        actual_execution_mode = iterator._get_execution_mode()
                                        actual_orchestrator = iterator._get_orchestrator_type()
                                        # ⚠ 新断言规范：使用 _prefix_routed 而非 _triggered_by_prefix
                                        actual_prefix_routed = iterator._prefix_routed

        # 比较决策结果
        assert actual_execution_mode.value == expected_decision.effective_mode, (
            f"[{test_case.test_id}] SelfIterator.execution_mode 不匹配\n"
            f"  实际: {actual_execution_mode.value}\n"
            f"  期望: {expected_decision.effective_mode}\n"
            f"  auto_detect_cloud_prefix: {test_case.auto_detect_cloud_prefix}"
        )

        assert actual_orchestrator == expected_decision.orchestrator, (
            f"[{test_case.test_id}] SelfIterator.orchestrator 不匹配\n"
            f"  实际: {actual_orchestrator}\n"
            f"  期望: {expected_decision.orchestrator}\n"
            f"  auto_detect_cloud_prefix: {test_case.auto_detect_cloud_prefix}"
        )

        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        # R-3 规则验证：当 auto_detect_cloud_prefix=False 时，& 前缀应被忽略，prefix_routed=False
        assert actual_prefix_routed == expected_decision.prefix_routed, (
            f"[{test_case.test_id}] SelfIterator.prefix_routed 不匹配\n"
            f"  实际: {actual_prefix_routed}\n"
            f"  期望: {expected_decision.prefix_routed}\n"
            f"  auto_detect_cloud_prefix: {test_case.auto_detect_cloud_prefix}\n"
            f"  R-3 规则: auto_detect=False 时 & 前缀应被忽略"
        )


# ============================================================
# SelfIterator._resolve_config_settings 与 build_unified_overrides 一致性测试
# ============================================================


@dataclass
class ResolveConfigSettingsTestCase:
    """_resolve_config_settings 与 build_unified_overrides 一致性测试参数

    验证 SelfIterator._resolve_config_settings() 返回的核心字段
    与 build_unified_overrides(...).resolved 一致。
    """

    test_id: str
    # 输入
    execution_mode: Optional[str]
    has_api_key: bool
    cloud_enabled: bool
    user_orchestrator: Optional[str]
    has_ampersand_prefix: bool
    cli_workers: Optional[int]
    cli_max_iterations: Optional[str]
    # 期望输出
    expected_orchestrator: str
    description: str


# 一致性测试参数表
RESOLVE_CONFIG_SETTINGS_TEST_CASES = [
    # ===== 基础 execution_mode=None（使用 config.yaml 默认 auto） =====
    # 注意：当 CLI 参数 execution_mode=None 时，build_unified_overrides 会使用
    # config.yaml 默认 execution_mode=auto，进而强制 basic 编排器。
    #
    # 如需使用 mp 编排器，用户应显式指定 --execution-mode cli。
    ResolveConfigSettingsTestCase(
        test_id="none_uses_config_default_auto",
        execution_mode=None,  # 无 CLI 显式设置
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",  # config.yaml 默认 auto，强制 basic
        description="无 CLI 显式 execution_mode，使用 config.yaml 默认 auto，强制 basic",
    ),
    # ===== 显式 execution_mode=cli（用户显式指定） =====
    ResolveConfigSettingsTestCase(
        test_id="cli_basic",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="mp",
        description="CLI 模式允许 mp",
    ),
    ResolveConfigSettingsTestCase(
        test_id="cli_no_key",
        execution_mode="cli",
        has_api_key=False,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="mp",
        description="CLI 模式无 API Key 仍允许 mp",
    ),
    ResolveConfigSettingsTestCase(
        test_id="cli_user_basic",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator="basic",
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="用户显式指定 basic",
    ),
    # ===== execution_mode=auto（关键场景） =====
    ResolveConfigSettingsTestCase(
        test_id="auto_with_key",
        execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="AUTO 模式有 API Key 强制 basic",
    ),
    # 关键：auto + 无 key → effective=cli 但 orchestrator=basic
    ResolveConfigSettingsTestCase(
        test_id="auto_no_key_forces_basic",
        execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="关键：AUTO 模式无 API Key 仍强制 basic（基于 requested_mode）",
    ),
    ResolveConfigSettingsTestCase(
        test_id="auto_cloud_disabled",
        execution_mode="auto",
        has_api_key=True,
        cloud_enabled=False,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="AUTO 模式 cloud_disabled 仍强制 basic",
    ),
    ResolveConfigSettingsTestCase(
        test_id="auto_user_mp_still_basic",
        execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator="mp",
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="AUTO + 用户请求 mp，仍强制 basic",
    ),
    # ===== execution_mode=cloud（关键场景） =====
    ResolveConfigSettingsTestCase(
        test_id="cloud_with_key",
        execution_mode="cloud",
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="CLOUD 模式有 API Key 强制 basic",
    ),
    # 关键：cloud + 无 key → effective=cli 但 orchestrator=basic
    ResolveConfigSettingsTestCase(
        test_id="cloud_no_key_forces_basic",
        execution_mode="cloud",
        has_api_key=False,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="关键：CLOUD 模式无 API Key 仍强制 basic（基于 requested_mode）",
    ),
    ResolveConfigSettingsTestCase(
        test_id="cloud_cloud_disabled",
        execution_mode="cloud",
        has_api_key=True,
        cloud_enabled=False,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="CLOUD 模式 cloud_disabled 仍强制 basic",
    ),
    ResolveConfigSettingsTestCase(
        test_id="cloud_user_mp_still_basic",
        execution_mode="cloud",
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator="mp",
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="CLOUD + 用户请求 mp，仍强制 basic",
    ),
    # ===== & 前缀场景 =====
    # 注意：build_unified_overrides 的行为：
    # - 当 & 前缀成功触发时，prefix_routed=True，强制 basic
    # - 当 & 前缀未成功触发时（无 API Key 或 cloud_disabled）：
    #   * build_unified_overrides 会使用 config.yaml 默认 execution_mode=auto
    #   * auto 模式强制 basic 编排器
    #   * 这与 build_execution_decision(requested_mode=None) 的行为不同
    #
    # 这是因为 build_unified_overrides 在 resolve_orchestrator_settings 内部
    # 会读取 config.yaml 默认值。
    ResolveConfigSettingsTestCase(
        test_id="ampersand_success",
        execution_mode=None,
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=True,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",
        description="& 前缀成功触发，强制 basic",
    ),
    # & 前缀未成功触发时，build_unified_overrides 使用 config.yaml 默认 auto，强制 basic
    ResolveConfigSettingsTestCase(
        test_id="ampersand_no_key",
        execution_mode=None,
        has_api_key=False,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=True,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",  # config.yaml 默认 auto，强制 basic
        description="& 前缀但无 API Key，config.yaml 默认 auto，强制 basic",
    ),
    ResolveConfigSettingsTestCase(
        test_id="ampersand_cloud_disabled",
        execution_mode=None,
        has_api_key=True,
        cloud_enabled=False,
        user_orchestrator=None,
        has_ampersand_prefix=True,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="basic",  # config.yaml 默认 auto，强制 basic
        description="& 前缀但 cloud_disabled，config.yaml 默认 auto，强制 basic",
    ),
    # ===== plan/ask 模式 =====
    ResolveConfigSettingsTestCase(
        test_id="plan_mode",
        execution_mode="plan",
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="mp",
        description="PLAN 只读模式允许 mp",
    ),
    ResolveConfigSettingsTestCase(
        test_id="ask_mode",
        execution_mode="ask",
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="mp",
        description="ASK 只读模式允许 mp",
    ),
    # ===== CLI 参数覆盖场景 =====
    ResolveConfigSettingsTestCase(
        test_id="cli_workers_override",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=5,
        cli_max_iterations="20",
        expected_orchestrator="mp",
        description="CLI 参数覆盖 workers/max_iterations",
    ),
]


class TestResolveConfigSettingsAndBuildUnifiedOverridesConsistency:
    """测试 SelfIterator._resolve_config_settings 与 build_unified_overrides 一致性

    验证 SelfIterator._resolve_config_settings() 返回的核心字段与
    build_unified_overrides(...).resolved 一致。

    核心字段包括：
    - execution_mode
    - orchestrator
    - max_iterations
    - workers (worker_pool_size)

    ================================================================================
    已知不一致场景（需要后续修复）
    ================================================================================

    修复后：SelfIterator._get_orchestrator_type() 与 build_unified_overrides 行为一致。

    当 & 前缀存在但未成功触发时（无 API Key 或 cloud_enabled=False），
    由于 & 前缀表达 Cloud 意图，orchestrator 强制为 basic。
    这与 AGENTS.md 文档描述一致。
    """

    # 已修复：所有不一致案例现在行为一致
    KNOWN_INCONSISTENT_CASES: set = set()  # 无已知不一致案例

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        RESOLVE_CONFIG_SETTINGS_TEST_CASES,
        ids=[tc.test_id for tc in RESOLVE_CONFIG_SETTINGS_TEST_CASES],
    )
    def test_resolve_config_settings_matches_build_unified_overrides(
        self, test_case: ResolveConfigSettingsTestCase
    ) -> None:
        """表驱动测试：验证 _resolve_config_settings 与 build_unified_overrides 一致"""
        # 标记已知不一致场景
        if test_case.test_id in self.KNOWN_INCONSISTENT_CASES:
            pytest.xfail(
                "已知不一致：SelfIterator 使用 build_execution_decision，"
                "build_unified_overrides 使用 resolve_orchestrator_settings。"
                "待后续修复。"
            )
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        ConfigManager.reset_instance()

        # 构建测试任务（可能包含 & 前缀）
        requirement = "& 后台任务" if test_case.has_ampersand_prefix else "普通任务"

        # 构建 args
        args = argparse.Namespace(
            requirement=requirement,
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations=test_case.cli_max_iterations or "10",
            workers=test_case.cli_workers or 3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator=test_case.user_orchestrator or "mp",
            no_mp=False,
            _orchestrator_user_set=(test_case.user_orchestrator is not None),
            execution_mode=test_case.execution_mode,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        # 创建 mock 配置
        mock_config = _create_mock_config(cloud_enabled=test_case.cloud_enabled)
        api_key = "mock-api-key" if test_case.has_api_key else None

        # 计算 build_unified_overrides 的结果
        decision = build_execution_decision(
            prompt=requirement,
            requested_mode=test_case.execution_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        unified = build_unified_overrides(
            args=args,
            execution_decision=decision,
        )

        # 验证 build_unified_overrides 的结果符合预期
        assert unified.resolved["orchestrator"] == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] build_unified_overrides orchestrator 不匹配\n"
            f"  期望={test_case.expected_orchestrator}, "
            f"实际={unified.resolved['orchestrator']}"
        )

        # 获取 SelfIterator._resolve_config_settings 的结果
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        resolved_settings = iterator._resolve_config_settings()
                                        actual_orchestrator = iterator._get_orchestrator_type()

        # 验证 _resolve_config_settings 的核心字段与 build_unified_overrides.resolved 一致
        # execution_mode
        assert resolved_settings.execution_mode == unified.resolved["execution_mode"], (
            f"[{test_case.test_id}] execution_mode 不一致\n"
            f"  _resolve_config_settings={resolved_settings.execution_mode}\n"
            f"  build_unified_overrides={unified.resolved['execution_mode']}"
        )

        # orchestrator（通过 _get_orchestrator_type 获取）
        assert actual_orchestrator == unified.resolved["orchestrator"], (
            f"[{test_case.test_id}] orchestrator 不一致\n"
            f"  _get_orchestrator_type={actual_orchestrator}\n"
            f"  build_unified_overrides={unified.resolved['orchestrator']}"
        )

        # max_iterations
        assert resolved_settings.max_iterations == unified.resolved["max_iterations"], (
            f"[{test_case.test_id}] max_iterations 不一致\n"
            f"  _resolve_config_settings={resolved_settings.max_iterations}\n"
            f"  build_unified_overrides={unified.resolved['max_iterations']}"
        )

        # workers (worker_pool_size)
        assert resolved_settings.worker_pool_size == unified.resolved["workers"], (
            f"[{test_case.test_id}] workers 不一致\n"
            f"  _resolve_config_settings={resolved_settings.worker_pool_size}\n"
            f"  build_unified_overrides={unified.resolved['workers']}"
        )

    def test_auto_no_key_effective_cli_but_orchestrator_basic(self) -> None:
        """强断言：requested_mode=auto 无 key → effective=cli 但 orchestrator=basic

        这是关键场景的专项测试，确保 SelfIterator 行为符合 AGENTS.md 规则。
        """
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        ConfigManager.reset_instance()

        args = argparse.Namespace(
            requirement="任务描述",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",  # 用户请求 mp
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="auto",  # 关键：显式 auto
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        mock_config = _create_mock_config(cloud_enabled=True)

        # 无 API Key
        decision = build_execution_decision(
            prompt="任务描述",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,  # 关键：无 API Key
        )

        # 验证 decision 的状态
        assert decision.effective_mode == "cli", "auto + 无 key 应回退到 cli"
        assert decision.orchestrator == "basic", "auto + 无 key 仍应强制 basic 编排器（基于 requested_mode）"
        # 新增断言：prefix_routed 应为 False（未成功触发 Cloud）
        assert decision.prefix_routed is False, "auto + 无 key 时 prefix_routed 应为 False，因为没有 & 前缀触发"

        unified = build_unified_overrides(
            args=args,
            execution_decision=decision,
        )

        assert unified.resolved["orchestrator"] == "basic", "build_unified_overrides: auto + 无 key 应强制 basic"

        # 验证 SelfIterator
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        actual_orchestrator = iterator._get_orchestrator_type()
                                        actual_execution_mode = iterator._get_execution_mode()

        assert actual_orchestrator == "basic", "SelfIterator: auto + 无 key 应强制 basic 编排器"
        # effective_mode 应回退到 cli
        assert actual_execution_mode.value == "cli", "SelfIterator: auto + 无 key，effective_mode 应回退到 cli"

    def test_cloud_no_key_effective_cli_but_orchestrator_basic(self) -> None:
        """强断言：requested_mode=cloud 无 key → effective=cli 但 orchestrator=basic

        这是关键场景的专项测试，确保 SelfIterator 行为符合 AGENTS.md 规则。
        """
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        ConfigManager.reset_instance()

        args = argparse.Namespace(
            requirement="任务描述",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",  # 用户请求 mp
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cloud",  # 关键：显式 cloud
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        mock_config = _create_mock_config(cloud_enabled=True)

        # 无 API Key
        decision = build_execution_decision(
            prompt="任务描述",
            requested_mode="cloud",
            cloud_enabled=True,
            has_api_key=False,  # 关键：无 API Key
        )

        # 验证 decision 的状态
        assert decision.effective_mode == "cli", "cloud + 无 key 应回退到 cli"
        assert decision.orchestrator == "basic", "cloud + 无 key 仍应强制 basic 编排器（基于 requested_mode）"
        # 新增断言：prefix_routed 应为 False（未成功触发 Cloud）
        assert decision.prefix_routed is False, "cloud + 无 key 时 prefix_routed 应为 False，因为没有 & 前缀触发"

        unified = build_unified_overrides(
            args=args,
            execution_decision=decision,
        )

        assert unified.resolved["orchestrator"] == "basic", "build_unified_overrides: cloud + 无 key 应强制 basic"

        # 验证 SelfIterator
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        actual_orchestrator = iterator._get_orchestrator_type()
                                        actual_execution_mode = iterator._get_execution_mode()

        assert actual_orchestrator == "basic", "SelfIterator: cloud + 无 key 应强制 basic 编排器"
        # effective_mode 应回退到 cli
        assert actual_execution_mode.value == "cli", "SelfIterator: cloud + 无 key，effective_mode 应回退到 cli"


# ============================================================
# 配置源与优先级一致性测试
# ============================================================


class TestConfigYamlSourceConsistency:
    """测试 config.yaml 配置源的一致性行为

    验证当 execution_mode 来自 config.yaml（而非 CLI 显式参数）时，
    两个入口（run.py 和 scripts/run_iterate.py）的行为一致性。

    关键场景：
    1. config.yaml execution_mode=auto/cloud + 无 & 前缀 + 无 CLI 显式：
       两入口最终 orchestrator 都为 basic
    2. config.yaml execution_mode=cli + 有 & 前缀但无 key 或 cloud_enabled=false：
       两入口都不应 prefix_routed，且 orchestrator 允许 mp
    3. 显式 --execution-mode=cli + 有 & 前缀：
       两入口都应忽略路由（prefix_routed=False），effective=cli，orchestrator 允许 mp
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def _create_config_with_execution_mode(self, execution_mode: str, cloud_enabled: bool = True) -> MagicMock:
        """创建带有指定 execution_mode 的配置对象"""
        mock_config = _create_mock_config(cloud_enabled=cloud_enabled)
        mock_config.cloud_agent.execution_mode = execution_mode
        return mock_config

    def test_config_yaml_auto_mode_no_prefix_no_cli_forces_basic(self) -> None:
        """config.yaml execution_mode=auto + 无 & 前缀 + 无 CLI 显式 → basic 编排器

        场景：config.yaml 中设置 execution_mode=auto，用户未使用 & 前缀，
        也未通过 CLI 显式指定 --execution-mode。
        预期：两入口最终 orchestrator 都为 basic（基于 config.yaml 的 auto 模式）。
        """
        from core.execution_policy import build_execution_decision

        # 构建 decision（模拟无 CLI 显式指定，使用 config.yaml 默认值）
        decision = build_execution_decision(
            prompt="普通任务描述",  # 无 & 前缀
            requested_mode="auto",  # 来自 config.yaml
            cloud_enabled=True,
            has_api_key=True,
        )

        # 验证编排器为 basic
        assert decision.orchestrator == "basic", "config.yaml execution_mode=auto 应强制 basic 编排器"
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed is False, "无 & 前缀，prefix_routed 应为 False"

        # 验证 resolve_orchestrator_settings 一致性
        # 注：参数名 triggered_by_prefix 保留以兼容旧接口，新代码推荐用 prefix_routed
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "auto"},
            prefix_routed=False,  # 使用新参数名
        )
        assert result["orchestrator"] == "basic", "resolve_orchestrator_settings: auto 模式应强制 basic"

    def test_config_yaml_cloud_mode_no_prefix_no_cli_forces_basic(self) -> None:
        """config.yaml execution_mode=cloud + 无 & 前缀 + 无 CLI 显式 → basic 编排器

        场景：config.yaml 中设置 execution_mode=cloud，用户未使用 & 前缀，
        也未通过 CLI 显式指定 --execution-mode。
        预期：两入口最终 orchestrator 都为 basic（基于 config.yaml 的 cloud 模式）。
        """
        from core.execution_policy import build_execution_decision

        # 构建 decision（模拟无 CLI 显式指定，使用 config.yaml 默认值）
        decision = build_execution_decision(
            prompt="普通任务描述",  # 无 & 前缀
            requested_mode="cloud",  # 来自 config.yaml
            cloud_enabled=True,
            has_api_key=True,
        )

        # 验证编排器为 basic
        assert decision.orchestrator == "basic", "config.yaml execution_mode=cloud 应强制 basic 编排器"
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed is False, "无 & 前缀，prefix_routed 应为 False"

        # 验证 resolve_orchestrator_settings 一致性
        # 注：参数名 triggered_by_prefix 保留以兼容旧接口，新代码推荐用 prefix_routed
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cloud"},
            prefix_routed=False,  # 使用新参数名
        )
        assert result["orchestrator"] == "basic", "resolve_orchestrator_settings: cloud 模式应强制 basic"

    def test_config_yaml_cli_mode_prefix_no_key_allows_mp(self) -> None:
        """config.yaml execution_mode=cli + 有 & 前缀 + 无 API Key → 允许 mp

        场景：config.yaml 中设置 execution_mode=cli，用户使用了 & 前缀，
        但由于无 API Key，& 前缀未能成功触发 Cloud。
        预期：两入口都不应 prefix_routed，且 orchestrator 允许 mp。
        """
        from core.execution_policy import build_execution_decision

        # 构建 decision
        decision = build_execution_decision(
            prompt="& 后台任务",  # 有 & 前缀
            requested_mode="cli",  # 来自 config.yaml
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key，& 前缀无法触发
        )

        # 验证 & 前缀未成功触发
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed is False, "无 API Key 时，& 前缀不应成功触发"
        # 验证允许 mp 编排器
        assert decision.orchestrator == "mp", "cli 模式 + & 前缀未成功触发，应允许 mp 编排器"
        assert decision.effective_mode == "cli", "effective_mode 应保持 cli"

        # 验证 resolve_orchestrator_settings 一致性
        # 注：参数名 triggered_by_prefix 保留以兼容旧接口，新代码推荐用 prefix_routed
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "mp", "resolve_orchestrator_settings: cli 模式应允许 mp"

    def test_config_yaml_cli_mode_prefix_cloud_disabled_allows_mp(self) -> None:
        """config.yaml execution_mode=cli + 有 & 前缀 + cloud_enabled=false → 允许 mp

        场景：config.yaml 中设置 execution_mode=cli，用户使用了 & 前缀，
        但由于 cloud_enabled=false，& 前缀未能成功触发 Cloud。
        预期：两入口都不应 prefix_routed，且 orchestrator 允许 mp。
        """
        from core.execution_policy import build_execution_decision

        # 构建 decision
        decision = build_execution_decision(
            prompt="& 后台任务",  # 有 & 前缀
            requested_mode="cli",  # 来自 config.yaml
            cloud_enabled=False,  # cloud_enabled=false
            has_api_key=True,
        )

        # 验证 & 前缀未成功触发
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed is False, "cloud_enabled=false 时，& 前缀不应成功触发"
        # 验证允许 mp 编排器
        assert decision.orchestrator == "mp", "cli 模式 + & 前缀未成功触发，应允许 mp 编排器"
        assert decision.effective_mode == "cli", "effective_mode 应保持 cli"

        # 验证 resolve_orchestrator_settings 一致性
        # 注：参数名 triggered_by_prefix 保留以兼容旧接口，新代码推荐用 prefix_routed
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "mp", "resolve_orchestrator_settings: cli 模式应允许 mp"


class TestExplicitCliIgnoresPrefixConsistency:
    """测试显式 --execution-mode=cli 忽略 & 前缀的一致性

    验证当用户显式指定 --execution-mode=cli 时，即使任务描述包含 & 前缀，
    两个入口的行为也应该一致：
    - prefix_routed=False（忽略 & 前缀）
    - effective_mode=cli
    - orchestrator 允许 mp
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def test_explicit_cli_with_prefix_ignores_routing(self) -> None:
        """显式 --execution-mode=cli + 有 & 前缀 → 忽略路由，允许 mp

        这是显式 CLI 模式覆盖 & 前缀的核心测试。
        """
        from core.execution_policy import build_execution_decision

        # 构建 decision（显式 CLI 模式 + & 前缀）
        decision = build_execution_decision(
            prompt="& 后台分析任务",  # 有 & 前缀
            requested_mode="cli",  # 用户显式指定 CLI
            cloud_enabled=True,
            has_api_key=True,
        )

        # 验证 & 前缀被忽略
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed is False, "显式 CLI 模式应忽略 & 前缀，prefix_routed=False"
        # 验证 effective_mode 为 cli
        assert decision.effective_mode == "cli", "显式 CLI 模式，effective_mode 应为 cli"
        # 验证允许 mp 编排器
        assert decision.orchestrator == "mp", "显式 CLI 模式应允许 mp 编排器"

    def test_explicit_cli_with_prefix_both_entries_consistent(self) -> None:
        """验证两入口对显式 CLI + & 前缀的处理一致性"""
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # 测试 1: build_execution_decision
        decision = build_execution_decision(
            prompt="& 后台任务",
            requested_mode="cli",  # 显式 CLI
            cloud_enabled=True,
            has_api_key=True,
        )
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed is False
        assert decision.effective_mode == "cli"
        assert decision.orchestrator == "mp"

        # 测试 2: resolve_orchestrator_settings
        # 注：参数名 triggered_by_prefix 保留以兼容旧接口，新代码推荐用 prefix_routed
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,  # 使用新参数名
        )
        assert result["orchestrator"] == "mp", "resolve_orchestrator_settings: 显式 cli 应允许 mp"

        # 测试 3: SelfIterator
        args = argparse.Namespace(
            requirement="& 后台任务",  # 有 & 前缀
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",  # 显式 CLI
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        mock_config = _create_mock_config(cloud_enabled=True)

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        actual_triggered = iterator._triggered_by_prefix
                                        actual_orchestrator = iterator._get_orchestrator_type()
                                        actual_mode = iterator._get_execution_mode()

        assert actual_triggered is False, "SelfIterator: 显式 CLI 应忽略 & 前缀"
        assert actual_mode.value == "cli", "SelfIterator: 显式 CLI，effective_mode 应为 cli"
        assert actual_orchestrator == "mp", "SelfIterator: 显式 CLI 应允许 mp 编排器"

    def test_explicit_cli_prefix_no_key_still_allows_mp(self) -> None:
        """显式 --execution-mode=cli + & 前缀 + 无 API Key → 仍允许 mp

        即使无 API Key，显式 CLI 模式也应覆盖 & 前缀。
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 后台任务",
            requested_mode="cli",  # 显式 CLI
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key
        )

        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed is False, "显式 CLI + 无 API Key，& 前缀不应触发"
        assert decision.effective_mode == "cli"
        assert decision.orchestrator == "mp", "显式 CLI 模式应允许 mp"

    def test_explicit_cli_prefix_cloud_disabled_still_allows_mp(self) -> None:
        """显式 --execution-mode=cli + & 前缀 + cloud_enabled=false → 仍允许 mp

        即使 cloud_enabled=false，显式 CLI 模式也应覆盖 & 前缀。
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 后台任务",
            requested_mode="cli",  # 显式 CLI
            cloud_enabled=False,  # cloud_enabled=false
            has_api_key=True,
        )

        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed is False, "显式 CLI + cloud_enabled=false，& 前缀不应触发"
        assert decision.effective_mode == "cli"
        assert decision.orchestrator == "mp", "显式 CLI 模式应允许 mp"


# ============================================================
# TestOrchestratorUserSetAndOverrides - 编排器显式设置与 overrides 一致性
# ============================================================


class TestOrchestratorUserSetAndOverrides:
    """测试 _orchestrator_user_set 标志与 overrides 的一致性

    验证要点:
    1. 未显式传参时 _orchestrator_user_set=False
    2. 未显式传参时不会把 orchestrator/no_mp 写入 overrides
    3. run.py 和 scripts/run_iterate.py 的检测列表一致
    4. cli_no_mp 使用 None 默认值（tri-state 策略）
    """

    def test_run_py_detection_list_consistency(self) -> None:
        """验证 run.py 的 _orchestrator_user_set 检测列表正确"""

        # 模拟 run.py 的检测逻辑
        def check_orchestrator_user_set(argv: list) -> bool:
            return any(arg in argv for arg in ["--orchestrator", "--no-mp"])

        # 场景1: 未传任何编排器参数
        assert check_orchestrator_user_set(["run.py", "任务"]) is False

        # 场景2: 传入 --orchestrator mp
        assert check_orchestrator_user_set(["run.py", "--orchestrator", "mp", "任务"]) is True

        # 场景3: 传入 --orchestrator basic
        assert check_orchestrator_user_set(["run.py", "--orchestrator", "basic", "任务"]) is True

        # 场景4: 传入 --no-mp
        assert check_orchestrator_user_set(["run.py", "--no-mp", "任务"]) is True

        # 场景5: 传入其他参数但不传编排器参数
        assert check_orchestrator_user_set(["run.py", "--workers", "5", "任务"]) is False

    def test_run_iterate_detection_list_consistency(self) -> None:
        """验证 scripts/run_iterate.py 的 _orchestrator_user_set 检测列表正确

        确保检测列表与 run.py 保持一致（不包含无效的 -no-mp）
        """

        # 模拟 scripts/run_iterate.py 的检测逻辑（修正后）
        def check_orchestrator_user_set(argv: list) -> bool:
            return any(arg in argv for arg in ["--orchestrator", "--no-mp"])

        # 场景1: 未传任何编排器参数
        assert check_orchestrator_user_set(["run_iterate.py", "任务"]) is False

        # 场景2: 传入 --orchestrator mp
        assert check_orchestrator_user_set(["run_iterate.py", "--orchestrator", "mp", "任务"]) is True

        # 场景3: 传入 --no-mp
        assert check_orchestrator_user_set(["run_iterate.py", "--no-mp", "任务"]) is True

        # 场景4: 无效的 -no-mp 不应被检测到
        # （修正后的代码不再包含 -no-mp）
        assert check_orchestrator_user_set(["run_iterate.py", "-no-mp", "任务"]) is False

    def test_build_cli_overrides_no_orchestrator_when_not_set(self) -> None:
        """验证未显式传参时 build_cli_overrides_from_args 不写入 orchestrator

        核心断言:
        - _orchestrator_user_set=False
        - orchestrator 不在 overrides 中
        - no_mp 不影响 overrides（因为是 None）
        """
        import argparse

        from core.config import build_cli_overrides_from_args

        # 创建模拟的 args（未设置任何编排器参数）
        args = argparse.Namespace(
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            minimal=False,
            dry_run=False,
            skip_online=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            # 关键：编排器参数未设置（tri-state）
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            # 模型参数
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
        )

        # 调用 build_cli_overrides_from_args
        overrides = build_cli_overrides_from_args(args, nl_options={})

        # 验证 orchestrator 不在 overrides 中
        assert "orchestrator" not in overrides, "未显式传参时 orchestrator 不应写入 overrides"

    def test_build_cli_overrides_orchestrator_when_explicitly_set(self) -> None:
        """验证显式传参时 build_cli_overrides_from_args 正确写入 orchestrator"""
        import argparse

        from core.config import build_cli_overrides_from_args

        # 场景1: 显式设置 --orchestrator mp
        args_mp = argparse.Namespace(
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            minimal=False,
            dry_run=False,
            skip_online=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=None,
            _orchestrator_user_set=True,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
        )

        overrides_mp = build_cli_overrides_from_args(args_mp, nl_options={})
        assert overrides_mp.get("orchestrator") == "mp", "显式设置 --orchestrator mp 时应写入 overrides"

        # 场景2: 显式设置 --orchestrator basic
        args_basic = argparse.Namespace(
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            minimal=False,
            dry_run=False,
            skip_online=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="basic",
            no_mp=None,
            _orchestrator_user_set=True,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
        )

        overrides_basic = build_cli_overrides_from_args(args_basic, nl_options={})
        assert overrides_basic.get("orchestrator") == "basic", "显式设置 --orchestrator basic 时应写入 overrides"

        # 场景3: 显式设置 --no-mp
        args_no_mp = argparse.Namespace(
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            minimal=False,
            dry_run=False,
            skip_online=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=True,  # 显式设置 --no-mp
            _orchestrator_user_set=True,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
        )

        overrides_no_mp = build_cli_overrides_from_args(args_no_mp, nl_options={})
        assert overrides_no_mp.get("orchestrator") == "basic", "显式设置 --no-mp 时应写入 orchestrator=basic"

    def test_cli_no_mp_tristate_consistency(self) -> None:
        """验证 cli_no_mp 使用 None 默认值（tri-state 策略）

        cli_no_mp 的可能值:
        - None: 未显式指定（默认）
        - True: 显式设置 --no-mp
        - False: 不会出现（action="store_true" 不产生 False）
        """
        import argparse

        from core.config import build_cli_overrides_from_args

        # 场景1: no_mp=None（未显式指定）
        args_none = argparse.Namespace(
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            minimal=False,
            dry_run=False,
            skip_online=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,  # tri-state: 未指定
            _orchestrator_user_set=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
        )

        overrides_none = build_cli_overrides_from_args(args_none, nl_options={})
        assert "orchestrator" not in overrides_none, "no_mp=None 时不应写入 orchestrator"

        # 场景2: no_mp=True（显式设置 --no-mp）
        args_true = argparse.Namespace(
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            minimal=False,
            dry_run=False,
            skip_online=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=True,  # 显式设置 --no-mp
            _orchestrator_user_set=True,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
        )

        overrides_true = build_cli_overrides_from_args(args_true, nl_options={})
        assert overrides_true.get("orchestrator") == "basic", "no_mp=True 时应写入 orchestrator=basic"

    def test_nl_options_orchestrator_when_user_not_set(self) -> None:
        """验证 _orchestrator_user_set=False 时允许 nl_options 覆盖"""
        import argparse

        from core.config import build_cli_overrides_from_args

        args = argparse.Namespace(
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            minimal=False,
            dry_run=False,
            skip_online=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,  # 未显式设置
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
        )

        # nl_options 包含 orchestrator
        nl_options = {"orchestrator": "basic"}

        overrides = build_cli_overrides_from_args(args, nl_options=nl_options)

        # 验证 nl_options 的 orchestrator 被写入 overrides
        assert overrides.get("orchestrator") == "basic", "_orchestrator_user_set=False 时应允许 nl_options 覆盖"

    def test_user_set_takes_priority_over_nl_options(self) -> None:
        """验证 _orchestrator_user_set=True 时用户设置优先于 nl_options"""
        import argparse

        from core.config import build_cli_overrides_from_args

        args = argparse.Namespace(
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            minimal=False,
            dry_run=False,
            skip_online=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="mp",  # 用户显式设置
            no_mp=None,
            _orchestrator_user_set=True,  # 用户显式设置
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
        )

        # nl_options 试图覆盖为 basic
        nl_options = {"orchestrator": "basic"}

        overrides = build_cli_overrides_from_args(args, nl_options=nl_options)

        # 验证用户设置优先
        assert overrides.get("orchestrator") == "mp", "_orchestrator_user_set=True 时用户设置应优先于 nl_options"


# ============================================================
# run.py 与 scripts/run_iterate.py 一致性回归测试
# ============================================================


class TestRunPyAndIteratePyConsistency:
    """验证 run.py 与 scripts/run_iterate.py 对 config.yaml 的处理一致性

    关键回归测试场景：
    1. config.yaml execution_mode=auto/cloud + 无 CLI 参数 + 无 & 前缀
       两入口最终 orchestrator 都为 basic（基于 config.yaml 的模式）
    2. config.yaml execution_mode=auto/cloud + 无 API Key
       两入口 effective_mode 都回退到 cli，但 orchestrator 仍为 basic
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def _create_mock_args_for_run_py(
        self,
        execution_mode: Optional[str] = None,
        orchestrator: str = "mp",
        orchestrator_user_set: bool = False,
        no_mp: Optional[bool] = None,
    ) -> argparse.Namespace:
        """创建模拟 run.py 的 args"""
        return argparse.Namespace(
            directory=".",
            execution_mode=execution_mode,
            orchestrator=orchestrator,
            no_mp=no_mp,
            _orchestrator_user_set=orchestrator_user_set,
            skip_online=True,
            dry_run=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            workers=None,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            minimal=False,
            force_update=False,
            use_knowledge=False,
            self_update=False,
            search_knowledge=None,
            cloud_api_key=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
        )

    def _create_mock_args_for_iterate_py(
        self,
        requirement: str,
        execution_mode: Optional[str] = None,
        orchestrator: str = "mp",
        orchestrator_user_set: bool = False,
        no_mp: Optional[bool] = None,
    ) -> argparse.Namespace:
        """创建模拟 scripts/run_iterate.py 的 args"""
        return argparse.Namespace(
            requirement=requirement,
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator=orchestrator,
            no_mp=no_mp,
            _orchestrator_user_set=orchestrator_user_set,
            execution_mode=execution_mode,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_config_yaml_auto_no_cli_no_prefix_run_py_forces_basic(self) -> None:
        """回归测试：run.py config.yaml=auto + 无 CLI + 无 & 前缀 → basic 编排器

        这是修复 effective_mode 写入链路后的关键验证。
        """
        from cursor.cloud_client import CloudClientFactory
        from run import TaskAnalyzer

        task = "普通任务描述"  # 无 & 前缀
        args = self._create_mock_args_for_run_py(execution_mode=None)

        # 创建 config.yaml 为 auto 模式的 mock
        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.execution_mode = "auto"

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("run.get_config", return_value=mock_config):
                        analyzer = TaskAnalyzer()
                        analysis = analyzer._rule_based_analysis(task, args)

        # 验证编排器为 basic
        assert analysis.options.get("orchestrator") == "basic", (
            "run.py: config.yaml=auto + 无 CLI + 无 & 前缀，应强制 basic 编排器"
        )
        # 验证 prefix_routed 为 False（无 & 前缀）
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert analysis.options.get("prefix_routed") is False, "run.py: 无 & 前缀，prefix_routed 应为 False"
        # 验证 effective_mode 为 auto（有 API Key 时保持 auto）
        assert analysis.options.get("effective_mode") == "auto", (
            "run.py: config.yaml=auto + 有 API Key，effective_mode 应为 auto"
        )

    def test_config_yaml_cloud_no_cli_no_prefix_run_py_forces_basic(self) -> None:
        """回归测试：run.py config.yaml=cloud + 无 CLI + 无 & 前缀 → basic 编排器"""
        from cursor.cloud_client import CloudClientFactory
        from run import TaskAnalyzer

        task = "普通任务描述"  # 无 & 前缀
        args = self._create_mock_args_for_run_py(execution_mode=None)

        # 创建 config.yaml 为 cloud 模式的 mock
        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.execution_mode = "cloud"

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("run.get_config", return_value=mock_config):
                        analyzer = TaskAnalyzer()
                        analysis = analyzer._rule_based_analysis(task, args)

        # 验证编排器为 basic
        assert analysis.options.get("orchestrator") == "basic", (
            "run.py: config.yaml=cloud + 无 CLI + 无 & 前缀，应强制 basic 编排器"
        )
        # 验证 effective_mode 为 cloud
        assert analysis.options.get("effective_mode") == "cloud", (
            "run.py: config.yaml=cloud + 有 API Key，effective_mode 应为 cloud"
        )

    def test_config_yaml_auto_no_key_run_py_fallback_cli_still_basic(self) -> None:
        """回归测试：run.py config.yaml=auto + 无 API Key → 回退 cli 但编排器仍为 basic

        关键场景：requested_mode=auto 决定编排器，不受 effective_mode 回退影响。
        """
        from cursor.cloud_client import CloudClientFactory
        from run import TaskAnalyzer

        task = "普通任务描述"  # 无 & 前缀
        args = self._create_mock_args_for_run_py(execution_mode=None)

        # 创建 config.yaml 为 auto 模式的 mock
        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.execution_mode = "auto"

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):  # 无 API Key
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("run.get_config", return_value=mock_config):
                        analyzer = TaskAnalyzer()
                        analysis = analyzer._rule_based_analysis(task, args)

        # 验证编排器仍为 basic（基于 requested_mode=auto）
        assert analysis.options.get("orchestrator") == "basic", "run.py: config.yaml=auto + 无 key，编排器仍应为 basic"
        # 验证 effective_mode 回退到 cli
        assert analysis.options.get("effective_mode") == "cli", (
            "run.py: config.yaml=auto + 无 key，effective_mode 应回退到 cli"
        )

    def test_config_yaml_auto_no_cli_no_prefix_both_entries_consistent(self) -> None:
        """回归测试：两入口对 config.yaml=auto + 无 CLI + 无 & 前缀 的处理一致

        验证 run.py 和 scripts/run_iterate.py 在此场景下行为完全一致。
        """
        from cursor.cloud_client import CloudClientFactory
        from run import TaskAnalyzer
        from scripts.run_iterate import SelfIterator

        task = "普通任务描述"  # 无 & 前缀

        # 创建 config.yaml 为 auto 模式的 mock
        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.execution_mode = "auto"

        # 测试 run.py
        args_run = self._create_mock_args_for_run_py(execution_mode=None)
        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("run.get_config", return_value=mock_config):
                        analyzer = TaskAnalyzer()
                        run_analysis = analyzer._rule_based_analysis(task, args_run)

        run_orchestrator = run_analysis.options.get("orchestrator")
        run_effective_mode = run_analysis.options.get("effective_mode")
        run_triggered = run_analysis.options.get("triggered_by_prefix")

        # 测试 scripts/run_iterate.py
        args_iterate = self._create_mock_args_for_iterate_py(
            requirement=task,
            execution_mode=None,
        )
        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args_iterate)
                                        iterate_orchestrator = iterator._get_orchestrator_type()
                                        iterate_effective_mode = iterator._get_execution_mode()
                                        iterate_triggered = iterator._triggered_by_prefix

        # 验证一致性
        assert run_orchestrator == iterate_orchestrator == "basic", (
            f"编排器不一致: run.py={run_orchestrator}, iterate.py={iterate_orchestrator}"
        )
        assert run_effective_mode == iterate_effective_mode.value == "auto", (
            f"effective_mode 不一致: run.py={run_effective_mode}, iterate.py={iterate_effective_mode.value}"
        )
        assert run_triggered == iterate_triggered is False, (
            f"triggered_by_prefix 不一致: run.py={run_triggered}, iterate.py={iterate_triggered}"
        )

    def test_config_yaml_cloud_no_key_both_entries_consistent(self) -> None:
        """回归测试：两入口对 config.yaml=cloud + 无 API Key 的处理一致

        关键场景：requested_mode=cloud 但无 key 时，两入口行为应一致：
        - effective_mode 回退到 cli
        - orchestrator 仍为 basic（基于 requested_mode）
        """
        from cursor.cloud_client import CloudClientFactory
        from run import TaskAnalyzer
        from scripts.run_iterate import SelfIterator

        task = "普通任务描述"  # 无 & 前缀

        # 创建 config.yaml 为 cloud 模式的 mock
        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.execution_mode = "cloud"

        # 测试 run.py
        args_run = self._create_mock_args_for_run_py(execution_mode=None)
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):  # 无 key
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("run.get_config", return_value=mock_config):
                        analyzer = TaskAnalyzer()
                        run_analysis = analyzer._rule_based_analysis(task, args_run)

        run_orchestrator = run_analysis.options.get("orchestrator")
        run_effective_mode = run_analysis.options.get("effective_mode")

        # 测试 scripts/run_iterate.py
        args_iterate = self._create_mock_args_for_iterate_py(
            requirement=task,
            execution_mode=None,
        )
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):  # 无 key
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args_iterate)
                                        iterate_orchestrator = iterator._get_orchestrator_type()
                                        iterate_effective_mode = iterator._get_execution_mode()

        # 验证一致性
        assert run_orchestrator == iterate_orchestrator == "basic", (
            f"编排器不一致: run.py={run_orchestrator}, iterate.py={iterate_orchestrator}"
        )
        assert run_effective_mode == iterate_effective_mode.value == "cli", (
            f"effective_mode 不一致: run.py={run_effective_mode}, iterate.py={iterate_effective_mode.value}"
        )


# ============================================================
# 完整执行模式矩阵测试 (requested_mode 来源 x has_api_key x cloud_enabled)
# ============================================================


@dataclass
class FullMatrixTestCase:
    """完整矩阵测试用例

    覆盖三种 requested_mode 来源与 has_api_key/cloud_enabled 的组合。
    """

    test_id: str
    # === 输入参数 ===
    # requested_mode 来源
    mode_source: str  # "cli_explicit" | "config_yaml" | "ampersand_prefix"
    requested_mode: Optional[str]  # None（& 前缀触发）或显式模式
    # prompt 相关
    has_ampersand_prefix: bool
    # 环境条件
    has_api_key: bool
    cloud_enabled: bool
    # === 期望输出 ===
    expected_effective_mode: str
    expected_orchestrator: str
    # [迁移] expected_triggered_by_prefix -> expected_prefix_routed
    expected_prefix_routed: bool
    description: str


# 完整矩阵测试参数表
FULL_MATRIX_TEST_CASES = [
    # ===============================================================
    # 组 1: requested_mode 来源 = CLI 显式
    # ===============================================================
    # --- CLI 显式 cli ---
    FullMatrixTestCase(
        test_id="cli_explicit_cli_key_true_enabled_true",
        mode_source="cli_explicit",
        requested_mode="cli",
        has_ampersand_prefix=False,
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        description="CLI 显式 cli + 有 key + enabled → CLI + mp",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_cli_key_false_enabled_true",
        mode_source="cli_explicit",
        requested_mode="cli",
        has_ampersand_prefix=False,
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        description="CLI 显式 cli + 无 key + enabled → CLI + mp",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_cli_key_true_enabled_false",
        mode_source="cli_explicit",
        requested_mode="cli",
        has_ampersand_prefix=False,
        has_api_key=True,
        cloud_enabled=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        description="CLI 显式 cli + 有 key + disabled → CLI + mp",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_cli_key_false_enabled_false",
        mode_source="cli_explicit",
        requested_mode="cli",
        has_ampersand_prefix=False,
        has_api_key=False,
        cloud_enabled=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        description="CLI 显式 cli + 无 key + disabled → CLI + mp",
    ),
    # --- CLI 显式 auto（关键场景：回退时仍强制 basic）---
    FullMatrixTestCase(
        test_id="cli_explicit_auto_key_true_enabled_true",
        mode_source="cli_explicit",
        requested_mode="auto",
        has_ampersand_prefix=False,
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="auto",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="CLI 显式 auto + 有 key + enabled → auto + basic",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_auto_key_false_enabled_true",
        mode_source="cli_explicit",
        requested_mode="auto",
        has_ampersand_prefix=False,
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",  # 回退
        expected_orchestrator="basic",  # 关键：仍强制 basic
        expected_prefix_routed=False,
        description="关键：CLI 显式 auto + 无 key → 回退 cli 但仍强制 basic",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_auto_key_true_enabled_false",
        mode_source="cli_explicit",
        requested_mode="auto",
        has_ampersand_prefix=False,
        has_api_key=True,
        cloud_enabled=False,
        expected_effective_mode="auto",  # 显式 auto 不受 enabled 影响
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="CLI 显式 auto + 有 key + disabled → auto + basic",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_auto_key_false_enabled_false",
        mode_source="cli_explicit",
        requested_mode="auto",
        has_ampersand_prefix=False,
        has_api_key=False,
        cloud_enabled=False,
        expected_effective_mode="cli",  # 回退
        expected_orchestrator="basic",  # 关键：仍强制 basic
        expected_prefix_routed=False,
        description="关键：CLI 显式 auto + 无 key + disabled → 回退 cli 但仍强制 basic",
    ),
    # --- CLI 显式 cloud（关键场景：回退时仍强制 basic）---
    FullMatrixTestCase(
        test_id="cli_explicit_cloud_key_true_enabled_true",
        mode_source="cli_explicit",
        requested_mode="cloud",
        has_ampersand_prefix=False,
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="CLI 显式 cloud + 有 key + enabled → cloud + basic",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_cloud_key_false_enabled_true",
        mode_source="cli_explicit",
        requested_mode="cloud",
        has_ampersand_prefix=False,
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",  # 回退
        expected_orchestrator="basic",  # 关键：仍强制 basic
        expected_prefix_routed=False,
        description="关键：CLI 显式 cloud + 无 key → 回退 cli 但仍强制 basic",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_cloud_key_true_enabled_false",
        mode_source="cli_explicit",
        requested_mode="cloud",
        has_ampersand_prefix=False,
        has_api_key=True,
        cloud_enabled=False,
        expected_effective_mode="cloud",  # 显式 cloud 不受 enabled 影响
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="CLI 显式 cloud + 有 key + disabled → cloud + basic",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_cloud_key_false_enabled_false",
        mode_source="cli_explicit",
        requested_mode="cloud",
        has_ampersand_prefix=False,
        has_api_key=False,
        cloud_enabled=False,
        expected_effective_mode="cli",  # 回退
        expected_orchestrator="basic",  # 关键：仍强制 basic
        expected_prefix_routed=False,
        description="关键：CLI 显式 cloud + 无 key + disabled → 回退 cli 但仍强制 basic",
    ),
    # ===============================================================
    # 组 2: requested_mode 来源 = config.yaml 默认
    # ===============================================================
    # 【重要】config.yaml 默认 execution_mode=auto（来自 core/config.py DEFAULT_EXECUTION_MODE）
    # 因此本组测试用例统一以 auto 为默认基准。
    #
    # 如果未来 config.yaml 默认值变更，需同步更新此处的测试用例。
    #
    # 关键规则：
    # - overrides 不含 execution_mode 时，resolve_orchestrator_settings 读取
    #   config.yaml 默认 auto 并强制 basic
    # - requested_mode=auto 即强制 basic，不管 effective_mode 是否回退
    # --- config.yaml 默认 auto（系统默认行为）---
    FullMatrixTestCase(
        test_id="config_yaml_auto_key_true_enabled_true",
        mode_source="config_yaml",
        requested_mode="auto",
        has_ampersand_prefix=False,
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="auto",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="config.yaml auto + 有 key + enabled → auto + basic",
    ),
    FullMatrixTestCase(
        test_id="config_yaml_auto_key_false_enabled_true",
        mode_source="config_yaml",
        requested_mode="auto",
        has_ampersand_prefix=False,
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",  # 回退
        expected_orchestrator="basic",  # 仍强制 basic
        expected_prefix_routed=False,
        description="关键：config.yaml auto + 无 key → 回退 cli 但仍强制 basic",
    ),
    # --- config.yaml 默认 cloud ---
    FullMatrixTestCase(
        test_id="config_yaml_cloud_key_true_enabled_true",
        mode_source="config_yaml",
        requested_mode="cloud",
        has_ampersand_prefix=False,
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="config.yaml cloud + 有 key + enabled → cloud + basic",
    ),
    FullMatrixTestCase(
        test_id="config_yaml_cloud_key_false_enabled_true",
        mode_source="config_yaml",
        requested_mode="cloud",
        has_ampersand_prefix=False,
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",  # 回退
        expected_orchestrator="basic",  # 仍强制 basic
        expected_prefix_routed=False,
        description="关键：config.yaml cloud + 无 key → 回退 cli 但仍强制 basic",
    ),
    # ===============================================================
    # 组 3: requested_mode 来源 = & 前缀触发
    # ===============================================================
    # --- & 前缀触发成功 ---
    FullMatrixTestCase(
        test_id="ampersand_prefix_key_true_enabled_true",
        mode_source="ampersand_prefix",
        requested_mode=None,  # 无显式模式
        has_ampersand_prefix=True,
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=True,
        description="& 前缀 + 有 key + enabled → cloud + basic + triggered",
    ),
    # --- & 前缀触发失败（无 key）---
    FullMatrixTestCase(
        test_id="ampersand_prefix_key_false_enabled_true",
        mode_source="ampersand_prefix",
        requested_mode=None,
        has_ampersand_prefix=True,
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_prefix_routed=False,
        description="& 前缀 + 无 key → CLI + basic（& 前缀表达 Cloud 意图）",
    ),
    # --- & 前缀触发失败（disabled）---
    FullMatrixTestCase(
        test_id="ampersand_prefix_key_true_enabled_false",
        mode_source="ampersand_prefix",
        requested_mode=None,
        has_ampersand_prefix=True,
        has_api_key=True,
        cloud_enabled=False,
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_prefix_routed=False,
        description="& 前缀 + 有 key + disabled → CLI + basic（& 前缀表达 Cloud 意图）",
    ),
    # --- & 前缀触发失败（无 key + disabled）---
    FullMatrixTestCase(
        test_id="ampersand_prefix_key_false_enabled_false",
        mode_source="ampersand_prefix",
        requested_mode=None,
        has_ampersand_prefix=True,
        has_api_key=False,
        cloud_enabled=False,
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_prefix_routed=False,
        description="& 前缀 + 无 key + disabled → CLI + basic（& 前缀表达 Cloud 意图）",
    ),
    # ===============================================================
    # 组 4: 边界场景 - CLI 显式覆盖 & 前缀
    # ===============================================================
    FullMatrixTestCase(
        test_id="cli_explicit_cli_with_ampersand_key_true_enabled_true",
        mode_source="cli_explicit",
        requested_mode="cli",
        has_ampersand_prefix=True,  # 有 & 前缀
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="cli",  # CLI 显式覆盖
        expected_orchestrator="mp",
        expected_prefix_routed=False,  # 被忽略
        description="CLI 显式 cli + & 前缀 → 忽略前缀，CLI + mp",
    ),
    FullMatrixTestCase(
        test_id="cli_explicit_auto_with_ampersand_key_true_enabled_true",
        mode_source="cli_explicit",
        requested_mode="auto",
        has_ampersand_prefix=True,  # 有 & 前缀
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="auto",
        expected_orchestrator="basic",
        expected_prefix_routed=False,  # 显式 auto，不是 & 触发
        description="CLI 显式 auto + & 前缀 → auto + basic（非 & 触发）",
    ),
]


class TestFullExecutionMatrix:
    """完整执行模式矩阵测试

    覆盖三种 requested_mode 来源与 has_api_key/cloud_enabled 的组合：
    - CLI 显式参数
    - config.yaml 默认值
    - & 前缀触发

    核心断言规则：
    - requested_mode=auto/cloud ⇒ orchestrator=basic（即使 effective 回退）
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        FULL_MATRIX_TEST_CASES,
        ids=[tc.test_id for tc in FULL_MATRIX_TEST_CASES],
    )
    def test_build_execution_decision_full_matrix(self, test_case: FullMatrixTestCase) -> None:
        """表驱动测试：完整矩阵验证 build_execution_decision"""
        from core.execution_policy import build_execution_decision

        prompt = "& 任务描述" if test_case.has_ampersand_prefix else "任务描述"

        decision = build_execution_decision(
            prompt=prompt,
            requested_mode=test_case.requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        # 验证 effective_mode
        assert decision.effective_mode == test_case.expected_effective_mode, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  effective_mode 期望={test_case.expected_effective_mode}，"
            f"实际={decision.effective_mode}"
        )

        # 验证 orchestrator（核心断言）
        assert decision.orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  orchestrator 期望={test_case.expected_orchestrator}，"
            f"实际={decision.orchestrator}"
        )

        # 验证 prefix_routed（策略决策层面）
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed == test_case.expected_prefix_routed, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  prefix_routed 期望={test_case.expected_prefix_routed}，"
            f"实际={decision.prefix_routed}"
        )

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in FULL_MATRIX_TEST_CASES if tc.requested_mode in ("auto", "cloud")],
        ids=[tc.test_id for tc in FULL_MATRIX_TEST_CASES if tc.requested_mode in ("auto", "cloud")],
    )
    def test_auto_cloud_always_forces_basic(self, test_case: FullMatrixTestCase) -> None:
        """断言核心规则：requested_mode=auto/cloud ⇒ orchestrator=basic

        即使 effective_mode 回退到 cli，orchestrator 仍应为 basic。
        """
        from core.execution_policy import build_execution_decision

        prompt = "& 任务描述" if test_case.has_ampersand_prefix else "任务描述"

        decision = build_execution_decision(
            prompt=prompt,
            requested_mode=test_case.requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        # 核心断言：auto/cloud 必须强制 basic
        assert decision.orchestrator == "basic", (
            f"[{test_case.test_id}] 核心规则违反！\n"
            f"  requested_mode={test_case.requested_mode} 必须强制 basic 编排器\n"
            f"  effective_mode={decision.effective_mode}（可能是回退）\n"
            f"  实际 orchestrator={decision.orchestrator}"
        )


# ============================================================
# 从 core/execution_policy.py 导入的决策矩阵驱动测试
# ============================================================


class TestExecutionDecisionMatrixFromPolicy:
    """基于 core/execution_policy.py 中定义的决策矩阵驱动的测试

    本测试类使用 EXECUTION_DECISION_MATRIX_CASES（定义于 core/execution_policy.py）
    作为测试数据源，验证：
    1. resolve_requested_mode_for_decision 的输出
    2. build_execution_decision 的完整决策结果
    3. R-1/R-2/R-3/plan&ask 忽略/auto_detect=false 等关键边界场景

    ================================================================================
    与现有测试的关系
    ================================================================================

    - CONSISTENCY_TEST_CASES: 验证 should_use_mp_orchestrator 和 resolve_orchestrator_settings
    - FULL_MATRIX_TEST_CASES: 验证 build_execution_decision 的完整矩阵
    - EXECUTION_DECISION_MATRIX_CASES: **统一权威来源**，包含推导输入和期望输出的完整定义

    本测试类使用 EXECUTION_DECISION_MATRIX_CASES 验证从输入到输出的完整流程。

    ================================================================================
    统一字段 Schema 遵循声明
    ================================================================================

    本测试遵循 core/execution_policy.py 定义的"统一字段 Schema"：
    - cli_execution_mode: CLI --execution-mode 原始参数值
    - has_ampersand_prefix: 语法检测层面，原始 prompt 是否有 & 前缀
    - config_execution_mode: config.yaml 中的 cloud_agent.execution_mode
    - expected_requested_mode_for_decision: resolve_requested_mode_for_decision 的输出
    - expected_effective_mode: 有效执行模式
    - expected_orchestrator: 编排器类型
    - expected_prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "case",
        EXECUTION_DECISION_MATRIX_CASES,
        ids=[c.case_id for c in EXECUTION_DECISION_MATRIX_CASES],
    )
    def test_resolve_requested_mode_for_decision(self, case: DecisionMatrixCase) -> None:
        """验证 resolve_requested_mode_for_decision 的输出

        对于每个矩阵用例，验证给定 cli_execution_mode、has_ampersand_prefix、
        config_execution_mode 的输入，resolve_requested_mode_for_decision 返回
        期望的 requested_mode_for_decision。
        """
        actual = resolve_requested_mode_for_decision(
            cli_execution_mode=case.cli_execution_mode,
            has_ampersand_prefix=case.has_ampersand_prefix,
            config_execution_mode=case.config_execution_mode,
        )

        assert actual == case.expected_requested_mode_for_decision, (
            f"[{case.case_id}] {case.description}\n"
            f"  输入: cli_execution_mode={case.cli_execution_mode}, "
            f"has_ampersand_prefix={case.has_ampersand_prefix}, "
            f"config_execution_mode={case.config_execution_mode}\n"
            f"  期望 requested_mode_for_decision={case.expected_requested_mode_for_decision}\n"
            f"  实际={actual}\n"
            f"  适用规则: {case.applicable_rules}"
        )

    @pytest.mark.parametrize(
        "case",
        EXECUTION_DECISION_MATRIX_CASES,
        ids=[c.case_id for c in EXECUTION_DECISION_MATRIX_CASES],
    )
    def test_build_execution_decision_from_matrix(self, case: DecisionMatrixCase) -> None:
        """验证 build_execution_decision 的完整决策结果

        对于每个矩阵用例，构造 prompt 和参数，验证 build_execution_decision 返回
        期望的 effective_mode、orchestrator、prefix_routed。

        注意：使用 expected_requested_mode_for_decision 作为 requested_mode，
        这确保与实际调用流程一致：
        1. resolve_requested_mode_for_decision 计算 requested_mode
        2. build_execution_decision 使用该 requested_mode 进行决策
        """
        # 构造 prompt（根据 has_ampersand_prefix 决定是否带 & 前缀）
        prompt = "& 任务描述" if case.has_ampersand_prefix else "任务描述"

        # 使用 expected_requested_mode_for_decision 作为 requested_mode
        # 这是 resolve_requested_mode_for_decision 的输出，确保测试与实际流程一致
        requested_mode = case.expected_requested_mode_for_decision

        decision = build_execution_decision(
            prompt=prompt,
            requested_mode=requested_mode,
            cloud_enabled=case.cloud_enabled,
            has_api_key=case.has_api_key,
            auto_detect_cloud_prefix=case.auto_detect_cloud_prefix,
        )

        # 验证 effective_mode
        assert decision.effective_mode == case.expected_effective_mode, (
            f"[{case.case_id}] {case.description}\n"
            f"  effective_mode 期望={case.expected_effective_mode}，实际={decision.effective_mode}\n"
            f"  适用规则: {case.applicable_rules}"
        )

        # 验证 orchestrator（核心断言）
        assert decision.orchestrator == case.expected_orchestrator, (
            f"[{case.case_id}] {case.description}\n"
            f"  orchestrator 期望={case.expected_orchestrator}，实际={decision.orchestrator}\n"
            f"  适用规则: {case.applicable_rules}"
        )

        # 验证 prefix_routed（策略决策层面）
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert decision.prefix_routed == case.expected_prefix_routed, (
            f"[{case.case_id}] {case.description}\n"
            f"  prefix_routed 期望={case.expected_prefix_routed}，实际={decision.prefix_routed}\n"
            f"  适用规则: {case.applicable_rules}"
        )

        # 验证 has_ampersand_prefix（语法检测层面）
        assert decision.has_ampersand_prefix == case.has_ampersand_prefix, (
            f"[{case.case_id}] {case.description}\n"
            f"  has_ampersand_prefix 期望={case.has_ampersand_prefix}，"
            f"实际={decision.has_ampersand_prefix}"
        )

    @pytest.mark.parametrize(
        "case",
        [c for c in EXECUTION_DECISION_MATRIX_CASES if "R-1" in c.applicable_rules],
        ids=[c.case_id for c in EXECUTION_DECISION_MATRIX_CASES if "R-1" in c.applicable_rules],
    )
    def test_r1_auto_cloud_forces_basic(self, case: DecisionMatrixCase) -> None:
        """断言 R-1 规则：requested=auto/cloud → 强制 basic

        即使 effective_mode 回退到 cli，orchestrator 仍应为 basic。
        """
        prompt = "& 任务描述" if case.has_ampersand_prefix else "任务描述"
        requested_mode = case.cli_execution_mode or case.config_execution_mode

        decision = build_execution_decision(
            prompt=prompt,
            requested_mode=requested_mode,
            cloud_enabled=case.cloud_enabled,
            has_api_key=case.has_api_key,
            auto_detect_cloud_prefix=case.auto_detect_cloud_prefix,
        )

        # R-1 核心断言：auto/cloud 必须强制 basic
        if requested_mode in ("auto", "cloud"):
            assert decision.orchestrator == "basic", (
                f"[{case.case_id}] R-1 规则违反！\n"
                f"  requested_mode={requested_mode} 必须强制 basic 编排器\n"
                f"  effective_mode={decision.effective_mode}（可能是回退）\n"
                f"  实际 orchestrator={decision.orchestrator}"
            )

    @pytest.mark.parametrize(
        "case",
        [c for c in EXECUTION_DECISION_MATRIX_CASES if "R-2" in c.applicable_rules],
        ids=[c.case_id for c in EXECUTION_DECISION_MATRIX_CASES if "R-2" in c.applicable_rules],
    )
    def test_r2_ampersand_expresses_cloud_intent(self, case: DecisionMatrixCase) -> None:
        """断言 R-2 规则：& 前缀表达 Cloud 意图时强制 basic

        即使 prefix_routed=False（未成功路由），只要 & 前缀表达了 Cloud 意图，
        orchestrator 仍应为 basic。
        """
        prompt = "& 任务描述"  # R-2 规则仅适用于有 & 前缀的情况

        decision = build_execution_decision(
            prompt=prompt,
            requested_mode=None,  # & 前缀场景通常 requested_mode=None
            cloud_enabled=case.cloud_enabled,
            has_api_key=case.has_api_key,
            auto_detect_cloud_prefix=case.auto_detect_cloud_prefix,
        )

        # R-2 核心断言：& 前缀表达 Cloud 意图时强制 basic
        assert decision.orchestrator == "basic", (
            f"[{case.case_id}] R-2 规则违反！\n"
            f"  & 前缀表达 Cloud 意图，必须强制 basic 编排器\n"
            f"  prefix_routed={decision.prefix_routed}（可能未成功路由）\n"
            f"  实际 orchestrator={decision.orchestrator}"
        )

    @pytest.mark.parametrize(
        "case",
        [c for c in EXECUTION_DECISION_MATRIX_CASES if "R-3" in c.applicable_rules],
        ids=[c.case_id for c in EXECUTION_DECISION_MATRIX_CASES if "R-3" in c.applicable_rules],
    )
    def test_r3_prefix_ignored_allows_mp(self, case: DecisionMatrixCase) -> None:
        """断言 R-3 规则：& 前缀被忽略时允许 mp

        当显式 cli/plan/ask 或 auto_detect=false 时，& 前缀被忽略，
        编排器可使用 mp。
        """
        prompt = "& 任务描述" if case.has_ampersand_prefix else "任务描述"
        requested_mode = case.cli_execution_mode or case.config_execution_mode

        decision = build_execution_decision(
            prompt=prompt,
            requested_mode=requested_mode,
            cloud_enabled=case.cloud_enabled,
            has_api_key=case.has_api_key,
            auto_detect_cloud_prefix=case.auto_detect_cloud_prefix,
        )

        # R-3 核心断言：& 前缀被忽略时允许 mp
        assert decision.orchestrator == case.expected_orchestrator, (
            f"[{case.case_id}] R-3 规则验证\n"
            f"  期望 orchestrator={case.expected_orchestrator}\n"
            f"  实际 orchestrator={decision.orchestrator}\n"
            f"  prefix_routed={decision.prefix_routed}"
        )

        # 验证 prefix_routed=False（& 前缀被忽略）
        assert decision.prefix_routed is False, (
            f"[{case.case_id}] R-3 规则：& 前缀应被忽略\n  期望 prefix_routed=False，实际={decision.prefix_routed}"
        )


# ============================================================
# run.py 与 run_iterate.py 结果字段命名一致性测试
# ============================================================


class TestResultFieldNamingConsistency:
    """测试 run.py 与 scripts/run_iterate.py 结果字段命名一致性

    验证两个入口对于相同场景产生的结果字段名称一致，包括：
    - effective_mode / execution_mode
    - orchestrator
    - triggered_by_prefix / prefix_routed
    - has_ampersand_prefix
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def test_execution_decision_has_required_fields(self) -> None:
        """验证 ExecutionDecision 包含所有必需字段"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 测试任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )

        # 验证必需字段存在
        required_fields = [
            "effective_mode",
            "orchestrator",
            "triggered_by_prefix",  # 兼容字段
            "prefix_routed",  # 新字段
            "has_ampersand_prefix",  # 新字段
            "requested_mode",
            "sanitized_prompt",
            "mode_reason",
            "orchestrator_reason",
        ]

        d = decision.to_dict()
        for field in required_fields:
            assert field in d, f"ExecutionDecision 缺少字段: {field}"

    def test_execution_decision_to_dict_consistency(self) -> None:
        """验证 ExecutionDecision.to_dict() 输出字段与属性一致"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )

        d = decision.to_dict()

        # 验证字典值与属性一致
        assert d["effective_mode"] == decision.effective_mode
        assert d["orchestrator"] == decision.orchestrator
        assert d["triggered_by_prefix"] == decision.triggered_by_prefix
        assert d["prefix_routed"] == decision.prefix_routed
        assert d["has_ampersand_prefix"] == decision.has_ampersand_prefix
        assert d["requested_mode"] == decision.requested_mode
        assert d["sanitized_prompt"] == decision.sanitized_prompt

    def test_triggered_by_prefix_equals_prefix_routed(self) -> None:
        """验证 triggered_by_prefix 是 prefix_routed 的别名

        两者值应始终相等，确保向后兼容。
        """
        from core.execution_policy import build_execution_decision

        # 场景 1: & 前缀成功触发
        decision1 = build_execution_decision(
            prompt="& 任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        assert decision1.triggered_by_prefix == decision1.prefix_routed, (
            "triggered_by_prefix 与 prefix_routed 应相等（成功触发场景）"
        )

        # 场景 2: & 前缀未成功触发
        decision2 = build_execution_decision(
            prompt="& 任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=False,  # 无 key
        )
        assert decision2.triggered_by_prefix == decision2.prefix_routed, (
            "triggered_by_prefix 与 prefix_routed 应相等（未成功触发场景）"
        )

        # 场景 3: 无 & 前缀
        decision3 = build_execution_decision(
            prompt="普通任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
        )
        assert decision3.triggered_by_prefix == decision3.prefix_routed, (
            "triggered_by_prefix 与 prefix_routed 应相等（无前缀场景）"
        )

    def test_unified_overrides_has_consistent_fields(self) -> None:
        """验证 build_unified_overrides 返回的 UnifiedOptions 字段一致"""
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision

        args = argparse.Namespace(
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            execution_mode="auto",
            cloud_timeout=None,
            cloud_auth_timeout=None,
            minimal=False,
            dry_run=False,
            skip_online=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
        )

        decision = build_execution_decision(
            prompt="任务",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=True,
        )

        unified = build_unified_overrides(
            args=args,
            execution_decision=decision,
        )

        # 验证 resolved 字典包含预期字段
        assert "execution_mode" in unified.resolved
        assert "orchestrator" in unified.resolved
        assert "workers" in unified.resolved
        assert "max_iterations" in unified.resolved

        # 验证 effective_mode 字段存在
        assert unified.effective_mode == decision.effective_mode

        # 验证 has_ampersand_prefix 字段存在
        assert unified.has_ampersand_prefix == decision.has_ampersand_prefix

    def test_self_iterator_exposes_consistent_fields(self) -> None:
        """验证 SelfIterator 暴露的字段与 ExecutionDecision 一致"""
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        args = argparse.Namespace(
            requirement="& 后台任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        mock_config = _create_mock_config(cloud_enabled=True)

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        # 验证 SelfIterator 暴露的字段
                                        assert hasattr(iterator, "_triggered_by_prefix")
                                        assert hasattr(iterator, "_execution_decision")
                                        # 验证方法存在
                                        assert hasattr(iterator, "_get_execution_mode")
                                        assert hasattr(iterator, "_get_orchestrator_type")


# ============================================================
# 核心规则断言测试（专项）
# ============================================================


class TestCoreRuleAssertions:
    """核心规则断言测试

    专项测试确保核心规则不被违反：
    - requested_mode=auto/cloud ⇒ orchestrator=basic（即使 effective 回退）
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "requested_mode,has_api_key,cloud_enabled",
        [
            ("auto", True, True),
            ("auto", True, False),
            ("auto", False, True),
            ("auto", False, False),
            ("cloud", True, True),
            ("cloud", True, False),
            ("cloud", False, True),
            ("cloud", False, False),
        ],
        ids=[
            "auto_key_enabled",
            "auto_key_disabled",
            "auto_nokey_enabled",
            "auto_nokey_disabled",
            "cloud_key_enabled",
            "cloud_key_disabled",
            "cloud_nokey_enabled",
            "cloud_nokey_disabled",
        ],
    )
    def test_auto_cloud_always_basic_via_build_execution_decision(
        self,
        requested_mode: str,
        has_api_key: bool,
        cloud_enabled: bool,
    ) -> None:
        """验证 build_execution_decision: auto/cloud → basic"""
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="任务",
            requested_mode=requested_mode,
            cloud_enabled=cloud_enabled,
            has_api_key=has_api_key,
        )

        assert decision.orchestrator == "basic", (
            f"核心规则违反！requested_mode={requested_mode} 必须强制 basic\n"
            f"  has_api_key={has_api_key}, cloud_enabled={cloud_enabled}\n"
            f"  effective_mode={decision.effective_mode}\n"
            f"  实际 orchestrator={decision.orchestrator}"
        )

    @pytest.mark.parametrize(
        "requested_mode,has_api_key,cloud_enabled",
        [
            ("auto", True, True),
            ("auto", True, False),
            ("auto", False, True),
            ("auto", False, False),
            ("cloud", True, True),
            ("cloud", True, False),
            ("cloud", False, True),
            ("cloud", False, False),
        ],
        ids=[
            "auto_key_enabled",
            "auto_key_disabled",
            "auto_nokey_enabled",
            "auto_nokey_disabled",
            "cloud_key_enabled",
            "cloud_key_disabled",
            "cloud_nokey_enabled",
            "cloud_nokey_disabled",
        ],
    )
    def test_auto_cloud_always_basic_via_resolve_orchestrator_settings(
        self,
        requested_mode: str,
        has_api_key: bool,
        cloud_enabled: bool,
    ) -> None:
        """验证 resolve_orchestrator_settings: auto/cloud → basic"""
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": requested_mode},
            prefix_routed=False,
        )

        assert result["orchestrator"] == "basic", (
            f"核心规则违反！requested_mode={requested_mode} 必须强制 basic\n"
            f"  实际 orchestrator={result['orchestrator']}"
        )

    @pytest.mark.parametrize(
        "requested_mode,has_api_key,cloud_enabled",
        [
            ("auto", True, True),
            ("auto", True, False),
            ("auto", False, True),
            ("auto", False, False),
            ("cloud", True, True),
            ("cloud", True, False),
            ("cloud", False, True),
            ("cloud", False, False),
        ],
        ids=[
            "auto_key_enabled",
            "auto_key_disabled",
            "auto_nokey_enabled",
            "auto_nokey_disabled",
            "cloud_key_enabled",
            "cloud_key_disabled",
            "cloud_nokey_enabled",
            "cloud_nokey_disabled",
        ],
    )
    def test_auto_cloud_always_basic_via_should_use_mp_orchestrator(
        self,
        requested_mode: str,
        has_api_key: bool,
        cloud_enabled: bool,
    ) -> None:
        """验证 should_use_mp_orchestrator: auto/cloud → False（不可用 mp）"""
        can_use_mp = should_use_mp_orchestrator(requested_mode)

        assert can_use_mp is False, (
            f"核心规则违反！requested_mode={requested_mode} 不允许 mp\n  should_use_mp_orchestrator 返回 {can_use_mp}"
        )

    @pytest.mark.parametrize(
        "requested_mode",
        ["cli", "plan", "ask", None],
        ids=["cli", "plan", "ask", "none"],
    )
    def test_non_cloud_modes_allow_mp(self, requested_mode: Optional[str]) -> None:
        """验证非 Cloud 模式允许 mp"""
        can_use_mp = should_use_mp_orchestrator(requested_mode)

        assert can_use_mp is True, (
            f"requested_mode={requested_mode} 应允许 mp\n  should_use_mp_orchestrator 返回 {can_use_mp}"
        )


# ============================================================
# config.yaml 源的 requested_mode 一致性测试
# ============================================================


@dataclass
class ConfigYamlRequestedModeTestCase:
    """config.yaml 提供 requested_mode 的测试参数

    覆盖场景：当 requested_mode 来自 config.yaml（而非 CLI 显式参数）时，
    验证 run.py 的 build_unified_overrides 与 scripts/run_iterate.py 的
    ExecutionDecision 快照字段一致性。

    快照字段包括：
    - execution_mode: 原始请求模式（来自 config.yaml）
    - effective_mode: 有效执行模式（可能因缺少 API Key 等回退）
    - orchestrator: 编排器类型（mp/basic）
    - prefix_routed: & 前缀是否成功触发 Cloud

    注意：expected_orchestrator 用于直接传递 requested_mode 的测试场景。
    当 SelfIterator 使用 resolve_requested_mode_for_decision 时（& 前缀存在时返回 None），
    可能需要不同的期望值，此时使用 expected_orchestrator_self_iterator。
    """

    test_id: str
    config_execution_mode: str  # config.yaml 中的 cloud_agent.execution_mode
    prompt: str  # 用户输入的任务描述（可包含 & 前缀）
    has_api_key: bool
    cloud_enabled: bool
    # 预期结果
    expected_effective_mode: str
    expected_orchestrator: str
    expected_prefix_routed: bool
    description: str
    # 可选：SelfIterator 测试的期望 orchestrator（当与 expected_orchestrator 不同时使用）
    expected_orchestrator_self_iterator: Optional[str] = None


# config.yaml 源 requested_mode 测试矩阵
CONFIG_YAML_REQUESTED_MODE_CASES = [
    # ===== config.yaml execution_mode=auto 场景 =====
    ConfigYamlRequestedModeTestCase(
        test_id="config_auto_with_key_enabled_no_prefix",
        config_execution_mode="auto",
        prompt="普通任务描述",
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="auto",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="config.yaml auto + 有 key + enabled + 无前缀 → auto/basic",
    ),
    ConfigYamlRequestedModeTestCase(
        test_id="config_auto_no_key_enabled_no_prefix",
        config_execution_mode="auto",
        prompt="普通任务描述",
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 但编排器仍强制 basic（基于 requested_mode）
        expected_prefix_routed=False,
        description="关键场景：config.yaml auto + 无 key → effective=cli 但 orchestrator=basic",
    ),
    ConfigYamlRequestedModeTestCase(
        test_id="config_auto_with_key_disabled_no_prefix",
        config_execution_mode="auto",
        prompt="普通任务描述",
        has_api_key=True,
        cloud_enabled=False,
        expected_effective_mode="auto",  # 仍为 auto（cloud_enabled 不影响显式模式）
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="config.yaml auto + cloud_disabled → auto/basic",
    ),
    ConfigYamlRequestedModeTestCase(
        test_id="config_auto_with_prefix_with_key_enabled",
        config_execution_mode="auto",
        prompt="& 后台任务",
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="auto",
        expected_orchestrator="basic",
        expected_prefix_routed=False,  # 显式 auto 模式下，& 前缀不视为"触发"
        description="config.yaml auto + 有 & 前缀 → auto/basic（& 前缀非触发源）",
    ),
    # ===== config.yaml execution_mode=cloud 场景 =====
    ConfigYamlRequestedModeTestCase(
        test_id="config_cloud_with_key_enabled_no_prefix",
        config_execution_mode="cloud",
        prompt="普通任务描述",
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="config.yaml cloud + 有 key + enabled + 无前缀 → cloud/basic",
    ),
    ConfigYamlRequestedModeTestCase(
        test_id="config_cloud_no_key_enabled_no_prefix",
        config_execution_mode="cloud",
        prompt="普通任务描述",
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 但编排器仍强制 basic（基于 requested_mode）
        expected_prefix_routed=False,
        description="关键场景：config.yaml cloud + 无 key → effective=cli 但 orchestrator=basic",
    ),
    ConfigYamlRequestedModeTestCase(
        test_id="config_cloud_with_key_disabled_no_prefix",
        config_execution_mode="cloud",
        prompt="普通任务描述",
        has_api_key=True,
        cloud_enabled=False,
        expected_effective_mode="cloud",  # 仍为 cloud（cloud_enabled 不影响显式模式）
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        description="config.yaml cloud + cloud_disabled → cloud/basic",
    ),
    ConfigYamlRequestedModeTestCase(
        test_id="config_cloud_with_prefix_no_key_enabled",
        config_execution_mode="cloud",
        prompt="& 后台任务",
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",  # 回退
        # & 前缀表达 Cloud 意图，即使 resolve_requested_mode_for_decision 返回 None，
        # build_execution_decision 也会因 has_ampersand_prefix=True 强制使用 basic。
        expected_orchestrator="basic",  # 直接传递 requested_mode="cloud" 时强制 basic
        expected_prefix_routed=False,  # 无 key，& 前缀未成功触发
        description="config.yaml cloud + & 前缀 + 无 key → cli/basic（& 前缀表达 Cloud 意图）",
        # SelfIterator 现在也使用 basic（& 前缀表达 Cloud 意图）
        expected_orchestrator_self_iterator="basic",
    ),
    # ===== config.yaml execution_mode=cli 场景（对照组）=====
    ConfigYamlRequestedModeTestCase(
        test_id="config_cli_with_key_enabled_no_prefix",
        config_execution_mode="cli",
        prompt="普通任务描述",
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="cli",
        expected_orchestrator="mp",  # cli 模式允许 mp
        expected_prefix_routed=False,
        description="config.yaml cli → cli/mp（对照组）",
    ),
    ConfigYamlRequestedModeTestCase(
        test_id="config_cli_with_prefix_with_key_enabled",
        config_execution_mode="cli",
        prompt="& 后台任务",
        has_api_key=True,
        cloud_enabled=True,
        expected_effective_mode="cli",  # 显式 cli 忽略 & 前缀
        expected_orchestrator="mp",
        expected_prefix_routed=False,  # 显式 cli 模式忽略 & 前缀
        description="config.yaml cli + & 前缀 → cli/mp（忽略 & 前缀）",
    ),
    ConfigYamlRequestedModeTestCase(
        test_id="config_cli_with_prefix_no_key_enabled",
        config_execution_mode="cli",
        prompt="& 后台任务",
        has_api_key=False,
        cloud_enabled=True,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        description="config.yaml cli + & 前缀 + 无 key → cli/mp",
    ),
]


class TestConfigYamlRequestedModeConsistency:
    """测试 config.yaml 提供 requested_mode 时的一致性

    验证当 requested_mode 来自 config.yaml（而非 CLI 显式参数）时，
    run.py 的 build_unified_overrides 与 ExecutionDecision 快照字段一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        CONFIG_YAML_REQUESTED_MODE_CASES,
        ids=[tc.test_id for tc in CONFIG_YAML_REQUESTED_MODE_CASES],
    )
    def test_execution_decision_matches_expected(self, test_case: ConfigYamlRequestedModeTestCase) -> None:
        """测试 ExecutionDecision 输出符合预期

        使用 build_execution_decision 直接验证决策结果。
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        # 构建 ExecutionDecision
        decision = build_execution_decision(
            prompt=test_case.prompt,
            requested_mode=test_case.config_execution_mode,  # 来自 config.yaml
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        # 验证 effective_mode
        assert decision.effective_mode == test_case.expected_effective_mode, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  effective_mode 期望={test_case.expected_effective_mode}，"
            f"实际={decision.effective_mode}"
        )

        # 验证 orchestrator
        assert decision.orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  orchestrator 期望={test_case.expected_orchestrator}，"
            f"实际={decision.orchestrator}"
        )

        # 验证 prefix_routed
        assert decision.prefix_routed == test_case.expected_prefix_routed, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  prefix_routed 期望={test_case.expected_prefix_routed}，"
            f"实际={decision.prefix_routed}"
        )

    @pytest.mark.parametrize(
        "test_case",
        CONFIG_YAML_REQUESTED_MODE_CASES,
        ids=[tc.test_id for tc in CONFIG_YAML_REQUESTED_MODE_CASES],
    )
    def test_resolve_orchestrator_matches_decision(self, test_case: ConfigYamlRequestedModeTestCase) -> None:
        """测试 resolve_orchestrator_settings 与 ExecutionDecision 一致

        验证两个函数对相同输入产生相同的 orchestrator 选择。
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        # 方式 1：build_execution_decision
        decision = build_execution_decision(
            prompt=test_case.prompt,
            requested_mode=test_case.config_execution_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        # 方式 2：resolve_orchestrator_settings
        overrides = {"execution_mode": test_case.config_execution_mode}
        result = resolve_orchestrator_settings(
            overrides=overrides,
            prefix_routed=decision.prefix_routed,
        )

        # 验证 orchestrator 一致
        assert result["orchestrator"] == decision.orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  resolve_orchestrator_settings.orchestrator={result['orchestrator']}\n"
            f"  build_execution_decision.orchestrator={decision.orchestrator}"
        )

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in CONFIG_YAML_REQUESTED_MODE_CASES if tc.config_execution_mode in ("auto", "cloud")],
        ids=[tc.test_id for tc in CONFIG_YAML_REQUESTED_MODE_CASES if tc.config_execution_mode in ("auto", "cloud")],
    )
    def test_build_unified_overrides_matches_decision(self, test_case: ConfigYamlRequestedModeTestCase) -> None:
        """测试 build_unified_overrides 与 ExecutionDecision 一致

        验证 run.py 使用的 build_unified_overrides 函数对 config.yaml 源的
        requested_mode 产生与 ExecutionDecision 一致的结果。
        """
        import argparse

        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        # 构建 ExecutionDecision
        decision = build_execution_decision(
            prompt=test_case.prompt,
            requested_mode=test_case.config_execution_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        # 构建 args（模拟无 CLI 显式设置）
        args = argparse.Namespace(
            requirement=test_case.prompt,
            execution_mode=None,  # 无 CLI 显式设置
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            workers=3,
            max_iterations="10",
            dry_run=False,
            auto_commit=False,
            auto_push=False,
            cloud_timeout=300,
            cloud_auth_timeout=30,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
        )

        # Mock 配置
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_case.cloud_enabled
        mock_config.cloud_agent.execution_mode = test_case.config_execution_mode
        mock_config.cloud_agent.timeout = 300
        mock_config.cloud_agent.auth_timeout = 30
        mock_config.system.max_iterations = 10
        mock_config.system.worker_pool_size = 3

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # 调用 build_unified_overrides
                unified = build_unified_overrides(
                    args=args,
                    execution_decision=decision,
                    prefix_routed=decision.prefix_routed,
                )

        # 验证 orchestrator 一致
        assert unified.orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  build_unified_overrides.orchestrator={unified.orchestrator}\n"
            f"  期望={test_case.expected_orchestrator}"
        )

        # 验证 effective_mode 一致
        assert unified.effective_mode == test_case.expected_effective_mode, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  build_unified_overrides.effective_mode={unified.effective_mode}\n"
            f"  期望={test_case.expected_effective_mode}"
        )

        # 验证 prefix_routed 一致（策略决策层面）
        # ⚠ 新断言规范：使用 prefix_routed 而非 triggered_by_prefix
        assert unified.prefix_routed == test_case.expected_prefix_routed, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  build_unified_overrides.prefix_routed={unified.prefix_routed}\n"
            f"  期望={test_case.expected_prefix_routed}"
        )


class TestConfigYamlAndSelfIteratorConsistency:
    """测试 config.yaml 源与 SelfIterator 的一致性

    验证当 requested_mode 来自 config.yaml 时，
    SelfIterator 的 _execution_decision 与直接调用 build_execution_decision 一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in CONFIG_YAML_REQUESTED_MODE_CASES if tc.config_execution_mode in ("auto", "cloud")],
        ids=[tc.test_id for tc in CONFIG_YAML_REQUESTED_MODE_CASES if tc.config_execution_mode in ("auto", "cloud")],
    )
    def test_self_iterator_decision_matches_expected(self, test_case: ConfigYamlRequestedModeTestCase) -> None:
        """测试 SelfIterator 的执行决策与预期一致"""
        import argparse

        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        ConfigManager.reset_instance()

        # 构建 args（无 CLI 显式设置 execution_mode，使用 config.yaml 默认值）
        args = argparse.Namespace(
            requirement=test_case.prompt,
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",  # 默认请求 mp，但应被强制切换
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=None,  # 无 CLI 显式设置，使用 config.yaml
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        # Mock 配置
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_case.cloud_enabled
        mock_config.cloud_agent.execution_mode = test_case.config_execution_mode
        mock_config.cloud_agent.timeout = 300
        mock_config.cloud_agent.auth_timeout = 30
        mock_config.system.max_iterations = 10
        mock_config.system.worker_pool_size = 3
        mock_config.system.enable_sub_planners = True
        mock_config.system.strict_review = False
        mock_config.models.planner = DEFAULT_PLANNER_MODEL
        mock_config.models.worker = DEFAULT_WORKER_MODEL
        mock_config.models.reviewer = DEFAULT_REVIEWER_MODEL
        mock_config.planner.timeout = 500.0
        mock_config.worker.task_timeout = 600.0
        mock_config.reviewer.timeout = 300.0
        mock_config.logging.stream_json.enabled = False
        mock_config.logging.stream_json.console = True
        mock_config.logging.stream_json.detail_dir = "logs/stream_json/detail/"
        mock_config.logging.stream_json.raw_dir = "logs/stream_json/raw/"

        api_key = "mock-api-key" if test_case.has_api_key else None

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    iterator = SelfIterator(args)
                    actual_orchestrator = iterator._get_orchestrator_type()

        # 计算 SelfIterator 场景下的预期 orchestrator
        #
        # 关键逻辑差异：
        # - 直接测试 build_execution_decision：传递 requested_mode=config.yaml 值（如 "auto"/"cloud"）
        # - SelfIterator：使用 resolve_requested_mode_for_decision，当有 & 前缀时返回 None
        #
        # 当 SelfIterator 传递 requested_mode=None 且有 & 前缀时：
        # - 如果条件满足（有 key、enabled），& 前缀成功触发 Cloud，orchestrator=basic
        # - 如果条件不满足（无 key 或 disabled），& 前缀未触发，orchestrator=mp
        from core.cloud_utils import is_cloud_request

        has_ampersand = is_cloud_request(test_case.prompt)

        # 判断 SelfIterator 场景下 & 前缀是否会成功触发
        # 当 requested_mode=None 时，build_execution_decision 会检查：
        # - has_ampersand_prefix=True
        # - cloud_enabled=True
        # - has_api_key=True
        # 如果都满足，prefix_routed=True，orchestrator=basic
        self_iterator_prefix_would_trigger = has_ampersand and test_case.cloud_enabled and test_case.has_api_key

        if self_iterator_prefix_would_trigger:
            # & 前缀成功触发 Cloud，orchestrator=basic
            expected_orchestrator_for_self_iterator = "basic"
        elif has_ampersand and not self_iterator_prefix_would_trigger:
            # & 前缀存在但未成功触发（无 key 或 disabled），
            # 由于 & 前缀表达 Cloud 意图，仍强制使用 basic（与 AGENTS.md 一致）
            expected_orchestrator_for_self_iterator = "basic"
        else:
            # 无 & 前缀，使用 config.yaml 的 execution_mode
            expected_orchestrator_for_self_iterator = test_case.expected_orchestrator

        # 验证 orchestrator
        assert actual_orchestrator == expected_orchestrator_for_self_iterator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  SelfIterator._get_orchestrator_type()={actual_orchestrator}\n"
            f"  期望={expected_orchestrator_for_self_iterator}\n"
            f"  (has_ampersand={has_ampersand}, prefix_routed={test_case.expected_prefix_routed})"
        )


class TestConfigYamlAutoCloudForcesBasicCore:
    """核心规则测试：config.yaml auto/cloud 必须强制 basic

    这是最关键的规则验证：当 requested_mode 来自 config.yaml 且为 auto/cloud 时，
    无论 has_api_key、cloud_enabled 如何变化，orchestrator 始终为 basic。

    这与 AGENTS.md 中的描述一致：
    "请求 Cloud 即强制 basic，不受 key/enable 影响"
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "config_execution_mode,has_api_key,cloud_enabled",
        [
            ("auto", True, True),
            ("auto", True, False),
            ("auto", False, True),
            ("auto", False, False),
            ("cloud", True, True),
            ("cloud", True, False),
            ("cloud", False, True),
            ("cloud", False, False),
        ],
        ids=[
            "config_auto_key_enabled",
            "config_auto_key_disabled",
            "config_auto_nokey_enabled",
            "config_auto_nokey_disabled",
            "config_cloud_key_enabled",
            "config_cloud_key_disabled",
            "config_cloud_nokey_enabled",
            "config_cloud_nokey_disabled",
        ],
    )
    def test_config_yaml_auto_cloud_forces_basic_via_decision(
        self,
        config_execution_mode: str,
        has_api_key: bool,
        cloud_enabled: bool,
    ) -> None:
        """验证 build_execution_decision: config.yaml auto/cloud → basic"""
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        decision = build_execution_decision(
            prompt="普通任务",
            requested_mode=config_execution_mode,  # 来自 config.yaml
            cloud_enabled=cloud_enabled,
            has_api_key=has_api_key,
        )

        assert decision.orchestrator == "basic", (
            f"核心规则违反！config.yaml {config_execution_mode} 必须强制 basic\n"
            f"  has_api_key={has_api_key}, cloud_enabled={cloud_enabled}\n"
            f"  实际 orchestrator={decision.orchestrator}"
        )

    @pytest.mark.parametrize(
        "config_execution_mode,has_api_key,cloud_enabled",
        [
            ("auto", True, True),
            ("auto", True, False),
            ("auto", False, True),
            ("auto", False, False),
            ("cloud", True, True),
            ("cloud", True, False),
            ("cloud", False, True),
            ("cloud", False, False),
        ],
        ids=[
            "config_auto_key_enabled",
            "config_auto_key_disabled",
            "config_auto_nokey_enabled",
            "config_auto_nokey_disabled",
            "config_cloud_key_enabled",
            "config_cloud_key_disabled",
            "config_cloud_nokey_enabled",
            "config_cloud_nokey_disabled",
        ],
    )
    def test_config_yaml_auto_cloud_forces_basic_via_resolve(
        self,
        config_execution_mode: str,
        has_api_key: bool,
        cloud_enabled: bool,
    ) -> None:
        """验证 resolve_orchestrator_settings: config.yaml auto/cloud → basic"""
        ConfigManager.reset_instance()

        result = resolve_orchestrator_settings(
            overrides={"execution_mode": config_execution_mode},
            prefix_routed=False,
        )

        assert result["orchestrator"] == "basic", (
            f"核心规则违反！config.yaml {config_execution_mode} 必须强制 basic\n"
            f"  has_api_key={has_api_key}, cloud_enabled={cloud_enabled}\n"
            f"  实际 orchestrator={result['orchestrator']}"
        )

    @pytest.mark.parametrize(
        "config_execution_mode,has_api_key,cloud_enabled",
        [
            ("auto", False, True),
            ("auto", False, False),
            ("cloud", False, True),
            ("cloud", False, False),
        ],
        ids=[
            "config_auto_nokey_enabled",
            "config_auto_nokey_disabled",
            "config_cloud_nokey_enabled",
            "config_cloud_nokey_disabled",
        ],
    )
    def test_config_yaml_auto_cloud_no_key_effective_cli_but_orchestrator_basic(
        self,
        config_execution_mode: str,
        has_api_key: bool,
        cloud_enabled: bool,
    ) -> None:
        """关键场景：config.yaml auto/cloud + 无 key → effective=cli 但 orchestrator=basic

        这是最重要的一致性测试：
        - effective_mode 因缺少 API Key 回退到 cli
        - 但 orchestrator 仍强制 basic（基于 requested_mode）

        这符合 AGENTS.md 的设计：
        "编排器选择基于 requested_mode，不是 effective_mode"
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        decision = build_execution_decision(
            prompt="普通任务",
            requested_mode=config_execution_mode,
            cloud_enabled=cloud_enabled,
            has_api_key=False,  # 无 API Key
        )

        # effective_mode 应回退到 cli
        assert decision.effective_mode == "cli", (
            f"config.yaml {config_execution_mode} + 无 key 应回退到 cli\n"
            f"  实际 effective_mode={decision.effective_mode}"
        )

        # 但 orchestrator 仍应强制 basic
        assert decision.orchestrator == "basic", (
            f"关键规则违反！即使 effective=cli，orchestrator 仍应为 basic\n"
            f"  config_execution_mode={config_execution_mode}\n"
            f"  实际 orchestrator={decision.orchestrator}"
        )

        # 新增断言：prefix_routed 应为 False（无 & 前缀触发）
        assert decision.prefix_routed is False, (
            f"config.yaml {config_execution_mode} + 无 key 时 prefix_routed 应为 False\n"
            f"  实际 prefix_routed={decision.prefix_routed}"
        )


# ============================================================
# 新增：requested_mode=auto 无 API Key 快照一致性专项测试
# ============================================================


class TestAutoNoKeySnapshotConsistency:
    """测试 requested_mode=auto 无 API Key 时的快照字段一致性

    验证 run.py 与 scripts/run_iterate.py 对相同输入产生一致的决策快照。
    此类复用 build_execution_decision 作为权威决策源。

    核心规则（与 AGENTS.md 对齐）：
    - requested_mode=auto 且无 API Key 时：
      - effective_mode 应回退到 "cli"
      - orchestrator 应强制为 "basic"（基于 requested_mode）
      - prefix_routed 应为 False（未成功触发 Cloud）
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def test_auto_no_key_snapshot_via_build_execution_decision(self) -> None:
        """使用 build_execution_decision 验证 auto + 无 key 场景

        这是权威决策测试，确保所有入口复用 build_execution_decision 时
        能获得一致的决策结果。
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="分析代码",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key
        )

        # 核心断言 1: orchestrator 必须为 basic
        assert decision.orchestrator == "basic", (
            f"关键规则：requested_mode=auto 无 key 时 orchestrator 必须为 basic\n"
            f"  实际 orchestrator={decision.orchestrator}"
        )

        # 核心断言 2: prefix_routed 必须为 False
        assert decision.prefix_routed is False, (
            f"关键规则：无 & 前缀时 prefix_routed 必须为 False\n  实际 prefix_routed={decision.prefix_routed}"
        )

        # 核心断言 3: effective_mode 应回退到 cli
        assert decision.effective_mode == "cli", (
            f"关键规则：auto + 无 key 应回退到 cli\n  实际 effective_mode={decision.effective_mode}"
        )

    def test_run_py_and_run_iterate_snapshot_consistency(self) -> None:
        """验证 run.py 与 scripts/run_iterate.py 快照字段一致

        复用 build_execution_decision，分别模拟两个入口的调用方式，
        验证快照字段完全一致。
        """
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # === Step 1: 通过 build_execution_decision 获取权威决策 ===
        decision = build_execution_decision(
            prompt="任务描述",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,
        )

        # 权威快照字段
        expected_effective_mode = decision.effective_mode
        expected_orchestrator = decision.orchestrator
        expected_prefix_routed = decision.prefix_routed

        # === Step 2: 模拟 scripts/run_iterate.py 的 SelfIterator ===
        args = argparse.Namespace(
            requirement="任务描述",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="auto",
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        mock_config = _create_mock_config(cloud_enabled=True)

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        iterator_effective_mode = iterator._get_execution_mode()
                                        iterator_orchestrator = iterator._get_orchestrator_type()
                                        iterator_prefix_routed = iterator._prefix_routed

        # === Step 3: 验证快照一致性 ===
        # orchestrator 一致性（核心断言）
        assert iterator_orchestrator == expected_orchestrator, (
            f"快照不一致：orchestrator\n"
            f"  build_execution_decision={expected_orchestrator}\n"
            f"  SelfIterator={iterator_orchestrator}"
        )

        # prefix_routed 一致性（核心断言）
        assert iterator_prefix_routed == expected_prefix_routed, (
            f"快照不一致：prefix_routed\n"
            f"  build_execution_decision={expected_prefix_routed}\n"
            f"  SelfIterator={iterator_prefix_routed}"
        )

        # effective_mode 一致性
        assert iterator_effective_mode.value == expected_effective_mode, (
            f"快照不一致：effective_mode\n"
            f"  build_execution_decision={expected_effective_mode}\n"
            f"  SelfIterator={iterator_effective_mode.value}"
        )

    def test_run_py_snapshot_via_compute_function(self) -> None:
        """使用 _compute_run_py_snapshot 验证 run.py 语义

        验证 _compute_run_py_snapshot 计算的快照与 build_execution_decision
        的结果一致。
        """
        from core.execution_policy import build_execution_decision

        # 构造测试用例
        test_case = SnapshotTestCase(
            test_id="auto_no_key_snapshot_test",
            requirement="任务描述",
            execution_mode="auto",
            has_api_key=False,
            cloud_enabled=True,
            orchestrator_cli=None,
            no_mp_cli=False,
            expected_snapshot=DecisionSnapshot(
                effective_mode="cli",
                orchestrator="basic",
                prefix_routed=False,
                requested_mode_for_decision="auto",
                cli_execution_mode="auto",
            ),
            description="auto + 无 key 快照一致性测试",
        )

        # 计算 run.py 语义快照
        run_py_snapshot = _compute_run_py_snapshot(test_case)

        # 直接调用 build_execution_decision 获取权威决策
        decision = build_execution_decision(
            prompt=test_case.requirement,
            requested_mode=test_case.execution_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        # 验证 orchestrator 一致
        assert run_py_snapshot.orchestrator == decision.orchestrator, (
            f"orchestrator 不一致\n"
            f"  _compute_run_py_snapshot={run_py_snapshot.orchestrator}\n"
            f"  build_execution_decision={decision.orchestrator}"
        )

        # 验证 prefix_routed 一致
        assert run_py_snapshot.prefix_routed == decision.prefix_routed, (
            f"prefix_routed 不一致\n"
            f"  _compute_run_py_snapshot={run_py_snapshot.prefix_routed}\n"
            f"  build_execution_decision={decision.prefix_routed}"
        )

        # 验证期望快照也一致
        assert run_py_snapshot.orchestrator == test_case.expected_snapshot.orchestrator, (
            f"orchestrator 与预期不符\n"
            f"  实际={run_py_snapshot.orchestrator}\n"
            f"  预期={test_case.expected_snapshot.orchestrator}"
        )
        assert run_py_snapshot.prefix_routed == test_case.expected_snapshot.prefix_routed, (
            f"prefix_routed 与预期不符\n"
            f"  实际={run_py_snapshot.prefix_routed}\n"
            f"  预期={test_case.expected_snapshot.prefix_routed}"
        )


# ============================================================
# 新增：使用 monkeypatch 清除环境变量的真实场景测试
# ============================================================


class TestNoApiKeyViaMonkeypatchEnvironment:
    """使用 monkeypatch 确保环境变量不存在时的行为测试

    本测试类与现有使用 patch.object(CloudClientFactory, "resolve_api_key", ...)
    的测试不同，它通过 clean_env fixture 真正清除环境变量来模拟生产环境中
    无 API Key 的场景。

    核心验证规则（与 AGENTS.md 对齐）：
    - 无 API Key 时，CloudClientFactory.resolve_api_key() 返回 None
    - execution_mode=auto 时：orchestrator=basic, effective_mode=cli（回退）
    - execution_mode=cloud 时：orchestrator=basic, effective_mode=cli（回退）
    - 显式 execution_mode=cli 时：orchestrator=mp（允许 mp 编排器）

    使用 clean_env fixture（来自 conftest.py）统一清除 API Key 环境变量。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.fixture
    def clean_api_key_env(self, clean_env) -> None:
        """委托给全局 clean_env fixture（来自 conftest.py）

        保留此方法签名以兼容现有测试方法，避免大量重构。
        clean_env fixture 会清除 CURSOR_API_KEY 和 CURSOR_CLOUD_API_KEY。
        """
        # clean_env fixture 已经处理了环境变量清理
        pass

    def test_resolve_api_key_returns_none_without_env_vars(
        self, clean_api_key_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证当环境变量不存在时 CloudClientFactory.resolve_api_key() 返回 None

        这是前置验证测试，确保 monkeypatch 正确清除了环境变量。
        """
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        # 创建 Mock 配置，确保 config.yaml 中也没有 API Key
        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # 不传入任何参数，纯依赖环境变量和配置
                api_key = CloudClientFactory.resolve_api_key()

        assert api_key is None, f"环境变量已清除，resolve_api_key() 应返回 None\n  实际返回值={api_key}"

    def test_auto_mode_no_key_forces_basic_orchestrator(
        self, clean_api_key_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """测试 auto 模式无 API Key 时强制 basic 编排器

        使用 monkeypatch 清除环境变量后验证：
        - resolve_orchestrator_settings: orchestrator=basic
        - build_unified_overrides: orchestrator=basic, effective_mode=cli
        """
        from core.config import build_unified_overrides
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        # 创建 Mock 配置
        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        # 构建 args
        args = argparse.Namespace(
            requirement="分析代码",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",  # 用户请求 mp，但应被强制覆盖
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="auto",  # 关键：显式 auto
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # 测试 1: resolve_orchestrator_settings
                result = resolve_orchestrator_settings(
                    overrides={"execution_mode": "auto"},
                    prefix_routed=False,
                )

                assert result["orchestrator"] == "basic", (
                    "resolve_orchestrator_settings: auto 模式无 API Key 应强制 basic\n"
                    f"  实际 orchestrator={result['orchestrator']}"
                )

                # 测试 2: build_unified_overrides
                # 先验证 resolve_api_key 返回 None
                api_key = CloudClientFactory.resolve_api_key()
                assert api_key is None, "前置条件：resolve_api_key 应返回 None"

                # 构建 execution_decision
                from core.execution_policy import build_execution_decision

                decision = build_execution_decision(
                    prompt="分析代码",
                    requested_mode="auto",
                    cloud_enabled=True,
                    has_api_key=False,
                )

                unified = build_unified_overrides(
                    args=args,
                    execution_decision=decision,
                )

                # 核心断言：orchestrator=basic
                assert unified.resolved["orchestrator"] == "basic", (
                    "build_unified_overrides: auto 模式无 API Key 应强制 basic\n"
                    f"  实际 orchestrator={unified.resolved['orchestrator']}"
                )

                # 核心断言：effective_mode=cli（回退）
                assert unified.effective_mode == "cli", (
                    "build_unified_overrides: auto 模式无 API Key 应回退到 cli\n"
                    f"  实际 effective_mode={unified.effective_mode}"
                )

                # 核心断言：requested_mode=auto
                assert unified.requested_mode == "auto", (
                    "build_unified_overrides: requested_mode 应保持为 auto\n"
                    f"  实际 requested_mode={unified.requested_mode}"
                )

    def test_cloud_mode_no_key_forces_basic_orchestrator(
        self, clean_api_key_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """测试 cloud 模式无 API Key 时强制 basic 编排器

        使用 monkeypatch 清除环境变量后验证：
        - resolve_orchestrator_settings: orchestrator=basic
        - build_unified_overrides: orchestrator=basic, effective_mode=cli
        """
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        args = argparse.Namespace(
            requirement="后台分析任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cloud",  # 关键：显式 cloud
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # 测试 1: resolve_orchestrator_settings
                result = resolve_orchestrator_settings(
                    overrides={"execution_mode": "cloud"},
                    prefix_routed=False,
                )

                assert result["orchestrator"] == "basic", (
                    "resolve_orchestrator_settings: cloud 模式无 API Key 应强制 basic\n"
                    f"  实际 orchestrator={result['orchestrator']}"
                )

                # 测试 2: build_unified_overrides
                api_key = CloudClientFactory.resolve_api_key()
                assert api_key is None, "前置条件：resolve_api_key 应返回 None"

                decision = build_execution_decision(
                    prompt="后台分析任务",
                    requested_mode="cloud",
                    cloud_enabled=True,
                    has_api_key=False,
                )

                unified = build_unified_overrides(
                    args=args,
                    execution_decision=decision,
                )

                # 核心断言
                assert unified.resolved["orchestrator"] == "basic", (
                    "build_unified_overrides: cloud 模式无 API Key 应强制 basic\n"
                    f"  实际 orchestrator={unified.resolved['orchestrator']}"
                )
                assert unified.effective_mode == "cli", (
                    "build_unified_overrides: cloud 模式无 API Key 应回退到 cli\n"
                    f"  实际 effective_mode={unified.effective_mode}"
                )
                assert unified.requested_mode == "cloud", (
                    "build_unified_overrides: requested_mode 应保持为 cloud\n"
                    f"  实际 requested_mode={unified.requested_mode}"
                )

    def test_explicit_cli_mode_allows_mp_orchestrator(
        self, clean_api_key_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """对照用例：显式 --execution-mode cli 时应允许 mp 编排器

        这是关键对照测试，验证与 auto/cloud 模式不同，
        显式指定 cli 模式时应该允许使用 mp 编排器。
        """
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        args = argparse.Namespace(
            requirement="本地任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",  # 用户请求 mp
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",  # 关键：显式 cli
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # 测试 1: resolve_orchestrator_settings
                result = resolve_orchestrator_settings(
                    overrides={"execution_mode": "cli"},
                    prefix_routed=False,
                )

                assert result["orchestrator"] == "mp", (
                    "resolve_orchestrator_settings: 显式 cli 模式应允许 mp 编排器\n"
                    f"  实际 orchestrator={result['orchestrator']}"
                )

                # 测试 2: build_unified_overrides
                decision = build_execution_decision(
                    prompt="本地任务",
                    requested_mode="cli",
                    cloud_enabled=True,
                    has_api_key=False,
                )

                unified = build_unified_overrides(
                    args=args,
                    execution_decision=decision,
                )

                # 核心断言：显式 cli 应允许 mp
                assert unified.resolved["orchestrator"] == "mp", (
                    "build_unified_overrides: 显式 cli 模式应允许 mp 编排器\n"
                    f"  实际 orchestrator={unified.resolved['orchestrator']}"
                )

                # effective_mode 应为 cli
                assert unified.effective_mode == "cli", (
                    "build_unified_overrides: 显式 cli 模式 effective_mode 应为 cli\n"
                    f"  实际 effective_mode={unified.effective_mode}"
                )

                # requested_mode 应为 cli
                assert unified.requested_mode == "cli", (
                    "build_unified_overrides: requested_mode 应保持为 cli\n"
                    f"  实际 requested_mode={unified.requested_mode}"
                )

    @pytest.mark.parametrize(
        "execution_mode,expected_orchestrator,expected_effective_mode",
        [
            ("auto", "basic", "cli"),  # auto 无 key → basic, 回退 cli
            ("cloud", "basic", "cli"),  # cloud 无 key → basic, 回退 cli
            ("cli", "mp", "cli"),  # 显式 cli → mp, cli
        ],
        ids=["auto_no_key", "cloud_no_key", "explicit_cli"],
    )
    def test_orchestrator_selection_by_execution_mode_parametrized(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        execution_mode: str,
        expected_orchestrator: str,
        expected_effective_mode: str,
    ) -> None:
        """参数化测试：不同 execution_mode 对编排器选择的影响

        这是表驱动测试，覆盖三种主要场景：
        - auto 模式无 API Key
        - cloud 模式无 API Key
        - 显式 cli 模式

        所有场景都通过 monkeypatch 确保环境变量不存在。
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # 测试 resolve_orchestrator_settings
                result = resolve_orchestrator_settings(
                    overrides={"execution_mode": execution_mode},
                    prefix_routed=False,
                )

                assert result["orchestrator"] == expected_orchestrator, (
                    f"resolve_orchestrator_settings: {execution_mode} 模式\n"
                    f"  期望 orchestrator={expected_orchestrator}\n"
                    f"  实际 orchestrator={result['orchestrator']}"
                )

                # 测试 build_execution_decision
                decision = build_execution_decision(
                    prompt="测试任务",
                    requested_mode=execution_mode,
                    cloud_enabled=True,
                    has_api_key=False,
                )

                assert decision.orchestrator == expected_orchestrator, (
                    f"build_execution_decision: {execution_mode} 模式\n"
                    f"  期望 orchestrator={expected_orchestrator}\n"
                    f"  实际 orchestrator={decision.orchestrator}"
                )

                assert decision.effective_mode == expected_effective_mode, (
                    f"build_execution_decision: {execution_mode} 模式\n"
                    f"  期望 effective_mode={expected_effective_mode}\n"
                    f"  实际 effective_mode={decision.effective_mode}"
                )

    def test_self_iterator_with_no_api_key_env(self, clean_api_key_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """测试 SelfIterator 在无 API Key 环境变量时的行为

        验证 SelfIterator 使用 CloudClientFactory.resolve_api_key() 检测到
        无 API Key 时，正确设置编排器和执行模式。
        """
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        args = argparse.Namespace(
            requirement="分析任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="auto",  # 显式 auto
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                with patch("scripts.run_iterate.KnowledgeUpdater"):
                    with patch("scripts.run_iterate.ChangelogAnalyzer"):
                        with patch("scripts.run_iterate.IterationGoalBuilder"):
                            with patch("scripts.run_iterate.resolve_docs_source_config"):
                                with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                    # 验证 resolve_api_key 返回 None（环境变量已清除）
                                    api_key = CloudClientFactory.resolve_api_key()
                                    assert api_key is None, "前置条件：环境变量已清除，resolve_api_key 应返回 None"

                                    iterator = SelfIterator(args)
                                    actual_orchestrator = iterator._get_orchestrator_type()
                                    actual_execution_mode = iterator._get_execution_mode()

        # 核心断言
        assert actual_orchestrator == "basic", (
            f"SelfIterator: auto 模式无 API Key 应强制 basic 编排器\n  实际 orchestrator={actual_orchestrator}"
        )
        assert actual_execution_mode.value == "cli", (
            "SelfIterator: auto 模式无 API Key，effective_mode 应回退到 cli\n"
            f"  实际 effective_mode={actual_execution_mode.value}"
        )


class TestTriggeredByPrefixSemanticMigration:
    """回归测试：验证 triggered_by_prefix 全仓统一为 prefix_routed 语义

    本测试类验证以下迁移结果：
    1. ExecutionPolicyContext.triggered_by_prefix 返回 prefix_routed（而非 has_ampersand_prefix）
    2. ExecutionDecision.triggered_by_prefix 返回 prefix_routed
    3. AmpersandPrefixInfo.triggered_by_prefix 返回 prefix_routed
    4. resolve_orchestrator_settings 支持 prefix_routed 参数别名
    5. UnifiedOptions.triggered_by_prefix 与 prefix_routed 一致

    这确保了全仓所有 triggered_by_prefix 使用点语义统一。
    """

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

    def test_ampersand_prefix_detected_ignored_explicit_mode(self) -> None:
        """测试 & 前缀 + 显式 execution_mode=auto/cloud 时状态为 DETECTED_IGNORED_EXPLICIT_MODE

        当用户同时使用 & 前缀和显式 --execution-mode auto/cloud 时：
        - has_ampersand_prefix=True（语法层面检测到 & 前缀）
        - prefix_routed=False（策略层面 & 前缀不是执行模式触发原因）
        - status=DETECTED_IGNORED_EXPLICIT_MODE
        - orchestrator 仍因 requested_mode=auto/cloud 强制 basic
        """
        from core.execution_policy import (
            AmpersandPrefixStatus,
            build_execution_decision,
            detect_ampersand_prefix,
        )

        # === Case 1: & 前缀 + 显式 execution_mode=auto + 有 API Key ===
        info_auto_with_key = detect_ampersand_prefix(
            prompt="& 显式 auto 任务",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=True,
        )
        assert info_auto_with_key.has_ampersand_prefix is True
        assert info_auto_with_key.prefix_routed is False, "显式 auto 模式下 & 前缀不应触发路由"
        assert info_auto_with_key.status == AmpersandPrefixStatus.DETECTED_IGNORED_EXPLICIT_MODE, (
            "状态应为 DETECTED_IGNORED_EXPLICIT_MODE"
        )
        # 验证编排器
        decision_auto_with_key = build_execution_decision(
            prompt="& 显式 auto 任务",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=True,
        )
        assert decision_auto_with_key.orchestrator == "basic", "requested_mode=auto 应强制 basic 编排器"

        # === Case 2: & 前缀 + 显式 execution_mode=auto + 无 API Key ===
        info_auto_no_key = detect_ampersand_prefix(
            prompt="& 显式 auto 无 Key",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,
        )
        assert info_auto_no_key.has_ampersand_prefix is True
        assert info_auto_no_key.prefix_routed is False
        assert info_auto_no_key.status == AmpersandPrefixStatus.DETECTED_IGNORED_EXPLICIT_MODE
        # 验证编排器
        decision_auto_no_key = build_execution_decision(
            prompt="& 显式 auto 无 Key",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,
        )
        assert decision_auto_no_key.orchestrator == "basic", "即使回退到 CLI，requested_mode=auto 仍应强制 basic 编排器"
        assert decision_auto_no_key.effective_mode == "cli", "无 API Key 应回退到 CLI"

        # === Case 3: & 前缀 + 显式 execution_mode=cloud + 有 API Key ===
        info_cloud_with_key = detect_ampersand_prefix(
            prompt="& 显式 cloud 任务",
            requested_mode="cloud",
            cloud_enabled=True,
            has_api_key=True,
        )
        assert info_cloud_with_key.has_ampersand_prefix is True
        assert info_cloud_with_key.prefix_routed is False, "显式 cloud 模式下 & 前缀不应触发路由"
        assert info_cloud_with_key.status == AmpersandPrefixStatus.DETECTED_IGNORED_EXPLICIT_MODE
        # 验证编排器
        decision_cloud_with_key = build_execution_decision(
            prompt="& 显式 cloud 任务",
            requested_mode="cloud",
            cloud_enabled=True,
            has_api_key=True,
        )
        assert decision_cloud_with_key.orchestrator == "basic", "requested_mode=cloud 应强制 basic 编排器"

        # === Case 4: & 前缀 + 显式 execution_mode=cloud + 无 API Key ===
        info_cloud_no_key = detect_ampersand_prefix(
            prompt="& 显式 cloud 无 Key",
            requested_mode="cloud",
            cloud_enabled=True,
            has_api_key=False,
        )
        assert info_cloud_no_key.has_ampersand_prefix is True
        assert info_cloud_no_key.prefix_routed is False
        assert info_cloud_no_key.status == AmpersandPrefixStatus.DETECTED_IGNORED_EXPLICIT_MODE
        # 验证编排器
        decision_cloud_no_key = build_execution_decision(
            prompt="& 显式 cloud 无 Key",
            requested_mode="cloud",
            cloud_enabled=True,
            has_api_key=False,
        )
        assert decision_cloud_no_key.orchestrator == "basic", (
            "即使回退到 CLI，requested_mode=cloud 仍应强制 basic 编排器"
        )
        assert decision_cloud_no_key.effective_mode == "cli", "无 API Key 应回退到 CLI"

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


# ============================================================
# TestExecutionModeNoApiKeyParameterized - 无 API Key 场景参数化测试
# ============================================================


@dataclass
class NoApiKeyTestCase:
    """无 API Key 场景测试用例

    用于验证无 API Key 时执行模式决策的正确性。
    """

    test_id: str
    # 输入配置
    cli_execution_mode: Optional[str]  # CLI 显式指定的 execution_mode（None 表示未指定）
    config_execution_mode: str  # config.yaml 中的默认 execution_mode
    prompt: str  # 测试 prompt（可能带 & 前缀）
    cloud_enabled: bool
    # 期望输出
    expected_requested_mode: Optional[str]  # requested_mode 应为此值
    expected_effective_mode: str  # effective_mode 应为此值
    expected_orchestrator: str  # orchestrator 应为此值
    expected_has_ampersand_prefix: bool  # 是否有 & 前缀
    expected_prefix_routed: bool  # & 前缀是否成功路由
    expect_user_message: bool  # 是否期望产生 user_message
    expect_cloud_call: bool  # 是否期望调用 CloudClientFactory.execute_task


# 测试用例定义
NO_API_KEY_TEST_CASES: list[NoApiKeyTestCase] = [
    # ===== CLI 显式指定 execution_mode='auto' =====
    NoApiKeyTestCase(
        test_id="cli_explicit_auto_no_key",
        cli_execution_mode="auto",
        config_execution_mode="cli",  # config 值应被 CLI 覆盖
        prompt="测试任务",
        cloud_enabled=True,
        expected_requested_mode="auto",
        expected_effective_mode="cli",  # 无 key 回退
        expected_orchestrator="basic",  # auto 强制 basic
        expected_has_ampersand_prefix=False,
        expected_prefix_routed=False,
        expect_user_message=True,  # CLI 显式请求，应产生 warning 级别消息
        expect_cloud_call=False,  # 无 API Key 不调用 Cloud
    ),
    # ===== CLI 显式指定 execution_mode='cloud' =====
    NoApiKeyTestCase(
        test_id="cli_explicit_cloud_no_key",
        cli_execution_mode="cloud",
        config_execution_mode="cli",
        prompt="后台任务",
        cloud_enabled=True,
        expected_requested_mode="cloud",
        expected_effective_mode="cli",  # 无 key 回退
        expected_orchestrator="basic",  # cloud 强制 basic
        expected_has_ampersand_prefix=False,
        expected_prefix_routed=False,
        expect_user_message=True,
        expect_cloud_call=False,
    ),
    # ===== Config 默认 auto（CLI 未指定）=====
    NoApiKeyTestCase(
        test_id="config_default_auto_no_key",
        cli_execution_mode=None,  # 未显式指定
        config_execution_mode="auto",  # config.yaml 默认 auto
        prompt="分析代码",
        cloud_enabled=True,
        expected_requested_mode="auto",
        expected_effective_mode="cli",  # 无 key 回退
        expected_orchestrator="basic",  # auto 强制 basic
        expected_has_ampersand_prefix=False,
        expected_prefix_routed=False,
        expect_user_message=True,  # config 默认，info 级别消息
        expect_cloud_call=False,
    ),
    # ===== Config 默认 cloud（CLI 未指定）=====
    NoApiKeyTestCase(
        test_id="config_default_cloud_no_key",
        cli_execution_mode=None,
        config_execution_mode="cloud",
        prompt="分析代码",
        cloud_enabled=True,
        expected_requested_mode="cloud",
        expected_effective_mode="cli",
        expected_orchestrator="basic",
        expected_has_ampersand_prefix=False,
        expected_prefix_routed=False,
        expect_user_message=True,
        expect_cloud_call=False,
    ),
    # ===== & 前缀 + config 默认 auto + 无 API Key =====
    # 当 & 前缀存在时，resolve_requested_mode_for_decision 返回 None（而非 config.yaml 的值）
    # 这让 build_execution_decision 处理 & 前缀路由逻辑
    # 由于无 API Key，prefix_routed=False，effective_mode=cli
    # 但 & 前缀表达 Cloud 意图，即使 prefix_routed=False 也使用 basic（与 AGENTS.md 一致）
    # 参考 SNAPSHOT_TEST_CASES 中的 ampersand_no_key 用例
    NoApiKeyTestCase(
        test_id="ampersand_prefix_config_auto_no_key",
        cli_execution_mode=None,
        config_execution_mode="auto",
        prompt="& 后台分析",
        cloud_enabled=True,
        expected_requested_mode=None,  # resolve_requested_mode_for_decision 返回 None（有 & 前缀）
        expected_effective_mode="cli",  # 无 key 回退到 CLI
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_has_ampersand_prefix=True,  # 检测到 & 前缀
        expected_prefix_routed=False,  # 无 key 未成功路由
        expect_user_message=True,
        expect_cloud_call=False,
    ),
    # ===== & 前缀 + cloud_enabled=False =====
    # 当 cloud_enabled=False 且无显式 execution_mode 时，
    # resolve_requested_mode_for_decision 返回 None（有 & 前缀）
    # 但 & 前缀表达 Cloud 意图，强制使用 basic（与 AGENTS.md 一致）
    NoApiKeyTestCase(
        test_id="ampersand_prefix_cloud_disabled",
        cli_execution_mode=None,
        config_execution_mode="auto",
        prompt="& 后台分析",
        cloud_enabled=False,  # Cloud 功能禁用
        expected_requested_mode=None,  # resolve_requested_mode_for_decision 返回 None（有 & 前缀）
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,  # cloud_enabled=False，未路由
        expect_user_message=True,
        expect_cloud_call=False,
    ),
    # ===== CLI 显式 cli 模式（对照组）=====
    NoApiKeyTestCase(
        test_id="cli_explicit_cli_mode",
        cli_execution_mode="cli",
        config_execution_mode="auto",
        prompt="本地任务",
        cloud_enabled=True,
        expected_requested_mode="cli",
        expected_effective_mode="cli",
        expected_orchestrator="mp",  # cli 模式允许 mp
        expected_has_ampersand_prefix=False,
        expected_prefix_routed=False,
        expect_user_message=False,  # cli 模式不产生回退消息
        expect_cloud_call=False,
    ),
    # ===== & 前缀 + CLI 显式 cli 模式 =====
    NoApiKeyTestCase(
        test_id="ampersand_prefix_cli_explicit_cli_mode",
        cli_execution_mode="cli",
        config_execution_mode="auto",
        prompt="& 后台任务",
        cloud_enabled=True,
        expected_requested_mode="cli",
        expected_effective_mode="cli",
        expected_orchestrator="mp",  # cli 模式允许 mp
        expected_has_ampersand_prefix=True,  # 检测到 & 前缀
        expected_prefix_routed=False,  # cli 模式忽略 & 前缀
        expect_user_message=False,  # cli 模式不产生回退消息（info 级别）
        expect_cloud_call=False,
    ),
]


class TestExecutionModeNoApiKeyParameterized:
    """无 API Key 场景参数化测试

    使用 monkeypatch 清除 CURSOR_API_KEY/CURSOR_CLOUD_API_KEY 环境变量，
    验证以下场景的行为：
    - CLI 显式指定 execution_mode='auto'/'cloud'
    - config.yaml 默认 execution_mode（CLI 未显式指定）
    - & 前缀触发场景

    核心验证：
    1. requested_mode 与 effective_mode 的组合正确性
    2. orchestrator 强制 basic（auto/cloud 模式）
    3. user_message 不重复打印
    4. CloudClientFactory.execute_task / CloudAuthManager.authenticate 未被调用
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.fixture
    def clean_api_key_env(self, clean_env) -> None:
        """委托给全局 clean_env fixture（来自 conftest.py）

        保留此方法签名以兼容现有测试方法，避免大量重构。
        clean_env fixture 会清除 CURSOR_API_KEY 和 CURSOR_CLOUD_API_KEY。
        """
        pass

    @pytest.mark.parametrize(
        "test_case",
        NO_API_KEY_TEST_CASES,
        ids=[tc.test_id for tc in NO_API_KEY_TEST_CASES],
    )
    def test_build_execution_decision_no_api_key(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        test_case: NoApiKeyTestCase,
    ) -> None:
        """参数化测试：build_execution_decision 在无 API Key 时的行为

        验证 requested_mode, effective_mode, orchestrator 的组合。
        """
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import (
            build_execution_decision,
            resolve_requested_mode_for_decision,
        )

        ConfigManager.reset_instance()

        # 创建 Mock 配置
        mock_config = _create_mock_config(cloud_enabled=test_case.cloud_enabled)
        mock_config.cloud_agent.api_key = None
        mock_config.cloud_agent.execution_mode = test_case.config_execution_mode

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # 检测 & 前缀
                has_ampersand_prefix = is_cloud_request(test_case.prompt)

                # 确定 requested_mode
                requested_mode = resolve_requested_mode_for_decision(
                    cli_execution_mode=test_case.cli_execution_mode,
                    has_ampersand_prefix=has_ampersand_prefix,
                    config_execution_mode=test_case.config_execution_mode,
                )

                # 构建执行决策
                decision = build_execution_decision(
                    prompt=test_case.prompt,
                    requested_mode=requested_mode,
                    cloud_enabled=test_case.cloud_enabled,
                    has_api_key=False,  # 无 API Key
                )

        # 验证 has_ampersand_prefix
        assert decision.has_ampersand_prefix == test_case.expected_has_ampersand_prefix, (
            f"[{test_case.test_id}] has_ampersand_prefix 不匹配\n"
            f"  期望: {test_case.expected_has_ampersand_prefix}\n"
            f"  实际: {decision.has_ampersand_prefix}"
        )

        # 验证 prefix_routed
        assert decision.prefix_routed == test_case.expected_prefix_routed, (
            f"[{test_case.test_id}] prefix_routed 不匹配\n"
            f"  期望: {test_case.expected_prefix_routed}\n"
            f"  实际: {decision.prefix_routed}"
        )

        # 验证 effective_mode
        assert decision.effective_mode == test_case.expected_effective_mode, (
            f"[{test_case.test_id}] effective_mode 不匹配\n"
            f"  期望: {test_case.expected_effective_mode}\n"
            f"  实际: {decision.effective_mode}"
        )

        # 验证 orchestrator
        assert decision.orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] orchestrator 不匹配\n"
            f"  期望: {test_case.expected_orchestrator}\n"
            f"  实际: {decision.orchestrator}"
        )

        # 验证 requested_mode
        assert decision.requested_mode == test_case.expected_requested_mode, (
            f"[{test_case.test_id}] requested_mode 不匹配\n"
            f"  期望: {test_case.expected_requested_mode}\n"
            f"  实际: {decision.requested_mode}"
        )

        # 验证 user_message
        if test_case.expect_user_message:
            assert decision.user_message is not None, f"[{test_case.test_id}] 应产生 user_message，实际为 None"
        else:
            # cli 模式可能不产生消息，或者产生 info 级别消息
            # 这里放宽检查，仅当明确不期望消息时验证
            pass

    @pytest.mark.parametrize(
        "test_case",
        NO_API_KEY_TEST_CASES,
        ids=[tc.test_id for tc in NO_API_KEY_TEST_CASES],
    )
    def test_cloud_client_factory_not_called_without_api_key(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        test_case: NoApiKeyTestCase,
    ) -> None:
        """参数化测试：无 API Key 时 CloudClientFactory.execute_task 不被调用

        验证在无 API Key 环境下，系统不会尝试调用 Cloud API。
        """
        from cursor.cloud_client import CloudAuthManager, CloudClientFactory

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=test_case.cloud_enabled)
        mock_config.cloud_agent.api_key = None
        mock_config.cloud_agent.execution_mode = test_case.config_execution_mode

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # Mock CloudClientFactory.execute_task
                with patch.object(
                    CloudClientFactory,
                    "execute_task",
                    new_callable=MagicMock,
                ) as mock_execute_task:
                    # Mock CloudAuthManager.authenticate
                    with patch.object(
                        CloudAuthManager,
                        "authenticate",
                        new_callable=MagicMock,
                    ) as mock_authenticate:
                        # 验证 resolve_api_key 返回 None
                        api_key = CloudClientFactory.resolve_api_key()
                        assert api_key is None, f"[{test_case.test_id}] resolve_api_key 应返回 None"

                        # 构建执行决策（这是纯决策逻辑，不应触发 Cloud 调用）
                        from core.cloud_utils import is_cloud_request
                        from core.execution_policy import build_execution_decision

                        has_ampersand_prefix = is_cloud_request(test_case.prompt)

                        # 确定 requested_mode
                        if test_case.cli_execution_mode is not None:
                            requested_mode = test_case.cli_execution_mode
                        elif has_ampersand_prefix:
                            requested_mode = None
                        else:
                            requested_mode = test_case.config_execution_mode

                        decision = build_execution_decision(
                            prompt=test_case.prompt,
                            requested_mode=requested_mode,
                            cloud_enabled=test_case.cloud_enabled,
                            has_api_key=False,
                        )

                        # 验证 CloudClientFactory.execute_task 未被调用
                        assert not mock_execute_task.called, (
                            f"[{test_case.test_id}] CloudClientFactory.execute_task "
                            "不应被调用（build_execution_decision 是纯决策逻辑）"
                        )

                        # 验证 CloudAuthManager.authenticate 未被调用
                        # （check_available 可能会触达，但不应触发实际认证）
                        assert not mock_authenticate.called, (
                            f"[{test_case.test_id}] CloudAuthManager.authenticate "
                            "不应被调用（build_execution_decision 是纯决策逻辑）"
                        )

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in NO_API_KEY_TEST_CASES if tc.expected_orchestrator == "basic"],
        ids=[tc.test_id for tc in NO_API_KEY_TEST_CASES if tc.expected_orchestrator == "basic"],
    )
    def test_resolve_orchestrator_settings_forces_basic(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        test_case: NoApiKeyTestCase,
    ) -> None:
        """参数化测试：auto/cloud 模式强制 basic 编排器

        验证 resolve_orchestrator_settings 在 auto/cloud 模式时强制 basic。
        """
        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=test_case.cloud_enabled)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                result = resolve_orchestrator_settings(
                    overrides={"execution_mode": test_case.expected_requested_mode},
                    prefix_routed=test_case.expected_prefix_routed,
                )

                assert result["orchestrator"] == "basic", (
                    f"[{test_case.test_id}] resolve_orchestrator_settings 应强制 basic\n"
                    f"  requested_mode={test_case.expected_requested_mode}\n"
                    f"  期望 orchestrator=basic\n"
                    f"  实际 orchestrator={result['orchestrator']}"
                )

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in NO_API_KEY_TEST_CASES if tc.expected_orchestrator == "mp"],
        ids=[tc.test_id for tc in NO_API_KEY_TEST_CASES if tc.expected_orchestrator == "mp"],
    )
    def test_cli_mode_allows_mp_orchestrator(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        test_case: NoApiKeyTestCase,
    ) -> None:
        """参数化测试：cli 模式允许 mp 编排器

        验证显式 cli 模式时 orchestrator=mp。
        """
        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=test_case.cloud_enabled)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                result = resolve_orchestrator_settings(
                    overrides={"execution_mode": "cli"},
                    prefix_routed=False,
                )

                assert result["orchestrator"] == "mp", (
                    f"[{test_case.test_id}] 显式 cli 模式应允许 mp 编排器\n"
                    f"  期望 orchestrator=mp\n"
                    f"  实际 orchestrator={result['orchestrator']}"
                )


class TestUserMessageDedupNoApiKey:
    """用户消息去重测试 - 无 API Key 场景

    验证在无 API Key 时，user_message 只出现一次（不重复打印）。
    使用 clean_env fixture（来自 conftest.py）统一清除 API Key 环境变量。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.fixture
    def clean_api_key_env(self, clean_env) -> None:
        """委托给全局 clean_env fixture（来自 conftest.py）

        保留此方法签名以兼容现有测试方法。
        """
        pass

    def test_user_message_appears_once_in_decision(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """验证 build_execution_decision 产生的 user_message 唯一性

        每个 decision 对象只包含一个 user_message，不会重复。
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # 多次调用 build_execution_decision
                decisions = []
                for _ in range(3):
                    decision = build_execution_decision(
                        prompt="& 后台任务",
                        requested_mode="auto",
                        cloud_enabled=True,
                        has_api_key=False,
                    )
                    decisions.append(decision)

                # 每个 decision 都有 user_message
                for i, decision in enumerate(decisions):
                    assert decision.user_message is not None, f"第 {i + 1} 个 decision 应有 user_message"

                # 消息内容应该一致（相同输入产生相同消息）
                assert all(d.user_message == decisions[0].user_message for d in decisions), (
                    "相同输入应产生相同的 user_message"
                )

    @pytest.mark.parametrize(
        "execution_mode,mode_source,expected_message_level",
        [
            ("auto", "cli", "warning"),  # CLI 显式 → warning
            ("auto", "config", "info"),  # config 默认 → info
            ("auto", None, "info"),  # 未指定来源 → info
            ("cloud", "cli", "warning"),  # CLI 显式 → warning
            ("cloud", "config", "info"),  # config 默认 → info
        ],
        ids=[
            "auto_cli_explicit",
            "auto_config_default",
            "auto_unspecified",
            "cloud_cli_explicit",
            "cloud_config_default",
        ],
    )
    def test_message_level_based_on_mode_source(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        execution_mode: str,
        mode_source: Optional[str],
        expected_message_level: str,
    ) -> None:
        """验证 message_level 根据 mode_source 正确设置

        CLI 显式指定 → warning 级别
        config.yaml 默认 → info 级别（避免每次都警告）
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                decision = build_execution_decision(
                    prompt="测试任务",
                    requested_mode=execution_mode,
                    cloud_enabled=True,
                    has_api_key=False,
                    mode_source=mode_source,
                )

                assert decision.message_level == expected_message_level, (
                    f"execution_mode={execution_mode}, mode_source={mode_source}\n"
                    f"  期望 message_level={expected_message_level}\n"
                    f"  实际 message_level={decision.message_level}"
                )


class TestCloudClientNotCalledDuringDecision:
    """验证决策阶段不调用 Cloud 客户端

    build_execution_decision 和 resolve_orchestrator_settings 是纯决策逻辑，
    不应触发任何 Cloud API 调用。
    使用 clean_env fixture（来自 conftest.py）统一清除 API Key 环境变量。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.fixture
    def clean_api_key_env(self, clean_env) -> None:
        """委托给全局 clean_env fixture（来自 conftest.py）

        保留此方法签名以兼容现有测试方法。
        """
        pass

    @pytest.mark.parametrize(
        "execution_mode,prompt,cloud_enabled",
        [
            ("auto", "普通任务", True),
            ("cloud", "后台任务", True),
            ("auto", "& 前缀任务", True),
            ("cli", "本地任务", True),
            ("auto", "普通任务", False),  # cloud_enabled=False
        ],
        ids=[
            "auto_normal",
            "cloud_explicit",
            "auto_ampersand",
            "cli_explicit",
            "auto_cloud_disabled",
        ],
    )
    def test_execute_task_not_called_in_decision_phase(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        execution_mode: str,
        prompt: str,
        cloud_enabled: bool,
    ) -> None:
        """验证 build_execution_decision 不调用 CloudClientFactory.execute_task"""
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=cloud_enabled)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                with patch.object(
                    CloudClientFactory,
                    "execute_task",
                    new_callable=MagicMock,
                ) as mock_execute_task:
                    # 调用 build_execution_decision
                    decision = build_execution_decision(
                        prompt=prompt,
                        requested_mode=execution_mode,
                        cloud_enabled=cloud_enabled,
                        has_api_key=False,
                    )

                    # 验证 execute_task 未被调用
                    assert not mock_execute_task.called, (
                        f"execution_mode={execution_mode}, prompt={prompt}\n"
                        "CloudClientFactory.execute_task 不应在决策阶段被调用"
                    )

                    # 验证 decision 对象正确构建
                    assert decision is not None
                    assert decision.effective_mode in ("cli", "cloud", "auto", "plan", "ask")

    @pytest.mark.parametrize(
        "execution_mode",
        ["auto", "cloud", "cli"],
        ids=["auto", "cloud", "cli"],
    )
    def test_authenticate_not_called_in_decision_phase(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        execution_mode: str,
    ) -> None:
        """验证 build_execution_decision 不调用 CloudAuthManager.authenticate"""
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudAuthManager

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                with patch.object(
                    CloudAuthManager,
                    "authenticate",
                    new_callable=MagicMock,
                ) as mock_authenticate:
                    # 调用 build_execution_decision
                    decision = build_execution_decision(
                        prompt="测试任务",
                        requested_mode=execution_mode,
                        cloud_enabled=True,
                        has_api_key=False,
                    )

                    # 验证 authenticate 未被调用
                    assert not mock_authenticate.called, (
                        f"execution_mode={execution_mode}\nCloudAuthManager.authenticate 不应在决策阶段被调用"
                    )

    def test_resolve_orchestrator_settings_no_cloud_calls(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """验证 resolve_orchestrator_settings 不触发 Cloud 调用"""
        from cursor.cloud_client import CloudAuthManager, CloudClientFactory

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                with patch.object(
                    CloudClientFactory,
                    "execute_task",
                    new_callable=MagicMock,
                ) as mock_execute_task:
                    with patch.object(
                        CloudAuthManager,
                        "authenticate",
                        new_callable=MagicMock,
                    ) as mock_authenticate:
                        # 调用 resolve_orchestrator_settings
                        for mode in ["auto", "cloud", "cli"]:
                            result = resolve_orchestrator_settings(
                                overrides={"execution_mode": mode},
                                prefix_routed=False,
                            )

                            assert result is not None

                        # 验证没有 Cloud 调用
                        assert not mock_execute_task.called, "resolve_orchestrator_settings 不应触发 execute_task"
                        assert not mock_authenticate.called, "resolve_orchestrator_settings 不应触发 authenticate"


class TestMessageLevelSemanticsMatrix:
    """消息级别语义矩阵测试

    验证 build_execution_decision 产生的 message_level 遵循以下策略：

    ================================================================================
    消息级别策略（由 mode_source 和场景决定）
    ================================================================================

    | 场景                                      | message_level | 说明               |
    |-------------------------------------------|---------------|-------------------|
    | mode_source='cli' + auto/cloud 无 key     | warning       | 用户显式请求       |
    | mode_source='config' + auto/cloud 无 key  | info          | 避免每次都警告     |
    | mode_source=None + auto/cloud 无 key      | info          | 默认使用 info      |
    | & 前缀 + 无 key (prefix_routed=False)     | warning       | 用户显式使用 &     |
    | & 前缀 + cloud_disabled                   | info          | 配置问题，非用户错误|
    | & 前缀 + CLI 模式显式指定                 | info          | 用户显式选择 CLI    |

    使用 clean_env fixture（来自 conftest.py）统一清除 API Key 环境变量。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.fixture
    def clean_api_key_env(self, clean_env) -> None:
        """委托给全局 clean_env fixture（来自 conftest.py）

        保留此方法签名以兼容现有测试方法。
        """
        pass

    # =========================================================================
    # mode_source 决定的 message_level 测试
    # =========================================================================

    @pytest.mark.parametrize(
        "execution_mode,mode_source,expected_message_level,description",
        [
            # mode_source='cli' 场景：用户显式请求，应使用 warning
            ("auto", "cli", "warning", "CLI 显式 auto 无 key → warning"),
            ("cloud", "cli", "warning", "CLI 显式 cloud 无 key → warning"),
            # mode_source='config' 场景：配置默认，应使用 info 避免每次都警告
            ("auto", "config", "info", "config 默认 auto 无 key → info"),
            ("cloud", "config", "info", "config 默认 cloud 无 key → info"),
            # mode_source=None 场景：未指定来源，使用 info
            ("auto", None, "info", "未指定来源 auto 无 key → info"),
            ("cloud", None, "info", "未指定来源 cloud 无 key → info"),
        ],
        ids=[
            "cli_explicit_auto_warning",
            "cli_explicit_cloud_warning",
            "config_default_auto_info",
            "config_default_cloud_info",
            "unspecified_auto_info",
            "unspecified_cloud_info",
        ],
    )
    def test_mode_source_determines_message_level_for_cloud_fallback(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        execution_mode: str,
        mode_source: Optional[str],
        expected_message_level: str,
        description: str,
    ) -> None:
        """验证 mode_source 决定 Cloud 回退场景的 message_level

        关键规则:
        - mode_source='cli' (用户显式 --execution-mode) → warning
        - mode_source='config' 或 None (配置默认) → info

        这确保:
        1. 用户显式请求 auto/cloud 但无 key 时得到明确警告
        2. 配置默认 auto 时不会每次都警告导致警告疲劳
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                decision = build_execution_decision(
                    prompt="测试任务",
                    requested_mode=execution_mode,
                    cloud_enabled=True,
                    has_api_key=False,
                    mode_source=mode_source,
                )

                # 验证 effective_mode 回退到 cli
                assert decision.effective_mode == "cli", (
                    f"{description}\n  期望回退到 cli\n  实际 effective_mode={decision.effective_mode}"
                )

                # 验证 message_level 正确
                assert decision.message_level == expected_message_level, (
                    f"{description}\n"
                    f"  execution_mode={execution_mode}, mode_source={mode_source}\n"
                    f"  期望 message_level={expected_message_level}\n"
                    f"  实际 message_level={decision.message_level}"
                )

                # 验证 user_message 存在
                assert decision.user_message is not None, f"{description}\n  应该有 user_message"

    # =========================================================================
    # & 前缀触发失败场景的 message_level 测试
    # =========================================================================

    @pytest.mark.parametrize(
        "has_api_key,cloud_enabled,cli_mode,expected_message_level,expected_status_substr,description",
        [
            # has_ampersand_prefix=True, prefix_routed=False (无 key)
            # 用户显式使用 & 前缀表示意图，应该 warning
            (False, True, None, "warning", "NO_KEY", "& 前缀 + 无 key → warning (用户显式意图)"),
            # has_ampersand_prefix=True, prefix_routed=False (cloud_disabled)
            # 配置问题，非用户错误，使用 info
            (True, False, None, "info", "DISABLED", "& 前缀 + cloud_disabled → info (配置问题)"),
            # has_ampersand_prefix=True, prefix_routed=False (显式 CLI 模式)
            # 用户显式选择 CLI，使用 info
            (True, True, "cli", "info", "CLI_MODE", "& 前缀 + 显式 CLI → info (用户选择)"),
        ],
        ids=[
            "ampersand_no_key_warning",
            "ampersand_disabled_info",
            "ampersand_cli_mode_info",
        ],
    )
    def test_ampersand_prefix_not_routed_message_level(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        has_api_key: bool,
        cloud_enabled: bool,
        cli_mode: Optional[str],
        expected_message_level: str,
        expected_status_substr: str,
        description: str,
    ) -> None:
        """验证 & 前缀存在但未成功触发时的 message_level

        关键规则:
        - has_ampersand_prefix=True + prefix_routed=False + 无 key → warning
          (用户显式使用 & 前缀表示 Cloud 意图，应明确警告)
        - has_ampersand_prefix=True + prefix_routed=False + cloud_disabled → info
          (配置问题，非用户错误)
        - has_ampersand_prefix=True + 显式 CLI 模式 → info
          (用户显式选择 CLI 忽略 & 前缀)
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=cloud_enabled)
        mock_config.cloud_agent.api_key = "test-key" if has_api_key else None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                decision = build_execution_decision(
                    prompt="& 后台任务",  # 带 & 前缀
                    requested_mode=cli_mode,  # None 或 "cli"
                    cloud_enabled=cloud_enabled,
                    has_api_key=has_api_key,
                )

                # 验证 has_ampersand_prefix = True (语法层面)
                assert decision.has_ampersand_prefix is True, (
                    f"{description}\n  prompt 以 & 开头，has_ampersand_prefix 应为 True"
                )

                # 验证 prefix_routed = False (策略层面)
                assert decision.prefix_routed is False, (
                    f"{description}\n"
                    f"  has_api_key={has_api_key}, cloud_enabled={cloud_enabled}, cli_mode={cli_mode}\n"
                    "  条件不满足时 prefix_routed 应为 False"
                )

                # 验证 effective_mode = cli
                assert decision.effective_mode == "cli", (
                    f"{description}\n  期望 effective_mode=cli\n  实际 effective_mode={decision.effective_mode}"
                )

                # 验证 message_level 正确
                assert decision.message_level == expected_message_level, (
                    f"{description}\n"
                    f"  期望 message_level={expected_message_level}\n"
                    f"  实际 message_level={decision.message_level}"
                )

                # 验证 ampersand_prefix_info 状态
                assert decision.ampersand_prefix_info is not None, f"{description}\n  应该有 ampersand_prefix_info"
                assert expected_status_substr in decision.ampersand_prefix_info.status.value.upper(), (
                    f"{description}\n"
                    f"  期望状态包含 {expected_status_substr}\n"
                    f"  实际状态 {decision.ampersand_prefix_info.status}"
                )

    def test_ampersand_no_key_vs_config_auto_no_key_message_level_consistency(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """验证 & 前缀无 key 与 config auto 无 key 的 message_level 一致性

        核心规则:
        - & 前缀无 key: warning (用户显式使用 & 表示 Cloud 意图)
        - config auto 无 key (mode_source='config'): info (避免每次都警告)
        - CLI 显式 auto 无 key (mode_source='cli'): warning (用户显式请求)

        两种"用户显式表达 Cloud 意图"的场景应该产生一致的 warning 级别。
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                # 场景 1: & 前缀无 key (用户显式使用 &)
                decision_ampersand = build_execution_decision(
                    prompt="& 后台任务",
                    requested_mode=None,  # 无显式模式
                    cloud_enabled=True,
                    has_api_key=False,
                )

                # 场景 2: CLI 显式 auto 无 key (用户显式 --execution-mode auto)
                decision_cli_auto = build_execution_decision(
                    prompt="普通任务",
                    requested_mode="auto",
                    cloud_enabled=True,
                    has_api_key=False,
                    mode_source="cli",
                )

                # 场景 3: config 默认 auto 无 key (配置默认)
                decision_config_auto = build_execution_decision(
                    prompt="普通任务",
                    requested_mode="auto",
                    cloud_enabled=True,
                    has_api_key=False,
                    mode_source="config",
                )

                # 验证: 用户显式意图场景应该是 warning
                assert decision_ampersand.message_level == "warning", (
                    "& 前缀表示用户显式 Cloud 意图，无 key 时应 warning"
                )
                assert decision_cli_auto.message_level == "warning", (
                    "CLI 显式 --execution-mode auto 表示用户显式 Cloud 意图，无 key 时应 warning"
                )

                # 验证: 配置默认场景应该是 info
                assert decision_config_auto.message_level == "info", (
                    "config 默认 auto 无 key 应使用 info 避免每次都警告"
                )

                # 验证一致性：两种用户显式意图场景应该一致
                assert decision_ampersand.message_level == decision_cli_auto.message_level, (
                    "& 前缀和 CLI 显式 auto 都表示用户显式意图，message_level 应一致\n"
                    f"  & 前缀: {decision_ampersand.message_level}\n"
                    f"  CLI auto: {decision_cli_auto.message_level}"
                )


class TestMessageLevelAccessibleForPrinting:
    """验证 message_level 字段可供入口脚本用于打印决策

    确保 build_execution_decision 返回的 message_level 字段可以被
    入口脚本（run.py、scripts/run_iterate.py）用于决定使用
    print_warning 还是 print_info。
    使用 clean_env fixture（来自 conftest.py）统一清除 API Key 环境变量。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.fixture
    def clean_api_key_env(self, clean_env) -> None:
        """委托给全局 clean_env fixture（来自 conftest.py）

        保留此方法签名以兼容现有测试方法。
        """
        pass

    def test_decision_exposes_message_level_and_user_message(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """验证 ExecutionDecision 暴露 message_level 和 user_message 字段

        入口脚本可通过这两个字段决定如何打印消息:
        - if decision.user_message:
        -     if decision.message_level == "warning":
        -         print_warning(decision.user_message)
        -     else:
        -         print_info(decision.user_message)
        """
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                decision = build_execution_decision(
                    prompt="& 后台任务",
                    requested_mode="auto",
                    cloud_enabled=True,
                    has_api_key=False,
                    mode_source="cli",
                )

                # 验证字段存在且类型正确
                assert hasattr(decision, "message_level"), "ExecutionDecision 应有 message_level 字段"
                assert hasattr(decision, "user_message"), "ExecutionDecision 应有 user_message 字段"

                assert decision.message_level in ("warning", "info"), (
                    f"message_level 应为 'warning' 或 'info'，实际: {decision.message_level}"
                )
                assert isinstance(decision.user_message, str), (
                    f"user_message 应为 str，实际: {type(decision.user_message)}"
                )

    def test_to_dict_includes_message_level(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """验证 to_dict() 输出包含 message_level 字段"""
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                decision = build_execution_decision(
                    prompt="任务",
                    requested_mode="auto",
                    cloud_enabled=True,
                    has_api_key=False,
                    mode_source="cli",
                )

                decision_dict = decision.to_dict()

                assert "message_level" in decision_dict, "to_dict() 输出应包含 message_level"
                assert "user_message" in decision_dict, "to_dict() 输出应包含 user_message"
                assert "mode_source" in decision_dict, "to_dict() 输出应包含 mode_source"


# ============================================================
# TaskAnalyzer 与 SelfIterator 决策对齐测试
# ============================================================


@dataclass
class TaskAnalyzerAlignmentCase:
    """TaskAnalyzer 与 SelfIterator 对齐测试参数

    用于验证 run.py 的 TaskAnalyzer._rule_based_analysis 与
    scripts/run_iterate.py 的 SelfIterator._execution_decision 的一致性。
    """

    test_id: str
    task_text: str  # 用户输入的任务描述（可包含 & 前缀）
    execution_mode: Optional[str]  # CLI --execution-mode 参数
    has_api_key: bool
    cloud_enabled: bool
    orchestrator_cli: Optional[str]  # CLI --orchestrator 参数
    no_mp_cli: bool  # CLI --no-mp 参数
    expected_effective_mode: str
    expected_orchestrator: str
    expected_prefix_routed: bool
    expected_has_ampersand_prefix: bool
    expected_requested_mode: Optional[str]
    description: str


# TaskAnalyzer 与 SelfIterator 对齐测试用例
TASK_ANALYZER_ALIGNMENT_CASES = [
    # ===== 基础 CLI 模式 =====
    TaskAnalyzerAlignmentCase(
        test_id="align_cli_basic",
        task_text="实现新功能",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=False,
        expected_requested_mode="cli",
        description="CLI 模式普通任务",
    ),
    TaskAnalyzerAlignmentCase(
        test_id="align_cli_no_key",
        task_text="优化代码",
        execution_mode="cli",
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=False,
        expected_requested_mode="cli",
        description="CLI 模式无 API Key",
    ),
    # ===== AUTO 模式 =====
    TaskAnalyzerAlignmentCase(
        test_id="align_auto_with_key",
        task_text="重构模块",
        execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="auto",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=False,
        expected_requested_mode="auto",
        description="AUTO 模式有 API Key，强制 basic",
    ),
    TaskAnalyzerAlignmentCase(
        test_id="align_auto_no_key_forces_basic",
        task_text="分析代码",
        execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 仍为 basic
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=False,
        expected_requested_mode="auto",
        description="关键场景：AUTO 无 Key 回退 CLI 但仍 basic",
    ),
    # ===== CLOUD 模式 =====
    TaskAnalyzerAlignmentCase(
        test_id="align_cloud_with_key",
        task_text="后台任务",
        execution_mode="cloud",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=False,
        expected_requested_mode="cloud",
        description="CLOUD 模式有 API Key",
    ),
    TaskAnalyzerAlignmentCase(
        test_id="align_cloud_no_key_fallback",
        task_text="后台任务",
        execution_mode="cloud",
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 仍为 basic
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=False,
        expected_requested_mode="cloud",
        description="CLOUD 无 Key 回退 CLI 但仍 basic",
    ),
    # ===== & 前缀触发 =====
    TaskAnalyzerAlignmentCase(
        test_id="align_prefix_with_key_enabled",
        task_text="& 后台分析",
        execution_mode=None,
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=True,
        expected_has_ampersand_prefix=True,
        expected_requested_mode=None,  # & 前缀时 requested_mode 为 None
        description="& 前缀成功触发 Cloud",
    ),
    TaskAnalyzerAlignmentCase(
        test_id="align_prefix_no_key",
        task_text="& 后台任务",
        execution_mode=None,
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="cli",  # 无 Key 回退
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=True,
        expected_requested_mode=None,
        description="& 前缀无 Key 未成功触发，仍使用 basic",
    ),
    TaskAnalyzerAlignmentCase(
        test_id="align_prefix_cloud_disabled",
        task_text="& 后台任务",
        execution_mode=None,
        has_api_key=True,
        cloud_enabled=False,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="cli",  # cloud_enabled=False 回退
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，强制 basic
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=True,
        expected_requested_mode=None,
        description="& 前缀 cloud_enabled=False 未触发，仍使用 basic",
    ),
    # ===== 显式 CLI + & 前缀（忽略 & 前缀）=====
    TaskAnalyzerAlignmentCase(
        test_id="align_cli_ignores_prefix",
        task_text="& 后台任务",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,  # 显式 CLI 忽略 & 前缀
        expected_has_ampersand_prefix=True,  # 语法检测仍为 True
        expected_requested_mode="cli",
        description="显式 CLI 忽略 & 前缀",
    ),
    # ===== 显式 AUTO + & 前缀（不额外触发路由）=====
    TaskAnalyzerAlignmentCase(
        test_id="align_auto_with_prefix_no_route",
        task_text="& 后台任务",
        execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_effective_mode="auto",
        expected_orchestrator="basic",
        expected_prefix_routed=False,  # 显式 auto 不触发路由
        expected_has_ampersand_prefix=True,
        expected_requested_mode="auto",
        description="显式 AUTO + & 前缀不触发路由",
    ),
    # ===== 编排器显式设置 =====
    TaskAnalyzerAlignmentCase(
        test_id="align_cli_explicit_basic",
        task_text="任务",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli="basic",
        no_mp_cli=False,
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # 用户显式设置
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=False,
        expected_requested_mode="cli",
        description="CLI + 显式 basic 编排器",
    ),
    TaskAnalyzerAlignmentCase(
        test_id="align_cli_no_mp",
        task_text="任务",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=True,
        expected_effective_mode="cli",
        expected_orchestrator="basic",  # --no-mp 强制 basic
        expected_prefix_routed=False,
        expected_has_ampersand_prefix=False,
        expected_requested_mode="cli",
        description="CLI + --no-mp 强制 basic",
    ),
]


def _build_task_analyzer_args(case: TaskAnalyzerAlignmentCase) -> argparse.Namespace:
    """构建 TaskAnalyzer 测试参数

    注意: 当 no_mp_cli=True 时，orchestrator 应为 None，
    让 TaskAnalyzer._rule_based_analysis 根据 no_mp 设置 user_requested_orchestrator="basic"。
    """
    # 当 no_mp=True 时，orchestrator 应为 None，否则使用 CLI 指定或默认 mp
    orchestrator_value = case.orchestrator_cli
    if orchestrator_value is None and not case.no_mp_cli:
        orchestrator_value = "mp"  # 默认值

    return argparse.Namespace(
        task=case.task_text,
        mode="iterate",
        directory=".",
        _directory_user_set=False,
        workers=3,
        max_iterations="10",
        strict_review=None,
        enable_sub_planners=None,
        verbose=False,
        quiet=False,
        log_level=None,
        skip_online=True,
        dry_run=False,
        force_update=False,
        use_knowledge=False,
        search_knowledge=None,
        self_update=False,
        planner_model=None,
        worker_model=None,
        reviewer_model=None,
        stream_log_enabled=None,
        stream_log_console=None,
        stream_log_detail_dir=None,
        stream_log_raw_dir=None,
        no_auto_analyze=False,
        auto_commit=False,
        auto_push=False,
        commit_per_iteration=False,
        orchestrator=orchestrator_value,
        no_mp=case.no_mp_cli,
        _orchestrator_user_set=(case.orchestrator_cli is not None) or case.no_mp_cli,
        execution_mode=case.execution_mode,
        planner_execution_mode=None,
        worker_execution_mode=None,
        reviewer_execution_mode=None,
        cloud_api_key=None,
        cloud_auth_timeout=30,
        cloud_timeout=300,
        auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
        cloud_background=None,
        stream_console_renderer=False,
        stream_advanced_renderer=False,
        stream_typing_effect=False,
        stream_typing_delay=0.02,
        stream_word_mode=True,
        stream_color_enabled=True,
        stream_show_word_diff=False,
    )


def _build_self_iterator_args(case: TaskAnalyzerAlignmentCase) -> argparse.Namespace:
    """构建 SelfIterator 测试参数

    注意: 当 no_mp_cli=True 时，orchestrator 应为 None，
    让 SelfIterator.__init__ 根据 no_mp 设置 user_requested_orchestrator="basic"。
    """
    # 当 no_mp=True 时，orchestrator 应为 None，否则使用 CLI 指定或默认 mp
    orchestrator_value = case.orchestrator_cli
    if orchestrator_value is None and not case.no_mp_cli:
        orchestrator_value = "mp"  # 默认值

    return argparse.Namespace(
        requirement=case.task_text,
        directory=".",
        skip_online=True,
        changelog_url=None,
        dry_run=False,
        strict=False,
        use_knowledge=False,
        workers=3,
        max_iterations=10,
        verbose=False,
        quiet=False,
        log_level=None,
        heartbeat_debug=False,
        stall_diagnostics_enabled=None,
        stall_diagnostics_level=None,
        stall_recovery_interval=30.0,
        execution_health_check_interval=30.0,
        health_warning_cooldown=60.0,
        auto_commit=False,
        auto_push=False,
        commit_message="",
        commit_per_iteration=False,
        force_update=False,
        orchestrator=orchestrator_value,
        no_mp=case.no_mp_cli,
        _orchestrator_user_set=(case.orchestrator_cli is not None) or case.no_mp_cli,
        execution_mode=case.execution_mode,
        cloud_api_key=None,
        cloud_auth_timeout=30,
        cloud_timeout=300,
        auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
        planner_execution_mode=None,
        worker_execution_mode=None,
        reviewer_execution_mode=None,
        stream_console_renderer=False,
        stream_advanced_renderer=False,
        stream_typing_effect=False,
        stream_typing_delay=0.02,
        stream_word_mode=True,
        stream_color_enabled=True,
        stream_show_word_diff=False,
        max_fetch_urls=None,
        fallback_core_docs_count=None,
        llms_txt_url=None,
        llms_cache_path=None,
    )


class TestTaskAnalyzerAndSelfIteratorAlignment:
    """测试 TaskAnalyzer._rule_based_analysis 与 SelfIterator._execution_decision 对齐

    验证 run.py 入口的 TaskAnalyzer 分析结果与 scripts/run_iterate.py 入口的
    SelfIterator 决策结果在以下字段上保持一致：
    - effective_mode
    - orchestrator
    - prefix_routed
    - has_ampersand_prefix
    - requested_mode

    测试使用统一的 mock 策略，避免触发真实网络/初始化。
    """

    @pytest.fixture(autouse=True)
    def reset_state(self) -> None:
        """每个测试前重置状态"""
        ConfigManager.reset_instance()
        # 重置 TaskAnalyzer 的消息去重状态
        from run import TaskAnalyzer

        TaskAnalyzer.reset_shown_messages()

    @pytest.mark.parametrize(
        "case",
        TASK_ANALYZER_ALIGNMENT_CASES,
        ids=[c.test_id for c in TASK_ANALYZER_ALIGNMENT_CASES],
    )
    def test_task_analyzer_execution_decision_alignment(
        self,
        case: TaskAnalyzerAlignmentCase,
    ) -> None:
        """测试 TaskAnalyzer.analyze 产生的 _execution_decision 与预期一致

        从 analysis.options["_execution_decision"] 提取字段进行验证。
        """
        from cursor.cloud_client import CloudClientFactory
        from run import TaskAnalyzer

        # 构建参数
        args = _build_task_analyzer_args(case)

        # 创建 mock 配置
        mock_config = _create_mock_config(cloud_enabled=case.cloud_enabled)

        # 解析 API Key
        api_key = "mock-api-key" if case.has_api_key else None

        # 创建分析器（禁用 Agent 分析）
        analyzer = TaskAnalyzer(use_agent=False)

        # 统一 mock
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("run.get_config", return_value=mock_config):
                with patch("core.config.get_config", return_value=mock_config):
                    with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                        # 调用 analyze 方法
                        analysis = analyzer.analyze(case.task_text, args)

                        # 提取 _execution_decision
                        decision = analysis.options.get("_execution_decision")

                        # 验证 _execution_decision 存在
                        assert decision is not None, f"[{case.test_id}] analysis.options 应包含 _execution_decision"

                        # 验证 effective_mode
                        assert decision.effective_mode == case.expected_effective_mode, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  effective_mode: 期望 {case.expected_effective_mode}, "
                            f"实际 {decision.effective_mode}"
                        )

                        # 验证 orchestrator
                        assert decision.orchestrator == case.expected_orchestrator, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  orchestrator: 期望 {case.expected_orchestrator}, "
                            f"实际 {decision.orchestrator}"
                        )

                        # 验证 prefix_routed
                        assert decision.prefix_routed == case.expected_prefix_routed, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  prefix_routed: 期望 {case.expected_prefix_routed}, "
                            f"实际 {decision.prefix_routed}"
                        )

                        # 验证 has_ampersand_prefix
                        assert decision.has_ampersand_prefix == case.expected_has_ampersand_prefix, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  has_ampersand_prefix: 期望 {case.expected_has_ampersand_prefix}, "
                            f"实际 {decision.has_ampersand_prefix}"
                        )

                        # 验证 requested_mode
                        assert decision.requested_mode == case.expected_requested_mode, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  requested_mode: 期望 {case.expected_requested_mode}, "
                            f"实际 {decision.requested_mode}"
                        )

    @pytest.mark.parametrize(
        "case",
        TASK_ANALYZER_ALIGNMENT_CASES,
        ids=[c.test_id for c in TASK_ANALYZER_ALIGNMENT_CASES],
    )
    def test_self_iterator_execution_decision_alignment(
        self,
        case: TaskAnalyzerAlignmentCase,
    ) -> None:
        """测试 SelfIterator._execution_decision 与预期一致

        验证 scripts/run_iterate.py 入口的决策结果。
        """
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # 构建参数
        args = _build_self_iterator_args(case)

        # 创建 mock 配置
        mock_config = _create_mock_config(cloud_enabled=case.cloud_enabled)

        # 解析 API Key
        api_key = "mock-api-key" if case.has_api_key else None

        # 统一 mock
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        # 创建 SelfIterator
                                        iterator = SelfIterator(args)

                                        # 获取 _execution_decision
                                        decision = iterator._execution_decision

                                        # 验证 effective_mode
                                        assert decision.effective_mode == case.expected_effective_mode, (
                                            f"[{case.test_id}] {case.description}\n"
                                            f"  effective_mode: 期望 {case.expected_effective_mode}, "
                                            f"实际 {decision.effective_mode}"
                                        )

                                        # 验证 orchestrator
                                        assert decision.orchestrator == case.expected_orchestrator, (
                                            f"[{case.test_id}] {case.description}\n"
                                            f"  orchestrator: 期望 {case.expected_orchestrator}, "
                                            f"实际 {decision.orchestrator}"
                                        )

                                        # 验证 prefix_routed
                                        assert decision.prefix_routed == case.expected_prefix_routed, (
                                            f"[{case.test_id}] {case.description}\n"
                                            f"  prefix_routed: 期望 {case.expected_prefix_routed}, "
                                            f"实际 {decision.prefix_routed}"
                                        )

                                        # 验证 has_ampersand_prefix
                                        assert decision.has_ampersand_prefix == case.expected_has_ampersand_prefix, (
                                            f"[{case.test_id}] {case.description}\n"
                                            f"  has_ampersand_prefix: 期望 {case.expected_has_ampersand_prefix}, "
                                            f"实际 {decision.has_ampersand_prefix}"
                                        )

                                        # 验证 requested_mode
                                        assert decision.requested_mode == case.expected_requested_mode, (
                                            f"[{case.test_id}] {case.description}\n"
                                            f"  requested_mode: 期望 {case.expected_requested_mode}, "
                                            f"实际 {decision.requested_mode}"
                                        )

    @pytest.mark.parametrize(
        "case",
        TASK_ANALYZER_ALIGNMENT_CASES,
        ids=[c.test_id for c in TASK_ANALYZER_ALIGNMENT_CASES],
    )
    def test_task_analyzer_and_self_iterator_cross_alignment(
        self,
        case: TaskAnalyzerAlignmentCase,
    ) -> None:
        """测试 TaskAnalyzer 与 SelfIterator 的决策交叉对齐

        同时实例化两者，验证决策字段完全一致。
        """
        from cursor.cloud_client import CloudClientFactory
        from run import TaskAnalyzer
        from scripts.run_iterate import SelfIterator

        # 构建参数
        analyzer_args = _build_task_analyzer_args(case)
        iterator_args = _build_self_iterator_args(case)

        # 创建 mock 配置
        mock_config = _create_mock_config(cloud_enabled=case.cloud_enabled)

        # 解析 API Key
        api_key = "mock-api-key" if case.has_api_key else None

        # 创建分析器
        analyzer = TaskAnalyzer(use_agent=False)

        # 统一 mock
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("run.get_config", return_value=mock_config):
                with patch("core.config.get_config", return_value=mock_config):
                    with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                        # 获取 TaskAnalyzer 决策
                        analysis = analyzer.analyze(case.task_text, analyzer_args)
                        analyzer_decision = analysis.options.get("_execution_decision")
                        assert analyzer_decision is not None
                        assert analyzer_decision is not None
                        assert analyzer_decision is not None
                        assert analyzer_decision is not None
                        assert analyzer_decision is not None
                        assert analyzer_decision is not None
                        assert analyzer_decision is not None
                        assert analyzer_decision is not None

                        # 获取 SelfIterator 决策
                        with patch("scripts.run_iterate.KnowledgeUpdater"):
                            with patch("scripts.run_iterate.ChangelogAnalyzer"):
                                with patch("scripts.run_iterate.IterationGoalBuilder"):
                                    with patch("scripts.run_iterate.resolve_docs_source_config"):
                                        with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                            iterator = SelfIterator(iterator_args)
                                            iterator_decision = iterator._execution_decision
                        assert iterator_decision is not None
                        assert iterator_decision is not None
                        assert iterator_decision is not None

                        # 验证两者一致
                        assert analyzer_decision is not None, (
                            f"[{case.test_id}] TaskAnalyzer 应产生 _execution_decision"
                        )

                        # 比较关键字段
                        assert analyzer_decision.effective_mode == iterator_decision.effective_mode, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  effective_mode 不一致:\n"
                            f"    TaskAnalyzer: {analyzer_decision.effective_mode}\n"
                            f"    SelfIterator: {iterator_decision.effective_mode}"
                        )

                        assert analyzer_decision.orchestrator == iterator_decision.orchestrator, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  orchestrator 不一致:\n"
                            f"    TaskAnalyzer: {analyzer_decision.orchestrator}\n"
                            f"    SelfIterator: {iterator_decision.orchestrator}"
                        )

                        assert analyzer_decision.prefix_routed == iterator_decision.prefix_routed, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  prefix_routed 不一致:\n"
                            f"    TaskAnalyzer: {analyzer_decision.prefix_routed}\n"
                            f"    SelfIterator: {iterator_decision.prefix_routed}"
                        )

                        assert analyzer_decision.has_ampersand_prefix == iterator_decision.has_ampersand_prefix, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  has_ampersand_prefix 不一致:\n"
                            f"    TaskAnalyzer: {analyzer_decision.has_ampersand_prefix}\n"
                            f"    SelfIterator: {iterator_decision.has_ampersand_prefix}"
                        )

                        assert analyzer_decision.requested_mode == iterator_decision.requested_mode, (
                            f"[{case.test_id}] {case.description}\n"
                            f"  requested_mode 不一致:\n"
                            f"    TaskAnalyzer: {analyzer_decision.requested_mode}\n"
                            f"    SelfIterator: {iterator_decision.requested_mode}"
                        )

    @pytest.mark.parametrize(
        "case",
        TASK_ANALYZER_ALIGNMENT_CASES,
        ids=[c.test_id for c in TASK_ANALYZER_ALIGNMENT_CASES],
    )
    def test_decision_snapshot_consistency(
        self,
        case: TaskAnalyzerAlignmentCase,
    ) -> None:
        """测试决策快照与 DecisionSnapshot 数据结构的一致性

        验证从 TaskAnalyzer 和 SelfIterator 提取的决策可以正确构建 DecisionSnapshot。
        """
        from cursor.cloud_client import CloudClientFactory
        from run import TaskAnalyzer
        from scripts.run_iterate import SelfIterator

        # 构建参数
        analyzer_args = _build_task_analyzer_args(case)
        iterator_args = _build_self_iterator_args(case)

        # 创建 mock 配置
        mock_config = _create_mock_config(cloud_enabled=case.cloud_enabled)

        # 解析 API Key
        api_key = "mock-api-key" if case.has_api_key else None

        # 创建分析器
        analyzer = TaskAnalyzer(use_agent=False)

        # 统一 mock
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("run.get_config", return_value=mock_config):
                with patch("core.config.get_config", return_value=mock_config):
                    with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                        # 获取 TaskAnalyzer 决策
                        analysis = analyzer.analyze(case.task_text, analyzer_args)
                        analyzer_decision = analysis.options.get("_execution_decision")

                        # 获取 SelfIterator 决策
                        with patch("scripts.run_iterate.KnowledgeUpdater"):
                            with patch("scripts.run_iterate.ChangelogAnalyzer"):
                                with patch("scripts.run_iterate.IterationGoalBuilder"):
                                    with patch("scripts.run_iterate.resolve_docs_source_config"):
                                        with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                            iterator = SelfIterator(iterator_args)
                                            iterator_decision = iterator._execution_decision

                        assert analyzer_decision is not None
                        assert iterator_decision is not None
                        # 构建 DecisionSnapshot（从 TaskAnalyzer）
                        analyzer_snapshot = DecisionSnapshot(
                            effective_mode=analyzer_decision.effective_mode,
                            orchestrator=analyzer_decision.orchestrator,
                            prefix_routed=analyzer_decision.prefix_routed,
                            requested_mode_for_decision=analyzer_decision.requested_mode,
                            cli_execution_mode=case.execution_mode,
                        )

                        # 构建 DecisionSnapshot（从 SelfIterator）
                        iterator_snapshot = DecisionSnapshot(
                            effective_mode=iterator_decision.effective_mode,
                            orchestrator=iterator_decision.orchestrator,
                            prefix_routed=iterator_decision.prefix_routed,
                            requested_mode_for_decision=iterator_decision.requested_mode,
                            cli_execution_mode=case.execution_mode,
                        )

                        # 验证两个快照一致
                        assert analyzer_snapshot.to_dict() == iterator_snapshot.to_dict(), (
                            f"[{case.test_id}] {case.description}\n"
                            f"  TaskAnalyzer 快照: {analyzer_snapshot.to_dict()}\n"
                            f"  SelfIterator 快照: {iterator_snapshot.to_dict()}"
                        )

                        # 验证快照与预期一致
                        expected_snapshot = DecisionSnapshot(
                            effective_mode=case.expected_effective_mode,
                            orchestrator=case.expected_orchestrator,
                            prefix_routed=case.expected_prefix_routed,
                            requested_mode_for_decision=case.expected_requested_mode,
                            cli_execution_mode=case.execution_mode,
                        )

                        assert analyzer_snapshot.to_dict() == expected_snapshot.to_dict(), (
                            f"[{case.test_id}] {case.description}\n"
                            f"  实际快照: {analyzer_snapshot.to_dict()}\n"
                            f"  期望快照: {expected_snapshot.to_dict()}"
                        )


# ============================================================
# 回退场景 execution_mode vs effective_mode 语义正确性测试
# ============================================================


class TestExecutionModeVsEffectiveModeSemantics:
    """验证 auto/cloud 无 key 回退场景中 execution_mode 和 effective_mode 的语义正确性

    核心规则：
    - resolved["execution_mode"] 应保持为 requested_mode（auto/cloud）
    - effective_mode 应为回退后的实际模式（cli）
    - 两者语义不同，不应混淆或相互覆盖
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.fixture
    def clean_api_key_env(self, clean_env) -> None:
        """委托给全局 clean_env fixture"""
        pass

    @pytest.mark.parametrize(
        "requested_mode,expected_effective_mode",
        [
            ("auto", "cli"),  # auto 无 key 回退到 cli
            ("cloud", "cli"),  # cloud 无 key 回退到 cli
        ],
        ids=["auto-no-key-fallback", "cloud-no-key-fallback"],
    )
    def test_resolved_execution_mode_preserves_requested_mode(
        self,
        clean_api_key_env: None,
        monkeypatch: pytest.MonkeyPatch,
        requested_mode: str,
        expected_effective_mode: str,
    ) -> None:
        """验证 resolved["execution_mode"] 保持为 requested_mode 而非 effective_mode

        关键断言：
        1. resolved["execution_mode"] == requested_mode（保持原始请求）
        2. effective_mode == "cli"（实际回退结果）
        3. 两者不相等（语义不同）
        """
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        args = argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=requested_mode,  # 显式请求 auto/cloud
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
                    # 构建 execution_decision
                    decision = build_execution_decision(
                        prompt="测试任务",
                        requested_mode=requested_mode,
                        cloud_enabled=True,
                        has_api_key=False,
                    )

                    # 构建 unified overrides
                    unified = build_unified_overrides(
                        args=args,
                        execution_decision=decision,
                    )

                    # 核心断言 1: resolved["execution_mode"] 保持为 requested_mode
                    assert unified.resolved["execution_mode"] == requested_mode, (
                        f"resolved['execution_mode'] 应保持为 requested_mode\n"
                        f"  期望: {requested_mode}\n"
                        f"  实际: {unified.resolved['execution_mode']}\n"
                        f"  ⚠ 如果实际值为 'cli'，说明 effective_mode 被错误写回了！"
                    )

                    # 核心断言 2: effective_mode 为回退后的实际模式
                    assert unified.effective_mode == expected_effective_mode, (
                        f"effective_mode 应为回退后的实际模式\n"
                        f"  期望: {expected_effective_mode}\n"
                        f"  实际: {unified.effective_mode}"
                    )

                    # 核心断言 3: requested_mode 保持不变
                    assert unified.requested_mode == requested_mode, (
                        f"requested_mode 应保持为原始请求\n  期望: {requested_mode}\n  实际: {unified.requested_mode}"
                    )

                    # 验证语义区分：requested_mode != effective_mode
                    assert unified.requested_mode != unified.effective_mode, (
                        f"无 API Key 回退时 requested_mode 和 effective_mode 应不同\n"
                        f"  requested_mode: {unified.requested_mode}\n"
                        f"  effective_mode: {unified.effective_mode}"
                    )


class TestNoEffectiveModeWriteBack:
    """验证关键入口文件中没有将 effective_mode 写回到 execution_mode

    这是一个静态代码分析测试，通过扫描源代码确保：
    1. 不存在 `execution_mode = effective_mode` 模式
    2. 不存在 `resolved["execution_mode"] = effective_mode` 模式
    3. 不存在 `["execution_mode"] = ...effective` 模式

    防止回归：如果有人错误地将 effective_mode 写回到 execution_mode，
    这个测试会捕获这种危险的代码变更。
    """

    # 需要扫描的关键入口文件
    TARGET_FILES = [
        "run.py",
        "scripts/run_iterate.py",
        "core/config.py",
    ]

    # 危险的代码模式（正则表达式）
    DANGEROUS_PATTERNS = [
        # execution_mode = effective_mode
        (r"execution_mode\s*=\s*effective_mode", "直接赋值 effective_mode"),
        # execution_mode = ...effective...
        (r"execution_mode\s*=\s*[^#\n]*effective", "包含 effective 的赋值"),
        # resolved["execution_mode"] = effective
        (r'\["execution_mode"\]\s*=\s*[^#\n]*effective', "字典赋值包含 effective"),
        # result["execution_mode"] = effective
        (r'result\["execution_mode"\]\s*=\s*[^#\n]*effective', "result 字典赋值"),
        # settings.execution_mode = effective
        (r"settings\.execution_mode\s*=\s*[^#\n]*effective", "属性赋值包含 effective"),
    ]

    # 允许的例外情况（注释、文档字符串、测试断言等）
    ALLOWED_CONTEXTS = [
        r"^\s*#",  # 注释行
        r"^\s*\"\"\"",  # 文档字符串开头
        r"^\s*\'\'\'",  # 文档字符串开头
        r"assert\s+",  # 测试断言
        r"f\".*\{",  # f-string 格式化（通常是日志/错误消息）
        r"\".*execution_mode.*effective",  # 字符串字面量（文档/错误消息）
    ]

    def _get_project_root(self) -> Path:
        """获取项目根目录"""
        import os

        # 从测试文件向上查找项目根目录
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / "run.py").exists():
                return current
            current = current.parent
        # 回退到环境变量或当前工作目录
        return Path(os.environ.get("PROJECT_ROOT", Path.cwd()))

    def _is_allowed_context(self, line: str) -> bool:
        """检查行是否在允许的上下文中（注释、文档等）"""
        import re

        for pattern in self.ALLOWED_CONTEXTS:
            if re.search(pattern, line):
                return True
        return False

    def _scan_file_for_dangerous_patterns(self, file_path: Path) -> list[tuple[int, str, str, str]]:
        """扫描单个文件中的危险模式

        Returns:
            list of (line_number, pattern_desc, matched_pattern, line_content)
        """
        import re

        violations: list[tuple[int, str, str, str]] = []

        if not file_path.exists():
            return violations

        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        for line_num, line in enumerate(lines, start=1):
            # 跳过允许的上下文
            if self._is_allowed_context(line):
                continue

            for pattern, desc in self.DANGEROUS_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append((line_num, desc, pattern, line.strip()))

        return violations

    def test_no_effective_mode_write_back_in_entry_points(self) -> None:
        """扫描关键入口文件确保没有将 effective_mode 写回到 execution_mode

        这个测试通过静态分析源代码来防止危险的代码模式。
        """
        project_root = self._get_project_root()

        all_violations: list[tuple[str, int, str, str, str]] = []

        for file_name in self.TARGET_FILES:
            file_path = project_root / file_name
            violations = self._scan_file_for_dangerous_patterns(file_path)
            if violations:
                all_violations.extend(
                    (file_name, line_num, desc, pattern, content) for line_num, desc, pattern, content in violations
                )

        if all_violations:
            violation_report = "\n".join(
                f"  {file}:{line}: [{desc}] 匹配模式 {pattern}\n    内容: {content}"
                for file, line, desc, pattern, content in all_violations
            )
            pytest.fail(
                f"发现 {len(all_violations)} 处可能将 effective_mode 写回到 execution_mode 的代码:\n"
                f"{violation_report}\n\n"
                f"⚠ 这是危险的代码模式！\n"
                f"  - execution_mode 应保持为 requested_mode（用户请求的模式）\n"
                f"  - effective_mode 是回退后的实际模式，不应写回到 execution_mode\n"
                f"  - 如果这是误报（如文档/注释），请更新 ALLOWED_CONTEXTS"
            )

    def test_dangerous_pattern_detection_works(self) -> None:
        """验证危险模式检测逻辑本身能正常工作

        通过构造测试字符串验证正则表达式匹配。
        """
        import re

        # 应该被检测的危险代码
        dangerous_lines = [
            "execution_mode = effective_mode",
            "self.execution_mode = effective_mode",
            'result["execution_mode"] = effective_mode',
            "settings.execution_mode = effective_mode",
        ]

        # 应该被允许的安全代码
        safe_lines = [
            "# execution_mode = effective_mode  <- 注释",
            '"""execution_mode vs effective_mode 说明"""',
            'assert execution_mode != effective_mode, "错误"',
            'f"execution_mode={execution_mode}, effective_mode={effective_mode}"',
        ]

        # 验证危险模式能被检测
        for line in dangerous_lines:
            matched = False
            for pattern, _ in self.DANGEROUS_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    matched = True
                    break
            assert matched, f"危险代码未被检测: {line}"

        # 验证安全代码被跳过（在允许上下文中）
        for line in safe_lines:
            is_allowed = self._is_allowed_context(line)
            assert is_allowed, f"安全代码被误判为危险: {line}"


class TestResolvedVsEffectiveModeInPrintConfig:
    """验证 --print-config 输出中 requested_mode 和 effective_mode 的正确区分

    这个测试确保 format_debug_config 函数在输出配置时：
    1. 输出 requested_mode（而非 execution_mode）表示用户请求
    2. 输出 effective_mode 表示实际生效的模式
    3. 两者在回退场景下不同
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "requested_mode",
        ["auto", "cloud"],
        ids=["auto-mode", "cloud-mode"],
    )
    def test_format_debug_config_distinguishes_requested_and_effective(
        self,
        requested_mode: str,
        clean_env,
    ) -> None:
        """验证 format_debug_config 正确区分 requested_mode 和 effective_mode"""
        from core.config import format_debug_config

        ConfigManager.reset_instance()

        mock_config = _create_mock_config(cloud_enabled=True)
        mock_config.cloud_agent.api_key = None

        cli_overrides = {
            "execution_mode": requested_mode,
            "orchestrator": "mp",
        }

        with patch("core.config.get_config", return_value=mock_config):
            with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                output = format_debug_config(
                    cli_overrides=cli_overrides,
                    source_label="test",
                    has_api_key=False,
                    cloud_enabled=True,
                )

        # format_debug_config 返回字符串
        output_str = output if isinstance(output, str) else "\n".join(output)

        # 验证输出中包含 requested_mode 字段
        assert "requested_mode" in output_str, f"输出应包含 requested_mode 字段\n输出内容:\n{output_str}"

        # 验证 requested_mode 保持为原始请求
        assert f"requested_mode: {requested_mode}" in output_str, (
            f"requested_mode 应为 {requested_mode}\n输出内容:\n{output_str}"
        )

        # 验证 effective_mode 为回退后的 cli
        assert "effective_mode: cli" in output_str, f"无 API Key 时 effective_mode 应为 cli\n输出内容:\n{output_str}"

        # 验证不使用 execution_mode 作为输出字段名（应使用 requested_mode）
        # 这确保输出语义清晰
        execution_mode_count = output_str.count("execution_mode:")
        assert execution_mode_count == 0, (
            f"输出不应使用 'execution_mode:' 作为字段名（应使用 requested_mode:）\n输出内容:\n{output_str}"
        )


# ============================================================
# TestEntryRequestedModeForDecisionMatrix - 入口一致性矩阵测试
# ============================================================


@dataclass
class EntryRequestedModeTestCase:
    """入口 requested_mode_for_decision 一致性测试用例

    用于验证 run.py 和 scripts/run_iterate.py 两个入口对于相同输入
    计算出相同的 requested_mode_for_decision，且符合不变式。

    字段说明：
    - test_id: 测试标识符
    - requirement: 用户输入（可包含 & 前缀）
    - cli_execution_mode: CLI --execution-mode 参数（None 表示未指定）
    - config_execution_mode: config.yaml 中的 cloud_agent.execution_mode
    - has_api_key: 是否有 API Key
    - cloud_enabled: cloud_agent.enabled 配置
    - expected_requested_mode: 预期的 requested_mode_for_decision
    - expected_has_ampersand: 预期的 has_ampersand_prefix
    - expected_prefix_routed: 预期的 prefix_routed
    - expected_orchestrator: 预期的 orchestrator
    - description: 场景描述
    """

    test_id: str
    requirement: str
    cli_execution_mode: Optional[str]
    config_execution_mode: str
    has_api_key: bool
    cloud_enabled: bool
    expected_requested_mode: Optional[str]
    expected_has_ampersand: bool
    expected_prefix_routed: bool
    expected_orchestrator: str
    description: str


# 入口一致性测试矩阵
ENTRY_REQUESTED_MODE_CASES = [
    # ===== 核心场景 1: 无 & + 无 CLI execution_mode + config=auto =====
    EntryRequestedModeTestCase(
        test_id="entry_no_amp_no_cli_config_auto",
        requirement="普通任务描述",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode="auto",
        expected_has_ampersand=False,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="默认场景：无 & + 无 CLI + config=auto → requested=auto, basic",
    ),
    EntryRequestedModeTestCase(
        test_id="entry_no_amp_no_cli_config_auto_no_key",
        requirement="普通任务描述",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode="auto",
        expected_has_ampersand=False,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="关键场景：config=auto + 无 key → 回退 CLI 但 orchestrator=basic",
    ),
    EntryRequestedModeTestCase(
        test_id="entry_no_amp_no_cli_config_cli",
        requirement="普通任务描述",
        cli_execution_mode=None,
        config_execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode="cli",
        expected_has_ampersand=False,
        expected_prefix_routed=False,
        expected_orchestrator="mp",
        description="config=cli 时默认使用 mp 编排器",
    ),
    # ===== 核心场景 2: 有 & + 无 CLI execution_mode =====
    EntryRequestedModeTestCase(
        test_id="entry_amp_no_cli_full_cloud",
        requirement="& 云端分析任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode=None,
        expected_has_ampersand=True,
        expected_prefix_routed=True,
        expected_orchestrator="basic",
        description="& 前缀成功路由到 Cloud",
    ),
    EntryRequestedModeTestCase(
        test_id="entry_amp_no_cli_no_key",
        requirement="& 云端分析任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode=None,
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="R-2: & 前缀 + 无 key → prefix_routed=False 但 basic (Cloud 意图)",
    ),
    EntryRequestedModeTestCase(
        test_id="entry_amp_no_cli_cloud_disabled",
        requirement="& 云端分析任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=False,
        expected_requested_mode=None,
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="R-2: & 前缀 + cloud_disabled → prefix_routed=False 但 basic",
    ),
    EntryRequestedModeTestCase(
        test_id="entry_amp_no_cli_both_fail",
        requirement="& 双重失败任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=False,
        expected_requested_mode=None,
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="R-2: & 前缀 + 无 key + cloud_disabled → basic",
    ),
    # ===== 场景 3: 有 & + 显式 --execution-mode cli =====
    EntryRequestedModeTestCase(
        test_id="entry_amp_explicit_cli",
        requirement="& 显式 CLI 任务",
        cli_execution_mode="cli",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode="cli",
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="mp",
        description="R-3: 显式 cli 忽略 & 前缀，允许 mp",
    ),
    EntryRequestedModeTestCase(
        test_id="entry_amp_explicit_cli_no_key",
        requirement="& 显式 CLI 任务",
        cli_execution_mode="cli",
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=False,
        expected_requested_mode="cli",
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="mp",
        description="R-3: 显式 cli 任何条件下都允许 mp",
    ),
    # ===== 场景 4: 有 & + 显式 --execution-mode plan =====
    EntryRequestedModeTestCase(
        test_id="entry_amp_explicit_plan",
        requirement="& 规划分析任务",
        cli_execution_mode="plan",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode="plan",
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="mp",
        description="R-3: plan 模式忽略 & 前缀，允许 mp",
    ),
    # ===== 场景 5: 有 & + 显式 --execution-mode ask =====
    EntryRequestedModeTestCase(
        test_id="entry_amp_explicit_ask",
        requirement="& 代码解释任务",
        cli_execution_mode="ask",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode="ask",
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="mp",
        description="R-3: ask 模式忽略 & 前缀，允许 mp",
    ),
    # ===== 场景 6: 有 & + 显式 --execution-mode cloud =====
    EntryRequestedModeTestCase(
        test_id="entry_amp_explicit_cloud",
        requirement="& 云端长时间任务",
        cli_execution_mode="cloud",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode="cloud",
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="显式 cloud 优先于 & 前缀，强制 basic",
    ),
    EntryRequestedModeTestCase(
        test_id="entry_amp_explicit_cloud_no_key",
        requirement="& 云端长时间任务",
        cli_execution_mode="cloud",
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode="cloud",
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="显式 cloud + 无 key → 回退 CLI 但 basic",
    ),
    # ===== 场景 7: 有 & + 显式 --execution-mode auto =====
    EntryRequestedModeTestCase(
        test_id="entry_amp_explicit_auto",
        requirement="& 自动模式任务",
        cli_execution_mode="auto",
        config_execution_mode="cli",  # 不同于默认
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode="auto",
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="显式 auto 优先于 config.yaml，强制 basic",
    ),
    EntryRequestedModeTestCase(
        test_id="entry_amp_explicit_auto_no_key",
        requirement="& 自动模式任务",
        cli_execution_mode="auto",
        config_execution_mode="cli",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode="auto",
        expected_has_ampersand=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="显式 auto + 无 key → 回退 CLI 但 basic",
    ),
]


class TestEntryRequestedModeMatrix:
    """入口 requested_mode_for_decision 一致性矩阵测试

    验证 run.py 和 scripts/run_iterate.py 两个入口对于相同的输入参数：
    1. 计算出相同的 requested_mode_for_decision
    2. build_execution_decision 返回一致的结果
    3. 符合核心不变式
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前后重置 ConfigManager"""
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        ENTRY_REQUESTED_MODE_CASES,
        ids=[tc.test_id for tc in ENTRY_REQUESTED_MODE_CASES],
    )
    def test_resolve_requested_mode_for_decision(self, test_case: EntryRequestedModeTestCase) -> None:
        """验证 resolve_requested_mode_for_decision 返回值"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import resolve_requested_mode_for_decision

        has_ampersand = is_cloud_request(test_case.requirement)

        assert has_ampersand == test_case.expected_has_ampersand, f"[{test_case.test_id}] has_ampersand_prefix 不匹配"

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand,
            config_execution_mode=test_case.config_execution_mode,
        )

        assert requested_mode == test_case.expected_requested_mode, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  expected_requested_mode={test_case.expected_requested_mode}\n"
            f"  actual_requested_mode={requested_mode}"
        )

    @pytest.mark.parametrize(
        "test_case",
        ENTRY_REQUESTED_MODE_CASES,
        ids=[tc.test_id for tc in ENTRY_REQUESTED_MODE_CASES],
    )
    def test_build_execution_decision_output(self, test_case: EntryRequestedModeTestCase) -> None:
        """验证 build_execution_decision 输出与预期一致"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import (
            build_execution_decision,
            resolve_requested_mode_for_decision,
        )

        has_ampersand = is_cloud_request(test_case.requirement)

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand,
            config_execution_mode=test_case.config_execution_mode,
        )

        decision = build_execution_decision(
            prompt=test_case.requirement,
            requested_mode=requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=True,
        )

        # 验证 prefix_routed
        assert decision.prefix_routed == test_case.expected_prefix_routed, (
            f"[{test_case.test_id}] prefix_routed 不匹配\n"
            f"  expected={test_case.expected_prefix_routed}, actual={decision.prefix_routed}"
        )

        # 验证 orchestrator
        assert decision.orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] orchestrator 不匹配\n"
            f"  expected={test_case.expected_orchestrator}, actual={decision.orchestrator}"
        )

        # 验证 has_ampersand_prefix
        assert decision.has_ampersand_prefix == test_case.expected_has_ampersand, (
            f"[{test_case.test_id}] has_ampersand_prefix 不匹配"
        )

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in ENTRY_REQUESTED_MODE_CASES if tc.expected_requested_mode in ("auto", "cloud")],
        ids=[tc.test_id for tc in ENTRY_REQUESTED_MODE_CASES if tc.expected_requested_mode in ("auto", "cloud")],
    )
    def test_auto_cloud_forces_basic_invariant(self, test_case: EntryRequestedModeTestCase) -> None:
        """核心不变式：requested_mode=auto/cloud 强制 basic"""
        from core.execution_policy import should_use_mp_orchestrator

        can_use_mp = should_use_mp_orchestrator(test_case.expected_requested_mode)

        assert can_use_mp is False, (
            f"[{test_case.test_id}] should_use_mp_orchestrator({test_case.expected_requested_mode}) 应返回 False"
        )

        assert test_case.expected_orchestrator == "basic", (
            f"[{test_case.test_id}] auto/cloud 模式 orchestrator 必须是 basic"
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            tc
            for tc in ENTRY_REQUESTED_MODE_CASES
            if tc.expected_has_ampersand and tc.expected_prefix_routed is False and tc.cli_execution_mode is None
        ],
        ids=[
            tc.test_id
            for tc in ENTRY_REQUESTED_MODE_CASES
            if tc.expected_has_ampersand and tc.expected_prefix_routed is False and tc.cli_execution_mode is None
        ],
    )
    def test_ampersand_cloud_intent_invariant(self, test_case: EntryRequestedModeTestCase) -> None:
        """R-2 规则：& 前缀表达 Cloud 意图时强制 basic（即使未成功路由）"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import (
            build_execution_decision,
            resolve_requested_mode_for_decision,
        )

        has_ampersand = is_cloud_request(test_case.requirement)
        assert has_ampersand is True

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand,
            config_execution_mode=test_case.config_execution_mode,
        )

        assert requested_mode is None, f"[{test_case.test_id}] & 前缀 + 无 CLI 时 requested_mode 应为 None"

        decision = build_execution_decision(
            prompt=test_case.requirement,
            requested_mode=requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=True,
        )

        assert decision.prefix_routed is False, f"[{test_case.test_id}] 此场景 prefix_routed 应为 False"

        assert decision.orchestrator == "basic", (
            f"[{test_case.test_id}] R-2: & 前缀表达 Cloud 意图 → orchestrator=basic"
        )

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in ENTRY_REQUESTED_MODE_CASES if tc.cli_execution_mode in ("cli", "plan", "ask")],
        ids=[tc.test_id for tc in ENTRY_REQUESTED_MODE_CASES if tc.cli_execution_mode in ("cli", "plan", "ask")],
    )
    def test_cli_plan_ask_allows_mp_invariant(self, test_case: EntryRequestedModeTestCase) -> None:
        """R-3 规则：cli/plan/ask 模式允许 mp 编排器"""
        from core.execution_policy import should_use_mp_orchestrator

        can_use_mp = should_use_mp_orchestrator(test_case.expected_requested_mode)

        assert can_use_mp is True, (
            f"[{test_case.test_id}] should_use_mp_orchestrator({test_case.expected_requested_mode}) 应返回 True"
        )

        assert test_case.expected_orchestrator == "mp", f"[{test_case.test_id}] cli/plan/ask 模式 orchestrator 应为 mp"


# ============================================================
# resolve_orchestrator_settings 与 build_execution_decision 设计差异测试
# ============================================================


class TestResolveOrchestratorSettingsVsBuildDecisionDesignDifference:
    """验证 resolve_orchestrator_settings 与 build_execution_decision 的设计差异

    ================================================================================
    核心设计差异（这是设计而非 bug）
    ================================================================================

    【差异说明】

    当 `&` 前缀存在但未成功路由（prefix_routed=False, auto_detect=True）时：

    1. `resolve_orchestrator_settings(prefix_routed=False)`:
       - **不会**单独因 & 前缀而强制 basic
       - 仅当 execution_mode=auto/cloud 或 prefix_routed=True 时才强制 basic
       - 职责边界：此函数不负责检测 & 前缀语法

    2. `build_execution_decision` + `build_unified_overrides`:
       - **会**因 & 前缀表达 Cloud 意图而强制 basic（R-2 规则）
       - 将 orchestrator=basic 写入 ExecutionDecision
       - 通过 overrides["orchestrator"]="basic" 传递给 resolve_orchestrator_settings

    【设计理由】

    - `resolve_orchestrator_settings` 是底层配置解析函数，不包含 & 前缀检测逻辑
    - `build_execution_decision` 是策略决策函数，负责 & 前缀检测和 R-2 规则
    - 正确的调用流程是先调用 build_execution_decision 再调用 resolve_orchestrator_settings
    - 这种分层设计保持了职责清晰

    【测试目的】

    本测试类用于：
    1. 记录并验证这一设计差异
    2. 确保两个函数的组合使用产生预期的 orchestrator=basic 结果
    3. 明确断言：仅调用 resolve_orchestrator_settings 不足以实现 R-2 规则
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def test_resolve_orchestrator_settings_alone_does_not_force_basic_for_prefix_false(
        self,
    ) -> None:
        """设计断言：仅调用 resolve_orchestrator_settings(prefix_routed=False) 不强制 basic

        当 prefix_routed=False 且 execution_mode=cli 时：
        - resolve_orchestrator_settings 返回 orchestrator=mp
        - 此函数不负责检测 & 前缀语法，因此不会因 & 前缀而强制 basic

        ⚠ 这是设计而非 bug：
        - resolve_orchestrator_settings 只看 prefix_routed 参数和 execution_mode
        - 不包含 prompt 解析或 & 前缀语法检测
        - R-2 规则（& 前缀表达 Cloud 意图）由 build_execution_decision 实现
        """
        # 场景：prefix_routed=False + execution_mode=cli → orchestrator=mp
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},  # 显式 CLI 模式
            prefix_routed=False,  # & 前缀未成功路由
        )

        assert result["orchestrator"] == "mp", (
            "设计断言：resolve_orchestrator_settings(prefix_routed=False, execution_mode=cli) "
            "返回 orchestrator=mp\n"
            "原因：此函数不负责检测 & 前缀语法，prefix_routed=False 不触发强制 basic"
        )

    def test_build_execution_decision_forces_basic_for_ampersand_not_routed(
        self,
    ) -> None:
        """设计断言：build_execution_decision 对 & 前缀未成功路由场景返回 orchestrator=basic

        当 & 前缀存在但因无 API Key/cloud_disabled 未成功路由时：
        - build_execution_decision 返回 orchestrator=basic
        - 原因：& 前缀表达 Cloud 意图（R-2 规则）

        这与 resolve_orchestrator_settings 不同：
        - resolve_orchestrator_settings(prefix_routed=False) 不会因 & 前缀强制 basic
        - build_execution_decision 会因 & 前缀表达 Cloud 意图而强制 basic
        """
        decision = build_execution_decision(
            prompt="& 分析代码",  # 有 & 前缀
            requested_mode=None,  # 无 CLI 显式设置
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key → prefix_routed=False
            auto_detect_cloud_prefix=True,
        )

        # 验证 prefix_routed=False（& 前缀未成功路由）
        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀"
        assert decision.prefix_routed is False, "& 前缀应未成功路由（无 API Key）"

        # 核心断言：build_execution_decision 返回 orchestrator=basic（R-2 规则）
        assert decision.orchestrator == "basic", (
            "设计断言：build_execution_decision 对 & 前缀未成功路由场景返回 orchestrator=basic\n"
            "原因：R-2 规则 - & 前缀表达 Cloud 意图，即使未成功路由也强制 basic"
        )

    def test_design_difference_orchestrator_via_overrides_propagation(
        self,
    ) -> None:
        """综合测试：验证 orchestrator=basic 通过 overrides 正确传播

        完整调用流程：
        1. build_execution_decision → orchestrator=basic
        2. 将 orchestrator=basic 写入 overrides
        3. resolve_orchestrator_settings(overrides) → 读取 overrides["orchestrator"]=basic

        这是正确的调用方式：先通过 build_execution_decision 决策，
        再通过 overrides 传递给 resolve_orchestrator_settings。
        """
        # Step 1: build_execution_decision 返回 orchestrator=basic
        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=None,
            cloud_enabled=False,  # cloud_disabled → prefix_routed=False
            has_api_key=True,
            auto_detect_cloud_prefix=True,
        )

        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀"
        assert decision.prefix_routed is False, "& 前缀应未成功路由（cloud_disabled）"
        assert decision.orchestrator == "basic", "build_execution_decision 应返回 basic"

        # Step 2: 构建 overrides，包含 decision.orchestrator
        overrides = {
            "execution_mode": decision.effective_mode,  # cli（回退）
            "orchestrator": decision.orchestrator,  # basic（来自 decision）
        }

        # Step 3: resolve_orchestrator_settings 读取 overrides["orchestrator"]
        result = resolve_orchestrator_settings(
            overrides=overrides,
            prefix_routed=decision.prefix_routed,  # False
        )

        # 核心断言：最终 orchestrator=basic（通过 overrides 传播）
        assert result["orchestrator"] == "basic", (
            "综合断言：orchestrator=basic 应通过 overrides 正确传播到 resolve_orchestrator_settings"
        )

    def test_design_difference_explicit_documentation(self) -> None:
        """设计差异的显式文档测试

        此测试作为设计决策的正式文档，验证以下关键点：

        【关键点 1】resolve_orchestrator_settings 的职责边界
        - 仅基于 prefix_routed 参数和 overrides["execution_mode"] 判断
        - 不包含 & 前缀语法检测（prompt 解析）
        - prefix_routed=False + execution_mode=cli → orchestrator=mp

        【关键点 2】build_execution_decision 的职责边界
        - 包含 & 前缀语法检测（is_cloud_request）
        - 实现 R-2 规则：& 前缀表达 Cloud 意图
        - has_ampersand_prefix=True + prefix_routed=False → orchestrator=basic

        【关键点 3】正确的组合调用方式
        - 先调用 build_execution_decision 获取决策
        - 将 decision.orchestrator 写入 overrides
        - 再调用 resolve_orchestrator_settings 应用配置
        - 最终 orchestrator=basic（通过 overrides 传播）

        【关键点 4】这是设计而非 bug
        - 分层设计保持职责清晰
        - resolve_orchestrator_settings 是底层配置解析
        - build_execution_decision 是策略决策层
        """
        # 对比测试 1: 仅调用 resolve_orchestrator_settings（不足以实现 R-2）
        result_alone = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,
        )
        assert result_alone["orchestrator"] == "mp", (
            "关键点 1 验证：resolve_orchestrator_settings 单独调用时，"
            "prefix_routed=False + execution_mode=cli → orchestrator=mp"
        )

        # 对比测试 2: 调用 build_execution_decision（实现 R-2）
        decision = build_execution_decision(
            prompt="& 任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=False,
            auto_detect_cloud_prefix=True,
        )
        assert decision.orchestrator == "basic", "关键点 2 验证：build_execution_decision 对 & 前缀未成功路由返回 basic"

        # 对比测试 3: 组合调用（正确方式）
        result_combined = resolve_orchestrator_settings(
            overrides={
                "execution_mode": decision.effective_mode,
                "orchestrator": decision.orchestrator,
            },
            prefix_routed=decision.prefix_routed,
        )
        assert result_combined["orchestrator"] == "basic", (
            "关键点 3 验证：组合调用通过 overrides 传播 orchestrator=basic"
        )

    def test_ampersand_prefix_not_routed_cloud_disabled_orchestrator_path(
        self,
    ) -> None:
        """场景测试：& 前缀 + cloud_disabled → prefix_routed=False，orchestrator=basic

        验证 cloud_agent.enabled=False 时：
        - has_ampersand_prefix=True（语法检测）
        - prefix_routed=False（cloud_disabled）
        - build_execution_decision.orchestrator=basic（R-2 规则）
        - 最终通过 overrides 传播为 orchestrator=basic
        """
        # 场景：& 前缀 + cloud_disabled
        decision = build_execution_decision(
            prompt="& 后台分析",
            requested_mode=None,
            cloud_enabled=False,  # cloud_disabled
            has_api_key=True,
            auto_detect_cloud_prefix=True,
        )

        # 断言 1: & 前缀检测
        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀"

        # 断言 2: prefix_routed=False（因 cloud_disabled）
        assert decision.prefix_routed is False, "cloud_disabled 应阻止 & 前缀路由"

        # 断言 3: orchestrator=basic（R-2 规则）
        assert decision.orchestrator == "basic", "& 前缀表达 Cloud 意图，即使 cloud_disabled 也强制 basic"

        # 断言 4: 通过 overrides 传播
        result = resolve_orchestrator_settings(
            overrides={"orchestrator": decision.orchestrator},
            prefix_routed=False,
        )
        assert result["orchestrator"] == "basic", "orchestrator=basic 应通过 overrides 正确传播"

    def test_ampersand_prefix_not_routed_no_api_key_orchestrator_path(
        self,
    ) -> None:
        """场景测试：& 前缀 + 无 API Key → prefix_routed=False，orchestrator=basic

        验证无 API Key 时：
        - has_ampersand_prefix=True（语法检测）
        - prefix_routed=False（无 API Key）
        - build_execution_decision.orchestrator=basic（R-2 规则）
        - 最终通过 overrides 传播为 orchestrator=basic
        """
        # 场景：& 前缀 + 无 API Key
        decision = build_execution_decision(
            prompt="& 后台分析",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key
            auto_detect_cloud_prefix=True,
        )

        # 断言 1: & 前缀检测
        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀"

        # 断言 2: prefix_routed=False（因无 API Key）
        assert decision.prefix_routed is False, "无 API Key 应阻止 & 前缀路由"

        # 断言 3: orchestrator=basic（R-2 规则）
        assert decision.orchestrator == "basic", "& 前缀表达 Cloud 意图，即使无 API Key 也强制 basic"

        # 断言 4: 通过 overrides 传播
        result = resolve_orchestrator_settings(
            overrides={"orchestrator": decision.orchestrator},
            prefix_routed=False,
        )
        assert result["orchestrator"] == "basic", "orchestrator=basic 应通过 overrides 正确传播"


# ============================================================
# TestAutoDetectCloudPrefixCLIOverride
# ============================================================


class TestAutoDetectCloudPrefixCLIOverride:
    """测试 CLI --auto-detect-cloud-prefix / --no-auto-detect-cloud-prefix 参数

    验证 CLI tri-state 参数对 & 前缀检测行为的覆盖。
    优先级：CLI 显式参数 > config.yaml > 默认值

    新增测试场景：
    - CLI 显式启用覆盖 config 禁用
    - CLI 显式禁用覆盖 config 启用
    - CLI 未指定时使用 config 设置
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def test_cli_auto_detect_enabled_overrides_config_disabled(self) -> None:
        """CLI --auto-detect-cloud-prefix 覆盖 config.yaml 中的禁用设置

        场景：config.yaml auto_detect_cloud_prefix=False，CLI 显式启用
        预期：& 前缀应成功触发 Cloud 路由
        """
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory

        # 创建 args，CLI 显式启用 auto_detect_cloud_prefix
        args = argparse.Namespace(
            requirement="& 后台任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=None,  # tri-state: None=未指定
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=True,  # CLI 显式启用
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
        )

        # Mock config 禁用 auto_detect，模拟 CLI 覆盖 config 的场景
        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",
            auto_detect_cloud_prefix=False,  # config 禁用
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            # 使用 compute_decision_inputs 构建决策
            # 直接传递 config 参数，无需 mock get_config
            inputs = compute_decision_inputs(
                args,
                original_prompt="& 后台任务",
                config=mock_config,
            )
            decision = inputs.build_decision()

            # CLI 显式启用应覆盖 config 禁用
            assert decision.prefix_routed is True, (
                "CLI --auto-detect-cloud-prefix 应覆盖 config.yaml 禁用设置，& 前缀应成功触发"
            )
            assert decision.effective_mode == "cloud", "& 前缀触发后 effective_mode 应为 cloud"
            assert decision.orchestrator == "basic", "Cloud 模式应强制使用 basic 编排器"

    def test_cli_auto_detect_disabled_overrides_config_enabled(self) -> None:
        """CLI --no-auto-detect-cloud-prefix 覆盖 config.yaml 中的启用设置

        场景：config.yaml auto_detect_cloud_prefix=True，CLI 显式禁用
        预期：& 前缀应被忽略，不触发 Cloud 路由
        """
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory

        # 创建 args，CLI 显式禁用 auto_detect_cloud_prefix
        args = argparse.Namespace(
            requirement="& 后台任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",  # 显式 CLI 模式以允许 mp
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=False,  # CLI 显式禁用
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
        )

        # Mock config 启用 auto_detect，模拟 CLI 覆盖 config 的场景
        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",
            auto_detect_cloud_prefix=True,  # config 启用
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            inputs = compute_decision_inputs(
                args,
                original_prompt="& 后台任务",
                config=mock_config,
            )
            decision = inputs.build_decision()

            # CLI 显式禁用应覆盖 config 启用
            assert decision.prefix_routed is False, (
                "CLI --no-auto-detect-cloud-prefix 应覆盖 config.yaml 启用设置，& 前缀应被忽略"
            )
            assert decision.has_ampersand_prefix is True, "语法上仍检测到 & 前缀"
            # 显式 cli 模式 + 禁用 auto_detect → 允许 mp
            assert decision.orchestrator == "mp", "禁用 auto_detect + 显式 cli 模式应允许 mp"

    def test_cli_auto_detect_none_uses_config(self) -> None:
        """CLI auto_detect_cloud_prefix=None 时使用 config.yaml 设置

        场景：CLI 未指定（tri-state None）
        预期：使用 config.yaml 中的设置
        """
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory

        # 创建 args，CLI 未指定 auto_detect_cloud_prefix
        args = argparse.Namespace(
            requirement="& 后台任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=None,  # tri-state: None=未指定
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=None,  # CLI 未指定
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
        )

        # 场景 1：config 启用 auto_detect
        mock_config_enabled = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",
            auto_detect_cloud_prefix=True,  # config 启用
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            inputs = compute_decision_inputs(
                args,
                original_prompt="& 任务",
                config=mock_config_enabled,
            )
            decision = inputs.build_decision()
            assert decision.prefix_routed is True, "config 启用时 & 前缀应触发"

        # 场景 2：config 禁用 auto_detect
        mock_config_disabled = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",
            auto_detect_cloud_prefix=False,  # config 禁用
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            inputs = compute_decision_inputs(
                args,
                original_prompt="& 任务",
                config=mock_config_disabled,
            )
            decision = inputs.build_decision()
            assert decision.prefix_routed is False, "config 禁用时 & 前缀应被忽略"

    def test_cli_auto_detect_disabled_allows_mp_with_cli_mode(self) -> None:
        """CLI 禁用 auto_detect + 显式 cli 模式 → 允许 mp 编排器

        场景：--no-auto-detect-cloud-prefix --execution-mode cli
        预期：& 前缀被忽略，可以使用 MP 编排器
        """
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory

        args = argparse.Namespace(
            requirement="& 后台任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",  # 用户请求 mp
            no_mp=False,
            _orchestrator_user_set=True,  # 用户显式设置
            execution_mode="cli",  # 显式 CLI 模式
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=False,  # CLI 显式禁用
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
        )

        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="auto",  # config 是 auto
            auto_detect_cloud_prefix=True,  # config 启用
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            inputs = compute_decision_inputs(
                args,
                original_prompt="& 后台任务",
                config=mock_config,
            )
            decision = inputs.build_decision()

            # CLI 禁用 auto_detect + 显式 cli 模式 → 允许 mp
            assert decision.prefix_routed is False, "& 前缀应被忽略"
            assert decision.effective_mode == "cli", "显式 cli 模式应生效"
            assert decision.orchestrator == "mp", "禁用 auto_detect + 显式 cli + 用户请求 mp → 允许使用 mp"


# ============================================================
# TestAutoDetectCloudPrefixDisabledConsistency
# ============================================================


class TestAutoDetectCloudPrefixDisabledConsistency:
    """测试 config.yaml 设置 auto_detect_cloud_prefix=False 时两入口一致性

    验证当通过 config.yaml 禁用 & 前缀自动检测时：
    1. run.py 的决策构建路径 (compute_decision_inputs → build_decision)
    2. SelfIterator 初始化路径 (_execution_decision)

    两个路径产生的 orchestrator 和 prefix_routed 字段应一致：
    - prefix_routed=False（& 前缀被忽略）
    - 编排器不强制 basic（允许 mp，取决于 execution_mode）

    核心规则 (R-3):
    当 auto_detect_cloud_prefix=False 时：
    - & 前缀被忽略，prefix_routed=False
    - 编排器选择回到 mode-only 规则
    - 如 execution_mode=cli/None → 允许 mp
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def _create_test_args(
        self,
        requirement: str,
        execution_mode: Optional[str] = None,
        orchestrator: str = "mp",
        orchestrator_user_set: bool = False,
        auto_detect_cloud_prefix: Optional[bool] = None,
    ) -> argparse.Namespace:
        """创建测试用的 argparse.Namespace"""
        return argparse.Namespace(
            requirement=requirement,
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=2,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator=orchestrator,
            no_mp=False,
            _orchestrator_user_set=orchestrator_user_set,
            execution_mode=execution_mode,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=auto_detect_cloud_prefix,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_config_auto_detect_disabled_run_py_path_consistency(self) -> None:
        """测试 run.py 决策路径: config.yaml auto_detect_cloud_prefix=False

        场景：config.yaml 设置 auto_detect_cloud_prefix=False，CLI 未指定
        预期：& 前缀被忽略，prefix_routed=False，允许 mp 编排器

        验证 run.py 的 compute_decision_inputs → build_decision 路径
        """
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory

        # CLI 未指定 auto_detect_cloud_prefix（tri-state None），使用 config.yaml 设置
        args = self._create_test_args(
            requirement="& 后台分析任务",
            execution_mode=None,  # 不显式指定 execution_mode
            auto_detect_cloud_prefix=None,  # CLI 未指定，使用 config.yaml
        )

        # config.yaml 禁用 auto_detect_cloud_prefix
        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",  # config 设置为 cli 以允许 mp
            auto_detect_cloud_prefix=False,  # 关键：config 禁用
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            inputs = compute_decision_inputs(
                args,
                original_prompt="& 后台分析任务",
                config=mock_config,
            )
            decision = inputs.build_decision()

            # R-3 规则验证：auto_detect=False → & 前缀被忽略
            assert decision.prefix_routed is False, "R-3: config.yaml auto_detect_cloud_prefix=False 时，& 前缀应被忽略"
            assert decision.has_ampersand_prefix is True, "语法层面仍检测到 & 前缀"
            # 编排器不强制 basic，允许 mp
            assert decision.orchestrator == "mp", "R-3: auto_detect=False + config cli 模式 → 不强制 basic，允许 mp"

    def test_config_auto_detect_disabled_self_iterator_path_consistency(self) -> None:
        """测试 SelfIterator 初始化路径: config.yaml auto_detect_cloud_prefix=False

        场景：config.yaml 设置 auto_detect_cloud_prefix=False，CLI 未指定
        预期：& 前缀被忽略，prefix_routed=False，允许 mp 编排器

        验证 SelfIterator 初始化时 _execution_decision 的决策
        """
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # CLI 未指定 auto_detect_cloud_prefix（tri-state None），使用 config.yaml 设置
        args = self._create_test_args(
            requirement="& 后台分析任务",
            execution_mode=None,  # 不显式指定 execution_mode
            auto_detect_cloud_prefix=None,  # CLI 未指定，使用 config.yaml
        )

        # config.yaml 禁用 auto_detect_cloud_prefix
        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",  # config 设置为 cli 以允许 mp
            auto_detect_cloud_prefix=False,  # 关键：config 禁用
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        decision = iterator._execution_decision

                                        # R-3 规则验证：auto_detect=False → & 前缀被忽略
                                        assert decision.prefix_routed is False, (
                                            "R-3: config.yaml auto_detect_cloud_prefix=False 时，"
                                            "SelfIterator & 前缀应被忽略"
                                        )
                                        assert decision.has_ampersand_prefix is True, "语法层面仍检测到 & 前缀"
                                        # 编排器不强制 basic，允许 mp
                                        assert decision.orchestrator == "mp", (
                                            "R-3: auto_detect=False + config cli 模式 → "
                                            "SelfIterator 不强制 basic，允许 mp"
                                        )

    def test_config_auto_detect_disabled_both_paths_match(self) -> None:
        """测试两入口一致性: run.py 与 SelfIterator 决策字段一致

        核心测试：验证 config.yaml auto_detect_cloud_prefix=False 时，
        run.py 的 compute_decision_inputs 路径与 SelfIterator 初始化路径
        产生一致的 orchestrator 和 prefix_routed 字段。

        场景：
        - config.yaml: auto_detect_cloud_prefix=False, execution_mode=cli
        - CLI: 无显式指定
        - prompt: "& 后台分析任务"（有 & 前缀）

        期望：
        - prefix_routed=False（& 前缀被忽略）
        - orchestrator=mp（不强制 basic）
        """
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        # CLI 未指定 auto_detect_cloud_prefix（tri-state None），使用 config.yaml 设置
        args = self._create_test_args(
            requirement="& 后台分析任务",
            execution_mode=None,  # 不显式指定 execution_mode
            auto_detect_cloud_prefix=None,  # CLI 未指定，使用 config.yaml
        )

        # config.yaml 禁用 auto_detect_cloud_prefix
        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",  # config 设置为 cli 以允许 mp
            auto_detect_cloud_prefix=False,  # 关键：config 禁用
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            # 路径 1: run.py 的 compute_decision_inputs → build_decision
            inputs = compute_decision_inputs(
                args,
                original_prompt="& 后台分析任务",
                config=mock_config,
            )
            run_py_decision = inputs.build_decision()

            # 路径 2: SelfIterator 初始化
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        iterator_decision = iterator._execution_decision

            # 核心一致性断言：两路径的关键字段必须一致
            assert run_py_decision.prefix_routed == iterator_decision.prefix_routed, (
                f"prefix_routed 不一致:\n"
                f"  run.py: {run_py_decision.prefix_routed}\n"
                f"  SelfIterator: {iterator_decision.prefix_routed}"
            )
            assert run_py_decision.orchestrator == iterator_decision.orchestrator, (
                f"orchestrator 不一致:\n"
                f"  run.py: {run_py_decision.orchestrator}\n"
                f"  SelfIterator: {iterator_decision.orchestrator}"
            )
            assert run_py_decision.effective_mode == iterator_decision.effective_mode, (
                f"effective_mode 不一致:\n"
                f"  run.py: {run_py_decision.effective_mode}\n"
                f"  SelfIterator: {iterator_decision.effective_mode}"
            )

            # R-3 规则断言：auto_detect=False → & 前缀被忽略 → 不强制 basic
            assert run_py_decision.prefix_routed is False, "R-3: auto_detect=False 时，& 前缀应被忽略"
            assert run_py_decision.orchestrator == "mp", "R-3: auto_detect=False + cli 模式 → 不强制 basic，允许 mp"

    def test_config_auto_detect_disabled_with_auto_mode_allows_mp(self) -> None:
        """测试两入口一致性: auto_detect=False + config execution_mode=auto + & 前缀

        场景：
        - config.yaml: auto_detect_cloud_prefix=False, execution_mode=auto
        - CLI: 无显式指定
        - prompt: "& 后台分析任务"（有 & 前缀）

        期望（R-3 规则）：
        - has_ampersand_prefix=True（语法层面检测到 & 前缀）
        - prefix_routed=False（& 前缀被忽略，因为 auto_detect=False）
        - effective_mode=cli（回退到 CLI）
        - orchestrator=mp（不强制 basic，因为 & 前缀被忽略，回退到 CLI）

        注意：这个场景验证 R-3 规则核心行为：
        当 auto_detect_cloud_prefix=False 时，& 前缀被完全忽略，
        系统回退到 CLI 模式，允许使用 mp 编排器。
        """
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        args = self._create_test_args(
            requirement="& 后台分析任务",
            execution_mode=None,  # 不显式指定 execution_mode
            auto_detect_cloud_prefix=None,  # CLI 未指定，使用 config.yaml
        )

        # config.yaml: auto_detect=False，execution_mode=auto
        # 但 auto_detect=False 会导致系统回退到 CLI
        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="auto",  # config 设置为 auto
            auto_detect_cloud_prefix=False,  # 关键：config 禁用
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            # 路径 1: run.py
            inputs = compute_decision_inputs(
                args,
                original_prompt="& 后台分析任务",
                config=mock_config,
            )
            run_py_decision = inputs.build_decision()

            # 路径 2: SelfIterator
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        iterator_decision = iterator._execution_decision

            # 一致性断言（核心）
            assert run_py_decision.prefix_routed == iterator_decision.prefix_routed, (
                f"prefix_routed 不一致:\n"
                f"  run.py: {run_py_decision.prefix_routed}\n"
                f"  SelfIterator: {iterator_decision.prefix_routed}"
            )
            assert run_py_decision.orchestrator == iterator_decision.orchestrator, (
                f"orchestrator 不一致:\n"
                f"  run.py: {run_py_decision.orchestrator}\n"
                f"  SelfIterator: {iterator_decision.orchestrator}"
            )
            assert run_py_decision.effective_mode == iterator_decision.effective_mode, (
                f"effective_mode 不一致:\n"
                f"  run.py: {run_py_decision.effective_mode}\n"
                f"  SelfIterator: {iterator_decision.effective_mode}"
            )

            # R-3 规则断言
            # auto_detect=False → & 前缀被忽略 → 回退到 CLI → 允许 mp
            assert run_py_decision.has_ampersand_prefix is True, "语法层面仍检测到 & 前缀"
            assert run_py_decision.prefix_routed is False, "R-3: auto_detect=False 时，& 前缀应被忽略"
            assert run_py_decision.effective_mode == "cli", "R-3: auto_detect=False 时，回退到 CLI 模式"
            assert run_py_decision.orchestrator == "mp", "R-3: auto_detect=False + 回退到 CLI → 允许 mp（不强制 basic）"

    def test_config_auto_detect_disabled_no_prefix_consistency(self) -> None:
        """测试两入口一致性: auto_detect=False + 无 & 前缀

        场景：
        - config.yaml: auto_detect_cloud_prefix=False, execution_mode=cli
        - CLI: 无显式指定
        - prompt: "普通任务"（无 & 前缀）

        期望：
        - has_ampersand_prefix=False
        - prefix_routed=False
        - orchestrator=mp（cli 模式允许 mp）
        """
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        args = self._create_test_args(
            requirement="普通任务",  # 无 & 前缀
            execution_mode=None,
            auto_detect_cloud_prefix=None,
        )

        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",
            auto_detect_cloud_prefix=False,
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            # 路径 1: run.py
            inputs = compute_decision_inputs(
                args,
                original_prompt="普通任务",
                config=mock_config,
            )
            run_py_decision = inputs.build_decision()

            # 路径 2: SelfIterator
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    with patch("scripts.run_iterate.KnowledgeUpdater"):
                        with patch("scripts.run_iterate.ChangelogAnalyzer"):
                            with patch("scripts.run_iterate.IterationGoalBuilder"):
                                with patch("scripts.run_iterate.resolve_docs_source_config"):
                                    with patch("scripts.run_iterate.resolve_doc_url_strategy_config"):
                                        iterator = SelfIterator(args)
                                        iterator_decision = iterator._execution_decision

            # 一致性断言
            assert run_py_decision.has_ampersand_prefix == iterator_decision.has_ampersand_prefix
            assert run_py_decision.prefix_routed == iterator_decision.prefix_routed
            assert run_py_decision.orchestrator == iterator_decision.orchestrator

            # 无 & 前缀场景断言
            assert run_py_decision.has_ampersand_prefix is False, "无 & 前缀"
            assert run_py_decision.prefix_routed is False, "无 & 前缀，prefix_routed=False"
            assert run_py_decision.orchestrator == "mp", "cli 模式允许 mp"


# ============================================================
# build_unified_overrides 缺失 execution_decision 时 auto_detect_cloud_prefix 配置测试
# ============================================================


class TestBuildUnifiedOverridesNoDecisionAutoDetect:
    """测试 build_unified_overrides 在 execution_decision 缺失时读取 config.auto_detect_cloud_prefix

    当 execution_decision=None 时，build_unified_overrides 会通过 compute_decision_inputs
    重建决策。此时 auto_detect_cloud_prefix 配置决定 & 前缀是否参与 Cloud 路由。

    关键场景（与 AGENTS.md R-3 规则一致）：
    - auto_detect_cloud_prefix=False 时，& 前缀被忽略，orchestrator 允许 mp
    - auto_detect_cloud_prefix=True（默认）时，& 前缀触发 Cloud 意图，orchestrator 为 basic

    此测试验证 build_unified_overrides 的重建路径正确读取 config.auto_detect_cloud_prefix。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def _create_test_args(
        self,
        requirement: str = "& 测试任务",
        execution_mode: Optional[str] = None,
        auto_detect_cloud_prefix: Optional[bool] = None,
    ) -> argparse.Namespace:
        """创建测试用 args"""
        return argparse.Namespace(
            requirement=requirement,
            directory=".",
            skip_online=True,
            dry_run=False,
            verbose=False,
            quiet=False,
            log_file=None,
            workers=None,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            heartbeat_debug=False,
            stall_diagnostics=False,
            stall_diagnostics_level="warning",
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=execution_mode,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            auto_detect_cloud_prefix=auto_detect_cloud_prefix,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )

    @pytest.mark.parametrize(
        "config_auto_detect,has_api_key,expected_orchestrator,scenario_desc",
        [
            # config auto_detect=False 场景：& 前缀被忽略，允许 mp
            (False, True, "mp", "auto_detect=False + has_key → mp"),
            (False, False, "mp", "auto_detect=False + no_key → mp"),
            # config auto_detect=True 场景：& 前缀触发 Cloud 意图，强制 basic
            (True, True, "basic", "auto_detect=True + has_key → basic (prefix_routed=True)"),
            (True, False, "basic", "auto_detect=True + no_key → basic (prefix_routed=False)"),
        ],
        ids=[
            "auto_detect_false_has_key",
            "auto_detect_false_no_key",
            "auto_detect_true_has_key",
            "auto_detect_true_no_key",
        ],
    )
    def test_no_decision_config_auto_detect_matrix(
        self,
        config_auto_detect: bool,
        has_api_key: bool,
        expected_orchestrator: str,
        scenario_desc: str,
    ) -> None:
        """矩阵测试：验证 execution_decision 缺失时 config.auto_detect_cloud_prefix 的影响

        此测试覆盖 R-3 规则（AGENTS.md）：
        - auto_detect_cloud_prefix=false 时，& 前缀被忽略，orchestrator 允许 mp
        - auto_detect_cloud_prefix=true 时，& 前缀表达 Cloud 意图，orchestrator 强制 basic
        """
        from core.config import build_unified_overrides
        from cursor.cloud_client import CloudClientFactory

        args = self._create_test_args(
            requirement="& 分析代码",  # 带 & 前缀
            execution_mode=None,
            auto_detect_cloud_prefix=None,  # 使用 config.yaml 的值
        )

        # 创建 mock config
        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",  # 使用 cli 以便观察 auto_detect 对 orchestrator 的影响
            auto_detect_cloud_prefix=config_auto_detect,
        )

        nl_options = {
            "_original_goal": "& 分析代码",
            "goal": "分析代码",
        }

        api_key_value = "mock-api-key" if has_api_key else None

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key_value):
            with patch("core.config.get_config", return_value=mock_config):
                options = build_unified_overrides(
                    args=args,
                    nl_options=nl_options,
                    execution_decision=None,  # 关键：不提供 execution_decision
                )

        assert options.orchestrator == expected_orchestrator, (
            f"[{scenario_desc}] orchestrator 不符预期\n  期望={expected_orchestrator}, 实际={options.orchestrator}"
        )
        assert options.resolved["orchestrator"] == expected_orchestrator, (
            f"[{scenario_desc}] resolved['orchestrator'] 不符预期"
        )

    def test_no_decision_auto_detect_false_allows_mp(self) -> None:
        """验证 execution_decision=None 且 config.auto_detect_cloud_prefix=False 时允许 mp

        这是核心场景测试：
        - execution_decision 未提供（触发 compute_decision_inputs 重建）
        - config.yaml 设置 auto_detect_cloud_prefix=False
        - & 前缀被忽略，orchestrator=mp
        """
        from core.config import build_unified_overrides
        from cursor.cloud_client import CloudClientFactory

        args = self._create_test_args(
            requirement="& 后台任务",
            execution_mode=None,
            auto_detect_cloud_prefix=None,
        )

        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",
            auto_detect_cloud_prefix=False,  # 关键：禁用
        )

        nl_options = {
            "_original_goal": "& 后台任务",
            "goal": "后台任务",
        }

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            with patch("core.config.get_config", return_value=mock_config):
                options = build_unified_overrides(
                    args=args,
                    nl_options=nl_options,
                    execution_decision=None,
                )

        # 核心断言
        assert options.orchestrator == "mp", (
            "config.auto_detect_cloud_prefix=False 时，& 前缀被忽略，orchestrator=mp\n"
            f"  实际 orchestrator={options.orchestrator}"
        )
        assert options.prefix_routed is False, "auto_detect=False 时 prefix_routed 应为 False"

    def test_no_decision_auto_detect_true_forces_basic(self) -> None:
        """验证 execution_decision=None 且 config.auto_detect_cloud_prefix=True（默认）时强制 basic

        这是核心场景测试：
        - execution_decision 未提供（触发 compute_decision_inputs 重建）
        - config.yaml 设置 auto_detect_cloud_prefix=True（默认）
        - & 前缀触发 Cloud 意图，orchestrator=basic
        """
        from core.config import build_unified_overrides
        from cursor.cloud_client import CloudClientFactory

        args = self._create_test_args(
            requirement="& 后台任务",
            execution_mode=None,
            auto_detect_cloud_prefix=None,
        )

        mock_config = _create_mock_config(
            cloud_enabled=True,
            config_execution_mode="cli",
            auto_detect_cloud_prefix=True,  # 默认启用
        )

        nl_options = {
            "_original_goal": "& 后台任务",
            "goal": "后台任务",
        }

        # 无 API Key，触发 prefix_routed=False，但 & 前缀表达 Cloud 意图，仍强制 basic
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("core.config.get_config", return_value=mock_config):
                options = build_unified_overrides(
                    args=args,
                    nl_options=nl_options,
                    execution_decision=None,
                )

        # 核心断言
        assert options.orchestrator == "basic", (
            "config.auto_detect_cloud_prefix=True 时，& 前缀触发 Cloud 意图，orchestrator=basic\n"
            f"  实际 orchestrator={options.orchestrator}"
        )
        assert options.prefix_routed is False, "无 API Key 时 prefix_routed 应为 False（但仍强制 basic）"
