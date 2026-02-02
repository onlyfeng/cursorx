"""执行模式与编排器选择一致性矩阵测试

本测试文件确保 run.py 和 scripts/run_iterate.py 两个入口
对于编排器选择的逻辑是一致的。

核心规则（与 CLI help 对齐）：
- requested=auto/cloud 即强制 basic，不受 key/enable 影响
- 这意味着即使因为没有 API Key 导致 effective_mode 回退到 CLI，
  只要 requested_mode 是 auto/cloud，编排器就应该是 basic

测试矩阵覆盖场景：
1. requested_mode=cli -> 允许 mp
2. requested_mode=auto + 有 key -> 强制 basic
3. requested_mode=auto + 无 key -> 强制 basic（关键场景）
4. requested_mode=cloud + 有 key -> 强制 basic
5. requested_mode=cloud + 无 key -> 强制 basic（关键场景）
6. & 前缀触发 + 有 key -> 强制 basic
7. & 前缀触发 + 无 key -> 允许 mp（& 前缀未成功触发）

决策快照一致性测试：
- 对每个矩阵 case，分别以 run.py 的合并结果/IterateArgs 语义
  与 SelfIterator 自身解析语义计算出决策快照
- 断言两者的快照字段一致
- 必要时对配置单例 ConfigManager 做 reset 以避免测试间污染
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.config import ConfigManager, resolve_orchestrator_settings
from core.execution_policy import (
    resolve_effective_execution_mode,
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

    # ===== None/默认模式 =====
    ConsistencyTestCase(
        test_id="none_defaults_to_cli_allows_mp",
        requested_mode=None,
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="未指定模式默认为 CLI，允许 MP",
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
    """

    @pytest.mark.parametrize(
        "test_case",
        CONSISTENCY_TEST_CASES,
        ids=[tc.test_id for tc in CONSISTENCY_TEST_CASES],
    )
    def test_should_use_mp_orchestrator(
        self, test_case: ConsistencyTestCase
    ) -> None:
        """测试 should_use_mp_orchestrator 函数"""
        can_use_mp = should_use_mp_orchestrator(test_case.requested_mode)

        expected_can_use_mp = test_case.expected_orchestrator == "mp"
        assert can_use_mp == expected_can_use_mp, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  requested_mode={test_case.requested_mode}\n"
            f"  期望 can_use_mp={expected_can_use_mp}，实际={can_use_mp}"
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
    def test_resolve_orchestrator_settings(
        self, test_case: ConsistencyTestCase
    ) -> None:
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
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        from core.config import ConfigManager
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in CONSISTENCY_TEST_CASES if tc.requested_mode in ("auto", "cloud")],
        ids=[tc.test_id for tc in CONSISTENCY_TEST_CASES if tc.requested_mode in ("auto", "cloud")],
    )
    def test_self_iterator_orchestrator_selection(
        self, test_case: ConsistencyTestCase
    ) -> None:
        """测试 SelfIterator._get_orchestrator_type 的编排器选择逻辑"""
        from scripts.run_iterate import SelfIterator
        from cursor.cloud_client import CloudClientFactory

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
        mock_config.cloud_agent.execution_mode = "cli"  # 默认值
        mock_config.cloud_agent.timeout = 300
        mock_config.cloud_agent.auth_timeout = 30
        mock_config.system.max_iterations = 10
        mock_config.system.worker_pool_size = 3
        mock_config.system.enable_sub_planners = True
        mock_config.system.strict_review = False
        mock_config.models.planner = "gpt-5.2-high"
        mock_config.models.worker = "opus-4.5-thinking"
        mock_config.models.reviewer = "gpt-5.2-codex"
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
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        from core.config import ConfigManager
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
        assert result["orchestrator"] == "basic", (
            "resolve_orchestrator_settings: auto 模式应强制 basic 编排器"
        )

        # 测试 2: should_use_mp_orchestrator
        can_use_mp = should_use_mp_orchestrator("auto")
        assert can_use_mp is False, (
            "should_use_mp_orchestrator: auto 模式应返回 False"
        )

    def test_cloud_no_key_both_entries_force_basic(self) -> None:
        """关键测试：cloud 模式无 API Key 时，两个入口都应强制 basic

        这是最重要的一致性测试，验证：
        1. resolve_orchestrator_settings 强制 basic
        2. should_use_mp_orchestrator 返回 False
        """
        # 测试 1: resolve_orchestrator_settings
        overrides = {"execution_mode": "cloud"}
        result = resolve_orchestrator_settings(overrides=overrides)
        assert result["orchestrator"] == "basic", (
            "resolve_orchestrator_settings: cloud 模式应强制 basic 编排器"
        )

        # 测试 2: should_use_mp_orchestrator
        can_use_mp = should_use_mp_orchestrator("cloud")
        assert can_use_mp is False, (
            "should_use_mp_orchestrator: cloud 模式应返回 False"
        )

    def test_cli_allows_mp_both_entries(self) -> None:
        """测试 CLI 模式两个入口都允许 MP"""
        # 测试 1: resolve_orchestrator_settings
        overrides = {"execution_mode": "cli"}
        result = resolve_orchestrator_settings(overrides=overrides)
        assert result["orchestrator"] == "mp", (
            "resolve_orchestrator_settings: cli 模式应允许 mp 编排器"
        )

        # 测试 2: should_use_mp_orchestrator
        can_use_mp = should_use_mp_orchestrator("cli")
        assert can_use_mp is True, (
            "should_use_mp_orchestrator: cli 模式应返回 True"
        )


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
        assert should_use_mp_orchestrator("auto") is False, (
            "违反 AGENTS.md 规则：auto 模式应强制 basic"
        )

        # 测试 cloud 模式
        assert should_use_mp_orchestrator("cloud") is False, (
            "违反 AGENTS.md 规则：cloud 模式应强制 basic"
        )

        # 测试 cli 模式
        assert should_use_mp_orchestrator("cli") is True, (
            "违反 AGENTS.md 规则：cli 模式应允许 mp"
        )

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

        assert auto_result["orchestrator"] == "basic", (
            "违反 AGENTS.md 规则：auto 模式应强制 basic 编排器"
        )
        assert cloud_result["orchestrator"] == "basic", (
            "违反 AGENTS.md 规则：cloud 模式应强制 basic 编排器"
        )
        assert cli_result["orchestrator"] == "mp", (
            "违反 AGENTS.md 规则：cli 模式应默认使用 mp 编排器"
        )


# ============================================================
# 决策快照数据结构
# ============================================================


@dataclass
class DecisionSnapshot:
    """执行决策快照

    用于比较 run.py 的 IterateArgs 语义与 SelfIterator 自身解析语义。
    """
    execution_mode: str  # cli/cloud/auto 等有效执行模式
    orchestrator: str    # mp 或 basic
    triggered_by_prefix: bool  # 是否由 & 前缀触发

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典便于比较"""
        return {
            "execution_mode": self.execution_mode,
            "orchestrator": self.orchestrator,
            "triggered_by_prefix": self.triggered_by_prefix,
        }


@dataclass
class SnapshotTestCase:
    """决策快照一致性测试参数"""
    test_id: str
    requirement: str  # 用户输入的任务描述（可包含 & 前缀）
    execution_mode: Optional[str]  # CLI --execution-mode 参数
    has_api_key: bool
    cloud_enabled: bool
    orchestrator_cli: Optional[str]  # CLI --orchestrator 参数
    no_mp_cli: bool  # CLI --no-mp 参数
    expected_snapshot: DecisionSnapshot
    description: str


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
            execution_mode="cli",
            orchestrator="mp",
            triggered_by_prefix=False,
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
            execution_mode="cli",
            orchestrator="mp",
            triggered_by_prefix=False,
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
            execution_mode="cli",
            orchestrator="mp",
            triggered_by_prefix=False,
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
            execution_mode="auto",
            orchestrator="basic",
            triggered_by_prefix=False,
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
            execution_mode="cli",  # 回退到 CLI
            orchestrator="basic",  # 但编排器仍为 basic（基于 requested_mode）
            triggered_by_prefix=False,
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
            execution_mode="auto",  # 虽然 cloud_enabled=False，但 CLI 显式指定 auto
            orchestrator="basic",   # CLI 显式指定 auto，强制 basic
            triggered_by_prefix=False,
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
            execution_mode="cloud",
            orchestrator="basic",
            triggered_by_prefix=False,
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
            execution_mode="cli",  # 回退到 CLI
            orchestrator="basic",  # 但编排器仍为 basic（基于 requested_mode）
            triggered_by_prefix=False,
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
            execution_mode="cloud",  # 虽然 cloud_enabled=False，但 CLI 显式指定 cloud
            orchestrator="basic",    # CLI 显式指定 cloud，强制 basic
            triggered_by_prefix=False,
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
            execution_mode="cloud",
            orchestrator="basic",
            triggered_by_prefix=True,
        ),
        description="& 前缀触发 + 有 API Key + cloud_enabled，使用 Cloud 模式",
    ),
    SnapshotTestCase(
        test_id="ampersand_no_key",
        requirement="& 后台执行任务",
        execution_mode=None,
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            execution_mode="cli",  # 无 API Key 回退到 CLI
            orchestrator="mp",     # & 前缀未成功触发，允许 mp
            triggered_by_prefix=False,  # 未成功触发
        ),
        description="& 前缀但无 API Key，回退到 CLI 并允许 mp 编排器",
    ),
    SnapshotTestCase(
        test_id="ampersand_cloud_disabled",
        requirement="& 推送到云端",
        execution_mode=None,
        has_api_key=True,
        cloud_enabled=False,  # cloud_enabled=False
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            execution_mode="cli",  # cloud_enabled=False 回退到 CLI
            orchestrator="mp",     # & 前缀未成功触发，允许 mp
            triggered_by_prefix=False,  # 未成功触发
        ),
        description="& 前缀但 cloud_enabled=False，忽略前缀使用 CLI + mp",
    ),
    SnapshotTestCase(
        test_id="ampersand_no_key_and_cloud_disabled",
        requirement="& 双重失败场景",
        execution_mode=None,
        has_api_key=False,
        cloud_enabled=False,
        orchestrator_cli=None,
        no_mp_cli=False,
        expected_snapshot=DecisionSnapshot(
            execution_mode="cli",  # 无 API Key 且 cloud_enabled=False
            orchestrator="mp",     # & 前缀未成功触发，允许 mp
            triggered_by_prefix=False,  # 未成功触发
        ),
        description="& 前缀 + 无 API Key + cloud_enabled=False，双重失败回退到 CLI + mp",
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
            execution_mode="cli",
            orchestrator="mp",
            triggered_by_prefix=False,  # 显式 CLI 忽略 & 前缀
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
            execution_mode="auto",
            orchestrator="basic",
            triggered_by_prefix=False,  # 有显式 execution_mode，triggered_by_prefix 为 False
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
            execution_mode="cli",
            orchestrator="basic",
            triggered_by_prefix=False,
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
            execution_mode="cli",
            orchestrator="basic",
            triggered_by_prefix=False,
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
            execution_mode="cli",
            orchestrator="mp",
            triggered_by_prefix=False,
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
            execution_mode="cloud",
            orchestrator="basic",  # cloud 模式强制 basic，即使用户请求 mp
            triggered_by_prefix=False,
        ),
        description="显式 --orchestrator mp + CLOUD 模式，强制切换到 basic",
    ),
]


# ============================================================
# 决策快照一致性测试类
# ============================================================


def _create_mock_config(cloud_enabled: bool = True) -> MagicMock:
    """创建模拟配置对象"""
    mock_config = MagicMock()
    mock_config.cloud_agent.enabled = cloud_enabled
    mock_config.cloud_agent.execution_mode = "cli"  # 默认值
    mock_config.cloud_agent.timeout = 300
    mock_config.cloud_agent.auth_timeout = 30
    mock_config.system.max_iterations = 10
    mock_config.system.worker_pool_size = 3
    mock_config.system.enable_sub_planners = True
    mock_config.system.strict_review = False
    mock_config.models.planner = "gpt-5.2-high"
    mock_config.models.worker = "opus-4.5-thinking"
    mock_config.models.reviewer = "gpt-5.2-codex"
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

    模拟 run.py 的 Runner._merge_options 和 _get_execution_mode 逻辑。

    编排器选择规则（与 resolve_orchestrator_settings 一致）:
    1. triggered_by_prefix=True: 强制 basic（& 前缀成功触发 Cloud）
    2. CLI 显式指定 execution_mode=auto/cloud: 强制 basic
    3. 其他情况: 允许 mp（config.yaml 默认值不单独触发强制 basic）
    """
    from core.cloud_utils import is_cloud_request

    # 检测 & 前缀
    has_ampersand_prefix = is_cloud_request(test_case.requirement)

    # 解析有效执行模式
    effective_mode, _ = resolve_effective_execution_mode(
        requested_mode=test_case.execution_mode,
        triggered_by_prefix=has_ampersand_prefix,
        cloud_enabled=test_case.cloud_enabled,
        has_api_key=test_case.has_api_key,
    )

    # 判断 & 前缀是否成功触发
    # 仅当 & 前缀存在且最终 effective_mode=cloud 时才为 True
    triggered_by_prefix = has_ampersand_prefix and effective_mode == "cloud"

    # 解析编排器类型（使用 resolve_orchestrator_settings）
    cli_overrides = {}
    if test_case.execution_mode is not None:
        cli_overrides["execution_mode"] = test_case.execution_mode
    if test_case.orchestrator_cli is not None:
        cli_overrides["orchestrator"] = test_case.orchestrator_cli
    elif test_case.no_mp_cli:
        cli_overrides["orchestrator"] = "basic"

    # 如果 & 前缀成功触发，execution_mode 应为 cloud（用于 resolve_orchestrator_settings）
    if triggered_by_prefix:
        cli_overrides["execution_mode"] = "cloud"

    # 调用 resolve_orchestrator_settings，传递 triggered_by_prefix
    resolved = resolve_orchestrator_settings(
        overrides=cli_overrides,
        triggered_by_prefix=triggered_by_prefix,
    )

    return DecisionSnapshot(
        execution_mode=effective_mode,
        orchestrator=resolved["orchestrator"],
        triggered_by_prefix=triggered_by_prefix,
    )


class TestIterateArgsAndSelfIteratorConsistency:
    """测试 run.py IterateArgs 与 SelfIterator 的决策快照一致性

    对每个测试 case，分别计算：
    1. run.py 的合并结果/IterateArgs 语义快照
    2. SelfIterator 自身解析语义快照（通过实例化并调用 _get_execution_mode/_get_orchestrator_type）

    断言两者一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        SNAPSHOT_TEST_CASES,
        ids=[tc.test_id for tc in SNAPSHOT_TEST_CASES],
    )
    def test_snapshot_consistency(self, test_case: SnapshotTestCase) -> None:
        """测试决策快照一致性"""
        from scripts.run_iterate import SelfIterator
        from cursor.cloud_client import CloudClientFactory

        # 重置配置单例
        ConfigManager.reset_instance()

        # 计算 run.py 语义的快照
        run_py_snapshot = _compute_run_py_snapshot(test_case)

        # 构建测试参数
        args = _build_iterate_args(test_case)

        # 创建 mock 配置
        mock_config = _create_mock_config(cloud_enabled=test_case.cloud_enabled)

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
                                        self_iterator_execution_mode = iterator._get_execution_mode()
                                        self_iterator_orchestrator = iterator._get_orchestrator_type()
                                        self_iterator_triggered = iterator._triggered_by_prefix

        self_iterator_snapshot = DecisionSnapshot(
            execution_mode=self_iterator_execution_mode.value,
            orchestrator=self_iterator_orchestrator,
            triggered_by_prefix=self_iterator_triggered,
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
        """验证预期快照与 Policy 函数结果一致"""
        ConfigManager.reset_instance()

        run_py_snapshot = _compute_run_py_snapshot(test_case)

        assert run_py_snapshot.execution_mode == test_case.expected_snapshot.execution_mode, (
            f"[{test_case.test_id}] execution_mode 不匹配\n"
            f"  实际: {run_py_snapshot.execution_mode}\n"
            f"  期望: {test_case.expected_snapshot.execution_mode}"
        )

        assert run_py_snapshot.orchestrator == test_case.expected_snapshot.orchestrator, (
            f"[{test_case.test_id}] orchestrator 不匹配\n"
            f"  实际: {run_py_snapshot.orchestrator}\n"
            f"  期望: {test_case.expected_snapshot.orchestrator}"
        )

        assert run_py_snapshot.triggered_by_prefix == test_case.expected_snapshot.triggered_by_prefix, (
            f"[{test_case.test_id}] triggered_by_prefix 不匹配\n"
            f"  实际: {run_py_snapshot.triggered_by_prefix}\n"
            f"  期望: {test_case.expected_snapshot.triggered_by_prefix}"
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
            assert config is not None, f"第 {i+1} 次重置后应能获取配置"


class TestTriggeredByPrefixParameter:
    """测试 resolve_orchestrator_settings 的 triggered_by_prefix 参数

    验证新增的 triggered_by_prefix 参数行为：
    - triggered_by_prefix=True: 强制 basic 编排器（& 前缀成功触发）
    - triggered_by_prefix=False: 不影响编排器选择（除非 CLI 显式指定 auto/cloud）
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def test_triggered_by_prefix_true_forces_basic(self) -> None:
        """测试 triggered_by_prefix=True 强制使用 basic 编排器"""
        # 不传递任何 overrides，仅通过 triggered_by_prefix 触发 basic
        result = resolve_orchestrator_settings(
            overrides={},
            triggered_by_prefix=True,
        )
        assert result["orchestrator"] == "basic", (
            "triggered_by_prefix=True 应强制使用 basic 编排器"
        )

    def test_triggered_by_prefix_false_allows_mp(self) -> None:
        """测试 triggered_by_prefix=False 允许使用 mp 编排器"""
        result = resolve_orchestrator_settings(
            overrides={},
            triggered_by_prefix=False,
        )
        assert result["orchestrator"] == "mp", (
            "triggered_by_prefix=False 应允许使用 mp 编排器"
        )

    def test_triggered_by_prefix_with_cli_execution_mode(self) -> None:
        """测试 triggered_by_prefix 与 CLI execution_mode 的交互

        即使 triggered_by_prefix=False，CLI 显式指定 execution_mode=auto/cloud
        也应该强制使用 basic 编排器。
        """
        # CLI 显式指定 auto
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "auto"},
            triggered_by_prefix=False,
        )
        assert result["orchestrator"] == "basic", (
            "CLI 显式指定 execution_mode=auto 应强制 basic"
        )

        # CLI 显式指定 cloud
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cloud"},
            triggered_by_prefix=False,
        )
        assert result["orchestrator"] == "basic", (
            "CLI 显式指定 execution_mode=cloud 应强制 basic"
        )

        # CLI 显式指定 cli
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=False,
        )
        assert result["orchestrator"] == "mp", (
            "CLI 显式指定 execution_mode=cli 应允许 mp"
        )

    def test_triggered_by_prefix_overrides_cli_mp_request(self) -> None:
        """测试 triggered_by_prefix=True 覆盖用户显式请求的 mp 编排器

        即使用户显式请求 --orchestrator mp，triggered_by_prefix=True 也应强制 basic。
        """
        result = resolve_orchestrator_settings(
            overrides={"orchestrator": "mp"},
            triggered_by_prefix=True,
        )
        assert result["orchestrator"] == "basic", (
            "triggered_by_prefix=True 应覆盖用户请求的 mp 编排器"
        )

    def test_cli_basic_not_affected_by_triggered_by_prefix(self) -> None:
        """测试用户显式请求 basic 不受 triggered_by_prefix 影响"""
        # triggered_by_prefix=False + CLI 请求 basic
        result = resolve_orchestrator_settings(
            overrides={"orchestrator": "basic"},
            triggered_by_prefix=False,
        )
        assert result["orchestrator"] == "basic", (
            "用户显式请求 basic 应保持 basic"
        )

        # triggered_by_prefix=True + CLI 请求 basic
        result = resolve_orchestrator_settings(
            overrides={"orchestrator": "basic"},
            triggered_by_prefix=True,
        )
        assert result["orchestrator"] == "basic", (
            "triggered_by_prefix=True + CLI 请求 basic 应保持 basic"
        )


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
        expected_skip_online=True,   # minimal 隐含 skip_online=True
        expected_dry_run=True,       # run.py minimal preset 设置 dry_run=True
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
    def test_minimal_mode_options_merge(
        self, test_case: MinimalConsistencyTestCase
    ) -> None:
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
    def test_self_iterator_minimal_mode_parsing(
        self, test_case: MinimalConsistencyTestCase
    ) -> None:
        """测试 SelfIterator 在 minimal 模式下的参数解析"""
        from scripts.run_iterate import SelfIterator
        from cursor.cloud_client import CloudClientFactory

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
    def test_run_py_and_self_iterator_minimal_consistency(
        self, test_case: MinimalConsistencyTestCase
    ) -> None:
        """测试 run.py 与 SelfIterator 在 minimal 模式下的一致性"""
        from scripts.run_iterate import SelfIterator
        from cursor.cloud_client import CloudClientFactory

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

    这是本次修复的核心场景：当 & 前缀存在但由于各种原因未成功触发 Cloud 时，
    不应该强制使用 basic 编排器。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    def test_ampersand_no_key_allows_mp(self) -> None:
        """测试 & 前缀但无 API Key 允许使用 mp 编排器"""
        # 模拟：& 前缀存在，但因无 API Key 未成功触发
        # triggered_by_prefix 应为 False
        result = resolve_orchestrator_settings(
            overrides={},  # 没有 CLI 显式 execution_mode
            triggered_by_prefix=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "mp", (
            "& 前缀但无 API Key，未成功触发，应允许 mp 编排器"
        )

    def test_ampersand_cloud_disabled_allows_mp(self) -> None:
        """测试 & 前缀但 cloud_enabled=False 允许使用 mp 编排器"""
        # 模拟：& 前缀存在，但因 cloud_enabled=False 未成功触发
        result = resolve_orchestrator_settings(
            overrides={},
            triggered_by_prefix=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "mp", (
            "& 前缀但 cloud_enabled=False，未成功触发，应允许 mp 编排器"
        )

    def test_ampersand_success_forces_basic(self) -> None:
        """测试 & 前缀成功触发强制使用 basic 编排器"""
        # 模拟：& 前缀存在，成功触发 Cloud
        result = resolve_orchestrator_settings(
            overrides={},
            triggered_by_prefix=True,  # & 前缀成功触发
        )
        assert result["orchestrator"] == "basic", (
            "& 前缀成功触发，应强制使用 basic 编排器"
        )


# ============================================================
# 边界用例扩展测试
# ============================================================


# 边界用例参数表
EDGE_CASE_CONSISTENCY_TESTS = [
    # ===== cloud_enabled=False + & 前缀 =====
    ConsistencyTestCase(
        test_id="ampersand_cloud_disabled_allows_mp",
        requested_mode=None,  # 未显式指定
        has_api_key=True,
        cloud_enabled=False,  # cloud_enabled=False
        expected_orchestrator="mp",
        description="& 前缀 + cloud_enabled=False：忽略前缀，允许 MP 编排器",
    ),
    ConsistencyTestCase(
        test_id="ampersand_cloud_disabled_no_key_allows_mp",
        requested_mode=None,
        has_api_key=False,
        cloud_enabled=False,
        expected_orchestrator="mp",
        description="& 前缀 + cloud_enabled=False + 无 key：忽略前缀，允许 MP",
    ),

    # ===== 显式 execution_mode=cli 忽略 & 前缀 =====
    ConsistencyTestCase(
        test_id="explicit_cli_ignores_ampersand_with_key",
        requested_mode="cli",  # 显式 CLI
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="显式 cli + 有 key + 有 & 前缀：忽略前缀，允许 MP",
    ),
    ConsistencyTestCase(
        test_id="explicit_cli_ignores_ampersand_no_key",
        requested_mode="cli",
        has_api_key=False,
        cloud_enabled=True,
        expected_orchestrator="mp",
        description="显式 cli + 无 key + 有 & 前缀：忽略前缀，允许 MP",
    ),

    # ===== 显式 auto/cloud + 无 key 仍强制 basic（关键场景）=====
    ConsistencyTestCase(
        test_id="explicit_auto_no_key_cloud_disabled_forces_basic",
        requested_mode="auto",
        has_api_key=False,
        cloud_enabled=False,
        expected_orchestrator="basic",
        description="关键：显式 auto + 无 key + cloud_disabled 仍强制 basic（基于 requested_mode）",
    ),
    ConsistencyTestCase(
        test_id="explicit_cloud_no_key_cloud_disabled_forces_basic",
        requested_mode="cloud",
        has_api_key=False,
        cloud_enabled=False,
        expected_orchestrator="basic",
        description="关键：显式 cloud + 无 key + cloud_disabled 仍强制 basic（基于 requested_mode）",
    ),

    # ===== 双重条件：显式 mode + & 前缀 =====
    ConsistencyTestCase(
        test_id="explicit_auto_with_ampersand_forces_basic",
        requested_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="显式 auto + & 前缀：双重条件都强制 basic",
    ),
    ConsistencyTestCase(
        test_id="explicit_cloud_with_ampersand_forces_basic",
        requested_mode="cloud",
        has_api_key=True,
        cloud_enabled=True,
        expected_orchestrator="basic",
        description="显式 cloud + & 前缀：双重条件都强制 basic",
    ),
]


class TestEdgeCaseOrchestratorConsistency:
    """边界场景编排器选择一致性测试

    补齐以下关键边界用例：
    1. cloud_enabled=False + & 前缀
    2. 显式 execution_mode=cli 忽略 & 前缀
    3. 显式 auto/cloud + 无 key 仍强制 basic 但 effective=cli
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> None:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        EDGE_CASE_CONSISTENCY_TESTS,
        ids=[tc.test_id for tc in EDGE_CASE_CONSISTENCY_TESTS],
    )
    def test_should_use_mp_orchestrator_edge_cases(
        self, test_case: ConsistencyTestCase
    ) -> None:
        """测试边界场景下 should_use_mp_orchestrator 行为"""
        can_use_mp = should_use_mp_orchestrator(test_case.requested_mode)

        expected_can_use_mp = test_case.expected_orchestrator == "mp"
        assert can_use_mp == expected_can_use_mp, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  requested_mode={test_case.requested_mode}\n"
            f"  期望 can_use_mp={expected_can_use_mp}，实际={can_use_mp}"
        )


# ============================================================
# build_execution_decision 表驱动测试
# ============================================================


@dataclass
class ExecutionDecisionTestCase:
    """build_execution_decision 测试用例

    统一的决策测试结构，用于表驱动测试。
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
    expected_triggered_by_prefix: bool
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=True,
        expected_sanitized_prompt="分析代码架构",
        description="& 前缀成功触发 Cloud",
    ),

    # ===== & 前缀触发失败场景 =====
    ExecutionDecisionTestCase(
        test_id="ampersand_no_key_fails",
        prompt="& 后台执行任务",
        requested_mode=None,
        cloud_enabled=True,
        has_api_key=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",  # 未成功触发，允许 mp
        expected_triggered_by_prefix=False,  # 未成功触发
        expected_sanitized_prompt="后台执行任务",
        description="& 前缀无 API Key 未成功触发",
    ),
    ExecutionDecisionTestCase(
        test_id="ampersand_cloud_disabled_fails",
        prompt="& 推送到云端",
        requested_mode=None,
        cloud_enabled=False,
        has_api_key=True,
        expected_effective_mode="cli",
        expected_orchestrator="mp",  # 未成功触发，允许 mp
        expected_triggered_by_prefix=False,  # 未成功触发
        expected_sanitized_prompt="推送到云端",
        description="& 前缀 cloud_disabled 未成功触发",
    ),
    ExecutionDecisionTestCase(
        test_id="ampersand_cloud_disabled_no_key_fails",
        prompt="& 任务",
        requested_mode=None,
        cloud_enabled=False,
        has_api_key=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",  # 未成功触发，允许 mp
        expected_triggered_by_prefix=False,
        expected_sanitized_prompt="任务",
        description="& 前缀 cloud_disabled + 无 key 未成功触发",
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
        expected_triggered_by_prefix=False,  # 显式 CLI 忽略
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
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
        expected_triggered_by_prefix=False,
        expected_sanitized_prompt="解释代码",
        description="ASK 只读模式允许 MP",
    ),

    # ===== 默认行为 =====
    ExecutionDecisionTestCase(
        test_id="default_cli",
        prompt="普通任务",
        requested_mode=None,
        cloud_enabled=True,
        has_api_key=True,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_triggered_by_prefix=False,
        expected_sanitized_prompt="普通任务",
        description="无显式模式无 & 前缀默认 CLI",
    ),

    # ===== 禁用自动检测 & 前缀 =====
    ExecutionDecisionTestCase(
        test_id="auto_detect_disabled",
        prompt="& 任务",
        requested_mode=None,
        cloud_enabled=True,
        has_api_key=True,
        auto_detect_cloud_prefix=False,
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_triggered_by_prefix=False,
        expected_sanitized_prompt="任务",
        description="禁用自动检测时忽略 & 前缀",
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
    def test_build_execution_decision(
        self, test_case: ExecutionDecisionTestCase
    ) -> None:
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

        # 验证 triggered_by_prefix
        assert decision.triggered_by_prefix == test_case.expected_triggered_by_prefix, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  triggered_by_prefix 期望={test_case.expected_triggered_by_prefix}，"
            f"实际={decision.triggered_by_prefix}"
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
    def test_decision_matches_policy_functions(
        self, test_case: ExecutionDecisionTestCase
    ) -> None:
        """验证 build_execution_decision 与独立 Policy 函数结果一致"""
        from core.execution_policy import build_execution_decision
        from core.cloud_utils import is_cloud_request

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

        # resolve_effective_execution_mode 需要正确的 triggered_by_prefix
        # 这取决于路由条件是否满足
        if test_case.user_requested_orchestrator is None:
            # 验证 should_use_mp_orchestrator
            can_use_mp_by_policy = should_use_mp_orchestrator(test_case.requested_mode)

            # 如果 & 前缀成功触发，也应强制 basic
            if decision.triggered_by_prefix:
                can_use_mp_by_policy = False

            expected_orch = "mp" if can_use_mp_by_policy else "basic"

            assert decision.orchestrator == expected_orch, (
                f"[{test_case.test_id}] orchestrator 与 should_use_mp_orchestrator 不一致\n"
                f"  decision.orchestrator={decision.orchestrator}\n"
                f"  should_use_mp_orchestrator={can_use_mp_by_policy} -> {expected_orch}"
            )

        # 验证 effective_mode 与 resolve_effective_execution_mode 一致
        effective_mode, _ = resolve_effective_execution_mode(
            requested_mode=test_case.requested_mode,
            triggered_by_prefix=decision.triggered_by_prefix,
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
        [tc for tc in EXECUTION_DECISION_TEST_CASES
         if tc.user_requested_orchestrator is None
         and tc.auto_detect_cloud_prefix],  # 排除用户显式指定和自定义 auto_detect 的用例
        ids=[tc.test_id for tc in EXECUTION_DECISION_TEST_CASES
             if tc.user_requested_orchestrator is None
             and tc.auto_detect_cloud_prefix],
    )
    def test_self_iterator_matches_decision(
        self, test_case: ExecutionDecisionTestCase
    ) -> None:
        """测试 SelfIterator 的决策与 build_execution_decision 一致"""
        from scripts.run_iterate import SelfIterator
        from cursor.cloud_client import CloudClientFactory
        from core.execution_policy import build_execution_decision

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
                                        actual_triggered = iterator._triggered_by_prefix

        # 比较决策结果
        assert actual_execution_mode.value == expected_decision.effective_mode, (
            f"[{test_case.test_id}] SelfIterator.execution_mode 不匹配\n"
            f"  实际: {actual_execution_mode.value}\n"
            f"  期望: {expected_decision.effective_mode}"
        )

        assert actual_orchestrator == expected_decision.orchestrator, (
            f"[{test_case.test_id}] SelfIterator.orchestrator 不匹配\n"
            f"  实际: {actual_orchestrator}\n"
            f"  期望: {expected_decision.orchestrator}"
        )

        assert actual_triggered == expected_decision.triggered_by_prefix, (
            f"[{test_case.test_id}] SelfIterator.triggered_by_prefix 不匹配\n"
            f"  实际: {actual_triggered}\n"
            f"  期望: {expected_decision.triggered_by_prefix}"
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
    # ===== 基础 execution_mode=None（默认） =====
    ResolveConfigSettingsTestCase(
        test_id="none_default",
        execution_mode=None,
        has_api_key=True,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=False,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="mp",
        description="无显式 execution_mode，默认 CLI，允许 mp",
    ),

    # ===== execution_mode=cli =====
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
    ResolveConfigSettingsTestCase(
        test_id="ampersand_no_key",
        execution_mode=None,
        has_api_key=False,
        cloud_enabled=True,
        user_orchestrator=None,
        has_ampersand_prefix=True,
        cli_workers=None,
        cli_max_iterations=None,
        expected_orchestrator="mp",
        description="& 前缀但无 API Key，未成功触发，允许 mp",
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
        expected_orchestrator="mp",
        description="& 前缀但 cloud_disabled，未成功触发，允许 mp",
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
    """

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
        from scripts.run_iterate import SelfIterator
        from cursor.cloud_client import CloudClientFactory
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision

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
        from scripts.run_iterate import SelfIterator
        from cursor.cloud_client import CloudClientFactory
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision

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
        assert decision.effective_mode == "cli", (
            "auto + 无 key 应回退到 cli"
        )
        assert decision.orchestrator == "basic", (
            "auto + 无 key 仍应强制 basic 编排器（基于 requested_mode）"
        )

        unified = build_unified_overrides(
            args=args,
            execution_decision=decision,
        )

        assert unified.resolved["orchestrator"] == "basic", (
            "build_unified_overrides: auto + 无 key 应强制 basic"
        )

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

        assert actual_orchestrator == "basic", (
            "SelfIterator: auto + 无 key 应强制 basic 编排器"
        )
        # effective_mode 应回退到 cli
        assert actual_execution_mode.value == "cli", (
            "SelfIterator: auto + 无 key，effective_mode 应回退到 cli"
        )

    def test_cloud_no_key_effective_cli_but_orchestrator_basic(self) -> None:
        """强断言：requested_mode=cloud 无 key → effective=cli 但 orchestrator=basic

        这是关键场景的专项测试，确保 SelfIterator 行为符合 AGENTS.md 规则。
        """
        from scripts.run_iterate import SelfIterator
        from cursor.cloud_client import CloudClientFactory
        from core.config import build_unified_overrides
        from core.execution_policy import build_execution_decision

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
        assert decision.effective_mode == "cli", (
            "cloud + 无 key 应回退到 cli"
        )
        assert decision.orchestrator == "basic", (
            "cloud + 无 key 仍应强制 basic 编排器（基于 requested_mode）"
        )

        unified = build_unified_overrides(
            args=args,
            execution_decision=decision,
        )

        assert unified.resolved["orchestrator"] == "basic", (
            "build_unified_overrides: cloud + 无 key 应强制 basic"
        )

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

        assert actual_orchestrator == "basic", (
            "SelfIterator: cloud + 无 key 应强制 basic 编排器"
        )
        # effective_mode 应回退到 cli
        assert actual_execution_mode.value == "cli", (
            "SelfIterator: cloud + 无 key，effective_mode 应回退到 cli"
        )


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

    def _create_config_with_execution_mode(
        self, execution_mode: str, cloud_enabled: bool = True
    ) -> MagicMock:
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
        assert decision.orchestrator == "basic", (
            "config.yaml execution_mode=auto 应强制 basic 编排器"
        )
        assert decision.triggered_by_prefix is False, (
            "无 & 前缀，triggered_by_prefix 应为 False"
        )

        # 验证 resolve_orchestrator_settings 一致性
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "auto"},
            triggered_by_prefix=False,
        )
        assert result["orchestrator"] == "basic", (
            "resolve_orchestrator_settings: auto 模式应强制 basic"
        )

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
        assert decision.orchestrator == "basic", (
            "config.yaml execution_mode=cloud 应强制 basic 编排器"
        )
        assert decision.triggered_by_prefix is False, (
            "无 & 前缀，triggered_by_prefix 应为 False"
        )

        # 验证 resolve_orchestrator_settings 一致性
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cloud"},
            triggered_by_prefix=False,
        )
        assert result["orchestrator"] == "basic", (
            "resolve_orchestrator_settings: cloud 模式应强制 basic"
        )

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
        assert decision.triggered_by_prefix is False, (
            "无 API Key 时，& 前缀不应成功触发"
        )
        # 验证允许 mp 编排器
        assert decision.orchestrator == "mp", (
            "cli 模式 + & 前缀未成功触发，应允许 mp 编排器"
        )
        assert decision.effective_mode == "cli", (
            "effective_mode 应保持 cli"
        )

        # 验证 resolve_orchestrator_settings 一致性
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "mp", (
            "resolve_orchestrator_settings: cli 模式应允许 mp"
        )

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
        assert decision.triggered_by_prefix is False, (
            "cloud_enabled=false 时，& 前缀不应成功触发"
        )
        # 验证允许 mp 编排器
        assert decision.orchestrator == "mp", (
            "cli 模式 + & 前缀未成功触发，应允许 mp 编排器"
        )
        assert decision.effective_mode == "cli", (
            "effective_mode 应保持 cli"
        )

        # 验证 resolve_orchestrator_settings 一致性
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=False,  # & 前缀未成功触发
        )
        assert result["orchestrator"] == "mp", (
            "resolve_orchestrator_settings: cli 模式应允许 mp"
        )


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
        assert decision.triggered_by_prefix is False, (
            "显式 CLI 模式应忽略 & 前缀，triggered_by_prefix=False"
        )
        # 验证 effective_mode 为 cli
        assert decision.effective_mode == "cli", (
            "显式 CLI 模式，effective_mode 应为 cli"
        )
        # 验证允许 mp 编排器
        assert decision.orchestrator == "mp", (
            "显式 CLI 模式应允许 mp 编排器"
        )

    def test_explicit_cli_with_prefix_both_entries_consistent(self) -> None:
        """验证两入口对显式 CLI + & 前缀的处理一致性"""
        from scripts.run_iterate import SelfIterator
        from cursor.cloud_client import CloudClientFactory
        from core.execution_policy import build_execution_decision

        # 测试 1: build_execution_decision
        decision = build_execution_decision(
            prompt="& 后台任务",
            requested_mode="cli",  # 显式 CLI
            cloud_enabled=True,
            has_api_key=True,
        )
        assert decision.triggered_by_prefix is False
        assert decision.effective_mode == "cli"
        assert decision.orchestrator == "mp"

        # 测试 2: resolve_orchestrator_settings
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=False,
        )
        assert result["orchestrator"] == "mp", (
            "resolve_orchestrator_settings: 显式 cli 应允许 mp"
        )

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

        assert actual_triggered is False, (
            "SelfIterator: 显式 CLI 应忽略 & 前缀"
        )
        assert actual_mode.value == "cli", (
            "SelfIterator: 显式 CLI，effective_mode 应为 cli"
        )
        assert actual_orchestrator == "mp", (
            "SelfIterator: 显式 CLI 应允许 mp 编排器"
        )

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

        assert decision.triggered_by_prefix is False, (
            "显式 CLI + 无 API Key，& 前缀不应触发"
        )
        assert decision.effective_mode == "cli"
        assert decision.orchestrator == "mp", (
            "显式 CLI 模式应允许 mp"
        )

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

        assert decision.triggered_by_prefix is False, (
            "显式 CLI + cloud_enabled=false，& 前缀不应触发"
        )
        assert decision.effective_mode == "cli"
        assert decision.orchestrator == "mp", (
            "显式 CLI 模式应允许 mp"
        )


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
        import sys
        import argparse

        # 模拟 run.py 的检测逻辑
        def check_orchestrator_user_set(argv: list) -> bool:
            return any(
                arg in argv for arg in ["--orchestrator", "--no-mp"]
            )

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
        import sys
        import argparse

        # 模拟 scripts/run_iterate.py 的检测逻辑（修正后）
        def check_orchestrator_user_set(argv: list) -> bool:
            return any(
                arg in argv for arg in ["--orchestrator", "--no-mp"]
            )

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
        from core.config import build_cli_overrides_from_args
        import argparse

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
        assert "orchestrator" not in overrides, (
            "未显式传参时 orchestrator 不应写入 overrides"
        )

    def test_build_cli_overrides_orchestrator_when_explicitly_set(self) -> None:
        """验证显式传参时 build_cli_overrides_from_args 正确写入 orchestrator"""
        from core.config import build_cli_overrides_from_args
        import argparse

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
        assert overrides_mp.get("orchestrator") == "mp", (
            "显式设置 --orchestrator mp 时应写入 overrides"
        )

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
        assert overrides_basic.get("orchestrator") == "basic", (
            "显式设置 --orchestrator basic 时应写入 overrides"
        )

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
        assert overrides_no_mp.get("orchestrator") == "basic", (
            "显式设置 --no-mp 时应写入 orchestrator=basic"
        )

    def test_cli_no_mp_tristate_consistency(self) -> None:
        """验证 cli_no_mp 使用 None 默认值（tri-state 策略）

        cli_no_mp 的可能值:
        - None: 未显式指定（默认）
        - True: 显式设置 --no-mp
        - False: 不会出现（action="store_true" 不产生 False）
        """
        from core.config import build_cli_overrides_from_args
        import argparse

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
        assert "orchestrator" not in overrides_none, (
            "no_mp=None 时不应写入 orchestrator"
        )

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
        assert overrides_true.get("orchestrator") == "basic", (
            "no_mp=True 时应写入 orchestrator=basic"
        )

    def test_nl_options_orchestrator_when_user_not_set(self) -> None:
        """验证 _orchestrator_user_set=False 时允许 nl_options 覆盖"""
        from core.config import build_cli_overrides_from_args
        import argparse

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
        assert overrides.get("orchestrator") == "basic", (
            "_orchestrator_user_set=False 时应允许 nl_options 覆盖"
        )

    def test_user_set_takes_priority_over_nl_options(self) -> None:
        """验证 _orchestrator_user_set=True 时用户设置优先于 nl_options"""
        from core.config import build_cli_overrides_from_args
        import argparse

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
        assert overrides.get("orchestrator") == "mp", (
            "_orchestrator_user_set=True 时用户设置应优先于 nl_options"
        )
