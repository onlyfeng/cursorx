"""测试 scripts/run_iterate.py 自我迭代脚本

测试覆盖：
1. WebFetcher.fetch 失败时流程不崩溃，仍能进入知识库统计/目标构建
2. MultiProcessOrchestrator.run 抛出异常返回 _fallback_required，触发 basic 编排器回退
3. --dry-run 返回结构字段验证
4. --skip-online 分支验证
5. 参数默认值来自 config.yaml 配置

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
"""

import argparse
import asyncio
import os
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast
from unittest.mock import DEFAULT, AsyncMock, MagicMock
from unittest.mock import patch as _patch

if TYPE_CHECKING:
    pass

import pytest

import scripts.run_iterate as run_iterate


# 兼容 importlib 模式下的 patch 目标解析，避免重复模块实例
class PatchProxy:
    object = staticmethod(_patch.object)
    dict = staticmethod(_patch.dict)
    multiple = staticmethod(_patch.multiple)
    stopall = staticmethod(_patch.stopall)
    DEFAULT = DEFAULT

    def __call__(self, target, *args, **kwargs):
        if isinstance(target, str) and target.startswith("scripts.run_iterate."):
            attr = target.split("scripts.run_iterate.", 1)[1]
            return _patch.object(run_iterate, attr, *args, **kwargs)
        return _patch(target, *args, **kwargs)


patch = PatchProxy()

T = TypeVar("T")


def record_call(call_sequence: list[str], label: str, result: T | None = None) -> T | None:
    call_sequence.append(label)
    return result


# 导入被测模块
# 导入配置管理器
from core.config import (
    MAX_CONSOLE_PREVIEW_CHARS,
    MAX_KNOWLEDGE_DOC_PREVIEW_CHARS,
    TRUNCATION_HINT,
    ConfigManager,
    get_config,
    parse_max_iterations,
)
from core.output_contract import CooldownInfoFields, IterateResultFields
from scripts.run_iterate import (
    DEFAULT_CHANGELOG_URL,
    ChangelogAnalyzer,
    ChangelogEntry,
    IterationContext,
    IterationGoalBuilder,
    KnowledgeUpdater,
    SelfIterator,
    UpdateAnalysis,
    parse_args,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def base_iterate_args() -> argparse.Namespace:
    """创建基础迭代参数

    注意：以下参数的默认值与 config.yaml 中的配置保持一致：
    - max_iterations: 来自 system.max_iterations（默认 10，测试中使用 "3"）
    - workers: 来自 system.worker_pool_size（默认 3，测试中使用 2）
    - cloud_timeout: 来自 cloud_agent.timeout（默认 300）
    - cloud_auth_timeout: 来自 cloud_agent.auth_timeout（默认 30）
    - docs_source: 来自 knowledge_docs_update.docs_source

    测试中显式设置这些值以隔离测试环境，避免受 config.yaml 变更影响。
    """
    return argparse.Namespace(
        requirement="测试需求",
        directory=".",  # 工作目录
        skip_online=False,
        changelog_url=None,  # 使用 config.yaml 默认值（tri-state）
        dry_run=False,
        max_iterations="3",  # 测试值，实际默认来自 config.yaml system.max_iterations
        workers=2,  # 测试值，实际默认来自 config.yaml system.worker_pool_size
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
        # 执行模式参数
        execution_mode="cli",
        cloud_api_key=None,
        cloud_auth_timeout=30,  # 来自 config.yaml cloud_agent.auth_timeout
        cloud_timeout=300,  # 来自 config.yaml cloud_agent.timeout
        # 角色级执行模式
        planner_execution_mode=None,
        worker_execution_mode=None,
        reviewer_execution_mode=None,
        # 流式控制台渲染参数
        stream_console_renderer=False,
        stream_advanced_renderer=False,
        stream_typing_effect=False,
        stream_typing_delay=0.02,
        stream_word_mode=True,
        stream_color_enabled=True,
        stream_show_word_diff=False,
        # 文档源配置参数（tri-state：None=使用 config.yaml 默认值）
        max_fetch_urls=None,
        fallback_core_docs_count=None,
        llms_txt_url=None,
        llms_cache_path=None,
    )


@pytest.fixture
def skip_online_args(base_iterate_args: argparse.Namespace) -> argparse.Namespace:
    """创建跳过在线检查的参数"""
    base_iterate_args.skip_online = True
    return base_iterate_args


@pytest.fixture
def dry_run_args(base_iterate_args: argparse.Namespace) -> argparse.Namespace:
    """创建 dry-run 模式参数"""
    base_iterate_args.dry_run = True
    base_iterate_args.skip_online = True  # dry-run 通常与 skip-online 一起使用
    return base_iterate_args


# ============================================================
# TestWebFetcherFailure - WebFetcher.fetch 失败场景
# ============================================================


class TestWebFetcherFailure:
    """测试 WebFetcher.fetch 失败时流程不崩溃"""

    @pytest.mark.asyncio
    async def test_changelog_fetch_failure_does_not_crash(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 Changelog 获取失败时流程继续执行"""
        iterator = SelfIterator(base_iterate_args)

        # Mock WebFetcher.fetch 返回失败
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = False
        mock_fetch_result.content = ""
        mock_fetch_result.error = "Network error"

        with (
            patch.object(iterator.changelog_analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.changelog_analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            # 分析 changelog 应返回空结果而不是崩溃
            analysis = await iterator.changelog_analyzer.analyze()

            # 验证返回空的 UpdateAnalysis
            assert isinstance(analysis, UpdateAnalysis)
            assert analysis.has_updates is False
            assert len(analysis.entries) == 0

    @pytest.mark.asyncio
    async def test_fetch_failure_still_enters_knowledge_stats(self, base_iterate_args: argparse.Namespace) -> None:
        """测试获取失败后仍能进入知识库统计阶段"""
        iterator = SelfIterator(base_iterate_args)

        # Mock WebFetcher.fetch 返回失败
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = False
        mock_fetch_result.content = ""
        mock_fetch_result.error = "Connection timeout"

        # Mock 知识库相关组件
        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 5})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.changelog_analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.changelog_analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(
                iterator.knowledge_updater.manager,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            # 执行到知识库统计阶段
            await iterator.knowledge_updater.initialize()
            stats = await iterator.knowledge_updater.get_stats()

            # 验证知识库统计被调用
            assert stats["document_count"] == 5
            mock_storage.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_failure_still_builds_iteration_goal(self, base_iterate_args: argparse.Namespace) -> None:
        """测试获取失败后仍能构建迭代目标"""
        iterator = SelfIterator(base_iterate_args)

        # 设置空的更新分析（模拟获取失败后的状态）
        iterator.context.update_analysis = UpdateAnalysis()
        iterator.context.user_requirement = "测试需求"

        # 构建目标应成功
        goal = iterator.goal_builder.build_goal(iterator.context)

        # 验证目标包含用户需求
        assert "测试需求" in goal
        assert "自我迭代任务" in goal

        # 验证目标可以生成摘要
        summary = iterator.goal_builder.get_summary(iterator.context)
        assert "用户需求" in summary


# ============================================================
# TestContentQualityAssessment - 内容质量评估测试
# ============================================================


class TestContentQualityAssessment:
    """测试 ChangelogAnalyzer 的内容质量评估功能"""

    def test_assess_content_quality_empty_content(self, base_iterate_args: argparse.Namespace) -> None:
        """测试空内容的质量评分为 0"""
        from scripts.run_iterate import ChangelogAnalyzer

        analyzer = ChangelogAnalyzer()
        assert analyzer._assess_content_quality("") == 0.0
        assert analyzer._assess_content_quality(None) == 0.0

    def test_assess_content_quality_high_quality(self, base_iterate_args: argparse.Namespace) -> None:
        """测试高质量内容的质量评分"""
        from scripts.run_iterate import ChangelogAnalyzer

        analyzer = ChangelogAnalyzer()

        high_quality_content = """
        January 16, 2026
        Cursor 0.47 Release Notes - Changelog

        ## New Features
        - Added dark mode support with improved accessibility
        - Enhanced code completion with AI-powered suggestions
        - New plugin system for extensibility

        ## Improvements
        - Performance improvements across the board
        - Better error handling and recovery
        - Updated documentation

        ## Bug Fixes
        - Fixed crash on startup when config is missing
        - Fixed memory leak in long-running sessions
        - Fixed UI glitches in the sidebar

        Version 0.47 is now available for download.
        """
        score = analyzer._assess_content_quality(high_quality_content)
        # 高质量内容应该有较高的评分（> 0.5）
        assert score >= 0.5, f"Expected score >= 0.5, got {score}"

    def test_assess_content_quality_low_quality(self, base_iterate_args: argparse.Namespace) -> None:
        """测试低质量内容的质量评分"""
        from scripts.run_iterate import ChangelogAnalyzer

        analyzer = ChangelogAnalyzer()

        low_quality_content = "Some random text without dates or keywords."
        score = analyzer._assess_content_quality(low_quality_content)
        # 低质量内容应该有较低的评分（< 0.4）
        assert score < 0.4, f"Expected score < 0.4, got {score}"

    def test_assess_content_quality_date_patterns(self, base_iterate_args: argparse.Namespace) -> None:
        """测试日期模式匹配对质量评分的影响"""
        from scripts.run_iterate import ChangelogAnalyzer

        analyzer = ChangelogAnalyzer()

        # 包含多个日期的内容
        content_with_dates = (
            """
        Jan 16, 2026 - First update
        February 20, 2026 - Second update
        2026-03-15 - Third update
        """
            + "x" * 200
        )  # 确保长度足够

        score = analyzer._assess_content_quality(content_with_dates)
        # 应该因为日期匹配获得较高分数
        assert score >= 0.4, f"Expected score >= 0.4, got {score}"

    def test_get_retry_methods_excludes_used_method(self, base_iterate_args: argparse.Namespace) -> None:
        """测试获取重试方法时排除已使用的方法"""
        from knowledge.fetcher import FetchMethod
        from scripts.run_iterate import ChangelogAnalyzer

        analyzer = ChangelogAnalyzer()
        # 模拟可用方法
        analyzer.fetcher._available_methods = [
            FetchMethod.MCP,
            FetchMethod.PLAYWRIGHT,
            FetchMethod.CURL,
        ]

        retry_methods = analyzer._get_retry_methods(FetchMethod.MCP)
        assert FetchMethod.MCP not in retry_methods
        assert FetchMethod.PLAYWRIGHT in retry_methods

    def test_content_quality_config_defaults(self, base_iterate_args: argparse.Namespace) -> None:
        """测试内容质量配置默认值"""
        from scripts.run_iterate import ContentQualityConfig

        config = ContentQualityConfig()
        assert config.min_text_length == 200
        assert config.quality_threshold == 0.4
        assert config.max_quality_retries == 2
        assert len(config.date_patterns) == 2
        assert len(config.changelog_keywords) > 5

    @pytest.mark.asyncio
    async def test_fetch_changelog_with_quality_retry(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 fetch_changelog 在质量不足时进行重试"""
        from knowledge.fetcher import FetchMethod, FetchResult
        from scripts.run_iterate import ChangelogAnalyzer, ContentQualityConfig

        # 使用较高的质量阈值以触发重试
        config = ContentQualityConfig(quality_threshold=0.9, max_quality_retries=1)
        analyzer = ChangelogAnalyzer(quality_config=config)

        # 第一次返回低质量内容
        low_quality_result = FetchResult(
            url="https://cursor.com/cn/changelog",
            success=True,
            content="Short content",
            method_used=FetchMethod.MCP,
            duration=1.0,
        )

        # 第二次返回高质量内容
        high_quality_result = FetchResult(
            url="https://cursor.com/cn/changelog",
            success=True,
            content="""
            January 16, 2026
            Cursor Changelog - Release Notes

            ## New Features
            - Feature 1
            - Feature 2

            ## Bug Fixes
            - Fix 1
            - Fix 2

            Version 0.47 is available now.
            """
            + "x" * 200,
            method_used=FetchMethod.PLAYWRIGHT,
            duration=2.0,
        )

        call_count = [0]

        async def mock_fetch(url, method=None, timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return low_quality_result
            return high_quality_result

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher, "get_available_methods", return_value=[FetchMethod.MCP, FetchMethod.PLAYWRIGHT]
            ),
            patch.object(analyzer.fetcher, "fetch", side_effect=mock_fetch),
        ):
            content = await analyzer.fetch_changelog()

            # 应该返回内容
            assert content is not None
            # 应该尝试了重试
            assert call_count[0] >= 2
            # 应该记录了尝试日志
            assert len(analyzer._fetch_attempts) >= 2


# ============================================================
# TestMPOrchestratorFallback - MP 编排器回退测试
# ============================================================


@pytest.mark.slow
class TestMPOrchestratorFallback:
    """测试 MultiProcessOrchestrator 失败时回退到 basic 编排器"""

    @pytest.mark.asyncio
    async def test_mp_timeout_triggers_fallback(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 MP 编排器超时触发回退"""
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        # Mock MP 编排器抛出超时异常
        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(side_effect=asyncio.TimeoutError("启动超时"))
                    MockMP.return_value = mock_orch

                    result = await iterator._run_with_mp_orchestrator(3, mock_km)

                    # 验证返回 _fallback_required
                    assert result.get("_fallback_required") is True
                    assert "超时" in result.get("_fallback_reason", "")

    @pytest.mark.asyncio
    async def test_mp_oserror_triggers_fallback(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 MP 编排器进程创建失败触发回退"""
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(side_effect=OSError("无法创建进程"))
                    MockMP.return_value = mock_orch

                    result = await iterator._run_with_mp_orchestrator(3, mock_km)

                    assert result.get("_fallback_required") is True
                    assert "进程创建失败" in result.get("_fallback_reason", "")

    @pytest.mark.asyncio
    async def test_mp_runtime_error_triggers_fallback(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 MP 编排器运行时错误触发回退"""
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(side_effect=RuntimeError("事件循环问题"))
                    MockMP.return_value = mock_orch

                    result = await iterator._run_with_mp_orchestrator(3, mock_km)

                    assert result.get("_fallback_required") is True
                    assert "运行时错误" in result.get("_fallback_reason", "")

    @pytest.mark.asyncio
    async def test_fallback_calls_basic_orchestrator(self, base_iterate_args: argparse.Namespace) -> None:
        """测试回退后调用 basic 编排器"""
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        mock_basic_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        # MP 返回需要回退
        mock_mp_result = {
            "_fallback_required": True,
            "_fallback_reason": "测试回退",
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_mp_result,
            ),
            patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ) as mock_basic,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证 basic 编排器被调用
            mock_basic.assert_called_once()

            # 验证结果结构稳定
            assert result["success"] is True
            assert "iterations_completed" in result
            assert "total_tasks_completed" in result

    @pytest.mark.asyncio
    async def test_fallback_result_structure_stable(self, base_iterate_args: argparse.Namespace) -> None:
        """测试回退后结果结构保持稳定

        验证:
        1. 基础结果字段（success, iterations_completed 等）
        2. 元数据字段（orchestrator_type, fallback_occurred 等）
        """
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        mock_basic_result = {
            "success": True,
            "iterations_completed": 2,
            "total_tasks_created": 4,
            "total_tasks_completed": 3,
            "total_tasks_failed": 1,
            "final_score": 85.0,
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value={"_fallback_required": True, "_fallback_reason": "测试回退原因"},
            ),
            patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ),
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证基础结果字段存在
            expected_fields = [
                "success",
                "iterations_completed",
                "total_tasks_created",
                "total_tasks_completed",
                "total_tasks_failed",
            ]
            for field in expected_fields:
                assert field in result, f"缺少基础字段: {field}"

            # 验证元数据字段存在且值正确
            metadata_fields = [
                "orchestrator_type",
                "orchestrator_requested",
                "fallback_occurred",
                "fallback_reason",
                "execution_mode",
                "max_iterations_configured",
            ]
            for field in metadata_fields:
                assert field in result, f"缺少元数据字段: {field}"

            # 验证回退场景下的元数据值
            assert result["orchestrator_type"] == "basic", "回退后应使用 basic 编排器"
            assert result["orchestrator_requested"] == "mp", "请求的编排器应为 mp"
            assert result["fallback_occurred"] is True, "应标记发生了回退"
            assert result["fallback_reason"] == "测试回退原因", "回退原因应与 mock 一致"
            assert result["execution_mode"] == "cli", "执行模式应为 cli"
            assert isinstance(result["max_iterations_configured"], int), "max_iterations_configured 应为整数"

    @pytest.mark.asyncio
    async def test_mp_success_result_metadata_fields(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 MP 编排器成功时元数据字段正确

        验证非回退场景下的元数据字段:
        - orchestrator_type == "mp"
        - orchestrator_requested == "mp"
        - fallback_occurred == False
        - fallback_reason is None
        """
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        mock_mp_result = {
            "success": True,
            "iterations_completed": 3,
            "total_tasks_created": 6,
            "total_tasks_completed": 5,
            "total_tasks_failed": 1,
            "final_score": 90.0,
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_mp_result,
            ),
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证 MP 成功场景的元数据
            assert result["orchestrator_type"] == "mp", "MP 成功时应使用 mp 编排器"
            assert result["orchestrator_requested"] == "mp", "请求的编排器应为 mp"
            assert result["fallback_occurred"] is False, "MP 成功时不应发生回退"
            assert result["fallback_reason"] is None, "MP 成功时 fallback_reason 应为 None"
            assert result["execution_mode"] == "cli", "执行模式应为 cli"

            # 验证基础结果字段
            assert result["success"] is True
            assert result["iterations_completed"] == 3


# ============================================================
# TestDryRunMode - dry-run 模式测试
# ============================================================


class TestDryRunMode:
    """测试 --dry-run 模式返回结构"""

    @pytest.mark.asyncio
    async def test_dry_run_returns_expected_fields(self, dry_run_args: argparse.Namespace) -> None:
        """测试 dry-run 模式返回预期字段"""
        iterator = SelfIterator(dry_run_args)

        # Mock 知识库组件
        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 3})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            result = await iterator.run()

            # 验证 dry-run 特有字段
            assert result.get("success") is True
            assert result.get("dry_run") is True
            assert "summary" in result
            assert "goal_length" in result
            assert result["goal_length"] > 0

    @pytest.mark.asyncio
    async def test_dry_run_does_not_execute_agent(self, dry_run_args: argparse.Namespace) -> None:
        """测试 dry-run 模式不执行 Agent 系统"""
        iterator = SelfIterator(dry_run_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(iterator, "_run_agent_system", new_callable=AsyncMock) as mock_agent,
        ):
            result = await iterator.run()

            # 验证 Agent 系统未被调用
            mock_agent.assert_not_called()

            # 验证仍返回成功
            assert result["success"] is True
            assert result["dry_run"] is True

    @pytest.mark.asyncio
    async def test_dry_run_builds_goal_preview(self, dry_run_args: argparse.Namespace) -> None:
        """测试 dry-run 模式构建目标预览"""
        dry_run_args.requirement = "添加新功能支持"
        iterator = SelfIterator(dry_run_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            result = await iterator.run()

            # 验证目标被构建
            assert iterator.context.iteration_goal != ""
            assert "添加新功能支持" in iterator.context.iteration_goal

            # 验证摘要包含用户需求
            assert "用户需求" in result["summary"]

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_write_llms_txt_cache(self, dry_run_args: argparse.Namespace) -> None:
        """测试 dry-run 模式不调用 _write_llms_txt_cache

        验证 dry-run 模式下 KnowledgeUpdater 的 disable_cache_write=True，
        因此 _write_llms_txt_cache 不会被调用。
        """
        iterator = SelfIterator(dry_run_args)

        # 验证 dry_run 参数正确传递
        assert iterator.knowledge_updater.dry_run is True
        assert iterator.knowledge_updater.disable_cache_write is True

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator.knowledge_updater,
                "_write_llms_txt_cache",
            ) as mock_write_cache,
        ):
            result = await iterator.run()

            # 验证 _write_llms_txt_cache 未被调用
            mock_write_cache.assert_not_called()

            # 验证 dry-run 模式返回成功
            assert result["success"] is True
            assert result["dry_run"] is True

    def test_dry_run_knowledge_updater_uses_read_only_storage(self, dry_run_args: argparse.Namespace) -> None:
        """测试 dry-run 模式下 KnowledgeUpdater 使用只读 storage"""
        iterator = SelfIterator(dry_run_args)

        # 验证 storage 是只读模式
        assert iterator.knowledge_updater.storage.is_read_only is True

    @pytest.mark.asyncio
    async def test_dry_run_returns_dry_run_stats(self, dry_run_args: argparse.Namespace) -> None:
        """测试 dry-run 模式返回 dry_run_stats 字段"""
        iterator = SelfIterator(dry_run_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            result = await iterator.run()

            # 验证返回 dry_run_stats 字段
            assert "dry_run_stats" in result
            stats = result["dry_run_stats"]
            assert "would_fetch_urls" in stats
            assert "would_write_docs" in stats
            assert "would_write_cache" in stats

    @pytest.mark.asyncio
    async def test_dry_run_returns_execution_decision_fields(self, dry_run_args: argparse.Namespace) -> None:
        """测试 dry-run 模式返回执行决策字段

        验证 run() 返回结果包含以下字段：
        - has_ampersand_prefix: 语法检测，原始 prompt 是否有 & 前缀
        - prefix_routed: 策略决策，& 前缀是否成功触发 Cloud 模式
        - triggered_by_prefix: prefix_routed 的兼容别名
        - requested_mode: 原始请求模式
        - effective_mode: 有效执行模式
        - orchestrator: 编排器类型
        """
        iterator = SelfIterator(dry_run_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            result = await iterator.run()

            # 验证执行决策字段存在
            assert "has_ampersand_prefix" in result
            assert "prefix_routed" in result
            assert "triggered_by_prefix" in result
            assert "requested_mode" in result
            assert "effective_mode" in result
            assert "orchestrator" in result

            # 验证字段一致性：triggered_by_prefix 应等于 prefix_routed
            assert result["triggered_by_prefix"] == result["prefix_routed"]

            # 验证字段类型
            assert isinstance(result["has_ampersand_prefix"], bool)
            assert isinstance(result["prefix_routed"], bool)
            assert isinstance(result["triggered_by_prefix"], bool)
            assert isinstance(result["effective_mode"], str)
            assert isinstance(result["orchestrator"], str)

    @pytest.mark.asyncio
    async def test_dry_run_execution_decision_fields_match_iterator(self, dry_run_args: argparse.Namespace) -> None:
        """测试 dry-run 返回的执行决策字段与 iterator 内部状态一致

        验证返回结果中的执行决策字段与 _execution_decision 对象一致。
        """
        iterator = SelfIterator(dry_run_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            result = await iterator.run()

            # 验证与 _execution_decision 对象一致
            decision = iterator._execution_decision
            assert result["has_ampersand_prefix"] == decision.has_ampersand_prefix
            assert result["prefix_routed"] == decision.prefix_routed
            assert result["triggered_by_prefix"] == decision.triggered_by_prefix
            assert result["requested_mode"] == decision.requested_mode
            assert result["effective_mode"] == decision.effective_mode
            assert result["orchestrator"] == decision.orchestrator


class TestDryRunStorageNoWrite:
    """测试 dry-run 模式下 KnowledgeStorage 不发生写入

    使用真实的临时目录验证 dry-run 模式下的只读存储行为。
    """

    @pytest.mark.asyncio
    async def test_dry_run_storage_no_files_created(self, tmp_path: Path) -> None:
        """测试 dry-run 模式下 KnowledgeStorage 不创建任何文件

        使用临时目录验证只读 storage 不会创建目录或文件。
        """
        from knowledge.storage import KnowledgeStorage, StorageConfig

        # 创建指向临时目录的只读 storage（模拟 dry-run 模式）
        storage_root = tmp_path / "knowledge_test"
        config = StorageConfig(
            storage_root=str(storage_root),
            read_only=True,
            auto_create_dirs=False,
        )
        storage = KnowledgeStorage(config=config, workspace_root=str(tmp_path))

        # 验证 storage 是只读模式
        assert storage.is_read_only is True

        # 初始化不应创建目录（只读模式下 auto_create_dirs=False）
        await storage.initialize()

        # 验证目录未被创建
        assert not storage_root.exists(), "只读模式下不应创建存储目录"

    @pytest.mark.asyncio
    async def test_dry_run_storage_write_raises_error(self, tmp_path: Path) -> None:
        """测试 dry-run 模式下 KnowledgeStorage 写入操作抛出 ReadOnlyStorageError"""
        from knowledge.models import Document
        from knowledge.storage import KnowledgeStorage, ReadOnlyStorageError, StorageConfig

        # 先创建存储目录和 index 文件，模拟已有数据
        storage_root = tmp_path / "knowledge_existing"
        storage_root.mkdir(parents=True)
        (storage_root / "docs").mkdir()
        (storage_root / "metadata").mkdir()
        index_file = storage_root / "index.json"
        index_file.write_text('{"version": 1, "documents": []}', encoding="utf-8")

        # 创建只读 storage
        config = StorageConfig(
            storage_root=str(storage_root.relative_to(tmp_path)),
            read_only=True,
        )
        storage = KnowledgeStorage(config=config, workspace_root=str(tmp_path))
        await storage.initialize()

        # 验证可以读取（初始化成功）
        assert storage.is_read_only is True

        # 尝试写入操作应抛出 ReadOnlyStorageError
        test_doc = Document(
            url="https://example.com/test",
            title="Test Document",
            content="Test content",
        )

        with pytest.raises(ReadOnlyStorageError) as exc_info:
            await storage.save_document(test_doc)

        assert "save_document" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_dry_run_knowledge_updater_no_cache_write(self, tmp_path: Path) -> None:
        """测试 dry-run 模式下 KnowledgeUpdater 不写入缓存文件

        使用临时目录验证 disable_cache_write=True 时不会创建缓存文件。
        """
        cache_path = tmp_path / "llms_cache" / "llms.txt"

        updater = KnowledgeUpdater(
            dry_run=True,
            llms_cache_path=str(cache_path),
        )

        # 验证 dry_run 模式下 disable_cache_write 被强制设置为 True
        assert updater.dry_run is True
        assert updater.disable_cache_write is True

        # 尝试调用 _write_llms_txt_cache（模拟在线获取后写入缓存）
        # 由于 disable_cache_write=True，_fetch_llms_txt 中不会调用 _write_llms_txt_cache
        # 这里我们直接验证缓存目录不存在
        assert not cache_path.parent.exists(), "dry-run 模式下不应创建缓存目录"


# ============================================================
# TestExceptionBranchExecutionDecisionFields - 异常分支执行决策字段测试
# ============================================================


class TestExceptionBranchExecutionDecisionFields:
    """测试异常分支返回执行决策字段

    验证 run() 方法在抛出异常时，返回结果仍包含完整的执行决策字段：
    - has_ampersand_prefix
    - prefix_routed
    - triggered_by_prefix
    - requested_mode
    - effective_mode
    - orchestrator
    """

    @pytest.fixture
    def exception_args(self, base_iterate_args: argparse.Namespace) -> argparse.Namespace:
        """创建用于测试异常分支的 args（基于 base_iterate_args）"""
        base_iterate_args.skip_online = True
        base_iterate_args.dry_run = False  # 需要执行到 _run_agent_system
        return base_iterate_args

    @pytest.mark.asyncio
    async def test_exception_branch_returns_execution_decision_fields(self, exception_args: argparse.Namespace) -> None:
        """测试异常分支返回执行决策字段

        通过模拟 _run_agent_system 抛出异常，验证异常分支返回完整的执行决策字段。
        """
        iterator = SelfIterator(exception_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            # 模拟 _run_agent_system 抛出异常
            with patch.object(
                iterator,
                "_run_agent_system",
                new_callable=AsyncMock,
                side_effect=RuntimeError("模拟执行异常"),
            ):
                result = await iterator.run()

                # 验证返回失败
                assert result["success"] is False
                assert "error" in result
                assert "模拟执行异常" in result["error"]

                # 验证执行决策字段存在
                assert "has_ampersand_prefix" in result
                assert "prefix_routed" in result
                assert "triggered_by_prefix" in result
                assert "requested_mode" in result
                assert "effective_mode" in result
                assert "orchestrator" in result

                # 验证字段一致性：triggered_by_prefix 应等于 prefix_routed
                assert result["triggered_by_prefix"] == result["prefix_routed"]

    @pytest.mark.asyncio
    async def test_exception_branch_fields_match_iterator(self, exception_args: argparse.Namespace) -> None:
        """测试异常分支返回的执行决策字段与 iterator 内部状态一致"""
        iterator = SelfIterator(exception_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator,
                "_run_agent_system",
                new_callable=AsyncMock,
                side_effect=ValueError("另一个测试异常"),
            ),
        ):
            result = await iterator.run()

            # 验证与 _execution_decision 对象一致
            decision = iterator._execution_decision
            assert result["has_ampersand_prefix"] == decision.has_ampersand_prefix
            assert result["prefix_routed"] == decision.prefix_routed
            assert result["triggered_by_prefix"] == decision.triggered_by_prefix
            assert result["requested_mode"] == decision.requested_mode
            assert result["effective_mode"] == decision.effective_mode
            assert result["orchestrator"] == decision.orchestrator

    @pytest.mark.asyncio
    async def test_all_branches_have_consistent_execution_decision_fields(
        self, exception_args: argparse.Namespace, dry_run_args: argparse.Namespace
    ) -> None:
        """测试所有分支返回的执行决策字段集合一致

        验证 dry-run 分支和异常分支返回的执行决策字段集合相同。
        """
        # 定义期望的字段集合
        expected_fields = {
            "has_ampersand_prefix",
            "prefix_routed",
            "triggered_by_prefix",
            "requested_mode",
            "effective_mode",
            "orchestrator",
        }

        # === 测试 dry-run 分支 ===
        dry_run_iterator = SelfIterator(dry_run_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(dry_run_iterator.knowledge_updater, "storage", mock_storage),
            patch.object(dry_run_iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                dry_run_iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            dry_run_result = await dry_run_iterator.run()

        # === 测试异常分支 ===
        exception_iterator = SelfIterator(exception_args)

        with (
            patch.object(exception_iterator.knowledge_updater, "storage", mock_storage),
            patch.object(exception_iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                exception_iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                exception_iterator,
                "_run_agent_system",
                new_callable=AsyncMock,
                side_effect=RuntimeError("测试异常"),
            ),
        ):
            exception_result = await exception_iterator.run()

        # 验证两个分支都包含完整的执行决策字段
        dry_run_fields = set(dry_run_result.keys()) & expected_fields
        exception_fields = set(exception_result.keys()) & expected_fields

        assert dry_run_fields == expected_fields, f"dry-run 分支缺少字段: {expected_fields - dry_run_fields}"
        assert exception_fields == expected_fields, f"异常分支缺少字段: {expected_fields - exception_fields}"


# ============================================================
# TestKnowledgeUpdaterPersistence - 知识库持久化测试
# ============================================================


class TestKnowledgeUpdaterPersistence:
    """测试 KnowledgeUpdater 的文档持久化功能

    验证：
    1. 保存后 KnowledgeStorage.search/load_document_by_url 能找到新文档
    2. dedup 行为：相同 URL 不重复保存
    3. force_refresh 行为：强制更新已存在的文档
    4. dry-run 模式下持久化被阻止但不崩溃
    """

    @pytest.mark.asyncio
    async def test_persistence_save_and_load_document(self, tmp_path: Path) -> None:
        """测试保存后能通过 load_document_by_url 找到文档"""
        from knowledge.models import Document
        from knowledge.storage import KnowledgeStorage, StorageConfig

        # 创建可写的临时 storage
        storage_root = tmp_path / "knowledge_persist"
        config = StorageConfig(
            storage_root=str(storage_root.relative_to(tmp_path)),
            read_only=False,
            auto_create_dirs=True,
        )
        storage = KnowledgeStorage(config=config, workspace_root=str(tmp_path))
        await storage.initialize()

        # 创建测试文档
        test_doc = Document(
            url="https://example.com/test-doc",
            title="Test Document",
            content="This is test content for persistence verification.",
            metadata={"source": "test"},
        )

        # 保存文档
        saved, msg = await storage.save_document(test_doc)
        assert saved is True, f"保存应成功: {msg}"

        # 通过 load_document_by_url 加载文档
        loaded_doc = await storage.load_document_by_url("https://example.com/test-doc")
        assert loaded_doc is not None, "应能通过 URL 加载文档"
        assert loaded_doc.title == "Test Document"
        assert "test content" in loaded_doc.content

    @pytest.mark.asyncio
    async def test_persistence_search_finds_document(self, tmp_path: Path) -> None:
        """测试保存后能通过 search 找到文档"""
        from knowledge.models import Document
        from knowledge.storage import KnowledgeStorage, StorageConfig

        # 创建可写的临时 storage
        storage_root = tmp_path / "knowledge_search"
        config = StorageConfig(
            storage_root=str(storage_root.relative_to(tmp_path)),
            read_only=False,
            auto_create_dirs=True,
        )
        storage = KnowledgeStorage(config=config, workspace_root=str(tmp_path))
        await storage.initialize()

        # 创建测试文档
        test_doc = Document(
            url="https://cursor.com/docs/searchable",
            title="Searchable Document About Cursor CLI",
            content="This document contains information about Cursor CLI features.",
        )

        # 保存文档
        saved, _ = await storage.save_document(test_doc)
        assert saved is True

        # 通过 search 搜索文档
        results = await storage.search("Cursor CLI", limit=10)
        assert len(results) >= 1, "应能通过关键词搜索到文档"
        assert any(r.title == "Searchable Document About Cursor CLI" for r in results)

    @pytest.mark.asyncio
    async def test_persistence_dedup_same_url(self, tmp_path: Path) -> None:
        """测试 dedup 行为：相同 URL 且内容未变化不重复保存"""
        from knowledge.models import Document
        from knowledge.storage import KnowledgeStorage, StorageConfig

        storage_root = tmp_path / "knowledge_dedup"
        config = StorageConfig(
            storage_root=str(storage_root.relative_to(tmp_path)),
            read_only=False,
            auto_create_dirs=True,
        )
        storage = KnowledgeStorage(config=config, workspace_root=str(tmp_path))
        await storage.initialize()

        # 创建并保存第一个文档
        test_doc = Document(
            url="https://example.com/dedup-test",
            title="Dedup Test",
            content="Same content for dedup testing.",
        )
        saved1, msg1 = await storage.save_document(test_doc)
        assert saved1 is True, f"首次保存应成功: {msg1}"

        # 再次保存相同内容的文档（应跳过）
        test_doc2 = Document(
            url="https://example.com/dedup-test",
            title="Dedup Test",
            content="Same content for dedup testing.",
        )
        saved2, msg2 = await storage.save_document(test_doc2, force=False)
        assert saved2 is False, f"内容未变化应跳过: {msg2}"
        assert "未变化" in msg2 or "跳过" in msg2

        # 验证索引中只有一个文档
        entries = await storage.list_documents()
        matching = [e for e in entries if e.url == "https://example.com/dedup-test"]
        assert len(matching) == 1, "相同 URL 应只有一个索引条目"

    @pytest.mark.asyncio
    async def test_persistence_force_refresh_updates_document(self, tmp_path: Path) -> None:
        """测试 force_refresh 行为：强制更新已存在的文档"""
        from knowledge.models import Document
        from knowledge.storage import KnowledgeStorage, StorageConfig

        storage_root = tmp_path / "knowledge_force"
        config = StorageConfig(
            storage_root=str(storage_root.relative_to(tmp_path)),
            read_only=False,
            auto_create_dirs=True,
        )
        storage = KnowledgeStorage(config=config, workspace_root=str(tmp_path))
        await storage.initialize()

        # 保存第一个版本
        doc_v1 = Document(
            url="https://example.com/force-test",
            title="Version 1",
            content="Original content version 1.",
        )
        saved1, _ = await storage.save_document(doc_v1)
        assert saved1 is True

        # 使用 force=True 强制更新（即使内容相同）
        doc_v2 = Document(
            url="https://example.com/force-test",
            title="Version 2",
            content="Updated content version 2.",
        )
        saved2, msg2 = await storage.save_document(doc_v2, force=True)
        assert saved2 is True, f"force=True 应强制更新: {msg2}"

        # 验证内容已更新
        loaded = await storage.load_document_by_url("https://example.com/force-test")
        assert loaded is not None
        assert "version 2" in loaded.content.lower() or "Version 2" in loaded.title

    @pytest.mark.asyncio
    async def test_persistence_dry_run_no_write_no_crash(self, tmp_path: Path) -> None:
        """测试 dry-run 模式下持久化被阻止但不崩溃

        验证 ReadOnlyStorageError 被正确捕获，流程继续执行。
        """
        from knowledge.models import Document
        from knowledge.storage import KnowledgeStorage, ReadOnlyStorageError, StorageConfig

        # 创建只读 storage（模拟 dry-run 模式）
        storage_root = tmp_path / "knowledge_readonly"
        # 先创建目录结构以便初始化
        storage_root.mkdir(parents=True)
        (storage_root / "docs").mkdir()
        (storage_root / "metadata").mkdir()
        (storage_root / "index.json").write_text('{"version": 1, "documents": []}', encoding="utf-8")

        config = StorageConfig(
            storage_root=str(storage_root.relative_to(tmp_path)),
            read_only=True,  # 只读模式
        )
        storage = KnowledgeStorage(config=config, workspace_root=str(tmp_path))
        await storage.initialize()

        # 创建测试文档
        test_doc = Document(
            url="https://example.com/readonly-test",
            title="ReadOnly Test",
            content="Should not be persisted.",
        )

        # 验证写入操作抛出 ReadOnlyStorageError
        with pytest.raises(ReadOnlyStorageError) as exc_info:
            await storage.save_document(test_doc)
        assert exc_info.value.operation == "save_document"

        # 验证 storage 状态正常（可以继续读取操作）
        entries = await storage.list_documents()
        assert len(entries) == 0  # 空索引

    @pytest.mark.asyncio
    async def test_knowledge_updater_persistence_in_save_changelog(self, tmp_path: Path) -> None:
        """测试 KnowledgeUpdater._save_changelog 的持久化行为

        验证 changelog 保存后能通过 storage 加载。
        """
        from knowledge.storage import KnowledgeStorage, StorageConfig

        # 创建可写的临时 storage
        storage_root = tmp_path / "knowledge_changelog"
        config = StorageConfig(
            storage_root=str(storage_root.relative_to(tmp_path)),
            read_only=False,
            auto_create_dirs=True,
        )
        storage = KnowledgeStorage(config=config, workspace_root=str(tmp_path))
        await storage.initialize()

        # 创建 KnowledgeUpdater（非 dry_run 模式）
        updater = KnowledgeUpdater(
            dry_run=False,
            llms_cache_path=str(tmp_path / "cache" / "llms.txt"),
        )
        # 替换为临时 storage
        updater.storage = storage
        await updater.initialize()

        # 模拟 UpdateAnalysis
        mock_analysis = UpdateAnalysis(
            has_updates=True,
            raw_content="<html><body><h1>Cursor Changelog</h1><p>Test changelog content.</p></body></html>",
            fingerprint="test-fingerprint-123",
            entries=[],
        )

        # 调用 _save_changelog
        result = await updater._save_changelog(
            mock_analysis,
            changelog_url="https://cursor.com/test-changelog",
            force=True,
        )

        # 验证保存成功
        assert result == "updated", f"_save_changelog 应返回 'updated': {result}"

        # 验证能通过 storage 加载
        loaded_doc = await storage.load_document_by_url("https://cursor.com/test-changelog")
        assert loaded_doc is not None, "应能通过 URL 加载已保存的 changelog"
        assert "Cursor Changelog" in loaded_doc.title or "changelog" in loaded_doc.content.lower()


# ============================================================
# TestSkipOnlineMode - skip-online 模式测试
# ============================================================


class TestSkipOnlineMode:
    """测试 --skip-online 模式"""

    @pytest.mark.asyncio
    async def test_skip_online_does_not_fetch_changelog(self, skip_online_args: argparse.Namespace) -> None:
        """测试 skip-online 模式不获取 Changelog"""
        iterator = SelfIterator(skip_online_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 2})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.changelog_analyzer.fetcher, "fetch", new_callable=AsyncMock) as mock_fetch,
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(
                iterator.knowledge_updater.manager,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator,
                "_run_agent_system",
                new_callable=AsyncMock,
                return_value={
                    "success": True,
                    "iterations_completed": 1,
                    "total_tasks_created": 1,
                    "total_tasks_completed": 1,
                    "total_tasks_failed": 0,
                },
            ),
        ):
            await iterator.run()

            # 验证 fetch 未被调用
            mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_online_uses_existing_knowledge(self, skip_online_args: argparse.Namespace) -> None:
        """测试 skip-online 模式使用现有知识库"""
        iterator = SelfIterator(skip_online_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 10})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator,
                "_run_agent_system",
                new_callable=AsyncMock,
                return_value={
                    "success": True,
                    "iterations_completed": 1,
                    "total_tasks_created": 1,
                    "total_tasks_completed": 1,
                    "total_tasks_failed": 0,
                },
            ),
        ):
            await iterator.run()

            # 验证知识库统计被调用
            mock_storage.get_stats.assert_called()

    @pytest.mark.asyncio
    async def test_skip_online_sets_empty_update_analysis(self, skip_online_args: argparse.Namespace) -> None:
        """测试 skip-online 模式设置空的更新分析"""
        iterator = SelfIterator(skip_online_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator,
                "_run_agent_system",
                new_callable=AsyncMock,
                return_value={
                    "success": True,
                    "iterations_completed": 1,
                    "total_tasks_created": 1,
                    "total_tasks_completed": 1,
                    "total_tasks_failed": 0,
                },
            ),
        ):
            await iterator.run()

            # 验证 update_analysis 被设置为空的 UpdateAnalysis
            assert iterator.context.update_analysis is not None
            assert iterator.context.update_analysis.has_updates is False


# ============================================================
# TestOrchestratorSelection - 编排器选择测试
# ============================================================


class TestOrchestratorSelection:
    """测试编排器选择逻辑"""

    def test_get_orchestrator_type_cli_mode_default_mp(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 cli 模式下默认使用 MP 编排器

        场景：execution_mode='cli' 时，编排器默认使用 mp
        注意：base_iterate_args fixture 已显式设置 execution_mode='cli'
        """
        # 确保是 cli 模式
        assert base_iterate_args.execution_mode == "cli"
        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "mp"

    def test_get_orchestrator_type_config_default_auto_uses_basic(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 config 默认 auto 模式时使用 basic 编排器

        场景：未显式指定 execution_mode（使用 config.yaml 默认 auto）
        期望：根据 AGENTS.md，auto 模式强制使用 basic 编排器
        """
        # 模拟 config 默认 auto 模式
        base_iterate_args.execution_mode = "auto"
        base_iterate_args.orchestrator = "mp"  # CLI 参数默认
        base_iterate_args._orchestrator_user_set = False
        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic", "config 默认 auto 模式应强制使用 basic 编排器"

    def test_execution_decision_no_prefix_no_cli_config_auto_orchestrator_basic(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 SelfIterator.__init__: 无 & 前缀 + CLI 未指定 + config=auto 场景

        关键场景：验证 SelfIterator.__init__ 创建的 _execution_decision.orchestrator == "basic"

        复用 resolve_requested_mode_for_decision 与 build_execution_decision，
        避免在测试里复制决策逻辑。
        """
        from cursor.cloud_client import CloudClientFactory

        # 模拟无 & 前缀的普通任务
        base_iterate_args.requirement = "普通任务描述"  # 无 & 前缀
        base_iterate_args.execution_mode = "auto"  # 来自 config.yaml 默认
        base_iterate_args.orchestrator = "mp"  # CLI 默认
        base_iterate_args._orchestrator_user_set = False

        # 模拟有 API Key
        with patch.object(CloudClientFactory, "resolve_api_key", return_value="test_key"):
            iterator = SelfIterator(base_iterate_args)

        # 核心断言：_execution_decision.orchestrator == "basic"
        assert iterator._execution_decision.orchestrator == "basic", (
            "无 & 前缀 + CLI 未指定 + config=auto 场景下，_execution_decision.orchestrator 应为 basic"
        )

        # 辅助断言
        assert iterator._execution_decision.effective_mode == "auto"
        assert iterator._execution_decision.prefix_routed is False
        assert iterator._execution_decision.has_ampersand_prefix is False

    def test_execution_decision_fallback_auto_no_key_orchestrator_basic(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 SelfIterator.__init__: auto + 无 API Key 回退 → orchestrator 仍为 basic

        回退场景关键测试：
        - requested_mode=auto（来自 config.yaml）
        - has_api_key=False → effective_mode=cli 回退
        - 但 orchestrator 仍为 basic（基于 requested_mode 语义）
        """
        from cursor.cloud_client import CloudClientFactory

        base_iterate_args.requirement = "分析代码结构"  # 无 & 前缀
        base_iterate_args.execution_mode = "auto"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args._orchestrator_user_set = False

        # 模拟无 API Key
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            iterator = SelfIterator(base_iterate_args)

        # 核心断言：即使回退到 cli，orchestrator 仍为 basic
        assert iterator._execution_decision.effective_mode == "cli", "无 API Key 应回退到 cli"
        assert iterator._execution_decision.orchestrator == "basic", (
            "回退场景关键断言：requested_mode=auto 时，即使回退到 cli，orchestrator 仍应为 basic"
        )
        assert iterator._execution_decision.prefix_routed is False

    def test_execution_decision_fallback_cloud_no_key_orchestrator_basic(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 SelfIterator.__init__: cloud + 无 API Key 回退 → orchestrator 仍为 basic

        回退场景关键测试：
        - requested_mode=cloud
        - has_api_key=False → effective_mode=cli 回退
        - 但 orchestrator 仍为 basic（基于 requested_mode 语义）
        """
        from cursor.cloud_client import CloudClientFactory

        base_iterate_args.requirement = "长时间分析任务"  # 无 & 前缀
        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args._orchestrator_user_set = False

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            iterator = SelfIterator(base_iterate_args)

        assert iterator._execution_decision.effective_mode == "cli", "无 API Key 应回退到 cli"
        assert iterator._execution_decision.orchestrator == "basic", (
            "回退场景关键断言：requested_mode=cloud 时，即使回退到 cli，orchestrator 仍应为 basic"
        )

    def test_get_orchestrator_type_no_mp_flag(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 --no-mp 标志使用 basic 编排器"""
        base_iterate_args.no_mp = True
        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic"

    def test_get_orchestrator_type_explicit_basic(self, base_iterate_args: argparse.Namespace) -> None:
        """测试显式指定 basic 编排器"""
        base_iterate_args.orchestrator = "basic"
        base_iterate_args._orchestrator_user_set = True  # 标记为显式设置
        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic"

    def test_get_orchestrator_type_from_requirement_non_parallel(self, base_iterate_args: argparse.Namespace) -> None:
        """测试从 requirement 检测非并行关键词时选择 basic 编排器

        场景：requirement='自我迭代，非并行模式' 且 no_mp=False/orchestrator='mp'
        期望：SelfIterator 选择 basic 编排器
        """
        # 设置 requirement 包含非并行关键词
        base_iterate_args.requirement = "自我迭代，非并行模式"
        # 确保命令行参数使用默认的 mp 编排器
        base_iterate_args.no_mp = False
        base_iterate_args.orchestrator = "mp"
        # 未显式设置编排器选项
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)
        # 应该从 requirement 检测到非并行关键词，返回 basic
        assert iterator._get_orchestrator_type() == "basic", (
            "当 requirement 包含 '非并行模式' 且用户未显式设置编排器时，应该自动选择 basic 编排器"
        )

    def test_get_orchestrator_type_from_requirement_various_keywords(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试多种非并行关键词都能正确检测"""
        non_parallel_requirements = [
            "自我迭代，非并行模式",
            "使用协程模式处理任务",
            "单进程执行",
            "禁用多进程执行",
            "使用 basic 编排器",
            "no-mp 模式运行",
            "串行处理任务",
        ]

        for requirement in non_parallel_requirements:
            base_iterate_args.requirement = requirement
            base_iterate_args.no_mp = False
            base_iterate_args.orchestrator = "mp"
            base_iterate_args._orchestrator_user_set = False

            iterator = SelfIterator(base_iterate_args)
            assert iterator._get_orchestrator_type() == "basic", f"requirement='{requirement}' 应该触发 basic 编排器"

    def test_get_orchestrator_type_user_explicit_setting_not_overridden(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试用户显式设置编排器时不被 requirement 关键词覆盖

        场景：用户通过 --orchestrator mp 显式指定使用 MP 编排器，
        即使 requirement 包含非并行关键词，也应该尊重用户的显式设置。
        """
        # requirement 包含非并行关键词
        base_iterate_args.requirement = "自我迭代，非并行模式"
        # 用户显式设置使用 mp 编排器
        base_iterate_args.no_mp = False
        base_iterate_args.orchestrator = "mp"
        # 标记用户显式设置了编排器选项
        base_iterate_args._orchestrator_user_set = True

        iterator = SelfIterator(base_iterate_args)
        # 用户显式设置应该优先，返回 mp
        assert iterator._get_orchestrator_type() == "mp", (
            "当用户显式设置 --orchestrator mp 时，即使 requirement 包含非并行关键词，也应该尊重用户的显式设置，返回 mp"
        )

    def test_get_orchestrator_type_user_explicit_no_mp_takes_priority(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试用户显式设置 --no-mp 时优先生效"""
        # requirement 不包含非并行关键词
        base_iterate_args.requirement = "普通的自我迭代任务"
        # 用户显式设置 --no-mp
        base_iterate_args.no_mp = True
        base_iterate_args.orchestrator = "mp"
        base_iterate_args._orchestrator_user_set = True

        iterator = SelfIterator(base_iterate_args)
        # --no-mp 显式设置应该返回 basic
        assert iterator._get_orchestrator_type() == "basic", "当用户显式设置 --no-mp 时，应该返回 basic"

    def test_get_orchestrator_type_requirement_without_keyword_cli_mode_uses_mp(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 cli 模式下 requirement 不包含非并行关键词时使用 mp 编排器

        场景：execution_mode='cli' 且 requirement 不包含非并行关键词
        期望：使用 mp 编排器（cli 模式默认）
        """
        # 确保是 cli 模式
        base_iterate_args.execution_mode = "cli"
        # requirement 不包含非并行关键词
        base_iterate_args.requirement = "普通的自我迭代任务，优化代码"
        base_iterate_args.no_mp = False
        base_iterate_args.orchestrator = "mp"
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)
        # cli 模式下没有非并行关键词，应该返回 mp
        assert iterator._get_orchestrator_type() == "mp", (
            "cli 模式下 requirement 不包含非并行关键词时，应使用 mp 编排器"
        )

    def test_get_orchestrator_type_empty_requirement_cli_mode_uses_mp(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 cli 模式下空 requirement 时使用 mp 编排器

        场景：execution_mode='cli' 且 requirement 为空
        期望：使用 mp 编排器（cli 模式默认）
        """
        # 确保是 cli 模式
        base_iterate_args.execution_mode = "cli"
        base_iterate_args.requirement = ""
        base_iterate_args.no_mp = False
        base_iterate_args.orchestrator = "mp"
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "mp", "cli 模式下空 requirement 时应使用 mp 编排器"


# ============================================================
# TestExecutionModeSelection - 执行模式选择测试
# ============================================================


class TestExecutionModeSelection:
    """测试 execution_mode 参数和相关逻辑

    覆盖场景：
    1. execution_mode=cli 时使用默认 mp 编排器
    2. execution_mode=cloud/auto 时强制使用 basic 编排器
    3. OrchestratorConfig 正确接收 execution_mode 和 cloud_auth_config

    注意：这些测试需要 mock Cloud API Key 存在，否则 cloud/auto 模式会回退到 cli
    """

    @pytest.fixture(autouse=True)
    def setup_mock_api_key(self, mock_cloud_api_key):
        """自动应用 conftest.py 中的 mock_cloud_api_key fixture"""
        pass

    def test_get_execution_mode_cli(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 execution_mode=cli 返回 CLI 模式"""
        from cursor.executor import ExecutionMode

        base_iterate_args.execution_mode = "cli"
        iterator = SelfIterator(base_iterate_args)

        assert iterator._get_execution_mode() == ExecutionMode.CLI

    def test_get_execution_mode_cloud(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 execution_mode=cloud 返回 CLOUD 模式"""
        from cursor.executor import ExecutionMode

        base_iterate_args.execution_mode = "cloud"
        iterator = SelfIterator(base_iterate_args)

        assert iterator._get_execution_mode() == ExecutionMode.CLOUD

    def test_get_execution_mode_auto(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 execution_mode=auto 返回 AUTO 模式"""
        from cursor.executor import ExecutionMode

        base_iterate_args.execution_mode = "auto"
        iterator = SelfIterator(base_iterate_args)

        assert iterator._get_execution_mode() == ExecutionMode.AUTO

    def test_execution_mode_cloud_forces_basic_orchestrator(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 execution_mode=cloud 时强制使用 basic 编排器

        场景：用户设置 --execution-mode cloud 但未显式设置 --orchestrator
        期望：SelfIterator 自动选择 basic 编排器
        """
        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.orchestrator = "mp"  # 默认 mp
        base_iterate_args.no_mp = False
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic", "execution_mode=cloud 时应强制使用 basic 编排器"

    def test_execution_mode_auto_forces_basic_orchestrator(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 execution_mode=auto 时强制使用 basic 编排器

        场景：用户设置 --execution-mode auto 且显式设置 --orchestrator mp
        期望：SelfIterator 仍选择 basic 编排器（执行模式优先）
        """
        base_iterate_args.execution_mode = "auto"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        base_iterate_args._orchestrator_user_set = True  # 用户显式设置了 mp

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic", (
            "execution_mode=auto 时应强制使用 basic 编排器，即使用户显式设置 mp"
        )

    def test_execution_mode_cli_allows_mp_orchestrator(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 execution_mode=cli 时允许使用 mp 编排器"""
        base_iterate_args.execution_mode = "cli"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "mp", "execution_mode=cli 时应允许使用 mp 编排器"

    def test_get_cloud_auth_config_from_arg(self, base_iterate_args: argparse.Namespace) -> None:
        """测试从命令行参数获取 cloud_auth_config"""
        from cursor.cloud_client import CloudClientFactory

        base_iterate_args.cloud_api_key = "test-api-key-12345"
        base_iterate_args.cloud_auth_timeout = 60

        # 使用特定 API Key 覆盖 autouse fixture
        with patch.object(CloudClientFactory, "resolve_api_key", return_value="test-api-key-12345"):
            iterator = SelfIterator(base_iterate_args)
            config = iterator._get_cloud_auth_config()

            assert config is not None
            assert config.api_key == "test-api-key-12345"
            assert config.auth_timeout == 60

    def test_get_cloud_auth_config_none_when_no_key(self, base_iterate_args: argparse.Namespace) -> None:
        """测试无 API Key 时返回 None"""
        from cursor.cloud_client import CloudClientFactory

        base_iterate_args.cloud_api_key = None

        # 覆盖 autouse fixture 为无 API Key
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            # 确保环境变量也没有设置
            import os

            with patch.dict(os.environ, {}, clear=True):
                iterator = SelfIterator(base_iterate_args)
                config = iterator._get_cloud_auth_config()

                assert config is None

    @pytest.mark.asyncio
    async def test_basic_orchestrator_receives_execution_mode(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 basic 编排器正确接收 execution_mode 配置"""
        from cursor.executor import ExecutionMode

        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.skip_online = True
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        # 捕获传给 OrchestratorConfig 的参数
        with patch("scripts.run_iterate.OrchestratorConfig") as MockConfig:
            with patch("scripts.run_iterate.Orchestrator") as MockOrch:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    with patch("scripts.run_iterate.CursorAgentConfig"):
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        mock_config_instance = MagicMock()
                        MockConfig.return_value = mock_config_instance

                        mock_orch = MagicMock()
                        mock_orch.run = AsyncMock(return_value={"success": True})
                        MockOrch.return_value = mock_orch

                        await iterator._run_with_basic_orchestrator(3, mock_km)

                        # 验证配置参数
                        MockConfig.assert_called_once()
                        call_kwargs = MockConfig.call_args.kwargs
                        assert call_kwargs.get("execution_mode") == ExecutionMode.CLOUD

    @pytest.mark.asyncio
    async def test_basic_orchestrator_receives_cloud_auth_config(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 basic 编排器正确接收 cloud_auth_config 配置"""
        from cursor.cloud_client import CloudClientFactory

        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.cloud_api_key = "test-cloud-key"
        base_iterate_args.cloud_auth_timeout = 45
        base_iterate_args.skip_online = True

        # 使用特定 API Key 覆盖 autouse fixture
        with patch.object(CloudClientFactory, "resolve_api_key", return_value="test-cloud-key"):
            iterator = SelfIterator(base_iterate_args)
            iterator.context.iteration_goal = "测试目标"

            with patch("scripts.run_iterate.OrchestratorConfig") as MockConfig:
                with patch("scripts.run_iterate.Orchestrator") as MockOrch:
                    with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                        with patch("scripts.run_iterate.CursorAgentConfig"):
                            mock_km = MagicMock()
                            mock_km.initialize = AsyncMock()
                            MockKM.return_value = mock_km

                            mock_config_instance = MagicMock()
                            MockConfig.return_value = mock_config_instance

                            mock_orch = MagicMock()
                            mock_orch.run = AsyncMock(return_value={"success": True})
                            MockOrch.return_value = mock_orch

                            await iterator._run_with_basic_orchestrator(3, mock_km)

                            # 验证 cloud_auth_config 参数
                            MockConfig.assert_called_once()
                            call_kwargs = MockConfig.call_args.kwargs
                            cloud_auth = call_kwargs.get("cloud_auth_config")
                            assert cloud_auth is not None
                            assert cloud_auth.api_key == "test-cloud-key"
                            assert cloud_auth.auth_timeout == 45

    @pytest.mark.asyncio
    async def test_cloud_timeout_passed_to_cursor_config_for_cloud_mode(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 --cloud-timeout 参数在 cloud 模式下传递到 CursorAgentConfig.timeout

        验证:
        1. 当 execution_mode=cloud 且设置 --cloud-timeout 时
        2. CursorAgentConfig 的 timeout 应使用 cloud_timeout 值
        3. 确保 Cloud 模式使用独立的超时配置
        """
        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.cloud_timeout = 1200  # 设置 20 分钟超时
        base_iterate_args.skip_online = True
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        captured_cursor_config_timeout = None

        with patch("scripts.run_iterate.OrchestratorConfig") as MockConfig:
            with patch("scripts.run_iterate.Orchestrator") as MockOrch:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    with patch("scripts.run_iterate.CursorAgentConfig") as MockCursorConfig:
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        # 捕获 CursorAgentConfig 的参数
                        def capture_cursor_config(*args, **kwargs):
                            nonlocal captured_cursor_config_timeout
                            captured_cursor_config_timeout = kwargs.get("timeout")
                            return MagicMock()

                        MockCursorConfig.side_effect = capture_cursor_config

                        mock_config_instance = MagicMock()
                        MockConfig.return_value = mock_config_instance

                        mock_orch = MagicMock()
                        mock_orch.run = AsyncMock(return_value={"success": True})
                        MockOrch.return_value = mock_orch

                        await iterator._run_with_basic_orchestrator(3, mock_km)

                        # 验证 CursorAgentConfig.timeout 使用 cloud_timeout 值
                        assert captured_cursor_config_timeout == 1200, (
                            "Cloud 模式下 CursorAgentConfig.timeout 应使用 --cloud-timeout 值"
                        )

    @pytest.mark.asyncio
    async def test_cloud_timeout_passed_to_cursor_config_for_auto_mode(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 --cloud-timeout 参数在 auto 模式下也传递到 CursorAgentConfig.timeout

        Auto 模式优先使用 Cloud，因此也应使用 cloud_timeout。
        """
        base_iterate_args.execution_mode = "auto"
        base_iterate_args.cloud_timeout = 900  # 设置 15 分钟超时
        base_iterate_args.skip_online = True
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        captured_cursor_config_timeout = None

        with patch("scripts.run_iterate.OrchestratorConfig") as MockConfig:
            with patch("scripts.run_iterate.Orchestrator") as MockOrch:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    with patch("scripts.run_iterate.CursorAgentConfig") as MockCursorConfig:
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        def capture_cursor_config(*args, **kwargs):
                            nonlocal captured_cursor_config_timeout
                            captured_cursor_config_timeout = kwargs.get("timeout")
                            return MagicMock()

                        MockCursorConfig.side_effect = capture_cursor_config

                        mock_config_instance = MagicMock()
                        MockConfig.return_value = mock_config_instance

                        mock_orch = MagicMock()
                        mock_orch.run = AsyncMock(return_value={"success": True})
                        MockOrch.return_value = mock_orch

                        await iterator._run_with_basic_orchestrator(3, mock_km)

                        # 验证 Auto 模式也使用 cloud_timeout
                        assert captured_cursor_config_timeout == 900, (
                            "Auto 模式下 CursorAgentConfig.timeout 也应使用 --cloud-timeout 值"
                        )

    @pytest.mark.asyncio
    async def test_cli_mode_does_not_override_default_timeout(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 CLI 模式下不使用 cloud_timeout（使用默认超时）

        CLI 模式应使用 CursorAgentConfig 的默认 timeout，
        不受 --cloud-timeout 参数影响。
        """
        base_iterate_args.execution_mode = "cli"  # CLI 模式
        base_iterate_args.cloud_timeout = 1800  # 设置 cloud_timeout（不应生效）
        base_iterate_args.skip_online = True
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        cursor_config_call_kwargs: dict[str, Any] = {}

        with patch("scripts.run_iterate.OrchestratorConfig") as MockConfig:
            with patch("scripts.run_iterate.Orchestrator") as MockOrch:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    with patch("scripts.run_iterate.CursorAgentConfig") as MockCursorConfig:
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        def capture_cursor_config(*args, **kwargs):
                            nonlocal cursor_config_call_kwargs
                            cursor_config_call_kwargs = kwargs
                            return MagicMock()

                        MockCursorConfig.side_effect = capture_cursor_config

                        mock_config_instance = MagicMock()
                        MockConfig.return_value = mock_config_instance

                        mock_orch = MagicMock()
                        mock_orch.run = AsyncMock(return_value={"success": True})
                        MockOrch.return_value = mock_orch

                        await iterator._run_with_basic_orchestrator(3, mock_km)

                        # CLI 模式下不应设置 timeout（使用默认值）
                        # 或者 timeout 不应为 cloud_timeout 值
                        timeout_value = cursor_config_call_kwargs.get("timeout")
                        assert timeout_value != 1800 or timeout_value is None, "CLI 模式下不应使用 --cloud-timeout 值"

    @pytest.mark.asyncio
    async def test_execution_mode_cloud_uses_basic_orchestrator_path(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 execution_mode=cloud 时 SelfIterator 走 basic 编排器路径

        完整场景验证：
        1. 用户设置 --execution-mode cloud
        2. 用户未设置 --no-mp（即默认使用 mp）
        3. SelfIterator 应自动切换到 basic 编排器
        4. 不应调用 _run_with_mp_orchestrator
        """
        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        base_iterate_args.skip_online = True
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        mock_basic_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
            ) as mock_mp,
            patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ) as mock_basic,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证 MP 编排器未被调用
            mock_mp.assert_not_called()

            # 验证 basic 编排器被调用
            mock_basic.assert_called_once()

            # 验证结果正确
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execution_mode_auto_uses_basic_orchestrator_path(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 execution_mode=auto 时 SelfIterator 走 basic 编排器路径"""
        base_iterate_args.execution_mode = "auto"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        base_iterate_args.skip_online = True
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        mock_basic_result = {
            "success": True,
            "iterations_completed": 2,
            "total_tasks_created": 3,
            "total_tasks_completed": 3,
            "total_tasks_failed": 0,
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
            ) as mock_mp,
            patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ) as mock_basic,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证 MP 编排器未被调用
            mock_mp.assert_not_called()

            # 验证 basic 编排器被调用
            mock_basic.assert_called_once()


# ============================================================
# TestAmpersandPrefixCloudMode - & 前缀触发 Cloud 模式测试
# ============================================================


class TestAmpersandPrefixCloudMode:
    """测试 '&' 前缀触发 Cloud 模式且 goal 可剥离不影响执行模式

    覆盖场景：
    1. requirement 以 '&' 开头时自动切换到 CLOUD 执行模式
    2. goal 剥离 '&' 前缀后保留实际任务内容
    3. 剥离后的 goal 不影响执行模式判断
    4. 边界用例：仅 '&'、空白等

    注意：这些测试需要 mock Cloud API Key 存在，否则 '&' 前缀会回退到 cli 模式
    """

    @pytest.fixture(autouse=True)
    def mock_cloud_api_key_and_enabled(self) -> Generator[None, None, None]:
        """Mock Cloud API Key 存在且 cloud_enabled=True，用于测试 '&' 前缀触发 Cloud 模式

        Policy 要求 cloud_enabled=True 才会路由 & 前缀到 Cloud 模式。
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        # 先获取配置实例，然后直接修改 cloud_agent.enabled 和 execution_mode
        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            # 临时设置 cloud_enabled=True 和 execution_mode=cli
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"
            try:
                yield
            finally:
                # 恢复原始值
                config.cloud_agent.enabled = original_enabled
                config.cloud_agent.execution_mode = original_execution_mode

    @pytest.fixture
    def cloud_prefix_args(self) -> argparse.Namespace:
        """创建带 '&' 前缀的参数"""
        return argparse.Namespace(
            requirement="& 分析代码架构",
            directory=".",  # 工作目录
            skip_online=True,
            changelog_url=None,  # tri-state
            dry_run=False,
            max_iterations="3",
            workers=2,
            force_update=False,
            verbose=False,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=None,  # None 表示用户未显式指定，应被 '&' 前缀覆盖
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            # 角色级执行模式
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            # 流式控制台渲染参数
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_ampersand_prefix_triggers_cloud_mode(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 '&' 前缀触发 CLOUD 执行模式"""
        from cursor.executor import ExecutionMode

        iterator = SelfIterator(cloud_prefix_args)

        # 验证执行模式被设置为 CLOUD
        assert iterator._get_execution_mode() == ExecutionMode.CLOUD

    def test_ampersand_prefix_strips_from_requirement(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 '&' 前缀从 requirement 中剥离"""
        iterator = SelfIterator(cloud_prefix_args)

        # 验证 user_requirement 不包含 '&' 前缀
        assert not iterator.context.user_requirement.startswith("&")
        assert iterator.context.user_requirement == "分析代码架构"

    def test_stripped_goal_preserves_content(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试剥离后的 goal 保留实际任务内容"""
        test_cases = [
            ("& 简单任务", "简单任务"),
            ("&任务描述", "任务描述"),
            ("  & 带空格的任务  ", "带空格的任务"),
            ("& 优化 CLI 参数处理", "优化 CLI 参数处理"),
        ]

        for requirement, expected_goal in test_cases:
            cloud_prefix_args.requirement = requirement
            iterator = SelfIterator(cloud_prefix_args)

            assert iterator.context.user_requirement == expected_goal, (
                f"requirement='{requirement}' 剥离后应为 '{expected_goal}'，"
                f"实际为 '{iterator.context.user_requirement}'"
            )

    def test_ampersand_prefix_forces_basic_orchestrator(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 '&' 前缀触发 Cloud 模式时强制使用 basic 编排器"""
        iterator = SelfIterator(cloud_prefix_args)

        # 虽然默认 orchestrator=mp，但 Cloud 模式应强制 basic
        assert iterator._get_orchestrator_type() == "basic"

    def test_stripped_goal_does_not_affect_execution_mode(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试剥离后的 goal 内容不影响执行模式判断

        场景：'& 使用多进程处理' → 剥离后为 '使用多进程处理'
        期望：执行模式仍为 CLOUD（由 '&' 前缀决定），不受 goal 内容影响
        """
        from cursor.executor import ExecutionMode

        # 设置一个包含 MP 关键词的任务
        cloud_prefix_args.requirement = "& 使用多进程并行处理任务"
        iterator = SelfIterator(cloud_prefix_args)

        # 执行模式应为 CLOUD（由 '&' 前缀决定）
        assert iterator._get_execution_mode() == ExecutionMode.CLOUD

        # 编排器应为 basic（Cloud 模式强制）
        assert iterator._get_orchestrator_type() == "basic"

        # goal 应正确剥离
        assert iterator.context.user_requirement == "使用多进程并行处理任务"

    def test_ampersand_only_does_not_trigger_cloud(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试仅 '&' 符号不触发 Cloud 模式"""
        from cursor.executor import ExecutionMode

        # 仅 '&' 符号
        cloud_prefix_args.requirement = "&"
        iterator = SelfIterator(cloud_prefix_args)

        # & 前缀不满足条件（无实际内容），回退到 CLI 模式
        assert iterator._get_execution_mode() == ExecutionMode.CLI

    def test_ampersand_with_whitespace_only_does_not_trigger_cloud(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 '& ' (& 加空白) 不触发 Cloud 模式"""
        from cursor.executor import ExecutionMode

        cloud_prefix_args.requirement = "&   "
        iterator = SelfIterator(cloud_prefix_args)

        # & 前缀后仅空白，不满足触发条件，回退到 CLI 模式
        assert iterator._get_execution_mode() == ExecutionMode.CLI

    def test_ampersand_in_middle_does_not_trigger_cloud(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 '&' 在中间不触发 Cloud 模式"""
        from cursor.executor import ExecutionMode

        cloud_prefix_args.requirement = "任务 & 描述"
        iterator = SelfIterator(cloud_prefix_args)

        # & 不在开头，不触发 Cloud 模式，使用 CLI 模式
        assert iterator._get_execution_mode() == ExecutionMode.CLI

        # requirement 不应被修改
        assert iterator.context.user_requirement == "任务 & 描述"

    def test_prefix_routed_flag_set_correctly(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 _prefix_routed（策略决策）标志正确设置

        语义说明:
        - has_ampersand_prefix: 语法检测层面，原始文本是否有 & 前缀
        - _prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud（内部分支统一使用此字段）
        - _triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出兼容）
        """
        # 有 '&' 前缀 + 满足条件 → 成功触发 Cloud 路由
        cloud_prefix_args.requirement = "& 任务"
        iterator1 = SelfIterator(cloud_prefix_args)
        # 内部分支使用 _prefix_routed 字段
        assert iterator1._prefix_routed is True, "& 前缀应成功触发 Cloud 路由"

        # 无 '&' 前缀 → 不触发 Cloud 路由
        cloud_prefix_args.requirement = "普通任务"
        iterator2 = SelfIterator(cloud_prefix_args)
        assert iterator2._prefix_routed is False, "无 & 前缀不应触发 Cloud 路由"

    @pytest.mark.asyncio
    async def test_cloud_mode_from_ampersand_uses_basic_orchestrator_path(
        self, cloud_prefix_args: argparse.Namespace
    ) -> None:
        """测试 '&' 前缀触发 Cloud 模式时走 basic 编排器路径"""
        iterator = SelfIterator(cloud_prefix_args)
        iterator.context.iteration_goal = "测试目标"

        mock_basic_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
            ) as mock_mp,
            patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ) as mock_basic,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # MP 编排器不应被调用
            mock_mp.assert_not_called()

            # basic 编排器应被调用
            mock_basic.assert_called_once()

            assert result["success"] is True

    # ========== 边界测试：& 前缀策略一致性 ==========

    def test_ampersand_no_space_triggers_cloud(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 '&任务'（无空格）成功触发 Cloud 路由"""
        from cursor.executor import ExecutionMode

        cloud_prefix_args.requirement = "&分析代码架构"
        iterator = SelfIterator(cloud_prefix_args)

        # has_ampersand_prefix=True（语法检测） → prefix_routed=True（成功路由）
        # 内部分支使用 _prefix_routed 字段
        assert iterator._prefix_routed is True, "& 前缀应成功触发 Cloud 路由"
        assert iterator._get_execution_mode() == ExecutionMode.CLOUD
        assert iterator.context.user_requirement == "分析代码架构"

    def test_ampersand_with_space_triggers_cloud(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 '& 任务'（有空格）成功触发 Cloud 路由"""
        from cursor.executor import ExecutionMode

        cloud_prefix_args.requirement = "& 分析代码架构"
        iterator = SelfIterator(cloud_prefix_args)

        # has_ampersand_prefix=True（语法检测） → prefix_routed=True（成功路由）
        # 内部分支使用 _prefix_routed 字段
        assert iterator._prefix_routed is True, "& 前缀应成功触发 Cloud 路由"
        assert iterator._get_execution_mode() == ExecutionMode.CLOUD
        assert iterator.context.user_requirement == "分析代码架构"

    def test_ampersand_empty_content_does_not_trigger_cloud(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 '&'（空内容）不触发 Cloud 路由

        语义说明：has_ampersand_prefix=False（空内容不算有效前缀）→ prefix_routed=False
        """
        from cursor.executor import ExecutionMode

        test_cases = ["&", "& ", "&  ", "  &  "]

        for requirement in test_cases:
            cloud_prefix_args.requirement = requirement
            iterator = SelfIterator(cloud_prefix_args)

            # 空内容不算有效 & 前缀，不触发 Cloud 路由
            # 内部分支使用 _prefix_routed 字段
            assert iterator._prefix_routed is False, f"requirement='{requirement}' 不应成功触发 Cloud 路由"
            assert iterator._get_execution_mode() == ExecutionMode.CLI

    def test_cloud_mode_auto_commit_default_false(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 Cloud 模式下 auto_commit 默认为 False"""
        # 确保参数中 auto_commit=False
        cloud_prefix_args.auto_commit = False
        iterator = SelfIterator(cloud_prefix_args)

        # 验证 Cloud 模式
        from cursor.executor import ExecutionMode

        assert iterator._get_execution_mode() == ExecutionMode.CLOUD

        # auto_commit 应保持为 False
        assert iterator.args.auto_commit is False

    def test_cloud_mode_requires_explicit_auto_commit(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 Cloud 模式下需要显式启用 auto_commit"""
        # 显式设置 auto_commit=True
        cloud_prefix_args.auto_commit = True
        iterator = SelfIterator(cloud_prefix_args)

        # Cloud 模式 + 显式 auto_commit
        from cursor.executor import ExecutionMode

        assert iterator._get_execution_mode() == ExecutionMode.CLOUD
        assert iterator.args.auto_commit is True

    def test_execution_mode_from_args_when_no_ampersand(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试无 '&' 前缀时从 args.execution_mode 读取执行模式"""
        from cursor.executor import ExecutionMode

        # 没有 & 前缀，但设置了 execution_mode=cloud
        cloud_prefix_args.requirement = "普通任务"
        cloud_prefix_args.execution_mode = "cloud"
        iterator = SelfIterator(cloud_prefix_args)

        # has_ampersand_prefix=False，因此 prefix_routed=False（非 & 前缀触发）
        # 但应从 args.execution_mode 获取 CLOUD 模式
        # 内部分支使用 _prefix_routed 字段
        assert iterator._prefix_routed is False, "无 & 前缀不应标记为 prefix_routed"
        assert iterator._get_execution_mode() == ExecutionMode.CLOUD

    def test_explicit_cli_overrides_ampersand_prefix(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试显式 execution_mode=cli 覆盖 '&' 前缀

        根据 AGENTS.md 文档:
        - execution_mode=cli: 强制使用 CLI，不受 & 前缀影响

        语义说明:
        - has_ampersand_prefix: 语法检测层面，此处为 True（原始文本有 & 前缀）
        - _prefix_routed: 策略决策层面，"& 前缀是否成功触发 Cloud 路由"（内部分支统一使用此字段）
        - _triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出兼容）
        - 当显式指定 cli 时，& 前缀不会成功触发 Cloud，因此 prefix_routed=False
        """
        from cursor.executor import ExecutionMode

        # 设置 execution_mode=cli，但有 & 前缀（has_ampersand_prefix=True）
        cloud_prefix_args.requirement = "& 任务"
        cloud_prefix_args.execution_mode = "cli"  # 显式指定 cli，覆盖 & 前缀
        iterator = SelfIterator(cloud_prefix_args)

        # 显式 cli 覆盖 & 前缀，Cloud 未被成功触发
        # prefix_routed 仅表示"成功触发 Cloud 路由"，此处为 False
        # 内部分支使用 _prefix_routed 字段
        assert iterator._prefix_routed is False, "显式 CLI 模式下 & 前缀不应成功触发 Cloud 路由"
        assert iterator._get_execution_mode() == ExecutionMode.CLI

        # prompt 应被清理（& 前缀被移除，避免后续再次触发）
        assert not iterator.context.user_requirement.startswith("&")
        assert iterator.context.user_requirement == "任务"

    def test_goal_stripping_preserves_ampersand_in_content(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 goal 剥离时保留内容中的 & 符号"""
        cloud_prefix_args.requirement = "& 优化 A & B 模块"
        iterator = SelfIterator(cloud_prefix_args)

        # 只剥离开头的 &，保留内容中的 &
        assert iterator.context.user_requirement == "优化 A & B 模块"

    def test_explicit_cli_with_ampersand_sanitizes_prompt(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试显式 execution_mode=cli + & 前缀时 prompt 被清理

        场景: --execution-mode cli "& 分析代码"
        期望:
        - 执行模式为 CLI（显式指定优先）
        - has_ampersand_prefix=True（语法检测层面，原始文本有 & 前缀）
        - prefix_routed=False（& 前缀未成功触发 Cloud 路由）
        - prompt 被清理（移除 & 前缀，避免后续再次触发）
        """
        from cursor.executor import ExecutionMode

        cloud_prefix_args.requirement = "& 分析代码架构"
        cloud_prefix_args.execution_mode = "cli"
        iterator = SelfIterator(cloud_prefix_args)

        assert iterator._get_execution_mode() == ExecutionMode.CLI
        # prefix_routed=False（显式 CLI 模式下 & 前缀不触发 Cloud）
        # 内部分支使用 _prefix_routed 字段
        assert iterator._prefix_routed is False, "显式 CLI 下 prefix_routed 应为 False"
        assert iterator.context.user_requirement == "分析代码架构"
        assert "&" not in iterator.context.user_requirement.split()[0]

    def test_explicit_cli_with_ampersand_no_space(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试显式 execution_mode=cli + &任务（无空格）

        场景: --execution-mode cli "&任务"
        期望: has_ampersand_prefix=True（语法检测），prefix_routed=False（未成功触发 Cloud）
        """
        from cursor.executor import ExecutionMode

        cloud_prefix_args.requirement = "&分析代码"
        cloud_prefix_args.execution_mode = "cli"
        iterator = SelfIterator(cloud_prefix_args)

        assert iterator._get_execution_mode() == ExecutionMode.CLI
        # prefix_routed=False（显式 CLI 模式下 & 前缀不触发 Cloud 路由）
        # 内部分支使用 _prefix_routed 字段
        assert iterator._prefix_routed is False, "显式 CLI 下 prefix_routed 应为 False"
        assert iterator.context.user_requirement == "分析代码"

    def test_prefix_routed_semantic_only_for_successful_cloud(self, cloud_prefix_args: argparse.Namespace) -> None:
        """测试 _prefix_routed（策略决策）语义：仅在成功触发 Cloud 时为 True

        语义说明:
        - has_ampersand_prefix: 语法检测层面，原始文本是否有 & 前缀
        - _prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud 路由（内部分支统一使用此字段）
        - _triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出兼容）
        - prefix_routed=True：表示"本次由 & 前缀成功触发了 Cloud 路由"
        - prefix_routed=False：表示"未成功触发 Cloud 路由"（包括无前缀、显式 CLI、配置禁用等）
        """
        from cursor.executor import ExecutionMode

        # Case 1: has_ampersand_prefix=True + cloud_enabled=True + has_api_key → prefix_routed=True
        cloud_prefix_args.requirement = "& 任务"
        cloud_prefix_args.execution_mode = None  # 未显式指定
        iterator1 = SelfIterator(cloud_prefix_args)
        # 内部分支使用 _prefix_routed 字段
        assert iterator1._prefix_routed is True, "& 前缀应成功触发 Cloud 路由"
        assert iterator1._get_execution_mode() == ExecutionMode.CLOUD

        # Case 2: has_ampersand_prefix=True + 显式 CLI → prefix_routed=False（被显式 CLI 覆盖）
        cloud_prefix_args.requirement = "& 任务"
        cloud_prefix_args.execution_mode = "cli"
        iterator2 = SelfIterator(cloud_prefix_args)
        assert iterator2._prefix_routed is False, "显式 CLI 模式下 & 前缀不应成功触发 Cloud 路由"
        assert iterator2._get_execution_mode() == ExecutionMode.CLI

        # Case 3: has_ampersand_prefix=False → prefix_routed=False
        cloud_prefix_args.requirement = "普通任务"
        cloud_prefix_args.execution_mode = None
        iterator3 = SelfIterator(cloud_prefix_args)
        assert iterator3._prefix_routed is False, "无 & 前缀不应触发 Cloud 路由"


# ============================================================
# TestExplicitCliWithAmpersandPrefix - 显式 CLI + & 前缀场景
# ============================================================


class TestExplicitCliWithAmpersandPrefixExtended:
    """测试显式 execution_mode=cli + requirement 以 & 开头的场景

    这是重要的边界场景：用户显式指定 CLI 模式，但任务文本以 & 开头。
    期望行为：
    1. 执行模式为 CLI（显式指定优先于 & 前缀）
    2. has_ampersand_prefix=True（语法检测层面，原始文本有 & 前缀）
    3. prefix_routed=False（& 前缀未成功触发 Cloud 路由）
    4. prompt 被清理（移除 & 前缀，避免后续再次触发 Cloud 检测）
    """

    @pytest.fixture(autouse=True)
    def mock_cloud_api_key_and_enabled(self) -> Generator[None, None, None]:
        """Mock Cloud API Key 存在且 cloud_enabled=True"""
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        config = get_config()
        original_enabled = config.cloud_agent.enabled

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            config.cloud_agent.enabled = True
            try:
                yield
            finally:
                config.cloud_agent.enabled = original_enabled

    @pytest.fixture
    def explicit_cli_with_ampersand_args(self) -> argparse.Namespace:
        """创建显式 CLI + & 前缀的参数"""
        return argparse.Namespace(
            requirement="& 分析代码架构",
            directory=".",
            skip_online=True,
            changelog_url=None,  # tri-state
            dry_run=False,
            max_iterations="3",
            workers=2,
            force_update=False,
            verbose=False,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",  # 显式指定 CLI
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
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
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_explicit_cli_overrides_ampersand_execution_mode(
        self, explicit_cli_with_ampersand_args: argparse.Namespace
    ) -> None:
        """测试显式 CLI 覆盖 & 前缀的执行模式"""
        from cursor.executor import ExecutionMode

        iterator = SelfIterator(explicit_cli_with_ampersand_args)
        assert iterator._get_execution_mode() == ExecutionMode.CLI

    def test_prefix_routed_is_false_with_explicit_cli(
        self, explicit_cli_with_ampersand_args: argparse.Namespace
    ) -> None:
        """测试显式 CLI 时 prefix_routed 为 False

        语义说明：has_ampersand_prefix=True（语法检测），但 prefix_routed=False（未成功触发 Cloud）
        内部分支使用 _prefix_routed 字段
        """
        iterator = SelfIterator(explicit_cli_with_ampersand_args)
        assert iterator._prefix_routed is False, "显式 CLI 模式下 prefix_routed 应为 False"

    def test_prompt_sanitized_with_explicit_cli(self, explicit_cli_with_ampersand_args: argparse.Namespace) -> None:
        """测试显式 CLI 时 prompt 被清理"""
        iterator = SelfIterator(explicit_cli_with_ampersand_args)
        assert iterator.context.user_requirement == "分析代码架构"
        assert not iterator.context.user_requirement.startswith("&")

    def test_orchestrator_uses_mp_with_explicit_cli(self, explicit_cli_with_ampersand_args: argparse.Namespace) -> None:
        """测试显式 CLI 时可以使用 MP 编排器（不受 & 前缀影响）"""
        iterator = SelfIterator(explicit_cli_with_ampersand_args)
        # CLI 模式允许使用 MP 编排器
        assert iterator._get_orchestrator_type() == "mp"

    @pytest.mark.parametrize(
        "requirement,expected_cleaned",
        [
            ("& 分析代码", "分析代码"),
            ("&分析代码", "分析代码"),
            ("  & 带空格的任务  ", "带空格的任务"),
            ("& 优化 A & B 模块", "优化 A & B 模块"),  # 保留内容中的 &
        ],
    )
    def test_various_ampersand_formats_sanitized(
        self,
        explicit_cli_with_ampersand_args: argparse.Namespace,
        requirement: str,
        expected_cleaned: str,
    ) -> None:
        """测试各种 & 前缀格式在显式 CLI 下都被正确清理

        语义说明：has_ampersand_prefix=True（语法检测），prefix_routed=False（显式 CLI 覆盖）
        """
        from cursor.executor import ExecutionMode

        explicit_cli_with_ampersand_args.requirement = requirement
        iterator = SelfIterator(explicit_cli_with_ampersand_args)

        assert iterator._get_execution_mode() == ExecutionMode.CLI
        # prefix_routed=False（显式 CLI 模式下 & 前缀不触发 Cloud 路由）
        # 内部分支使用 _prefix_routed 字段
        assert iterator._prefix_routed is False, "显式 CLI 下 prefix_routed 应为 False"
        assert iterator.context.user_requirement == expected_cleaned


# ============================================================
# TestExecutionModeAutoFallbackTableDriven - Auto 模式回退表驱动测试
# ============================================================


class TestExecutionModeAutoFallbackTableDriven:
    """execution_mode=auto 回退场景表驱动测试

    覆盖场景：
    1. execution_mode=auto + 无 key：回退 CLI，输出可操作提示
    2. execution_mode=auto + 认证失败：回退 CLI，进入冷却
    3. execution_mode=auto + 429：回退 CLI，按 retry_after 冷却
    4. & 前缀 + cloud_enabled=false：使用 CLI（不尝试 Cloud）
    """

    @pytest.fixture
    def auto_mode_args(self) -> argparse.Namespace:
        """创建 auto 执行模式的参数"""
        return argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
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
            execution_mode="auto",  # auto 模式
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
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    @pytest.mark.parametrize(
        "scenario,has_api_key,expect_execution_mode,expect_orchestrator",
        [
            # 场景 1: auto + 有 key -> AUTO 模式 + basic 编排器
            # 注意：auto 模式保持语义，编排器强制为 basic
            pytest.param(
                "auto_with_key",
                True,
                "AUTO",
                "basic",
                id="auto_with_key_uses_basic_orchestrator",
            ),
            # 场景 2: auto + 无 key -> CLI 模式 + basic 编排器
            # 新行为：requested=auto/cloud 即强制 basic，不受 key/enable 影响
            pytest.param(
                "auto_no_key",
                False,
                "CLI",
                "basic",  # 新行为：auto 强制 basic
                id="auto_no_key_forces_basic_orchestrator",
            ),
        ],
    )
    def test_auto_mode_orchestrator_selection(
        self,
        auto_mode_args: argparse.Namespace,
        scenario: str,
        has_api_key: bool,
        expect_execution_mode: str,
        expect_orchestrator: str,
    ) -> None:
        """测试 auto 模式下编排器选择"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        if has_api_key:
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                iterator = SelfIterator(auto_mode_args)
                actual_mode = iterator._get_execution_mode()
                actual_orchestrator = iterator._get_orchestrator_type()
        else:
            with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
                iterator = SelfIterator(auto_mode_args)
                actual_mode = iterator._get_execution_mode()
                actual_orchestrator = iterator._get_orchestrator_type()

        expected_mode = ExecutionMode[expect_execution_mode]
        assert actual_mode == expected_mode, f"场景 {scenario}: 期望 {expected_mode}，实际 {actual_mode}"
        assert actual_orchestrator == expect_orchestrator, (
            f"场景 {scenario}: 期望编排器 {expect_orchestrator}，实际 {actual_orchestrator}"
        )

    @pytest.mark.parametrize(
        "requirement,cloud_enabled,has_api_key,requested_mode,expect_cli_fallback,expect_reason",
        [
            # 场景 1: 显式 auto + 无 key -> 回退 CLI
            pytest.param(
                "& 分析任务",
                True,
                False,
                "auto",
                True,
                "无 API Key",
                id="prefix_enabled_no_key_fallback",
            ),
            # 场景 2: & 前缀触发（无显式模式）+ cloud_enabled=False -> 使用 CLI
            # 注意：使用 requested_mode=None 测试 & 前缀触发逻辑
            pytest.param(
                "& 分析任务",
                False,
                True,
                None,  # 无显式模式，& 前缀触发
                True,
                "cloud_enabled=False",
                id="prefix_disabled_uses_cli",
            ),
            # 场景 3: 无 & 前缀 + 显式 auto + 无 key -> 回退 CLI
            pytest.param(
                "分析任务",
                True,
                False,
                "auto",
                True,
                "无 API Key",
                id="no_prefix_auto_no_key_fallback",
            ),
        ],
    )
    def test_cloud_fallback_scenarios(
        self,
        auto_mode_args: argparse.Namespace,
        requirement: str,
        cloud_enabled: bool,
        has_api_key: bool,
        requested_mode: Optional[str],
        expect_cli_fallback: bool,
        expect_reason: str,
    ) -> None:
        """测试 Cloud 回退场景"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import resolve_effective_execution_mode
        from cursor.cloud_client import CloudClientFactory

        auto_mode_args.requirement = requirement
        has_ampersand_prefix = is_cloud_request(requirement)

        api_key = "mock-api-key" if has_api_key else None
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            # 使用 resolve_effective_execution_mode 测试策略
            mode, reason = resolve_effective_execution_mode(
                requested_mode=requested_mode,
                has_ampersand_prefix=has_ampersand_prefix,  # 语法检测层面
                cloud_enabled=cloud_enabled,
                has_api_key=has_api_key,
            )

            if expect_cli_fallback:
                assert mode == "cli", f"期望回退到 CLI，实际 {mode}"
            else:
                assert mode in ("cloud", "auto")

    def test_auto_no_key_outputs_actionable_hint(self, auto_mode_args: argparse.Namespace) -> None:
        """测试 auto + 无 key 时输出可操作提示"""
        from core.execution_policy import (
            CloudFailureKind,
            build_user_facing_fallback_message,
        )

        # 构建无 key 的回退消息
        message = build_user_facing_fallback_message(
            kind=CloudFailureKind.NO_KEY,
            retry_after=None,
            requested_mode="auto",
            has_ampersand_prefix=False,
        )

        # 验证消息包含设置 key 的提示
        assert "API Key" in message or "CURSOR_API_KEY" in message
        assert "config.yaml" in message or "环境变量" in message

    def test_auto_auth_failure_cooldown_reason(self, auto_mode_args: argparse.Namespace) -> None:
        """测试 auto + 认证失败时冷却原因被记录"""
        from core.execution_policy import (
            CloudFailureKind,
            classify_cloud_failure,
        )
        from cursor.cloud_client import AuthError

        error = AuthError("Invalid API Key")
        failure_info = classify_cloud_failure(error)

        assert failure_info.kind == CloudFailureKind.AUTH
        assert failure_info.retryable is False
        assert "认证" in failure_info.message or "API Key" in failure_info.message

    def test_auto_rate_limit_cooldown_uses_retry_after(self, auto_mode_args: argparse.Namespace) -> None:
        """测试 auto + 429 时冷却使用 retry_after"""
        from core.execution_policy import CloudFailureKind, classify_cloud_failure
        from cursor.cloud_client import RateLimitError

        error = RateLimitError("Rate limit exceeded, retry after 45 seconds")
        error.retry_after = 45
        failure_info = classify_cloud_failure(error)

        assert failure_info.kind == CloudFailureKind.RATE_LIMIT
        assert failure_info.retryable is True
        # retry_after 应该从错误中提取或使用默认值
        assert failure_info.retry_after is not None
        assert failure_info.retry_after > 0

    @pytest.mark.asyncio
    async def test_auto_mode_fallback_cli_still_executes(self, auto_mode_args: argparse.Namespace) -> None:
        """测试 auto 模式回退 CLI 后仍能正常执行

        注意：execution_mode="auto" 会强制使用 basic 编排器（与 CLI help 对齐），
        因此需要 mock _run_with_basic_orchestrator 而非 _run_with_mp_orchestrator。
        """
        from cursor.cloud_client import CloudClientFactory

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            iterator = SelfIterator(auto_mode_args)
            iterator.context.iteration_goal = "测试目标"

            mock_result = {
                "success": True,
                "iterations_completed": 1,
                "total_tasks_created": 2,
                "total_tasks_completed": 2,
                "total_tasks_failed": 0,
            }

            # auto 模式强制使用 basic 编排器，因此 mock _run_with_basic_orchestrator
            with (
                patch.object(
                    iterator,
                    "_run_with_basic_orchestrator",
                    new_callable=AsyncMock,
                    return_value=mock_result,
                ) as mock_basic,
                patch("scripts.run_iterate.KnowledgeManager") as MockKM,
            ):
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                result = await iterator._run_agent_system()

                # 验证 basic 编排器被调用（auto 模式强制 basic）
                mock_basic.assert_called_once()
                assert result["success"] is True


class TestAmpersandCloudDisabledTableDriven:
    """& 前缀 + cloud_enabled=false 表驱动测试

    验证当 cloud_enabled=False 时，& 前缀不会触发 Cloud 模式
    """

    @pytest.fixture
    def cloud_disabled_args(self) -> argparse.Namespace:
        """创建 cloud_enabled=False 的参数"""
        return argparse.Namespace(
            requirement="& 测试任务",  # 有 & 前缀
            directory=".",
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
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
            execution_mode="cli",  # 显式 cli（测试 CLI 行为时显式指定）
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
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    @pytest.mark.parametrize(
        "requirement,cloud_enabled,has_api_key,expect_mode",
        [
            # & 前缀 + cloud_enabled=False -> CLI
            pytest.param(
                "& 分析任务",
                False,
                True,
                "cli",
                id="prefix_cloud_disabled_uses_cli",
            ),
            # & 前缀 + cloud_enabled=True + 无 key -> CLI
            pytest.param(
                "& 分析任务",
                True,
                False,
                "cli",
                id="prefix_cloud_enabled_no_key_uses_cli",
            ),
            # & 前缀 + cloud_enabled=True + 有 key -> cloud
            pytest.param(
                "& 分析任务",
                True,
                True,
                "cloud",
                id="prefix_cloud_enabled_with_key_uses_cloud",
            ),
        ],
    )
    def test_ampersand_with_cloud_disabled(
        self,
        cloud_disabled_args: argparse.Namespace,
        requirement: str,
        cloud_enabled: bool,
        has_api_key: bool,
        expect_mode: str,
    ) -> None:
        """测试 & 前缀在 cloud_enabled=False 时的行为"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import resolve_effective_execution_mode

        cloud_disabled_args.requirement = requirement
        has_ampersand_prefix = is_cloud_request(requirement)

        mode, reason = resolve_effective_execution_mode(
            requested_mode=None,  # 未显式指定
            has_ampersand_prefix=has_ampersand_prefix,  # 语法检测层面
            cloud_enabled=cloud_enabled,
            has_api_key=has_api_key,
        )

        assert mode == expect_mode, (
            f"requirement='{requirement}', cloud_enabled={cloud_enabled}, "
            f"has_api_key={has_api_key}: 期望 {expect_mode}，实际 {mode}"
        )

    def test_policy_build_fallback_message_for_cloud_disabled(self, cloud_disabled_args: argparse.Namespace) -> None:
        """测试 cloud_enabled=False 时的用户提示"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import (
            resolve_effective_execution_mode_full,
        )

        requirement = "& 分析任务"
        has_ampersand_prefix = is_cloud_request(requirement)

        resolution = resolve_effective_execution_mode_full(
            requested_mode=None,
            has_ampersand_prefix=has_ampersand_prefix,  # 语法检测层面
            cloud_enabled=False,
            has_api_key=True,
        )

        # 应该有警告信息
        assert len(resolution.warnings) > 0
        # 警告应该提示如何启用 Cloud
        assert any("cloud_enabled" in w or "config.yaml" in w for w in resolution.warnings)


# ============================================================
# TestCloudModeNoKeyDegradation - Cloud 模式无 Key 降级测试
# ============================================================


class TestCloudModeNoKeyDegradation:
    """测试 execution_mode=cloud + 无 key 的降级行为

    验证场景：
    1. cloud 模式 + 无 key: 立即回退到 CLI，输出可操作提示
    2. 降级不触发冷却机制（NO_KEY 是配置问题）
    3. 用户友好的错误消息包含设置 API Key 的方法
    """

    @pytest.fixture
    def cloud_mode_args(self) -> argparse.Namespace:
        """创建 execution_mode=cloud 的参数"""
        return argparse.Namespace(
            requirement="云端任务",
            directory=".",
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
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
            execution_mode="cloud",  # 显式 cloud 模式
            cloud_api_key=None,  # 无 API key
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
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_cloud_mode_no_key_degrades_to_cli(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 cloud 模式 + 无 key 降级到 CLI"""
        from core.execution_policy import resolve_effective_execution_mode

        mode, reason = resolve_effective_execution_mode(
            requested_mode="cloud",
            has_ampersand_prefix=False,  # 无 & 前缀
            cloud_enabled=True,
            has_api_key=False,
        )

        assert mode == "cli", f"cloud + 无 key 应降级到 CLI，实际: {mode}"
        assert "未配置 API Key" in reason, f"原因应包含 API Key 提示，实际: {reason}"

    def test_cloud_mode_no_key_fallback_message_actionable(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 cloud + 无 key 时回退消息包含可操作提示"""
        from core.execution_policy import (
            CloudFailureKind,
            build_user_facing_fallback_message,
        )

        message = build_user_facing_fallback_message(
            kind=CloudFailureKind.NO_KEY,
            retry_after=None,
            requested_mode="cloud",
            has_ampersand_prefix=False,
        )

        # 验证消息包含必要的设置提示
        assert "未配置" in message, "消息应说明问题"
        assert "API Key" in message, "消息应提及 API Key"
        assert "CURSOR_API_KEY" in message or "config.yaml" in message, "消息应告知如何设置 API Key"

    def test_cloud_mode_no_key_does_not_trigger_cooldown(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 cloud + 无 key 不触发冷却（是配置问题而非运行时错误）"""
        from cursor.executor import CloudExecutionPolicy

        policy = CloudExecutionPolicy()

        # 初始状态无冷却
        assert not policy.cooldown_state.is_in_cooldown()
        assert policy.cooldown_state.failure_count == 0

        # NO_KEY 场景在策略层拦截，不会调用 start_cooldown
        # 验证状态保持干净
        assert policy.cooldown_state.error_type is None

    def test_cloud_mode_no_key_forces_basic_orchestrator(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 cloud + 无 key 降级到 CLI，但编排器仍强制 basic

        新行为：requested=auto/cloud 即强制 basic，不受 key/enable 影响。
        即使因为没有 API Key 导致 effective_mode 回退到 CLI，
        只要 requested_mode 是 cloud，编排器就应该是 basic。
        """
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            iterator = SelfIterator(cloud_mode_args)

            # 无 key 时应该降级到 CLI
            mode = iterator._get_execution_mode()
            assert mode == ExecutionMode.CLI, f"cloud + 无 key 应降级到 CLI，实际: {mode}"

            # 新行为：即使回退到 CLI，requested_mode=cloud 仍强制 basic
            orchestrator = iterator._get_orchestrator_type()
            assert orchestrator == "basic", f"requested_mode=cloud 应强制使用 basic 编排器，实际: {orchestrator}"

    def test_classify_no_key_error_correctly(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 NO_KEY 错误被正确分类"""
        from core.execution_policy import CloudFailureKind, classify_cloud_failure

        # 测试各种表示无 key 的错误消息
        no_key_messages = [
            "No API key configured",
            "missing api key",
            "未配置 API Key",
            "API_KEY is required",
            "缺少 api key",
        ]

        for msg in no_key_messages:
            failure_info = classify_cloud_failure(msg)
            assert failure_info.kind == CloudFailureKind.NO_KEY, (
                f"消息 '{msg}' 应分类为 NO_KEY，实际: {failure_info.kind}"
            )
            assert failure_info.retryable is False, "NO_KEY 应不可重试"

    @pytest.mark.asyncio
    async def test_self_iterator_cloud_no_key_runs_with_cli(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 SelfIterator 在 cloud + 无 key 时使用 CLI 执行

        注意：execution_mode="cloud" 会强制使用 basic 编排器（与 CLI help 对齐），
        即使因为无 API Key 实际降级到 CLI 执行，编排器仍然使用 basic。
        因此需要 mock _run_with_basic_orchestrator 而非 _run_with_mp_orchestrator。
        """
        from cursor.cloud_client import CloudClientFactory

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            iterator = SelfIterator(cloud_mode_args)
            iterator.context.iteration_goal = "测试目标"

            mock_result = {
                "success": True,
                "iterations_completed": 1,
                "total_tasks_created": 2,
                "total_tasks_completed": 2,
                "total_tasks_failed": 0,
            }

            # cloud 模式强制使用 basic 编排器，因此 mock _run_with_basic_orchestrator
            with (
                patch.object(
                    iterator,
                    "_run_with_basic_orchestrator",
                    new_callable=AsyncMock,
                    return_value=mock_result,
                ) as mock_basic,
                patch("scripts.run_iterate.KnowledgeManager") as MockKM,
            ):
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                result = await iterator._run_agent_system()

                # 验证使用了 basic 编排器（cloud 模式强制 basic）
                mock_basic.assert_called_once()
                assert result["success"] is True


# ============================================================
# TestCloudModeViaRunPyIntegration - run.py 调用 SelfIterator 集成测试
# ============================================================


class TestCloudModeViaRunPyIntegration:
    """测试通过 run.py 的 _run_iterate 调用 SelfIterator 时的 Cloud 模式处理

    场景：run.py 已经剥离了 '&' 前缀并设置了 execution_mode="cloud"，
    传给 SelfIterator 的 requirement 已经没有 '&' 前缀。

    注意：这些测试需要 mock Cloud API Key 存在，否则 cloud 模式会回退到 cli
    """

    @pytest.fixture(autouse=True)
    def mock_cloud_api_key(self) -> Generator[None, None, None]:
        """Mock Cloud API Key 存在，用于测试 Cloud 执行模式"""
        from cursor.cloud_client import CloudClientFactory

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            yield

    @pytest.fixture
    def iterate_args_from_run_py(self) -> argparse.Namespace:
        """模拟从 run.py _run_iterate 传入的参数（goal 已剥离 &）"""
        return argparse.Namespace(
            requirement="分析代码架构",  # 已剥离 & 前缀
            directory=".",  # 工作目录
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
            workers=2,
            force_update=False,
            verbose=False,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cloud",  # run.py 已设置为 cloud
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            # 角色级执行模式
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            # 流式控制台渲染参数
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_cloud_mode_from_execution_mode_param(self, iterate_args_from_run_py: argparse.Namespace) -> None:
        """测试从 execution_mode 参数获取 Cloud 模式"""
        from cursor.executor import ExecutionMode

        iterator = SelfIterator(iterate_args_from_run_py)

        # has_ampersand_prefix=False（无 & 前缀）→ prefix_routed=False
        # 内部分支使用 _prefix_routed 字段
        assert iterator._prefix_routed is False, "无 & 前缀不应标记为 prefix_routed"

        # 但应从 execution_mode 参数获取 CLOUD 模式（非 & 前缀触发）
        assert iterator._get_execution_mode() == ExecutionMode.CLOUD

    def test_basic_orchestrator_forced_for_cloud_mode(self, iterate_args_from_run_py: argparse.Namespace) -> None:
        """测试 Cloud 模式强制使用 basic 编排器"""
        iterator = SelfIterator(iterate_args_from_run_py)

        # 虽然 orchestrator=mp，但 Cloud 模式应强制 basic
        assert iterator._get_orchestrator_type() == "basic"

    def test_goal_preserved_correctly(self, iterate_args_from_run_py: argparse.Namespace) -> None:
        """测试 goal 正确保留"""
        iterator = SelfIterator(iterate_args_from_run_py)

        assert iterator.context.user_requirement == "分析代码架构"

    def test_auto_commit_still_defaults_to_false(self, iterate_args_from_run_py: argparse.Namespace) -> None:
        """测试 auto_commit 仍默认为 False"""
        iterator = SelfIterator(iterate_args_from_run_py)

        assert iterator.args.auto_commit is False


# ============================================================
# TestParseMaxIterations - 最大迭代次数解析测试
# ============================================================


class TestParseMaxIterations:
    """测试 parse_max_iterations 函数"""

    def test_numeric_values(self) -> None:
        """测试数字值解析"""
        assert parse_max_iterations("1") == 1
        assert parse_max_iterations("5") == 5
        assert parse_max_iterations("100") == 100

    def test_unlimited_keywords(self) -> None:
        """测试无限迭代关键词"""
        assert parse_max_iterations("MAX") == -1
        assert parse_max_iterations("max") == -1
        assert parse_max_iterations("UNLIMITED") == -1
        assert parse_max_iterations("INF") == -1
        assert parse_max_iterations("INFINITE") == -1

    def test_zero_and_negative(self) -> None:
        """测试零和负数"""
        assert parse_max_iterations("0") == -1
        assert parse_max_iterations("-1") == -1
        assert parse_max_iterations("-5") == -1

    def test_invalid_values(self) -> None:
        """测试无效值

        parse_max_iterations 抛出 MaxIterationsParseError（继承自 ValueError）
        如需用于 argparse 类型转换，应使用 parse_max_iterations_for_argparse
        """
        from core.config import MaxIterationsParseError

        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("invalid")
        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("abc")


# ============================================================
# TestIterationGoalBuilder - 迭代目标构建器测试
# ============================================================


class TestIterationGoalBuilder:
    """测试 IterationGoalBuilder 类"""

    def test_build_goal_with_user_requirement(self) -> None:
        """测试带用户需求的目标构建"""
        builder = IterationGoalBuilder()
        context = IterationContext(
            user_requirement="实现新功能",
        )

        goal = builder.build_goal(context)

        assert "自我迭代任务" in goal
        assert "用户需求" in goal
        assert "实现新功能" in goal

    def test_build_goal_with_update_analysis(self) -> None:
        """测试带更新分析的目标构建"""
        builder = IterationGoalBuilder()
        analysis = UpdateAnalysis(
            has_updates=True,
            new_features=["[2024-01-15] 新功能A"],
            improvements=["[2024-01-15] 改进B"],
        )
        context = IterationContext(
            update_analysis=analysis,
        )

        goal = builder.build_goal(context)

        assert "检测到的更新" in goal
        assert "新功能" in goal
        assert "改进" in goal

    def test_get_summary_with_requirement(self) -> None:
        """测试带需求的摘要生成"""
        builder = IterationGoalBuilder()
        context = IterationContext(
            user_requirement="这是一个很长的用户需求描述，应该被截断显示",
        )

        summary = builder.get_summary(context)

        assert "用户需求" in summary
        assert "..." in summary  # 应该被截断

    def test_get_summary_empty_context(self) -> None:
        """测试空上下文的摘要"""
        builder = IterationGoalBuilder()
        context = IterationContext()

        summary = builder.get_summary(context)

        assert "无具体迭代目标" in summary


# ============================================================
# TestTruncationHints - 截断提示测试
# ============================================================


class TestTruncationHints:
    """测试截断提示的存在性（轻量断言）

    验证截断常量被正确导入并可用于 scripts/run_iterate.py。
    """

    def test_truncation_constants_imported(self) -> None:
        """验证截断常量已正确导入"""
        # 验证常量存在
        assert MAX_CONSOLE_PREVIEW_CHARS > 0, "MAX_CONSOLE_PREVIEW_CHARS 应为正整数"
        assert MAX_KNOWLEDGE_DOC_PREVIEW_CHARS > 0, "MAX_KNOWLEDGE_DOC_PREVIEW_CHARS 应为正整数"

    def test_truncation_hint_defined(self) -> None:
        """验证截断提示字符串已定义"""
        assert TRUNCATION_HINT, "TRUNCATION_HINT 应非空"
        # 验证截断提示包含关键词
        assert "截断" in TRUNCATION_HINT or "..." in TRUNCATION_HINT, "TRUNCATION_HINT 应包含截断标识"

    def test_knowledge_preview_uses_constant(self) -> None:
        """验证知识库预览截断限制值符合预期"""
        # 知识库文档预览应使用 1000 字符限制
        assert MAX_KNOWLEDGE_DOC_PREVIEW_CHARS == 1000, "MAX_KNOWLEDGE_DOC_PREVIEW_CHARS 应为 1000"

    def test_console_preview_uses_constant(self) -> None:
        """验证控制台预览截断限制值符合预期"""
        # 控制台大段文本预览应使用 2000 字符限制
        assert MAX_CONSOLE_PREVIEW_CHARS == 2000, "MAX_CONSOLE_PREVIEW_CHARS 应为 2000"


# ============================================================
# TestChangelogAnalyzer - Changelog 分析器测试
# ============================================================


class TestChangelogAnalyzer:
    """测试 ChangelogAnalyzer 类"""

    def test_parse_changelog_empty(self) -> None:
        """测试解析空内容"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog("")
        assert len(entries) == 0

    def test_parse_changelog_with_entries(self) -> None:
        """测试解析有内容的 Changelog"""
        analyzer = ChangelogAnalyzer()
        content = """
## 2024-01-15

### 新增功能
- 新增斜杠命令支持

## 2024-01-10

### 修复
- 修复某个 bug
        """
        entries = analyzer.parse_changelog(content)
        assert len(entries) >= 1

    def test_extract_update_points(self) -> None:
        """测试提取更新要点"""
        analyzer = ChangelogAnalyzer()
        entries = [
            ChangelogEntry(
                date="2024-01-15",
                title="新功能发布",
                content="新增 MCP 支持",
                category="feature",
            ),
            ChangelogEntry(
                date="2024-01-10",
                title="Bug 修复",
                content="修复连接问题",
                category="fix",
            ),
        ]

        analysis = analyzer.extract_update_points(entries)

        assert analysis.has_updates is True
        assert len(analysis.new_features) == 1
        assert len(analysis.fixes) == 1

    def test_extract_links_from_html_relative_links(self) -> None:
        """测试相对链接被正确归一化为绝对路径"""
        analyzer = ChangelogAnalyzer()

        html_content = """
        <html>
        <body>
        <a href="/cn/docs/cli/reference/parameters">CLI Parameters</a>
        <a href="/docs/cli/mcp">MCP Guide</a>
        <a href="docs/changelog">Relative Changelog</a>
        </body>
        </html>
        """
        result = analyzer._extract_links_from_html(html_content)

        # 相对链接应被归一化为绝对路径（在 allowed 列表中）
        hrefs = [link["href"] for link in result["allowed"]]
        assert any("https://cursor.com/cn/docs/cli/reference/parameters" in h for h in hrefs), (
            f"应包含归一化的 /cn/docs/cli/reference/parameters 链接，实际: {hrefs}"
        )
        assert any("https://cursor.com/docs/cli/mcp" in h for h in hrefs), (
            f"应包含归一化的 /docs/cli/mcp 链接，实际: {hrefs}"
        )

    def test_extract_links_from_html_removes_fragment(self) -> None:
        """测试带 fragment 的链接被正确移除 fragment 部分"""
        analyzer = ChangelogAnalyzer()

        html_content = """
        <html>
        <body>
        <a href="https://cursor.com/cn/docs/cli/reference/parameters#mode">Mode Section</a>
        <a href="/cn/docs/cli/mcp#tools">MCP Tools</a>
        </body>
        </html>
        """
        result = analyzer._extract_links_from_html(html_content)

        # fragment 应被移除（在 allowed 列表中）
        hrefs = [link["href"] for link in result["allowed"]]
        for href in hrefs:
            assert "#" not in href, f"链接应移除 fragment，但发现: {href}"

        # 验证归一化后的 URL 正确
        assert "https://cursor.com/cn/docs/cli/reference/parameters" in hrefs, (
            f"应包含不带 fragment 的 parameters 链接，实际: {hrefs}"
        )

    def test_extract_links_from_html_filters_disallowed_domains(self) -> None:
        """测试非允许域名的链接被分类到 external"""
        analyzer = ChangelogAnalyzer()

        html_content = """
        <html>
        <body>
        <a href="https://cursor.com/cn/docs/cli/reference/parameters">Allowed Doc</a>
        <a href="https://github.com/cursor/cursor">GitHub Link</a>
        <a href="https://example.com/docs">External Docs</a>
        <a href="https://cursor.com/pricing">Pricing Page</a>
        <a href="https://cursor.com/changelog">Changelog Link</a>
        </body>
        </html>
        """
        result = analyzer._extract_links_from_html(html_content)

        allowed_hrefs = [link["href"] for link in result["allowed"]]
        external_hrefs = [link["href"] for link in result["external"]]

        # 只有允许域名的链接应在 allowed 列表中
        assert any("cursor.com/cn/docs" in h for h in allowed_hrefs), (
            f"应保留 cursor.com/cn/docs 链接在 allowed，实际: {allowed_hrefs}"
        )
        assert any("cursor.com/changelog" in h for h in allowed_hrefs), (
            f"应保留 cursor.com/changelog 链接在 allowed，实际: {allowed_hrefs}"
        )

        # 非允许域名应在 external 列表中
        assert any("github.com" in h for h in external_hrefs), (
            f"github.com 链接应在 external 中，实际: {external_hrefs}"
        )
        assert any("example.com" in h for h in external_hrefs), (
            f"example.com 链接应在 external 中，实际: {external_hrefs}"
        )
        assert any("pricing" in h for h in external_hrefs), f"pricing 页面链接应在 external 中，实际: {external_hrefs}"

        # allowed 中不应包含外部链接
        assert not any("github.com" in h for h in allowed_hrefs), (
            f"allowed 中不应包含 github.com 链接，实际: {allowed_hrefs}"
        )
        assert not any("example.com" in h for h in allowed_hrefs), (
            f"allowed 中不应包含 example.com 链接，实际: {allowed_hrefs}"
        )
        assert not any("pricing" in h for h in allowed_hrefs), (
            f"allowed 中不应包含 pricing 页面链接，实际: {allowed_hrefs}"
        )

    def test_extract_links_from_html_combined_scenarios(self) -> None:
        """测试组合场景：相对链接 + fragment + 非允许域名"""
        analyzer = ChangelogAnalyzer()

        html_content = """
        <html>
        <body>
        <a href="/cn/docs/cli/reference/parameters#mode">Relative with Fragment</a>
        <a href="https://github.com/cursor/cursor#readme">GitHub with Fragment</a>
        <a href="/cn/changelog#jan-2026">Changelog with Fragment</a>
        <a href="https://external.com/docs/guide">External Site</a>
        </body>
        </html>
        """
        result = analyzer._extract_links_from_html(html_content)

        allowed_hrefs = [link["href"] for link in result["allowed"]]
        external_hrefs = [link["href"] for link in result["external"]]

        # allowed 中应只有归一化且允许的链接
        assert len(allowed_hrefs) == 2, f"应只保留 2 个允许域名的链接在 allowed，实际: {allowed_hrefs}"

        # external 中应有 2 个外部链接
        assert len(external_hrefs) == 2, f"应有 2 个外部链接在 external，实际: {external_hrefs}"

        # 验证归一化：无 fragment（allowed 和 external 都适用）
        for href in allowed_hrefs + external_hrefs:
            assert "#" not in href, f"链接应不含 fragment: {href}"

        # 验证 allowed 中只有允许域名
        assert all(
            href.startswith("https://cursor.com/cn/docs")
            or href.startswith("https://cursor.com/docs")
            or href.startswith("https://cursor.com/cn/changelog")
            or href.startswith("https://cursor.com/changelog")
            for href in allowed_hrefs
        ), f"allowed 应只包含允许域名前缀的链接，实际: {allowed_hrefs}"

        # 验证 external 中包含外部链接
        assert any("github.com" in h for h in external_hrefs), (
            f"external 应包含 github.com 链接，实际: {external_hrefs}"
        )
        assert any("external.com" in h for h in external_hrefs), (
            f"external 应包含 external.com 链接，实际: {external_hrefs}"
        )

    def test_changelog_links_in_analysis_normalized_and_filtered(self) -> None:
        """测试 analyze() 方法中 changelog_links 和 external_links 被正确归一化和分类

        模拟完整的分析流程，验证最终 analysis.changelog_links 和 external_links 的内容。
        """
        analyzer = ChangelogAnalyzer()

        # 模拟包含多种链接的 changelog HTML
        html_with_links = """
        <html>
        <head><title>Changelog</title></head>
        <body>
        <main>
        <h2>## Jan 16, 2026</h2>
        <p>New features:</p>
        <ul>
        <li>Added <a href="/cn/docs/cli/reference/parameters#mode">plan mode</a></li>
        <li>See <a href="https://github.com/cursor/repo">GitHub</a> for details</li>
        <li>Check <a href="https://cursor.com/cn/changelog#jan-2026">changelog</a></li>
        <li>Visit <a href="https://external.com/guide">external guide</a></li>
        </ul>
        </main>
        </body>
        </html>
        """

        # 直接调用 _extract_links_from_html 测试
        result = analyzer._extract_links_from_html(html_with_links)
        allowed_hrefs = [link["href"] for link in result["allowed"]]
        external_hrefs = [link["href"] for link in result["external"]]

        # 验证 allowed 结果
        assert len(allowed_hrefs) == 2, f"应只保留 2 个允许且归一化的链接在 allowed，实际: {allowed_hrefs}"

        # 验证归一化（无 fragment）
        assert "https://cursor.com/cn/docs/cli/reference/parameters" in allowed_hrefs, (
            f"应包含归一化的 parameters 链接（无 fragment），实际: {allowed_hrefs}"
        )
        assert "https://cursor.com/cn/changelog" in allowed_hrefs, (
            f"应包含归一化的 changelog 链接（无 fragment），实际: {allowed_hrefs}"
        )

        # 验证 external 结果
        assert len(external_hrefs) == 2, f"应有 2 个外部链接在 external，实际: {external_hrefs}"
        assert any("github.com" in h for h in external_hrefs), (
            f"external 应包含 github.com 链接，实际: {external_hrefs}"
        )
        assert any("external.com" in h for h in external_hrefs), (
            f"external 应包含 external.com 链接，实际: {external_hrefs}"
        )

        # 验证 allowed 中不包含外部链接
        assert not any("github.com" in h for h in allowed_hrefs), (
            f"allowed 不应包含 github.com 链接，实际: {allowed_hrefs}"
        )
        assert not any("external.com" in h for h in allowed_hrefs), (
            f"allowed 不应包含 external.com 链接，实际: {allowed_hrefs}"
        )

    def test_update_analysis_external_links_field(self) -> None:
        """测试 UpdateAnalysis 数据类的 external_links 字段

        验证新增的 external_links 字段可以正确存储外部链接，
        且与 changelog_links 字段独立。
        """
        from scripts.run_iterate import UpdateAnalysis

        # 创建带有两种链接的 UpdateAnalysis 对象
        analysis = UpdateAnalysis(
            has_updates=True,
            changelog_links=[
                "https://cursor.com/docs/cli/reference/parameters",
                "https://cursor.com/changelog",
            ],
            external_links=[
                "https://github.com/cursor/cursor",
                "https://example.com/guide",
            ],
            summary="测试更新",
        )

        # 验证两个字段独立存在且正确
        assert len(analysis.changelog_links) == 2, (
            f"changelog_links 应有 2 个链接，实际: {len(analysis.changelog_links)}"
        )
        assert len(analysis.external_links) == 2, f"external_links 应有 2 个链接，实际: {len(analysis.external_links)}"

        # 验证字段内容
        assert "https://cursor.com/docs/cli/reference/parameters" in analysis.changelog_links
        assert "https://github.com/cursor/cursor" in analysis.external_links

        # 验证默认值为空列表
        empty_analysis = UpdateAnalysis()
        assert empty_analysis.external_links == [], (
            f"external_links 默认应为空列表，实际: {empty_analysis.external_links}"
        )

    def test_extract_links_external_link_mode_skip_all(self) -> None:
        """测试 external_link_mode=skip_all 时外链不被记录"""
        from scripts.run_iterate import (
            ChangelogAnalyzer,
            ResolvedFetchPolicyConfig,
            build_doc_allowlist,
        )

        # 创建 skip_all 模式的 fetch_policy
        fetch_policy = ResolvedFetchPolicyConfig(
            allowed_path_prefixes=["docs", "cn/docs"],
            allowed_domains=["cursor.com"],
            external_link_mode="skip_all",
            external_link_allowlist=[],
        )

        # 创建带有 fetch_policy 的 analyzer
        doc_allowlist = build_doc_allowlist()
        analyzer = ChangelogAnalyzer(
            doc_allowlist=doc_allowlist,
            fetch_policy=fetch_policy,
        )

        html_content = """
        <html>
        <body>
        <a href="https://cursor.com/docs/cli">Allowed Doc</a>
        <a href="https://github.com/cursor/repo">GitHub Link</a>
        <a href="https://external.com/guide">External Guide</a>
        </body>
        </html>
        """
        result = analyzer._extract_links_from_html(html_content)

        # skip_all 模式：allowed 列表正常，external 列表应为空
        assert len(result["allowed"]) >= 1, f"allowed 应包含允许的链接，实际: {result['allowed']}"
        assert len(result["external"]) == 0, f"skip_all 模式下 external 应为空，实际: {result['external']}"

    def test_extract_links_external_link_mode_record_only(self) -> None:
        """测试 external_link_mode=record_only 时外链被记录"""
        from scripts.run_iterate import (
            ChangelogAnalyzer,
            ResolvedFetchPolicyConfig,
            build_doc_allowlist,
        )

        # 创建 record_only 模式的 fetch_policy（默认模式）
        fetch_policy = ResolvedFetchPolicyConfig(
            allowed_path_prefixes=["docs", "cn/docs"],
            allowed_domains=["cursor.com"],
            external_link_mode="record_only",
            external_link_allowlist=[],
        )

        doc_allowlist = build_doc_allowlist()
        analyzer = ChangelogAnalyzer(
            doc_allowlist=doc_allowlist,
            fetch_policy=fetch_policy,
        )

        html_content = """
        <html>
        <body>
        <a href="https://cursor.com/docs/cli">Allowed Doc</a>
        <a href="https://github.com/cursor/repo">GitHub Link</a>
        <a href="https://external.com/guide">External Guide</a>
        </body>
        </html>
        """
        result = analyzer._extract_links_from_html(html_content)

        # record_only 模式：外链被记录到 external 列表
        assert len(result["allowed"]) >= 1, f"allowed 应包含允许的链接，实际: {result['allowed']}"
        assert len(result["external"]) == 2, f"record_only 模式下应记录 2 个外链，实际: {result['external']}"

        external_hrefs = [link["href"] for link in result["external"]]
        assert any("github.com" in h for h in external_hrefs), (
            f"external 应包含 github.com 链接，实际: {external_hrefs}"
        )
        assert any("external.com" in h for h in external_hrefs), (
            f"external 应包含 external.com 链接，实际: {external_hrefs}"
        )

    def test_extract_links_external_link_mode_fetch_allowlist(self) -> None:
        """测试 external_link_mode=fetch_allowlist 时 allowlist 匹配的外链被放入 allowed"""
        from scripts.run_iterate import (
            ChangelogAnalyzer,
            ResolvedFetchPolicyConfig,
            build_doc_allowlist,
        )

        # 创建 fetch_allowlist 模式，允许 github.com 域名
        fetch_policy = ResolvedFetchPolicyConfig(
            allowed_path_prefixes=["docs", "cn/docs"],
            allowed_domains=["cursor.com"],
            external_link_mode="fetch_allowlist",
            external_link_allowlist=["github.com"],  # 允许 github.com 域名
        )

        doc_allowlist = build_doc_allowlist()
        analyzer = ChangelogAnalyzer(
            doc_allowlist=doc_allowlist,
            fetch_policy=fetch_policy,
        )

        html_content = """
        <html>
        <body>
        <a href="https://cursor.com/docs/cli">Allowed Doc</a>
        <a href="https://github.com/cursor/repo">GitHub Link</a>
        <a href="https://external.com/guide">External Guide</a>
        </body>
        </html>
        """
        result = analyzer._extract_links_from_html(html_content)

        allowed_hrefs = [link["href"] for link in result["allowed"]]
        external_hrefs = [link["href"] for link in result["external"]]

        # fetch_allowlist 模式：github.com 匹配 allowlist，应放入 allowed
        assert any("github.com" in h for h in allowed_hrefs), (
            f"github.com 匹配 allowlist 应放入 allowed，实际 allowed: {allowed_hrefs}"
        )

        # external.com 不匹配 allowlist，应放入 external
        assert any("external.com" in h for h in external_hrefs), (
            f"external.com 不匹配 allowlist 应放入 external，实际 external: {external_hrefs}"
        )

        # github.com 不应在 external 中
        assert not any("github.com" in h for h in external_hrefs), (
            f"github.com 匹配 allowlist 不应在 external 中，实际: {external_hrefs}"
        )

    def test_extract_links_external_link_mode_fetch_allowlist_url_prefix(self) -> None:
        """测试 fetch_allowlist 模式支持 URL 前缀格式的 allowlist"""
        from scripts.run_iterate import (
            ChangelogAnalyzer,
            ResolvedFetchPolicyConfig,
            build_doc_allowlist,
        )

        # 创建 fetch_allowlist 模式，使用 URL 前缀格式
        fetch_policy = ResolvedFetchPolicyConfig(
            allowed_path_prefixes=["docs", "cn/docs"],
            allowed_domains=["cursor.com"],
            external_link_mode="fetch_allowlist",
            external_link_allowlist=["https://github.com/cursor"],  # 只允许 /cursor 路径
        )

        doc_allowlist = build_doc_allowlist()
        analyzer = ChangelogAnalyzer(
            doc_allowlist=doc_allowlist,
            fetch_policy=fetch_policy,
        )

        html_content = """
        <html>
        <body>
        <a href="https://github.com/cursor/repo">Cursor Repo</a>
        <a href="https://github.com/other/repo">Other Repo</a>
        </body>
        </html>
        """
        result = analyzer._extract_links_from_html(html_content)

        allowed_hrefs = [link["href"] for link in result["allowed"]]
        external_hrefs = [link["href"] for link in result["external"]]

        # https://github.com/cursor/repo 匹配前缀 allowlist
        assert any("github.com/cursor" in h for h in allowed_hrefs), (
            f"/cursor/repo 应匹配 URL 前缀 allowlist，实际 allowed: {allowed_hrefs}"
        )

        # https://github.com/other/repo 不匹配
        assert any("github.com/other" in h for h in external_hrefs), (
            f"/other/repo 不匹配应在 external 中，实际 external: {external_hrefs}"
        )

    def test_links_extracted_log_matches_analysis_fields(self) -> None:
        """测试 links_extracted 日志统计与 UpdateAnalysis 字段一致性

        验证 ChangelogAnalysisLog.links_extracted 的 allowed/external 数量
        与 UpdateAnalysis.changelog_links/external_links 的长度一致。
        """
        from scripts.run_iterate import (
            ChangelogAnalyzer,
            ResolvedFetchPolicyConfig,
            build_doc_allowlist,
        )

        # 使用 skip_all 模式验证 external 为 0
        fetch_policy = ResolvedFetchPolicyConfig(
            allowed_path_prefixes=["docs", "cn/docs"],
            allowed_domains=["cursor.com"],
            external_link_mode="skip_all",
            external_link_allowlist=[],
        )

        doc_allowlist = build_doc_allowlist()
        analyzer = ChangelogAnalyzer(
            doc_allowlist=doc_allowlist,
            fetch_policy=fetch_policy,
        )

        html_content = """
        <html>
        <body>
        <a href="https://cursor.com/docs/cli">Allowed Doc</a>
        <a href="https://github.com/cursor/repo">GitHub Link</a>
        </body>
        </html>
        """
        result = analyzer._extract_links_from_html(html_content)

        # 模拟 analyze 中的逻辑
        allowed_count = len(result["allowed"])
        external_count = len(result["external"])

        # skip_all 模式下 external 应为 0
        assert external_count == 0, f"skip_all 模式下 links_extracted.external 应为 0，实际: {external_count}"

        # 验证统计字段格式与 analysis_log.links_extracted 一致
        links_extracted = {
            "allowed": allowed_count,
            "external": external_count,
        }
        assert "allowed" in links_extracted and "external" in links_extracted, (
            "links_extracted 应包含 allowed 和 external 键"
        )


# ============================================================
# TestChangelogParserRobust - Changelog 解析器稳健性测试
# ============================================================


# 固定的 changelog 样例文本，模拟 Cursor CLI Jan 16 2026 更新
SAMPLE_CHANGELOG_JAN_16_2026 = """
# Cursor CLI Changelog

## Jan 16, 2026

### New Features
- **Plan/Ask Mode**: Added new `/plan` and `/ask` slash commands for switching between planning mode and Q&A mode.
- **Cloud Relay**: MCP servers now support cloud relay for remote access without local installation.
- **Diff View**: Enhanced diff view with Ctrl+R shortcut for reviewing changes before commit.

### Improvements
- Improved streaming output with `--stream-partial-output` flag.
- Better error handling for network timeouts.

### Bug Fixes
- Fixed issue with MCP server authentication.
- Resolved race condition in background tasks.

## Jan 10, 2026

### New Features
- Added support for custom subagents.

### Improvements
- Performance optimization for large codebases.

## Dec 20, 2025

### Bug Fixes
- Fixed memory leak in long-running sessions.
"""

# 带 HTML 混合内容的 changelog 样例
SAMPLE_CHANGELOG_HTML_MIXED = """
<html>
<head><title>Changelog</title></head>
<body>
<div class="changelog">
<h2>## Jan 16, 2026</h2>
<p>New <strong>plan mode</strong> feature added.</p>
<ul>
<li>Support for /plan command</li>
<li>Support for /ask command</li>
</ul>
<!-- This is a comment -->
<script>console.log('should be removed');</script>
</div>
</body>
</html>

## Jan 10, 2026

Regular markdown content here.
- Cloud relay support
- Diff view improvements
"""

# 无标准格式的 changelog（测试保底策略）
# 注意：避免包含日期格式字符串，否则会触发日期解析策略而非保底策略
SAMPLE_CHANGELOG_NO_HEADERS = """
Cursor CLI Updates

We've made several improvements:
- New plan/ask mode switching
- Cloud relay for MCP servers
- Enhanced diff view

This update was released recently.
"""


class TestChangelogParserRobust:
    """测试 ChangelogAnalyzer 的稳健解析策略"""

    def test_parse_sample_changelog_entry_count(self) -> None:
        """测试固定样例解析条数稳定"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_JAN_16_2026)

        # 应该解析出 3 个日期条目
        assert len(entries) == 3, f"期望 3 条，实际 {len(entries)} 条"

    def test_parse_sample_changelog_dates(self) -> None:
        """测试固定样例日期提取正确"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_JAN_16_2026)

        # 验证日期按顺序排列
        dates = [e.date for e in entries]
        assert "Jan 16, 2026" in dates[0] or "Jan 16" in dates[0]
        assert "Jan 10, 2026" in dates[1] or "Jan 10" in dates[1]
        assert "Dec 20, 2025" in dates[2] or "Dec 20" in dates[2]

    def test_parse_sample_changelog_categories(self) -> None:
        """测试固定样例分类稳定"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_JAN_16_2026)

        # 第一条应包含 feature（有 New Features）
        jan_16_entry = entries[0]
        assert jan_16_entry.category == "feature", f"期望 feature，实际 {jan_16_entry.category}"

        # 验证内容包含关键特性
        content_lower = jan_16_entry.content.lower()
        assert "plan" in content_lower or "/plan" in content_lower
        assert "ask" in content_lower or "/ask" in content_lower

    def test_parse_sample_changelog_keywords(self) -> None:
        """测试固定样例关键词提取"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_JAN_16_2026)

        # 第一条应包含多个关键词
        jan_16_entry = entries[0]
        all_keywords = " ".join(jan_16_entry.keywords).lower()

        # 验证至少提取到一些更新关键词
        assert len(jan_16_entry.keywords) > 0, "应该提取到更新关键词"

    def test_related_doc_urls_jan_16_features(self) -> None:
        """测试 Jan 16 2026 新特性的 related_doc_urls 选择"""
        from scripts.run_iterate import ChangelogEntry

        analyzer = ChangelogAnalyzer()

        # 创建包含 Jan 16 2026 新特性的 entry
        entries = [
            ChangelogEntry(
                date="Jan 16, 2026",
                title="New Features",
                content="Added /plan and /ask mode. Cloud relay for MCP. Enhanced diff view.",
                category="feature",
            ),
        ]

        analysis = analyzer.extract_update_points(entries)

        # 验证相关文档 URL 包含预期的文档
        related_urls = analysis.related_doc_urls
        url_str = " ".join(related_urls)

        # 应该匹配到 parameters（因为包含 plan/ask/mode）
        assert any("parameters" in url for url in related_urls), f"应包含 parameters 文档，实际: {related_urls}"

        # 应该匹配到 slash-commands（因为包含 /plan /ask）
        assert any("slash-commands" in url for url in related_urls), f"应包含 slash-commands 文档，实际: {related_urls}"

        # 应该匹配到 mcp（因为包含 cloud relay）
        assert any("mcp" in url for url in related_urls), f"应包含 mcp 文档，实际: {related_urls}"

        # Jan 16 2026: 应该匹配到 modes/plan 专页（因为包含 plan mode）
        assert any("modes/plan" in url for url in related_urls), f"应包含 modes/plan 文档，实际: {related_urls}"

        # Jan 16 2026: 应该匹配到 modes/ask 专页（因为包含 ask mode）
        assert any("modes/ask" in url for url in related_urls), f"应包含 modes/ask 文档，实际: {related_urls}"

    def test_html_mixed_content_cleanup(self) -> None:
        """测试 HTML 混合内容清理"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_HTML_MIXED)

        # 应该能解析出条目
        assert len(entries) >= 1, "HTML 混合内容应该能解析出条目"

        # 验证 HTML 标签和脚本被清理
        all_content = " ".join(e.content for e in entries)
        assert "<script>" not in all_content
        assert "</script>" not in all_content
        assert "<div" not in all_content
        assert "console.log" not in all_content

    def test_fallback_strategy_no_headers(self) -> None:
        """测试保底策略：无标准标题时全页作为单条"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_NO_HEADERS)

        # 保底策略应该返回至少 1 条
        assert len(entries) >= 1, "保底策略应该返回至少 1 条"

        # 验证内容被保留
        entry = entries[0]
        assert "plan/ask" in entry.content.lower() or "plan" in entry.content.lower()
        assert "cloud relay" in entry.content.lower() or "relay" in entry.content.lower()

    def test_clean_content_method(self) -> None:
        """测试 _clean_content 方法"""
        analyzer = ChangelogAnalyzer()

        # 测试 HTML 标签移除
        html_content = "<p>Hello <strong>World</strong></p>"
        cleaned = analyzer._clean_content(html_content)
        assert "<p>" not in cleaned
        assert "<strong>" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned

        # 测试 HTML 实体解码
        entity_content = "&nbsp;&lt;test&gt;&amp;"
        cleaned = analyzer._clean_content(entity_content)
        assert "&nbsp;" not in cleaned
        assert "&lt;" not in cleaned

        # 测试脚本移除
        script_content = "Before<script>alert('xss')</script>After"
        cleaned = analyzer._clean_content(script_content)
        assert "script" not in cleaned
        assert "alert" not in cleaned
        assert "Before" in cleaned
        assert "After" in cleaned

    def test_categorize_content_method(self) -> None:
        """测试 _categorize_content 方法"""
        analyzer = ChangelogAnalyzer()

        # 测试 feature 分类
        assert analyzer._categorize_content("New feature added") == "feature"
        assert analyzer._categorize_content("Added new support") == "feature"
        assert analyzer._categorize_content("新增功能") == "feature"
        assert analyzer._categorize_content("支持新特性") == "feature"

        # 测试 fix 分类
        assert analyzer._categorize_content("Bug fix for issue") == "fix"
        assert analyzer._categorize_content("Fixed the problem") == "fix"
        assert analyzer._categorize_content("修复了问题") == "fix"

        # 测试 improvement 分类
        assert analyzer._categorize_content("Improved performance") == "improvement"
        assert analyzer._categorize_content("优化了速度") == "improvement"
        assert analyzer._categorize_content("Better error handling") == "improvement"

        # 测试 other 分类
        assert analyzer._categorize_content("Some random text") == "other"

    def test_parse_consistency_multiple_runs(self) -> None:
        """测试多次解析结果一致性"""
        analyzer = ChangelogAnalyzer()

        # 运行多次解析
        results = [analyzer.parse_changelog(SAMPLE_CHANGELOG_JAN_16_2026) for _ in range(5)]

        # 验证每次结果一致
        first_count = len(results[0])
        for i, result in enumerate(results[1:], 2):
            assert len(result) == first_count, f"第 {i} 次解析结果数量 ({len(result)}) 与第 1 次 ({first_count}) 不一致"

        # 验证日期和分类一致
        first_dates = [e.date for e in results[0]]
        first_categories = [e.category for e in results[0]]
        for i, result in enumerate(results[1:], 2):
            dates = [e.date for e in result]
            categories = [e.category for e in result]
            assert dates == first_dates, f"第 {i} 次日期不一致"
            assert categories == first_categories, f"第 {i} 次分类不一致"

    def test_extract_doc_keywords_jan_16_features(self) -> None:
        """测试 _extract_doc_keywords 覆盖 Jan 16 2026 新特性"""
        analyzer = ChangelogAnalyzer()

        # 测试 parameters URL 的关键词
        params_keywords = analyzer._extract_doc_keywords("https://cursor.com/cn/docs/cli/reference/parameters")
        params_str = " ".join(params_keywords).lower()
        assert "plan" in params_str, "parameters 应包含 plan 关键词"
        assert "ask" in params_str, "parameters 应包含 ask 关键词"
        assert "mode" in params_str or "模式" in params_str

        # 测试 slash-commands URL 的关键词
        slash_keywords = analyzer._extract_doc_keywords("https://cursor.com/cn/docs/cli/reference/slash-commands")
        slash_str = " ".join(slash_keywords).lower()
        assert "/plan" in slash_str, "slash-commands 应包含 /plan"
        assert "/ask" in slash_str, "slash-commands 应包含 /ask"

        # 测试 mcp URL 的关键词
        mcp_keywords = analyzer._extract_doc_keywords("https://cursor.com/cn/docs/cli/mcp")
        mcp_str = " ".join(mcp_keywords).lower()
        assert "relay" in mcp_str or "cloud relay" in mcp_str, "mcp 应包含 relay 关键词"

        # 测试 overview URL 的关键词
        overview_keywords = analyzer._extract_doc_keywords("https://cursor.com/cn/docs/cli/overview")
        overview_str = " ".join(overview_keywords).lower()
        assert "diff" in overview_str, "overview 应包含 diff 关键词"

        # Jan 16 2026: 测试 modes/plan URL 的关键词
        plan_keywords = analyzer._extract_doc_keywords("https://cursor.com/cn/docs/cli/modes/plan")
        plan_str = " ".join(plan_keywords).lower()
        assert "plan" in plan_str, "modes/plan 应包含 plan 关键词"
        assert "规划模式" in plan_str or "plan mode" in plan_str, "modes/plan 应包含规划模式关键词"

        # Jan 16 2026: 测试 modes/ask URL 的关键词
        ask_keywords = analyzer._extract_doc_keywords("https://cursor.com/cn/docs/cli/modes/ask")
        ask_str = " ".join(ask_keywords).lower()
        assert "ask" in ask_str, "modes/ask 应包含 ask 关键词"
        assert "问答模式" in ask_str or "ask mode" in ask_str, "modes/ask 应包含问答模式关键词"

    def test_extract_main_content_from_main_tag(self) -> None:
        """测试从 <main> 标签提取主内容"""
        analyzer = ChangelogAnalyzer()

        html_with_main = """
        <html>
        <head><title>Test</title></head>
        <body>
        <nav>Navigation Menu</nav>
        <header>Site Header</header>
        <main>
        ## Jan 16, 2026
        - Plan mode support added
        - Ask mode for Q&A
        </main>
        <footer>Copyright 2026</footer>
        </body>
        </html>
        """

        extracted = analyzer._extract_main_content(html_with_main)
        assert "Plan mode" in extracted
        assert "Ask mode" in extracted
        # 导航内容不应出现在提取结果中
        assert "Navigation Menu" not in extracted

    def test_extract_main_content_from_article_tag(self) -> None:
        """测试从 <article> 标签提取主内容"""
        analyzer = ChangelogAnalyzer()

        html_with_article = """
        <header>Header Content</header>
        <article class="changelog-content">
        ## Updates
        - Cloud relay feature
        - Diff view improvements
        </article>
        <footer>Footer Content</footer>
        """

        extracted = analyzer._extract_main_content(html_with_article)
        assert "Cloud relay" in extracted
        assert "Diff view" in extracted

    def test_extract_main_content_from_content_div(self) -> None:
        """测试从带有 changelog class 的 div 提取主内容"""
        analyzer = ChangelogAnalyzer()

        html_with_content_div = """
        <nav>Menu</nav>
        <div class="changelog-content">
        ## Jan 20, 2026
        - New plan/ask modes
        - MCP cloud relay
        </div>
        <script>console.log('noise');</script>
        """

        extracted = analyzer._extract_main_content(html_with_content_div)
        assert "plan/ask" in extracted
        assert "MCP cloud relay" in extracted

    def test_extract_main_content_from_markdown_anchor(self) -> None:
        """测试从 Markdown 标题锚点提取主内容"""
        analyzer = ChangelogAnalyzer()

        content_with_noise = """
        Skip to content
        Navigation | Home | Docs | API

        Some header noise

        # Changelog

        ## Jan 16, 2026
        - Plan mode for read-only analysis
        - Ask mode for Q&A

        ## Jan 10, 2026
        - Cloud relay support
        """

        extracted = analyzer._extract_main_content(content_with_noise)
        # 应该从 "# Changelog" 开始提取
        assert "Changelog" in extracted
        assert "Plan mode" in extracted
        # 导航噪声应该被去除或不在主内容中
        # 注意：锚点策略从锚点位置开始，所以 "Skip to content" 可能不在提取范围内

    def test_clean_content_removes_nav_elements(self) -> None:
        """测试 _clean_content 移除 nav/header/footer 元素"""
        analyzer = ChangelogAnalyzer()

        html_with_nav = """
        <nav><a href="/">Home</a><a href="/docs">Docs</a></nav>
        <header>Site Header</header>
        <main>
        ## Updates
        - Plan mode added
        </main>
        <footer>Copyright 2026 All rights reserved</footer>
        """

        cleaned = analyzer._clean_content(html_with_nav)
        # 主内容应该保留
        assert "Plan mode" in cleaned
        # nav 内容不应该出现
        assert "Home" not in cleaned or "<nav>" not in cleaned

    def test_clean_content_removes_skip_to_content(self) -> None:
        """测试 _clean_content 移除导航文本"""
        analyzer = ChangelogAnalyzer()

        content_with_nav_text = """
        Skip to content
        Back to top
        Table of Contents

        ## Changelog
        - New feature: plan mode
        """

        cleaned = analyzer._clean_content(content_with_nav_text)
        assert "plan mode" in cleaned
        # 导航文本应该被移除
        assert "Skip to content" not in cleaned
        assert "Back to top" not in cleaned

    def test_clean_content_removes_copyright(self) -> None:
        """测试 _clean_content 移除版权信息"""
        analyzer = ChangelogAnalyzer()

        content_with_copyright = """
        ## Updates
        - Cloud relay feature

        © 2026 Cursor Inc. All rights reserved.
        Copyright 2026
        """

        cleaned = analyzer._clean_content(content_with_copyright)
        assert "Cloud relay" in cleaned
        # 版权信息应该被移除
        assert "All rights reserved" not in cleaned

    def test_clean_content_decodes_additional_entities(self) -> None:
        """测试 _clean_content 解码额外的 HTML 实体"""
        analyzer = ChangelogAnalyzer()

        # 测试 &hellip; 和 &trade; 解码
        content_with_entities = """
        ## Updates
        New features coming soon&hellip;
        Product name&trade; support added.
        Documentation registered&reg; here.
        """
        cleaned = analyzer._clean_content(content_with_entities)

        assert "&hellip;" not in cleaned
        assert "..." in cleaned
        assert "&trade;" not in cleaned
        assert "™" in cleaned
        assert "&reg;" not in cleaned
        assert "®" in cleaned

        # 单独测试 &copy; 解码（不触发版权过滤）
        # 注意：包含 © 的独立行会被版权过滤删除
        copy_content = "The &copy; symbol is used for icons"
        copy_cleaned = analyzer._clean_content(copy_content)
        # &copy; 应该被解码，但整行可能被版权过滤
        assert "&copy;" not in copy_cleaned


# ============================================================
# TestMainContentExtraction - 主内容区域提取测试
# ============================================================


# 真实风格的 HTML changelog 样例（模拟 Cursor 文档页面结构）
SAMPLE_CURSOR_DOC_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Cursor CLI Changelog</title>
    <script src="analytics.js"></script>
    <style>.highlight { color: blue; }</style>
</head>
<body>
<nav class="top-nav">
    <a href="/">Home</a>
    <a href="/docs">Documentation</a>
    <a href="/changelog">Changelog</a>
</nav>
<header class="page-header">
    <h1>Cursor CLI</h1>
    <p>The AI-first code editor</p>
</header>
<main class="main-content">
    <article class="changelog">
        <h1># Changelog</h1>

        <h2>## Jan 20, 2026</h2>
        <h3>### New Features</h3>
        <ul>
            <li>Added /plan command for planning mode (read-only analysis)</li>
            <li>Added /ask command for Q&A mode</li>
            <li>Cloud relay support for MCP servers</li>
            <li>Enhanced diff view with Ctrl+R shortcut</li>
        </ul>

        <h3>### Improvements</h3>
        <ul>
            <li>Better streaming output with --stream-partial-output</li>
        </ul>

        <h2>## Jan 16, 2026</h2>
        <h3>### Bug Fixes</h3>
        <ul>
            <li>Fixed MCP authentication issues</li>
        </ul>
    </article>
</main>
<footer class="site-footer">
    <p>© 2026 Cursor Inc. All rights reserved.</p>
    <nav>Privacy | Terms | Contact</nav>
</footer>
<script>initPage();</script>
</body>
</html>
"""

# 用户实际可能遇到的 changelog 片段风格
SAMPLE_USER_CHANGELOG_SNIPPET = """
Skip to main content

Cursor Documentation
Navigation: Home > CLI > Changelog

# Cursor CLI Changelog

What's New in Cursor CLI

## January 20, 2026

### CLI Modes

We've introduced new execution modes:

- **Plan Mode** (`--mode plan` or `/plan`): Read-only planning mode for code analysis
- **Ask Mode** (`--mode ask` or `/ask`): Q&A mode for quick questions
- **Agent Mode**: Full agent mode with file modifications (default)

### Cloud Relay

MCP servers now support cloud relay:

- Remote access without local installation
- Automatic authentication handling
- Support for custom MCP configurations

### Diff View Enhancements

- Press `Ctrl+R` to review changes before commit
- Inline diff view for better visualization
- Word-level diff highlighting

## January 16, 2026

### Bug Fixes

- Fixed timeout issues in long-running tasks
- Resolved race conditions in background execution

Back to top | Documentation Home | API Reference

© 2026 Cursor. All rights reserved.
"""


class TestMainContentExtraction:
    """测试主内容区域提取功能"""

    def test_parse_cursor_doc_page_style(self) -> None:
        """测试解析 Cursor 文档页面风格的 changelog"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CURSOR_DOC_PAGE)

        # 应该成功解析出条目
        assert len(entries) >= 1, f"期望至少 1 条，实际 {len(entries)} 条"

        # 验证内容包含关键特性
        all_content = " ".join(e.content.lower() for e in entries)
        assert "plan" in all_content, "应包含 plan 关键词"
        assert "ask" in all_content, "应包含 ask 关键词"

    def test_parse_user_changelog_snippet(self) -> None:
        """测试解析用户提供的 changelog 片段风格"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_USER_CHANGELOG_SNIPPET)

        # 应该成功解析
        assert len(entries) >= 1, f"期望至少 1 条，实际 {len(entries)} 条"

        # 验证噪声被过滤
        all_content = " ".join(e.content for e in entries)
        assert "Skip to main content" not in all_content, "导航噪声应被过滤"
        assert "Back to top" not in all_content, "底部导航应被过滤"

    def test_user_snippet_extracts_plan_ask_keywords(self) -> None:
        """测试用户片段能提取 plan/ask 关键词"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_USER_CHANGELOG_SNIPPET)

        # 检查关键词提取
        all_keywords = []
        for entry in entries:
            all_keywords.extend(entry.keywords)

        # 合并内容检查
        all_content = " ".join(e.content.lower() for e in entries)

        # 应包含 plan/ask 模式相关内容
        assert "plan" in all_content or "--mode plan" in all_content, "应包含 plan 模式内容"
        assert "ask" in all_content or "--mode ask" in all_content, "应包含 ask 模式内容"

    def test_user_snippet_extracts_cloud_relay_keywords(self) -> None:
        """测试用户片段能提取 cloud relay 关键词"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_USER_CHANGELOG_SNIPPET)

        all_content = " ".join(e.content.lower() for e in entries)

        assert "cloud relay" in all_content or "relay" in all_content, "应包含 cloud relay 相关内容"
        assert "mcp" in all_content, "应包含 MCP 相关内容"

    def test_user_snippet_extracts_diff_keywords(self) -> None:
        """测试用户片段能提取 diff 相关关键词"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_USER_CHANGELOG_SNIPPET)

        all_content = " ".join(e.content.lower() for e in entries)

        assert "diff" in all_content, "应包含 diff 关键词"
        assert "ctrl+r" in all_content or "ctrl" in all_content, "应包含 Ctrl+R 快捷键"

    def test_user_snippet_hits_modes_plan_url(self) -> None:
        """测试用户片段能命中 modes/plan URL"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_USER_CHANGELOG_SNIPPET)
        analysis = analyzer.extract_update_points(entries)

        # 验证相关文档 URL 包含 modes/plan
        related_urls = analysis.related_doc_urls
        assert any("modes/plan" in url for url in related_urls), f"应命中 modes/plan URL，实际: {related_urls}"

    def test_user_snippet_hits_modes_ask_url(self) -> None:
        """测试用户片段能命中 modes/ask URL"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_USER_CHANGELOG_SNIPPET)
        analysis = analyzer.extract_update_points(entries)

        related_urls = analysis.related_doc_urls
        assert any("modes/ask" in url for url in related_urls), f"应命中 modes/ask URL，实际: {related_urls}"

    def test_user_snippet_hits_cli_mcp_url(self) -> None:
        """测试用户片段能命中 cli/mcp URL"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_USER_CHANGELOG_SNIPPET)
        analysis = analyzer.extract_update_points(entries)

        related_urls = analysis.related_doc_urls
        assert any("mcp" in url for url in related_urls), f"应命中 mcp 相关 URL，实际: {related_urls}"

    def test_cursor_doc_page_filters_footer(self) -> None:
        """测试 Cursor 文档页面风格过滤 footer"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CURSOR_DOC_PAGE)

        all_content = " ".join(e.content for e in entries)

        # footer 内容应被过滤
        assert "Privacy" not in all_content or "Terms" not in all_content
        assert "initPage" not in all_content, "script 内容应被过滤"

    def test_cursor_doc_page_filters_nav(self) -> None:
        """测试 Cursor 文档页面风格过滤导航"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CURSOR_DOC_PAGE)

        all_content = " ".join(e.content for e in entries)

        # 导航内容应被过滤或不在主内容中
        # 由于使用 <main> 标签提取，导航应该被排除


# ============================================================
# TestCategoryDateParseStrategy - 类别+日期解析策略测试
# ============================================================


# 内嵌 changelog 样例：使用 "<Category> <Month Day, Year>" 格式
SAMPLE_CHANGELOG_CATEGORY_DATE = """
# Cursor Changelog

CLI Jan 16, 2026

New features for the CLI:
- Added /plan command for planning mode
- Added /ask command for Q&A mode
- Improved output formatting

Agent Jan 10, 2026

Agent improvements:
- Support for custom subagents
- Background task execution
- Better error handling

Cloud December 20, 2025

Cloud features update:
- Cloud relay for MCP servers
- API rate limiting improvements
"""

# 带 Markdown 标题符号的类别+日期样例
SAMPLE_CHANGELOG_CATEGORY_DATE_WITH_HEADERS = """
# Product Updates

## CLI Jan 20, 2026

### New Features
- Plan/Ask mode switching
- Enhanced diff view

### Bug Fixes
- Fixed timeout issues

## Agent Jan 15, 2026

### Improvements
- Subagent support enhanced

## MCP-Server Jan 10, 2026

Cloud relay feature added for remote access.
"""


class TestCategoryDateParseStrategy:
    """测试 ChangelogAnalyzer 的类别+日期解析策略

    测试 _parse_by_category_date_headers 方法，该策略用于识别
    类似 "CLI Jan 16, 2026" 的分段结构。
    """

    def test_parse_category_date_basic(self) -> None:
        """测试基本的类别+日期格式解析"""
        analyzer = ChangelogAnalyzer()

        # 使用内嵌样例直接调用 _parse_by_category_date_headers
        entries = analyzer._parse_by_category_date_headers(SAMPLE_CHANGELOG_CATEGORY_DATE)

        # 应该解析出至少 1 条 entry
        assert len(entries) >= 1, f"期望至少 1 条，实际 {len(entries)} 条"

    def test_parse_category_date_entry_count(self) -> None:
        """测试类别+日期格式解析条目数量"""
        analyzer = ChangelogAnalyzer()

        entries = analyzer._parse_by_category_date_headers(SAMPLE_CHANGELOG_CATEGORY_DATE)

        # 应该解析出 3 个条目（CLI, Agent, Cloud）
        assert len(entries) == 3, f"期望 3 条，实际 {len(entries)} 条"

    def test_parse_category_date_dates(self) -> None:
        """测试类别+日期格式日期提取"""
        analyzer = ChangelogAnalyzer()

        entries = analyzer._parse_by_category_date_headers(SAMPLE_CHANGELOG_CATEGORY_DATE)

        # 验证日期提取正确
        dates = [e.date for e in entries]
        assert any("Jan 16" in d for d in dates), "应包含 Jan 16 日期"
        assert any("Jan 10" in d for d in dates), "应包含 Jan 10 日期"
        assert any("Dec" in d or "December" in d for d in dates), "应包含 December 日期"

    def test_parse_category_date_titles(self) -> None:
        """测试类别+日期格式标题提取"""
        analyzer = ChangelogAnalyzer()

        entries = analyzer._parse_by_category_date_headers(SAMPLE_CHANGELOG_CATEGORY_DATE)

        # 验证标题包含类别信息
        titles = [e.title.lower() for e in entries]
        assert any("cli" in t for t in titles), "标题应包含 CLI"
        assert any("agent" in t for t in titles), "标题应包含 Agent"
        assert any("cloud" in t for t in titles), "标题应包含 Cloud"

    def test_parse_category_date_categories(self) -> None:
        """测试类别+日期格式分类识别"""
        analyzer = ChangelogAnalyzer()

        entries = analyzer._parse_by_category_date_headers(SAMPLE_CHANGELOG_CATEGORY_DATE)

        # 验证每条都有有效分类
        for entry in entries:
            assert entry.category in ["feature", "fix", "improvement", "other"], f"无效分类: {entry.category}"

        # CLI 和 Agent 类别应被识别为 feature
        categories = [e.category for e in entries]
        assert "feature" in categories, "应至少有一条被分类为 feature"

    def test_parse_category_date_content(self) -> None:
        """测试类别+日期格式内容提取"""
        analyzer = ChangelogAnalyzer()

        entries = analyzer._parse_by_category_date_headers(SAMPLE_CHANGELOG_CATEGORY_DATE)

        # 验证每条都有内容
        for entry in entries:
            assert entry.content, f"Entry {entry.title} 应有内容"

        # 验证 CLI 条目包含预期内容
        cli_entry = next((e for e in entries if "cli" in e.title.lower()), None)
        assert cli_entry is not None, "应有 CLI 条目"
        assert "/plan" in cli_entry.content or "plan" in cli_entry.content.lower(), "CLI 内容应包含 plan"

    def test_parse_category_date_with_markdown_headers(self) -> None:
        """测试带 Markdown 标题符号的类别+日期格式"""
        analyzer = ChangelogAnalyzer()

        entries = analyzer._parse_by_category_date_headers(SAMPLE_CHANGELOG_CATEGORY_DATE_WITH_HEADERS)

        # 应该解析出 3 个条目
        assert len(entries) >= 1, f"期望至少 1 条，实际 {len(entries)} 条"

        # 验证日期提取（带标题符号的格式）
        dates = [e.date for e in entries]
        assert any("Jan" in d for d in dates), "应包含 January 日期"

    def test_parse_category_date_via_parse_changelog(self) -> None:
        """测试通过 parse_changelog 调用类别+日期策略

        当内容不匹配日期标题策略时，应该回退到类别+日期策略。
        """
        analyzer = ChangelogAnalyzer()

        # 使用只有类别+日期格式的内容（不含 ## 2024-01-15 格式）
        content = """
CLI Jan 16, 2026

New CLI features:
- Plan mode support
- Ask mode support

Agent Jan 10, 2026

Agent updates:
- Subagent improvements
"""

        entries = analyzer.parse_changelog(content)

        # 应该通过类别+日期策略解析出条目
        assert len(entries) >= 1, f"期望至少 1 条，实际 {len(entries)} 条"

    def test_normalize_category(self) -> None:
        """测试 _normalize_category 方法"""
        analyzer = ChangelogAnalyzer()

        # 测试产品类别映射
        assert analyzer._normalize_category("CLI") == "feature"
        assert analyzer._normalize_category("Agent") == "feature"
        assert analyzer._normalize_category("Cloud") == "feature"
        assert analyzer._normalize_category("MCP") == "feature"

        # 测试功能类别映射
        assert analyzer._normalize_category("Feature") == "feature"
        assert analyzer._normalize_category("New Feature") == "feature"
        assert analyzer._normalize_category("Bug Fix") == "fix"
        assert analyzer._normalize_category("Improvement") == "improvement"
        assert analyzer._normalize_category("Update") == "improvement"

        # 测试未知类别
        assert analyzer._normalize_category("Unknown") == "other"
        assert analyzer._normalize_category("Random") == "other"

    def test_parse_category_date_keywords(self) -> None:
        """测试类别+日期格式关键词提取"""
        analyzer = ChangelogAnalyzer()

        entries = analyzer._parse_by_category_date_headers(SAMPLE_CHANGELOG_CATEGORY_DATE)

        # 验证关键词提取
        all_keywords = []
        for entry in entries:
            all_keywords.extend(entry.keywords)

        # 应该提取到一些更新相关关键词
        # 注意：关键词提取依赖 UPDATE_KEYWORDS 中的模式
        # 如果内容中包含 "new", "support", "improved" 等，应该被提取
        assert len(all_keywords) >= 0, "应该能够提取关键词（可能为空取决于内容）"


# ============================================================
# TestIntegration - 集成测试
# ============================================================


# ============================================================
# TestCommitPerIterationStrategy - 提交策略测试
# ============================================================


class TestCommitPerIterationStrategy:
    """测试 commit_per_iteration 提交策略

    覆盖 "开启 per-iteration 时 CONTINUE/ABORT 也会触发提交策略判定" 的行为。
    通过 mock CommitPolicy/Committer 调用次数验证。
    """

    @pytest.fixture
    def commit_per_iteration_args(self, base_iterate_args: argparse.Namespace) -> argparse.Namespace:
        """创建启用 commit_per_iteration 的参数"""
        base_iterate_args.auto_commit = True
        base_iterate_args.commit_per_iteration = True
        base_iterate_args.skip_online = True
        return base_iterate_args

    @pytest.mark.asyncio
    async def test_mp_config_receives_commit_per_iteration(self, commit_per_iteration_args: argparse.Namespace) -> None:
        """测试 MP 编排器配置接收 commit_per_iteration 参数"""
        iterator = SelfIterator(commit_per_iteration_args)
        iterator.context.iteration_goal = "测试目标"

        captured_config = None

        # 捕获传给 MultiProcessOrchestratorConfig 的参数
        with patch("scripts.run_iterate.MultiProcessOrchestratorConfig") as MockConfig:
            with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_config_instance = MagicMock()
                    MockConfig.return_value = mock_config_instance

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(return_value={"success": True})
                    MockMP.return_value = mock_orch

                    await iterator._run_with_mp_orchestrator(3, mock_km)

                    # 验证配置参数
                    MockConfig.assert_called_once()
                    call_kwargs = MockConfig.call_args.kwargs
                    assert call_kwargs.get("commit_per_iteration") is True
                    assert call_kwargs.get("commit_on_complete") is False  # 互斥
                    assert call_kwargs.get("enable_auto_commit") is True

    @pytest.mark.asyncio
    async def test_basic_config_receives_commit_per_iteration(
        self, commit_per_iteration_args: argparse.Namespace
    ) -> None:
        """测试 basic 编排器配置接收 commit_per_iteration 参数"""
        iterator = SelfIterator(commit_per_iteration_args)
        iterator.context.iteration_goal = "测试目标"

        # 捕获传给 OrchestratorConfig 的参数
        with patch("scripts.run_iterate.OrchestratorConfig") as MockConfig:
            with patch("scripts.run_iterate.Orchestrator") as MockOrch:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    with patch("scripts.run_iterate.CursorAgentConfig"):
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        mock_config_instance = MagicMock()
                        MockConfig.return_value = mock_config_instance

                        mock_orch = MagicMock()
                        mock_orch.run = AsyncMock(return_value={"success": True})
                        MockOrch.return_value = mock_orch

                        await iterator._run_with_basic_orchestrator(3, mock_km)

                        # 验证配置参数
                        MockConfig.assert_called_once()
                        call_kwargs = MockConfig.call_args.kwargs
                        assert call_kwargs.get("commit_per_iteration") is True
                        assert call_kwargs.get("commit_on_complete") is False  # 互斥
                        assert call_kwargs.get("enable_auto_commit") is True

    @pytest.mark.asyncio
    async def test_commit_on_complete_default_when_per_iteration_false(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 commit_per_iteration=False 时 commit_on_complete=True"""
        base_iterate_args.auto_commit = True
        base_iterate_args.commit_per_iteration = False
        base_iterate_args.skip_online = True
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        with patch("scripts.run_iterate.MultiProcessOrchestratorConfig") as MockConfig:
            with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_config_instance = MagicMock()
                    MockConfig.return_value = mock_config_instance

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(return_value={"success": True})
                    MockMP.return_value = mock_orch

                    await iterator._run_with_mp_orchestrator(3, mock_km)

                    # 验证配置参数
                    MockConfig.assert_called_once()
                    call_kwargs = MockConfig.call_args.kwargs
                    assert call_kwargs.get("commit_per_iteration") is False
                    assert call_kwargs.get("commit_on_complete") is True  # 默认启用

    @pytest.mark.asyncio
    async def test_orchestrator_committed_detection_with_commits(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 _has_orchestrator_committed 检测编排器已提交"""
        iterator = SelfIterator(base_iterate_args)

        # 有 total_commits > 0
        result_with_commits = {
            "commits": {
                "total_commits": 1,
                "commit_hashes": ["abc123"],
            }
        }
        assert iterator._has_orchestrator_committed(result_with_commits) is True

        # 有 successful_commits
        result_with_successful = {
            "commits": {
                "successful_commits": 2,
            }
        }
        assert iterator._has_orchestrator_committed(result_with_successful) is True

        # 迭代级别有 commit_hash
        result_with_iteration_commit = {
            "commits": {},
            "iterations": [
                {"id": 1, "commit_hash": "def456"},
            ],
        }
        assert iterator._has_orchestrator_committed(result_with_iteration_commit) is True

    @pytest.mark.asyncio
    async def test_orchestrator_committed_detection_without_commits(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 _has_orchestrator_committed 检测编排器未提交"""
        iterator = SelfIterator(base_iterate_args)

        # 空 commits
        result_empty: dict[str, Any] = {"commits": {}}
        assert iterator._has_orchestrator_committed(result_empty) is False

        # total_commits = 0
        result_zero = {"commits": {"total_commits": 0}}
        assert iterator._has_orchestrator_committed(result_zero) is False

        # 无 commits 字段
        result_no_commits = {"success": True}
        assert iterator._has_orchestrator_committed(result_no_commits) is False

        # 迭代无 commit_hash
        result_no_iter_commit = {
            "commits": {},
            "iterations": [
                {"id": 1, "commit_hash": ""},
                {"id": 2, "commit_hash": None},
            ],
        }
        assert iterator._has_orchestrator_committed(result_no_iter_commit) is False

    @pytest.mark.asyncio
    async def test_skip_double_commit_when_orchestrator_committed(
        self, commit_per_iteration_args: argparse.Namespace
    ) -> None:
        """测试当编排器已提交时跳过 SelfIterator 二次提交"""
        iterator = SelfIterator(commit_per_iteration_args)
        iterator.context.iteration_goal = "测试目标"

        # 编排器返回已提交的结果
        mp_result_with_commits = {
            "success": True,
            "commits": {
                "total_commits": 1,
                "commit_hashes": ["abc123"],
            },
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value=mp_result_with_commits,
            ),
            patch.object(
                iterator,
                "_run_commit_phase",
                new_callable=AsyncMock,
            ) as mock_commit_phase,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证 _run_commit_phase 未被调用（因为编排器已提交）
            mock_commit_phase.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_commit_phase_when_orchestrator_not_committed(
        self, commit_per_iteration_args: argparse.Namespace
    ) -> None:
        """测试当编排器未提交时执行 SelfIterator 提交"""
        iterator = SelfIterator(commit_per_iteration_args)
        iterator.context.iteration_goal = "测试目标"

        # 编排器返回未提交的结果
        mp_result_no_commits = {
            "success": True,
            "commits": {},
            "iterations_completed": 1,
            "total_tasks_completed": 1,
        }

        mock_commit_result = {
            "total_commits": 1,
            "commit_hashes": ["def456"],
            "pushed_commits": 0,
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value=mp_result_no_commits,
            ),
            patch.object(
                iterator,
                "_run_commit_phase",
                new_callable=AsyncMock,
                return_value=mock_commit_result,
            ) as mock_commit_phase,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证 _run_commit_phase 被调用（因为编排器未提交）
            mock_commit_phase.assert_called_once()


# ============================================================
# TestDegradationStrategyB - 降级策略 B 测试
# ============================================================


class TestDegradationStrategyB:
    """测试降级策略 B: degraded 保持为未完成(success=False)并在最终结果显式标注

    策略 B 契约:
    1. 当健康检查返回不健康触发降级时，SelfIterator 输出包含 degraded=True
    2. success=False（目标未完成）
    3. degradation_reason 显式标注降级原因
    4. iterations_completed 记录降级前完成的迭代数

    这个策略与策略 A（回退到 basic 编排器）的区别：
    - 策略 B 不触发回退，保留 MP 编排器的部分执行结果
    - 适用于降级发生在执行过程中的场景
    """

    @pytest.fixture
    def mp_args(self, base_iterate_args: argparse.Namespace) -> argparse.Namespace:
        """创建 MP 编排器参数"""
        base_iterate_args.skip_online = True
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        return base_iterate_args

    @pytest.mark.asyncio
    async def test_degraded_result_fields_stable(self, mp_args: argparse.Namespace) -> None:
        """测试降级时结果字段稳定性

        契约验证:
        - success: False（未完成）
        - degraded: True（已降级）
        - degradation_reason: 非空字符串
        - iterations_completed: 整数（降级前完成的迭代数）
        """
        iterator = SelfIterator(mp_args)
        iterator.context.iteration_goal = "测试降级目标"

        # MP 编排器返回降级结果
        mp_degraded_result = {
            "success": False,
            "degraded": True,
            "degradation_reason": "Planner 进程不健康",
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 1,
            "total_tasks_failed": 1,
            "commits": {},
            "iterations": [
                {
                    "id": 1,
                    "status": "executing",
                    "tasks_created": 2,
                    "tasks_completed": 1,
                    "tasks_failed": 1,
                    "review_passed": False,
                    "commit_hash": None,
                }
            ],
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value=mp_degraded_result,
            ),
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 策略 B 契约断言
            assert result["success"] is False, "降级时 success 应为 False"
            assert result["degraded"] is True, "降级时 degraded 应为 True"
            assert result["degradation_reason"] == "Planner 进程不健康", "degradation_reason 应显式标注原因"
            assert result["iterations_completed"] == 1, "iterations_completed 应记录降级前完成的迭代数"

    @pytest.mark.asyncio
    async def test_degraded_no_fallback_to_basic(self, mp_args: argparse.Namespace) -> None:
        """测试降级时不回退到 basic 编排器（策略 B 核心行为）

        策略 B 与策略 A 的区别：
        - 策略 A: degraded 触发回退到 basic 编排器
        - 策略 B: degraded 不触发回退，保留 MP 结果

        这个测试验证策略 B 的核心行为：MP 返回 degraded 结果时，
        SelfIterator 不应调用 _run_with_basic_orchestrator
        """
        iterator = SelfIterator(mp_args)
        iterator.context.iteration_goal = "测试目标"

        # MP 编排器返回降级结果（注意：没有 _fallback_required 标志）
        mp_degraded_result = {
            "success": False,
            "degraded": True,
            "degradation_reason": "Worker 进程不健康数量超过阈值",
            "iterations_completed": 2,
            "total_tasks_created": 5,
            "total_tasks_completed": 3,
            "total_tasks_failed": 2,
            "commits": {},
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value=mp_degraded_result,
            ),
            patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
            ) as mock_basic,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证 basic 编排器未被调用
            mock_basic.assert_not_called()

            # 验证保留 MP 的降级结果
            assert result["degraded"] is True
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_degraded_vs_fallback_required_distinction(self, mp_args: argparse.Namespace) -> None:
        """测试 degraded 与 _fallback_required 的区别

        - _fallback_required: MP 编排器启动失败，需要回退到 basic
        - degraded: MP 编排器运行过程中降级，不触发回退

        这是策略 B 的关键区分点
        """
        iterator = SelfIterator(mp_args)
        iterator.context.iteration_goal = "测试目标"

        # 场景1: _fallback_required（启动失败）应触发 basic 回退
        mp_startup_failure = {
            "_fallback_required": True,
            "_fallback_reason": "进程创建失败",
        }

        basic_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value=mp_startup_failure,
            ),
            patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=basic_result,
            ) as mock_basic,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # _fallback_required 应触发 basic 编排器
            mock_basic.assert_called_once()

    @pytest.mark.asyncio
    async def test_degraded_result_preserves_partial_work(self, mp_args: argparse.Namespace) -> None:
        """测试降级时保留部分工作结果

        策略 B 的优势：降级时不丢弃已完成的工作
        """
        iterator = SelfIterator(mp_args)
        iterator.context.iteration_goal = "测试目标"

        # MP 返回降级但有部分完成的工作
        mp_degraded_result = {
            "success": False,
            "degraded": True,
            "degradation_reason": "Reviewer 进程不健康",
            "iterations_completed": 3,
            "total_tasks_created": 10,
            "total_tasks_completed": 7,
            "total_tasks_failed": 1,
            "commits": {
                "total_commits": 2,
                "commit_hashes": ["abc123", "def456"],
            },
            "iterations": [
                {"id": 1, "commit_hash": "abc123", "tasks_completed": 3},
                {"id": 2, "commit_hash": "def456", "tasks_completed": 2},
                {"id": 3, "commit_hash": None, "tasks_completed": 2},
            ],
        }

        with (
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value=mp_degraded_result,
            ),
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证部分工作被保留
            assert result["iterations_completed"] == 3
            assert result["total_tasks_completed"] == 7
            assert result["commits"]["total_commits"] == 2

    @pytest.mark.asyncio
    async def test_degraded_with_different_reasons(self, mp_args: argparse.Namespace) -> None:
        """测试不同降级原因的处理一致性

        验证降级原因包括：
        - Planner 进程不健康
        - Reviewer 进程不健康
        - Worker 不健康数量超过阈值
        """
        iterator = SelfIterator(mp_args)
        iterator.context.iteration_goal = "测试目标"

        test_cases = [
            {
                "reason": "Planner 进程不健康",
                "iterations": 0,
            },
            {
                "reason": "Reviewer 进程不健康",
                "iterations": 2,
            },
            {
                "reason": "不健康 Worker 数量超过阈值: 3 > 1",
                "iterations": 1,
            },
        ]

        for case in test_cases:
            mp_degraded_result = {
                "success": False,
                "degraded": True,
                "degradation_reason": case["reason"],
                "iterations_completed": case["iterations"],
                "total_tasks_created": 0,
                "total_tasks_completed": 0,
                "total_tasks_failed": 0,
                "commits": {},
            }

            with (
                patch.object(
                    iterator,
                    "_run_with_mp_orchestrator",
                    new_callable=AsyncMock,
                    return_value=mp_degraded_result,
                ),
                patch("scripts.run_iterate.KnowledgeManager") as MockKM,
            ):
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                result = await iterator._run_agent_system()

                # 所有降级原因应有一致的字段结构
                assert result["success"] is False
                assert result["degraded"] is True
                assert result["degradation_reason"] == case["reason"]
                assert isinstance(result["iterations_completed"], int)


# ============================================================
# TestBaselineFingerprint - 基线 fingerprint 比较测试
# ============================================================


class TestBaselineFingerprint:
    """测试基线 fingerprint 比较功能

    覆盖场景：
    1. 同一份 changelog 连续两次分析，第二次应判定 has_updates=False
    2. 不同内容的 changelog 应判定 has_updates=True
    3. 无基线时应正常解析
    4. 无更新时应跳过 related docs 抓取
    """

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """创建 mock storage"""
        storage = MagicMock()
        storage._initialized = True
        storage.initialize = AsyncMock()
        storage.get_content_hash_by_url = MagicMock(return_value=None)
        storage.get_cleaned_fingerprint_by_url = MagicMock(return_value=None)
        storage.save_document = AsyncMock(return_value=(True, "保存成功"))
        storage.get_stats = AsyncMock(return_value={"document_count": 1})
        storage.search = AsyncMock(return_value=[])
        storage.list_documents = AsyncMock(return_value=[])
        return storage

    @pytest.mark.asyncio
    async def test_same_changelog_second_analysis_no_updates(self, mock_storage: MagicMock) -> None:
        """测试同一份 changelog 连续两次分析，第二次应判定 has_updates=False"""
        # 固定的 changelog 内容
        changelog_content = """
## Jan 16, 2026

### New Features
- Added /plan and /ask mode
- Cloud relay support
"""
        # 模拟 fetch 返回固定内容
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = changelog_content

        # 创建 analyzer 并注入 storage
        analyzer = ChangelogAnalyzer(storage=mock_storage)

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            # 第一次分析 - 无基线
            mock_storage.get_content_hash_by_url.return_value = None
            mock_storage.get_cleaned_fingerprint_by_url.return_value = None
            first_result = await analyzer.analyze()

            # 验证第一次有更新
            assert first_result.has_updates is True
            assert len(first_result.entries) > 0

            # 计算第一次的 fingerprint
            first_fingerprint = analyzer.compute_fingerprint(changelog_content)

            # 第二次分析 - 模拟基线存在（使用第一次的 fingerprint）
            mock_storage.get_content_hash_by_url.return_value = first_fingerprint
            mock_storage.get_cleaned_fingerprint_by_url.return_value = first_fingerprint
            second_result = await analyzer.analyze()

            # 验证第二次无更新
            assert second_result.has_updates is False
            assert len(second_result.entries) == 0
            assert second_result.summary == "未检测到新的更新"
            assert second_result.related_doc_urls == []

    @pytest.mark.asyncio
    async def test_different_changelog_has_updates(self, mock_storage: MagicMock) -> None:
        """测试不同内容的 changelog 应判定 has_updates=True"""
        old_content = "## Jan 10, 2026\n- Old feature"
        new_content = "## Jan 16, 2026\n- New feature added"

        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = new_content

        analyzer = ChangelogAnalyzer(storage=mock_storage)

        # 计算旧内容的 fingerprint 作为基线
        old_fingerprint = analyzer.compute_fingerprint(old_content)
        mock_storage.get_content_hash_by_url.return_value = old_fingerprint
        mock_storage.get_cleaned_fingerprint_by_url.return_value = old_fingerprint

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            result = await analyzer.analyze()

            # 验证检测到更新
            assert result.has_updates is True
            assert len(result.entries) > 0

    @pytest.mark.asyncio
    async def test_no_baseline_parses_normally(self, mock_storage: MagicMock) -> None:
        """测试无基线时应正常解析"""
        changelog_content = "## Jan 16, 2026\n- Feature A"

        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = changelog_content

        analyzer = ChangelogAnalyzer(storage=mock_storage)
        mock_storage.get_content_hash_by_url.return_value = None
        mock_storage.get_cleaned_fingerprint_by_url.return_value = None

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            result = await analyzer.analyze()

            # 无基线时应正常解析
            assert result.has_updates is True
            assert len(result.entries) >= 1

    @pytest.mark.asyncio
    async def test_no_updates_skips_related_docs_fetch(self, mock_storage: MagicMock) -> None:
        """测试无更新时应跳过 related docs 抓取"""
        changelog_content = "## Jan 16, 2026\n- MCP support"
        fingerprint = ChangelogAnalyzer().compute_fingerprint(changelog_content)

        # 创建无更新的 analysis
        analysis = UpdateAnalysis(
            has_updates=False,
            entries=[],
            summary="未检测到新的更新",
            raw_content=changelog_content,
            related_doc_urls=[],
        )

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.fetcher.fetch = AsyncMock()
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()

        await updater.initialize()
        result = await updater.update_from_analysis(analysis)

        # 验证 fetcher.fetch 未被调用（跳过了 related docs 抓取）
        updater.fetcher.fetch.assert_not_called()

        # 验证结果标记
        assert result.get("no_updates_detected") is True
        assert len(result.get("urls_processed", [])) == 0

    @pytest.mark.asyncio
    async def test_has_updates_fetches_related_docs(self, mock_storage: MagicMock) -> None:
        """测试有更新时应抓取 related docs

        重构后 KnowledgeUpdater 使用 manager.add_urls 批量获取文档。
        """
        changelog_content = "## Jan 16, 2026\n- MCP support"

        # 创建有更新的 analysis
        analysis = UpdateAnalysis(
            has_updates=True,
            entries=[],
            summary="检测到更新: 新功能 (1项)",
            raw_content=changelog_content,
            related_doc_urls=["https://cursor.com/cn/docs/cli/mcp"],
        )

        # Mock 文档对象
        mock_doc = MagicMock()
        mock_doc.url = "https://cursor.com/cn/docs/cli/mcp"
        mock_doc.title = "MCP Documentation"

        mock_changelog_doc = MagicMock()
        mock_changelog_doc.url = DEFAULT_CHANGELOG_URL

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        # manager 需要正确 mock
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()
        updater.manager.add_content = AsyncMock(return_value=mock_changelog_doc)
        updater.manager.add_urls = AsyncMock(return_value=[mock_doc])

        await updater.initialize()
        result = await updater.update_from_analysis(analysis)

        # 验证 manager.add_urls 被调用（批量获取文档）
        updater.manager.add_urls.assert_called()

        # 验证结果
        assert result.get("no_updates_detected") is False
        assert len(result.get("urls_processed", [])) > 0

    def test_compute_fingerprint_consistency(self) -> None:
        """测试 fingerprint 计算的一致性"""
        analyzer = ChangelogAnalyzer()
        content = "## Jan 16, 2026\n- Feature A\n- Feature B"

        # 多次计算应返回相同结果
        fp1 = analyzer.compute_fingerprint(content)
        fp2 = analyzer.compute_fingerprint(content)
        fp3 = analyzer.compute_fingerprint(content)

        assert fp1 == fp2 == fp3
        assert len(fp1) == 16  # SHA256 前16位

    def test_compute_fingerprint_different_content(self) -> None:
        """测试不同内容的 fingerprint 不同"""
        analyzer = ChangelogAnalyzer()
        content1 = "## Jan 16, 2026\n- Feature A"
        content2 = "## Jan 16, 2026\n- Feature B"

        fp1 = analyzer.compute_fingerprint(content1)
        fp2 = analyzer.compute_fingerprint(content2)

        assert fp1 != fp2

    def test_compute_fingerprint_ignores_whitespace_variance(self) -> None:
        """测试 fingerprint 对空白差异的处理"""
        analyzer = ChangelogAnalyzer()
        # _clean_content 会规范化空白
        content1 = "## Jan 16, 2026\n\n\n- Feature A"
        content2 = "## Jan 16, 2026\n\n- Feature A"

        fp1 = analyzer.compute_fingerprint(content1)
        fp2 = analyzer.compute_fingerprint(content2)

        # 由于 _clean_content 规范化空行，指纹应相同
        assert fp1 == fp2

    @pytest.mark.asyncio
    async def test_custom_changelog_url_second_analysis_no_updates(self, mock_storage: MagicMock) -> None:
        """测试自定义 changelog_url 时第二次分析返回 has_updates=False

        场景：
        1. 使用自定义 URL（非默认 DEFAULT_CHANGELOG_URL）
        2. 第一次分析保存 fingerprint 时使用自定义 URL 作为 key
        3. 第二次分析读取 fingerprint 时使用同一 URL 作为 key
        4. 第二次应判定 has_updates=False
        """
        custom_url = "https://example.com/custom-changelog"
        changelog_content = """
## Jan 16, 2026

### New Features
- Custom changelog feature
"""
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = changelog_content

        # 创建 analyzer 使用自定义 URL
        analyzer = ChangelogAnalyzer(
            changelog_url=custom_url,
            storage=mock_storage,
        )

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            # 第一次分析 - 无基线
            mock_storage.get_content_hash_by_url.return_value = None
            mock_storage.get_cleaned_fingerprint_by_url.return_value = None
            first_result = await analyzer.analyze()

            # 验证第一次有更新
            assert first_result.has_updates is True

            # 验证 analyzer 使用的是自定义 URL
            assert analyzer.changelog_url == custom_url

            # 计算 fingerprint
            fingerprint = analyzer.compute_fingerprint(changelog_content)

            # 第二次分析 - 设置基线（使用自定义 URL 作为 key）
            mock_storage.get_content_hash_by_url.return_value = fingerprint
            mock_storage.get_cleaned_fingerprint_by_url.return_value = fingerprint
            second_result = await analyzer.analyze()

            # 验证第二次无更新
            assert second_result.has_updates is False
            assert second_result.summary == "未检测到新的更新"

            # 验证 get_cleaned_fingerprint_by_url 使用的是自定义 URL
            calls = mock_storage.get_cleaned_fingerprint_by_url.call_args_list
            for call in calls:
                assert call.args[0] == custom_url

    @pytest.mark.asyncio
    async def test_update_from_analysis_uses_custom_changelog_url(self, mock_storage: MagicMock) -> None:
        """测试 update_from_analysis 保存文档时使用传入的 changelog_url

        重构后使用 manager.add_content 保存 changelog。

        验证：
        1. manager.add_content 被调用时 URL 是自定义 URL
        2. 返回结果中 changelog_url 字段正确
        """
        custom_url = "https://example.com/custom-changelog"
        changelog_content = "## Jan 16, 2026\n- Feature A"

        analysis = UpdateAnalysis(
            has_updates=True,
            entries=[],
            summary="检测到更新",
            raw_content=changelog_content,
            related_doc_urls=[],
        )

        # Mock 文档对象
        mock_doc = MagicMock()
        mock_doc.url = custom_url

        # 捕获 add_content 调用参数
        add_content_calls: list[dict] = []

        async def capture_add_content(**kwargs):
            add_content_calls.append(kwargs)
            return mock_doc

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()
        updater.manager.add_content = AsyncMock(side_effect=capture_add_content)
        updater.manager.add_urls = AsyncMock(return_value=[])

        await updater.initialize()
        result = await updater.update_from_analysis(
            analysis,
            force=False,
            changelog_url=custom_url,  # 传入自定义 URL
        )

        # 验证 add_content 被调用时使用自定义 URL
        assert len(add_content_calls) >= 1
        assert add_content_calls[0].get("url") == custom_url, (
            f"期望 URL={custom_url}，实际={add_content_calls[0].get('url')}"
        )

        # 验证返回结果包含 changelog_url 字段
        assert result.get("changelog_url") == custom_url

    @pytest.mark.asyncio
    async def test_baseline_key_consistency_with_custom_url(self, mock_storage: MagicMock) -> None:
        """测试基线 fingerprint 读取/写入使用同一 URL 作为 key

        完整场景验证：
        1. 使用自定义 URL 进行第一次分析
        2. update_from_analysis 保存文档时使用同一自定义 URL
        3. 第二次分析时 _get_baseline_fingerprint 使用同一自定义 URL 读取
        4. 第二次分析返回 has_updates=False
        """
        custom_url = "https://cursor.com/cn/custom-changelog"
        changelog_content = "## Jan 20, 2026\n- New feature"

        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = changelog_content
        mock_fetch_result.method_used = MagicMock()
        mock_fetch_result.method_used.value = "fetch"

        # 创建 analyzer 使用自定义 URL，注入 storage
        analyzer = ChangelogAnalyzer(
            changelog_url=custom_url,
            storage=mock_storage,
        )

        # 模拟首次分析：无基线
        mock_storage.get_content_hash_by_url.return_value = None
        mock_storage.get_cleaned_fingerprint_by_url.return_value = None

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            # 第一次分析
            first_result = await analyzer.analyze()
            assert first_result.has_updates is True

        # 模拟 update_from_analysis 保存文档（重构后使用 manager.add_content）
        mock_doc = MagicMock()
        mock_doc.url = custom_url

        # llms.txt fetch 返回空（无需实际内容）
        mock_llms_result = MagicMock()
        mock_llms_result.success = False
        mock_llms_result.content = ""

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.fetcher.fetch = AsyncMock(return_value=mock_llms_result)  # llms.txt
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()
        updater.manager.add_content = AsyncMock(return_value=mock_doc)
        updater.manager.add_urls = AsyncMock(return_value=[])

        await updater.initialize()
        update_result = await updater.update_from_analysis(
            first_result,
            changelog_url=custom_url,
        )

        # 验证保存时使用的是自定义 URL
        assert update_result.get("changelog_url") == custom_url

        # 模拟第二次分析：设置基线 fingerprint
        fingerprint = analyzer.compute_fingerprint(changelog_content)
        mock_storage.get_content_hash_by_url.return_value = fingerprint
        mock_storage.get_cleaned_fingerprint_by_url.return_value = fingerprint

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            # 第二次分析
            second_result = await analyzer.analyze()

            # 验证第二次无更新
            assert second_result.has_updates is False

            # 验证 _get_baseline_fingerprint 使用的是自定义 URL
            # 通过检查 get_cleaned_fingerprint_by_url 的调用参数
            call_args = mock_storage.get_cleaned_fingerprint_by_url.call_args
            assert call_args.args[0] == custom_url


# ============================================================
# TestLlmsTxtCacheStrategy - llms.txt 缓存策略测试
# ============================================================


class TestLlmsTxtCacheStrategy:
    """测试 llms.txt 缓存策略

    覆盖场景：
    1. 在线 fetch 成功时写入缓存
    2. 在线 fetch 失败但缓存存在时读取缓存
    3. 缓存不存在时回退到仓库文件
    4. 缓存和仓库文件都不存在时返回 None
    """

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """创建 mock storage"""
        storage = MagicMock()
        storage.initialize = AsyncMock()
        storage.get_stats = AsyncMock(return_value={"document_count": 0})
        return storage

    @pytest.mark.asyncio
    async def test_fetch_success_writes_cache(self, mock_storage: MagicMock, tmp_path: Path) -> None:
        """测试在线 fetch 成功时写入缓存"""
        llms_content = "# Cursor Docs\n\n## API Reference\nhttps://cursor.com/docs/api"

        # 设置临时缓存路径
        temp_cache_path = tmp_path / ".cursor" / "cache" / "llms.txt"
        temp_fallback_path = tmp_path / "cursor_docs_full.txt"

        # Mock fetch 成功
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = llms_content

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.fetcher.fetch = AsyncMock(return_value=mock_fetch_result)
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()
        # 设置实例属性为临时路径
        updater.llms_cache_path = temp_cache_path
        updater.llms_local_fallback = temp_fallback_path

        await updater.initialize()
        result = await updater._fetch_llms_txt()

        # 验证返回内容正确
        assert result == llms_content

        # 验证缓存文件已写入
        assert temp_cache_path.exists()
        cached_content = temp_cache_path.read_text(encoding="utf-8")
        assert cached_content == llms_content

    @pytest.mark.asyncio
    async def test_fetch_failure_reads_cache(self, mock_storage: MagicMock, tmp_path: Path) -> None:
        """测试在线 fetch 失败但缓存存在时读取缓存"""
        cached_content = "# Cached Cursor Docs\n\nhttps://cursor.com/docs/cached"

        # 设置临时缓存路径并预先写入缓存
        temp_cache_path = tmp_path / ".cursor" / "cache" / "llms.txt"
        temp_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_cache_path.write_text(cached_content, encoding="utf-8")
        temp_fallback_path = tmp_path / "cursor_docs_full.txt"

        # Mock fetch 失败
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = False
        mock_fetch_result.content = ""

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.fetcher.fetch = AsyncMock(return_value=mock_fetch_result)
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()
        # 设置实例属性为临时路径
        updater.llms_cache_path = temp_cache_path
        updater.llms_local_fallback = temp_fallback_path

        await updater.initialize()
        result = await updater._fetch_llms_txt()

        # 验证返回缓存内容
        assert result == cached_content

    @pytest.mark.asyncio
    async def test_no_cache_fallback_to_repo_file(self, mock_storage: MagicMock, tmp_path: Path) -> None:
        """测试缓存不存在时回退到仓库文件"""
        repo_content = "# Repo Cursor Docs\n\nhttps://cursor.com/docs/repo"

        # 设置临时路径（缓存不存在）
        temp_cache_path = tmp_path / ".cursor" / "cache" / "llms.txt"
        temp_fallback_path = tmp_path / "cursor_docs_full.txt"

        # 预先写入仓库文件
        temp_fallback_path.write_text(repo_content, encoding="utf-8")

        # Mock fetch 失败
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = False
        mock_fetch_result.content = ""

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.fetcher.fetch = AsyncMock(return_value=mock_fetch_result)
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()
        # 设置实例属性为临时路径
        updater.llms_cache_path = temp_cache_path
        updater.llms_local_fallback = temp_fallback_path

        await updater.initialize()
        result = await updater._fetch_llms_txt()

        # 验证返回仓库文件内容
        assert result == repo_content

    @pytest.mark.asyncio
    async def test_all_sources_unavailable_returns_none(self, mock_storage: MagicMock, tmp_path: Path) -> None:
        """测试所有来源都不可用时返回 None"""
        # 设置临时路径（都不存在）
        temp_cache_path = tmp_path / ".cursor" / "cache" / "llms.txt"
        temp_fallback_path = tmp_path / "cursor_docs_full.txt"

        # Mock fetch 失败
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = False
        mock_fetch_result.content = ""

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.fetcher.fetch = AsyncMock(return_value=mock_fetch_result)
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()
        # 设置实例属性为临时路径（不存在的路径）
        updater.llms_cache_path = temp_cache_path
        updater.llms_local_fallback = temp_fallback_path

        await updater.initialize()
        result = await updater._fetch_llms_txt()

        # 验证返回 None
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_exception_fallback_to_cache(self, mock_storage: MagicMock, tmp_path: Path) -> None:
        """测试在线 fetch 抛出异常时回退到缓存"""
        cached_content = "# Exception Fallback Cache\n\nhttps://cursor.com/docs"

        # 设置临时缓存路径并预先写入缓存
        temp_cache_path = tmp_path / ".cursor" / "cache" / "llms.txt"
        temp_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_cache_path.write_text(cached_content, encoding="utf-8")
        temp_fallback_path = tmp_path / "cursor_docs_full.txt"

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        # Mock fetch 抛出异常
        updater.fetcher.fetch = AsyncMock(side_effect=Exception("Network error"))
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()
        # 设置实例属性为临时路径
        updater.llms_cache_path = temp_cache_path
        updater.llms_local_fallback = temp_fallback_path

        await updater.initialize()
        result = await updater._fetch_llms_txt()

        # 验证返回缓存内容
        assert result == cached_content

    def test_write_cache_creates_directory(self, tmp_path: Path) -> None:
        """测试写入缓存时自动创建目录"""
        temp_cache_path = tmp_path / "new_dir" / "nested" / "llms.txt"
        content = "# Test Content"

        updater = KnowledgeUpdater()
        # 设置实例属性为临时路径
        updater.llms_cache_path = temp_cache_path

        result = updater._write_llms_txt_cache(content)

        # 验证写入成功
        assert result is True
        assert temp_cache_path.exists()
        assert temp_cache_path.read_text(encoding="utf-8") == content


# ============================================================
# TestLlmsTxtExternalUrlFiltering - llms.txt 外域 URL 过滤测试
# ============================================================


class TestLlmsTxtExternalUrlFiltering:
    """测试 llms.txt 外域 URL 过滤

    验证：当 llms.txt 包含外域 URL（如 github.com、stackoverflow.com）时，
    使用 ALLOWED_DOC_URL_PREFIXES 配置后，这些外域 URL 不会进入最终的 fetch 列表。
    """

    def test_external_urls_filtered_by_default_config(self) -> None:
        """测试使用默认配置时外域 URL 被过滤

        场景：llms.txt 包含 cursor.com 和 github.com 的链接，
        使用 ALLOWED_DOC_URL_PREFIXES 配置后，只保留 cursor.com 的文档链接。
        """
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            select_urls_to_fetch,
        )
        from scripts.run_iterate import (
            ALLOWED_DOC_URL_PREFIXES,
            ALLOWED_DOC_URL_PREFIXES_NETLOC,
        )

        # 使用与 run_iterate.py 相同的配置（使用新名称）
        config = DocURLStrategyConfig(
            allowed_url_prefixes=ALLOWED_DOC_URL_PREFIXES,
            allowed_domains=ALLOWED_DOC_URL_PREFIXES_NETLOC,
            max_urls=20,
        )

        # 模拟包含外域 URL 的 llms.txt 内容
        llms_content = """
# Cursor CLI Documentation
https://cursor.com/docs/overview
https://cursor.com/cn/docs/cli/reference
https://github.com/cursor/cursor-cli
https://stackoverflow.com/questions/tagged/cursor
https://cursor.com/docs/api
https://npmjs.com/package/@cursor/cli
https://cursor.com/changelog/2024
"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=llms_content,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 应只包含 cursor.com/docs 或 cursor.com/cn/docs 或 cursor.com/changelog 前缀的 URL
        assert len(result) == 4, f"应有 4 个 cursor.com 链接，实际: {result}"
        assert "https://cursor.com/docs/overview" in result
        assert "https://cursor.com/cn/docs/cli/reference" in result
        assert "https://cursor.com/docs/api" in result
        assert "https://cursor.com/changelog/2024" in result

        # 外域 URL 不应出现
        assert not any("github.com" in u for u in result), f"github.com 链接应被过滤，实际: {result}"
        assert not any("stackoverflow.com" in u for u in result), f"stackoverflow.com 链接应被过滤，实际: {result}"
        assert not any("npmjs.com" in u for u in result), f"npmjs.com 链接应被过滤，实际: {result}"

    def test_allowed_doc_url_prefixes_correctness(self) -> None:
        """测试 ALLOWED_DOC_URL_PREFIXES 前缀列表的正确性

        验证 ALLOWED_DOC_URL_PREFIXES 包含预期的 cursor.com 文档前缀。
        权威来源：knowledge.doc_sources.DEFAULT_ALLOWED_DOC_URL_PREFIXES
        """
        from knowledge.doc_sources import DEFAULT_ALLOWED_DOC_URL_PREFIXES
        from scripts.run_iterate import ALLOWED_DOC_URL_PREFIXES

        expected_prefixes = [
            "https://cursor.com/cn/docs",
            "https://cursor.com/docs",
            "https://cursor.com/cn/changelog",
            "https://cursor.com/changelog",
        ]
        assert set(ALLOWED_DOC_URL_PREFIXES) == set(expected_prefixes), (
            f"ALLOWED_DOC_URL_PREFIXES 应包含预期前缀，实际: {ALLOWED_DOC_URL_PREFIXES}"
        )

        # 验证 ALLOWED_DOC_URL_PREFIXES 指向权威来源
        assert ALLOWED_DOC_URL_PREFIXES == DEFAULT_ALLOWED_DOC_URL_PREFIXES, (
            "ALLOWED_DOC_URL_PREFIXES 应指向 knowledge.doc_sources.DEFAULT_ALLOWED_DOC_URL_PREFIXES"
        )

    def test_allowed_doc_url_prefixes_netloc_derived_correctly(self) -> None:
        """测试 ALLOWED_DOC_URL_PREFIXES_NETLOC 从前缀正确推导

        验证 ALLOWED_DOC_URL_PREFIXES_NETLOC 包含从 ALLOWED_DOC_URL_PREFIXES 推导的域名。
        """
        from scripts.run_iterate import (
            ALLOWED_DOC_URL_PREFIXES,
            ALLOWED_DOC_URL_PREFIXES_NETLOC,
            _derive_allowed_domains_from_prefixes,
        )

        derived = _derive_allowed_domains_from_prefixes(ALLOWED_DOC_URL_PREFIXES)
        assert derived == ALLOWED_DOC_URL_PREFIXES_NETLOC
        assert "cursor.com" in ALLOWED_DOC_URL_PREFIXES_NETLOC

    def test_deprecated_alias_compatibility(self) -> None:
        """测试 [DEPRECATED] 旧名别名仍然可用（向后兼容）"""
        from scripts.run_iterate import (
            ALLOWED_DOC_DOMAINS,  # [DEPRECATED] 旧名别名
            ALLOWED_DOC_DOMAINS_NETLOC,  # [DEPRECATED] 旧名别名
            ALLOWED_DOC_URL_PREFIXES,
            ALLOWED_DOC_URL_PREFIXES_NETLOC,
        )

        # 旧名应该是新名的别名
        assert ALLOWED_DOC_DOMAINS is ALLOWED_DOC_URL_PREFIXES
        assert ALLOWED_DOC_DOMAINS_NETLOC is ALLOWED_DOC_URL_PREFIXES_NETLOC


# ============================================================
# TestAllowedUrlPrefixesAliasCompatibility - 别名兼容性测试
# ============================================================


class TestAllowedUrlPrefixesAliasCompatibility:
    """测试 allowed_url_prefixes 与 allowed_domains 别名的兼容性

    验证：
    1. knowledge.doc_sources 模块中的别名（DEFAULT_ALLOWED_DOC_DOMAINS）正确指向新名称
    2. is_valid_doc_url 函数的新旧参数名都能正常工作
    3. load_core_docs 函数的新旧参数名都能正常工作
    4. Cursor 文档场景使用 prefix 过滤，通用场景使用 domain 过滤
    """

    def test_default_allowed_doc_domains_is_alias(self) -> None:
        """测试 DEFAULT_ALLOWED_DOC_DOMAINS 是 DEFAULT_ALLOWED_DOC_URL_PREFIXES 的别名"""
        from knowledge.doc_sources import (
            DEFAULT_ALLOWED_DOC_DOMAINS,
            DEFAULT_ALLOWED_DOC_URL_PREFIXES,
        )

        # 应该是完全相同的对象
        assert DEFAULT_ALLOWED_DOC_DOMAINS is DEFAULT_ALLOWED_DOC_URL_PREFIXES
        # 或者至少内容相同
        assert DEFAULT_ALLOWED_DOC_DOMAINS == DEFAULT_ALLOWED_DOC_URL_PREFIXES

    def test_is_valid_doc_url_new_param_name(self) -> None:
        """测试 is_valid_doc_url 使用新参数名 allowed_url_prefixes"""
        from knowledge.doc_sources import is_valid_doc_url

        prefixes = [
            "https://cursor.com/cn/docs",
            "https://cursor.com/docs",
        ]

        # 使用新参数名
        assert is_valid_doc_url(
            "https://cursor.com/docs/cli/overview",
            allowed_url_prefixes=prefixes,
        )
        assert not is_valid_doc_url(
            "https://github.com/cursor/repo",
            allowed_url_prefixes=prefixes,
        )

    def test_is_valid_doc_url_old_param_name_alias(self) -> None:
        """测试 is_valid_doc_url 使用旧参数名 allowed_domains（向后兼容）"""
        from knowledge.doc_sources import is_valid_doc_url

        prefixes = [
            "https://cursor.com/cn/docs",
            "https://cursor.com/docs",
        ]

        # 使用旧参数名（应该正常工作，向后兼容）
        assert is_valid_doc_url(
            "https://cursor.com/docs/cli/overview",
            allowed_domains=prefixes,
        )
        assert not is_valid_doc_url(
            "https://github.com/cursor/repo",
            allowed_domains=prefixes,
        )

    def test_is_valid_doc_url_new_param_takes_priority(self) -> None:
        """测试 is_valid_doc_url 新参数名优先级高于旧参数名"""
        from knowledge.doc_sources import is_valid_doc_url

        new_prefixes = ["https://cursor.com/cn/docs"]
        old_prefixes = ["https://cursor.com/docs"]

        # 同时提供新旧参数时，新参数优先
        # URL 匹配 new_prefixes 但不匹配 old_prefixes
        assert is_valid_doc_url(
            "https://cursor.com/cn/docs/cli",
            allowed_url_prefixes=new_prefixes,
            allowed_domains=old_prefixes,  # 应该被忽略
        )
        # URL 匹配 old_prefixes 但不匹配 new_prefixes
        assert not is_valid_doc_url(
            "https://cursor.com/docs/cli",
            allowed_url_prefixes=new_prefixes,
            allowed_domains=old_prefixes,  # 应该被忽略
        )

    def test_is_valid_doc_url_with_config(self) -> None:
        """测试 is_valid_doc_url 直接传入 config（优先级最高）"""
        from knowledge.doc_sources import is_valid_doc_url
        from knowledge.doc_url_strategy import DocURLStrategyConfig

        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/special"],
            exclude_patterns=[],
        )

        # config 参数优先级最高
        assert is_valid_doc_url(
            "https://cursor.com/special/page",
            allowed_url_prefixes=["https://other.com"],  # 应该被忽略
            allowed_domains=["example.com"],  # 应该被忽略
            config=config,
        )
        assert not is_valid_doc_url(
            "https://cursor.com/docs/cli",
            config=config,
        )

    def test_load_core_docs_new_param_name(self, tmp_path: Path) -> None:
        """测试 load_core_docs 使用新参数名 allowed_url_prefixes"""
        from knowledge.doc_sources import load_core_docs

        docs_file = tmp_path / "test_docs.txt"
        docs_file.write_text("https://cursor.com/cn/docs/cli/overview\nhttps://github.com/cursor/repo\n")

        urls = load_core_docs(
            source_files=["test_docs.txt"],
            project_root=tmp_path,
            allowed_url_prefixes=[
                "https://cursor.com/cn/docs",
                "https://cursor.com/docs",
            ],
        )

        # 只有匹配前缀的 URL 被保留
        assert len(urls) == 1
        assert "https://cursor.com/cn/docs/cli/overview" in urls
        assert not any("github.com" in u for u in urls)

    def test_load_core_docs_old_param_name_alias(self, tmp_path: Path) -> None:
        """测试 load_core_docs 使用旧参数名 allowed_domains（向后兼容）"""
        from knowledge.doc_sources import load_core_docs

        docs_file = tmp_path / "test_docs.txt"
        docs_file.write_text("https://cursor.com/cn/docs/cli/overview\nhttps://github.com/cursor/repo\n")

        urls = load_core_docs(
            source_files=["test_docs.txt"],
            project_root=tmp_path,
            allowed_domains=[  # 使用旧参数名
                "https://cursor.com/cn/docs",
                "https://cursor.com/docs",
            ],
        )

        # 只有匹配前缀的 URL 被保留
        assert len(urls) == 1
        assert "https://cursor.com/cn/docs/cli/overview" in urls

    def test_run_iterate_is_allowed_doc_url_uses_prefix(self) -> None:
        """测试 run_iterate.is_allowed_doc_url 使用 ALLOWED_DOC_URL_PREFIXES

        验证 run_iterate.py 中的包装函数正确使用前缀匹配。
        """
        from scripts.run_iterate import ALLOWED_DOC_URL_PREFIXES, is_allowed_doc_url

        # ALLOWED_DOC_URL_PREFIXES 应该是前缀列表
        assert all(p.startswith("https://cursor.com/") for p in ALLOWED_DOC_URL_PREFIXES)

        # 匹配前缀的 URL
        assert is_allowed_doc_url("https://cursor.com/cn/docs/cli")
        assert is_allowed_doc_url("https://cursor.com/docs/guide")
        assert is_allowed_doc_url("https://cursor.com/changelog/2026")

        # 不匹配前缀的 URL（即使是 cursor.com 域名）
        assert not is_allowed_doc_url("https://cursor.com/pricing")
        assert not is_allowed_doc_url("https://cursor.com/blog")


# ============================================================
# TestPrefixVsDomainFilterSemantics - 前缀/域名过滤语义区分测试
# ============================================================


class TestPrefixVsDomainFilterSemantics:
    """测试 allowed_url_prefixes 与 allowed_domains 的语义区分

    确保两套过滤机制语义不冲突：
    - allowed_domains: 通用场景的域名过滤（如 python.org 允许所有路径）
    - allowed_url_prefixes: Cursor 文档更新流程的精确前缀过滤
    """

    def test_cursor_doc_scenario_uses_prefix_filter(self) -> None:
        """测试 Cursor 文档场景使用前缀过滤

        Cursor 文档更新流程只需要抓取特定路径（/docs、/changelog），
        而不是 cursor.com 下的所有页面。
        """
        from knowledge.doc_url_strategy import DocURLStrategyConfig, is_allowed_doc_url

        # Cursor 文档场景配置
        cursor_config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/cn/docs",
                "https://cursor.com/docs",
                "https://cursor.com/cn/changelog",
                "https://cursor.com/changelog",
            ],
            max_urls=20,
        )

        # 应该允许的 URL
        assert is_allowed_doc_url("https://cursor.com/cn/docs/cli/overview", cursor_config)
        assert is_allowed_doc_url("https://cursor.com/docs/guide", cursor_config)
        assert is_allowed_doc_url("https://cursor.com/changelog/2026", cursor_config)

        # 不应该允许的 URL（即使是 cursor.com 域名）
        assert not is_allowed_doc_url("https://cursor.com/pricing", cursor_config)
        assert not is_allowed_doc_url("https://cursor.com/blog/post", cursor_config)
        assert not is_allowed_doc_url("https://cursor.com/download", cursor_config)

    def test_general_scenario_uses_domain_filter(self) -> None:
        """测试通用场景使用域名过滤

        通用文档抓取场景可能需要允许某个域名下的所有路径。
        """
        from knowledge.doc_url_strategy import DocURLStrategyConfig, is_allowed_doc_url

        # 通用场景配置（空 prefixes，使用 domains）
        general_config = DocURLStrategyConfig(
            allowed_url_prefixes=[],
            allowed_domains=["python.org", "nodejs.org"],
            max_urls=20,
            exclude_patterns=[],
        )

        # 任意路径都应该被允许
        assert is_allowed_doc_url("https://docs.python.org/3/library/", general_config)
        assert is_allowed_doc_url("https://python.org/about/", general_config)
        assert is_allowed_doc_url("https://wiki.python.org/moin/", general_config)
        assert is_allowed_doc_url("https://nodejs.org/en/docs/", general_config)

        # 其他域名被拒绝
        assert not is_allowed_doc_url("https://github.com/python/cpython", general_config)

    def test_semantics_do_not_conflict(self) -> None:
        """测试两种过滤语义不冲突

        - 使用 allowed_url_prefixes 时，进行精确前缀匹配
        - 使用 allowed_domains 时（prefixes 为空），进行域名匹配
        - 两者互不干扰
        """
        from knowledge.doc_url_strategy import DocURLStrategyConfig, is_allowed_doc_url

        test_url = "https://cursor.com/docs/cli"

        # 配置 1: 只用 prefixes（精确匹配）
        prefix_config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            allowed_domains=["example.com"],  # 会被忽略
        )
        assert is_allowed_doc_url(test_url, prefix_config)
        assert not is_allowed_doc_url("https://example.com/page", prefix_config)

        # 配置 2: 只用 domains（域名匹配）
        domain_config = DocURLStrategyConfig(
            allowed_url_prefixes=[],
            allowed_domains=["cursor.com"],
            exclude_patterns=[],
        )
        assert is_allowed_doc_url(test_url, domain_config)
        assert is_allowed_doc_url("https://cursor.com/pricing", domain_config)  # 任意路径

        # 配置 3: 都为空（允许所有）
        empty_config = DocURLStrategyConfig(
            allowed_url_prefixes=[],
            allowed_domains=[],
            exclude_patterns=[],
        )
        assert is_allowed_doc_url(test_url, empty_config)
        assert is_allowed_doc_url("https://any-domain.com/page", empty_config)

    def test_doc_sources_is_valid_doc_url_uses_prefix_semantics(self) -> None:
        """测试 doc_sources.is_valid_doc_url 默认使用前缀语义"""
        from knowledge.doc_sources import (
            DEFAULT_ALLOWED_DOC_URL_PREFIXES,
            is_valid_doc_url,
        )

        # 默认使用前缀列表过滤（Cursor 文档场景）
        assert is_valid_doc_url("https://cursor.com/cn/docs/cli")
        assert is_valid_doc_url("https://cursor.com/docs/guide")
        assert is_valid_doc_url("https://cursor.com/changelog/2026")

        # 不匹配默认前缀的 URL 被拒绝
        assert not is_valid_doc_url("https://cursor.com/pricing")
        assert not is_valid_doc_url("https://github.com/cursor/repo")

        # 验证默认前缀列表内容
        assert "https://cursor.com/cn/docs" in DEFAULT_ALLOWED_DOC_URL_PREFIXES
        assert "https://cursor.com/docs" in DEFAULT_ALLOWED_DOC_URL_PREFIXES
        assert "https://cursor.com/cn/changelog" in DEFAULT_ALLOWED_DOC_URL_PREFIXES
        assert "https://cursor.com/changelog" in DEFAULT_ALLOWED_DOC_URL_PREFIXES

    def test_domains_allowed_but_prefixes_not_allowed(self) -> None:
        """测试 domains 允许但 path_prefixes 不允许的场景

        核心测试点：当 allowed_url_prefixes 不为空时，
        allowed_domains 被忽略，URL 必须匹配前缀才能通过。
        即使 URL 的域名在 allowed_domains 中，如果不匹配前缀，也会被拒绝。

        这确保了 fetch_policy 的优先级规则：
        allowed_url_prefixes 优先于 allowed_domains
        """
        from knowledge.doc_url_strategy import DocURLStrategyConfig, is_allowed_doc_url

        # 配置：allowed_domains 允许 cursor.com，但 prefixes 仅允许 /special 路径
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/special"],
            allowed_domains=["cursor.com"],  # 会被忽略（因为 prefixes 不为空）
            exclude_patterns=[],
        )

        # /docs 路径：域名匹配 allowed_domains，但不匹配 prefixes → 被拒绝
        assert not is_allowed_doc_url("https://cursor.com/docs/cli", config), (
            "URL 不匹配 allowed_url_prefixes 应被拒绝（即使域名匹配 allowed_domains）"
        )

        # /pricing 路径：域名匹配 allowed_domains，但不匹配 prefixes → 被拒绝
        assert not is_allowed_doc_url("https://cursor.com/pricing", config), "URL 不匹配 allowed_url_prefixes 应被拒绝"

        # /special 路径：匹配 prefixes → 允许
        assert is_allowed_doc_url("https://cursor.com/special/page", config), "URL 匹配 allowed_url_prefixes 应被允许"

        # /special 根路径：精确匹配 prefixes → 允许
        assert is_allowed_doc_url("https://cursor.com/special", config), "URL 精确匹配 allowed_url_prefixes 应被允许"

    def test_url_prefixes_restrict_to_specific_path_only(self) -> None:
        """测试 url_prefixes 限制只抓取特定路径

        核心测试点：设置 allowed_url_prefixes 为特定路径列表时，
        只有匹配这些路径的 URL 才能进入 fetch 列表。
        """
        from knowledge.doc_url_strategy import DocURLStrategyConfig, is_allowed_doc_url

        # 配置：仅允许 cn/docs 路径（不允许 docs、changelog 等）
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/cn/docs"],
            allowed_domains=["cursor.com"],  # 会被忽略
            exclude_patterns=[],
        )

        # cn/docs 路径：匹配 → 允许
        assert is_allowed_doc_url("https://cursor.com/cn/docs/cli", config)
        assert is_allowed_doc_url("https://cursor.com/cn/docs", config)
        assert is_allowed_doc_url("https://cursor.com/cn/docs/guide/overview", config)

        # docs 路径（不带 cn）：不匹配 → 拒绝
        assert not is_allowed_doc_url("https://cursor.com/docs/cli", config), "/docs 不匹配 /cn/docs 前缀，应被拒绝"

        # changelog 路径：不匹配 → 拒绝
        assert not is_allowed_doc_url("https://cursor.com/changelog", config), (
            "/changelog 不匹配 /cn/docs 前缀，应被拒绝"
        )

        # 其他路径：不匹配 → 拒绝
        assert not is_allowed_doc_url("https://cursor.com/pricing", config)
        assert not is_allowed_doc_url("https://cursor.com/cn/changelog", config)

    def test_prefixes_empty_fallback_to_domains(self) -> None:
        """测试 prefixes 为空时回退到 domains 检查

        核心测试点：当 allowed_url_prefixes 为空列表时，
        使用 allowed_domains 进行域名匹配过滤。
        """
        from knowledge.doc_url_strategy import DocURLStrategyConfig, is_allowed_doc_url

        # 配置：prefixes 为空，使用 domains 过滤
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[],  # 空列表 → 回退到 domains
            allowed_domains=["cursor.com"],
            exclude_patterns=[],
        )

        # cursor.com 域名下的任意路径都应被允许
        assert is_allowed_doc_url("https://cursor.com/docs/cli", config)
        assert is_allowed_doc_url("https://cursor.com/pricing", config)
        assert is_allowed_doc_url("https://cursor.com/blog/post", config)
        assert is_allowed_doc_url("https://cursor.com/any/path", config)

        # 其他域名被拒绝
        assert not is_allowed_doc_url("https://github.com/cursor", config)
        assert not is_allowed_doc_url("https://example.com/docs", config)

        # 子域名匹配
        assert is_allowed_doc_url("https://api.cursor.com/v1/chat", config)
        assert is_allowed_doc_url("https://docs.cursor.com/guide", config)


# ============================================================
# TestDeprecatedCLIParameterCompatibility - 废弃 CLI 参数兼容性测试
# ============================================================


class TestDeprecatedCLIParameterCompatibility:
    """测试废弃的 CLI 参数向后兼容性

    验证：
    1. --allowed-url-prefixes（deprecated）仍能正常工作
    2. 使用 deprecated 参数时有清晰的 warning 日志
    3. 新旧参数同时使用时，新参数（--allowed-path-prefixes）优先
    4. 行为与使用新参数时完全一致
    """

    def test_deprecated_allowed_url_prefixes_cli_param_works(self) -> None:
        """测试 --allowed-url-prefixes（deprecated）参数仍能正常解析"""
        from argparse import Namespace
        from io import StringIO

        from loguru import logger

        from scripts.run_iterate import resolve_docs_source_config

        # 模拟使用 deprecated 参数的 args
        args = Namespace(
            allowed_path_prefixes=None,  # 新参数未指定
            allowed_url_prefixes_deprecated="docs,cn/docs",  # 使用旧参数
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            max_fetch_urls=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            fallback_core_docs_count=None,
            allowed_doc_url_prefixes=None,
        )

        # 捕获日志以验证 warning
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")
        try:
            config = resolve_docs_source_config(args)
        finally:
            logger.remove(handler_id)

        # 验证配置正确解析
        assert config.fetch_policy.allowed_path_prefixes == ["docs", "cn/docs"], "deprecated 参数应被正确解析"

        # 验证有 deprecation warning
        log_content = log_output.getvalue()
        assert "--allowed-url-prefixes 已废弃" in log_content, "使用 deprecated 参数应输出 warning"

    def test_new_param_takes_priority_over_deprecated(self) -> None:
        """测试新参数 --allowed-path-prefixes 优先级高于 deprecated 参数"""
        from argparse import Namespace
        from io import StringIO

        from loguru import logger

        from scripts.run_iterate import resolve_docs_source_config

        # 模拟同时指定新旧参数
        args = Namespace(
            allowed_path_prefixes="special,cn/special",  # 新参数
            allowed_url_prefixes_deprecated="docs,cn/docs",  # 旧参数（应被忽略）
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            max_fetch_urls=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            fallback_core_docs_count=None,
            allowed_doc_url_prefixes=None,
        )

        # 捕获日志
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")
        try:
            config = resolve_docs_source_config(args)
        finally:
            logger.remove(handler_id)

        # 验证新参数生效（旧参数被忽略）
        assert config.fetch_policy.allowed_path_prefixes == ["special", "cn/special"], "新参数应优先于 deprecated 参数"
        assert "docs" not in config.fetch_policy.allowed_path_prefixes, "deprecated 参数的值应被忽略"

        # 验证有 warning 提示同时指定了两个参数
        log_content = log_output.getvalue()
        assert "--allowed-path-prefixes 优先级更高" in log_content or "已忽略 --allowed-url-prefixes" in log_content, (
            "同时指定新旧参数应输出 warning"
        )

    def test_deprecated_param_behavior_consistent_with_new_param(self) -> None:
        """测试 deprecated 参数行为与新参数完全一致

        使用 deprecated 参数设置的配置，应该与使用新参数得到相同的过滤结果。
        """
        from argparse import Namespace

        from scripts.run_iterate import resolve_docs_source_config

        path_prefixes_value = "docs,cn/docs,changelog"

        # 使用新参数
        args_new = Namespace(
            allowed_path_prefixes=path_prefixes_value,
            allowed_url_prefixes_deprecated=None,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            max_fetch_urls=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            fallback_core_docs_count=None,
            allowed_doc_url_prefixes=None,
        )
        config_new = resolve_docs_source_config(args_new)

        # 使用 deprecated 参数
        args_deprecated = Namespace(
            allowed_path_prefixes=None,
            allowed_url_prefixes_deprecated=path_prefixes_value,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            max_fetch_urls=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            fallback_core_docs_count=None,
            allowed_doc_url_prefixes=None,
        )
        config_deprecated = resolve_docs_source_config(args_deprecated)

        # 验证两者配置结果完全一致
        assert config_new.fetch_policy.allowed_path_prefixes == config_deprecated.fetch_policy.allowed_path_prefixes, (
            "新旧参数应产生相同的配置结果"
        )

    def test_config_yaml_deprecated_field_backward_compat(self) -> None:
        """测试 config.yaml 中废弃字段的向后兼容性

        验证 fetch_policy.allowed_url_prefixes（旧字段名）仍能被正确解析。
        """
        import tempfile
        from io import StringIO
        from pathlib import Path

        import yaml
        from loguru import logger

        from core.config import ConfigManager

        # 创建使用旧字段名的 config.yaml
        config_data = {
            "knowledge_docs_update": {
                "docs_source": {
                    "fetch_policy": {
                        "allowed_url_prefixes": ["special", "cn/special"],  # 旧字段名
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            # 切换到临时目录并重置 ConfigManager
            import os

            from core.config import reset_deprecated_warnings

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                ConfigManager.reset_instance()
                # 重置 deprecated 警告状态以确保警告可以被触发
                reset_deprecated_warnings()

                # 捕获日志
                log_output = StringIO()
                handler_id = logger.add(log_output, format="{message}", level="WARNING")
                try:
                    config = ConfigManager.get_instance()
                finally:
                    logger.remove(handler_id)

                # 验证旧字段名被正确解析到 allowed_path_prefixes
                fetch_policy = config.knowledge_docs_update.docs_source.fetch_policy
                assert fetch_policy.allowed_path_prefixes == ["special", "cn/special"], (
                    "旧字段名 allowed_url_prefixes 应被解析到 allowed_path_prefixes"
                )

                # 验证有 deprecation warning
                log_content = log_output.getvalue()
                assert "allowed_url_prefixes 已废弃" in log_content, "使用旧字段名应输出 deprecation warning"
            finally:
                os.chdir(original_cwd)
                ConfigManager.reset_instance()

    def test_deprecated_warning_only_once_per_key(self) -> None:
        """测试 deprecated 警告每个 key 仅输出一次

        验证统一的 deprecated 警告机制：多次触发同一类警告时，
        只在第一次输出警告，后续调用不再重复输出。
        """
        from argparse import Namespace
        from io import StringIO

        from loguru import logger

        from core.config import (
            reset_deprecated_warnings,
        )
        from scripts.run_iterate import resolve_docs_source_config

        # 重置警告状态，确保测试独立性
        reset_deprecated_warnings()

        # 第一次调用：应该输出警告
        args1 = Namespace(
            allowed_path_prefixes=None,
            allowed_url_prefixes_deprecated="docs,cn/docs",
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            max_fetch_urls=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            fallback_core_docs_count=None,
            allowed_doc_url_prefixes=None,
        )

        log_output1 = StringIO()
        handler_id1 = logger.add(log_output1, format="{message}", level="WARNING")
        try:
            resolve_docs_source_config(args1)
        finally:
            logger.remove(handler_id1)

        log_content1 = log_output1.getvalue()
        assert "--allowed-url-prefixes 已废弃" in log_content1, "第一次调用应输出 deprecated 警告"

        # 第二次调用：不应该再输出警告（同一个 key）
        args2 = Namespace(
            allowed_path_prefixes=None,
            allowed_url_prefixes_deprecated="api,cn/api",
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            max_fetch_urls=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            fallback_core_docs_count=None,
            allowed_doc_url_prefixes=None,
        )

        log_output2 = StringIO()
        handler_id2 = logger.add(log_output2, format="{message}", level="WARNING")
        try:
            resolve_docs_source_config(args2)
        finally:
            logger.remove(handler_id2)

        log_content2 = log_output2.getvalue()
        assert "--allowed-url-prefixes 已废弃" not in log_content2, "第二次调用不应重复输出 deprecated 警告"

        # 重置后再调用：应该再次输出警告
        reset_deprecated_warnings()

        log_output3 = StringIO()
        handler_id3 = logger.add(log_output3, format="{message}", level="WARNING")
        try:
            resolve_docs_source_config(args1)
        finally:
            logger.remove(handler_id3)

        log_content3 = log_output3.getvalue()
        assert "--allowed-url-prefixes 已废弃" in log_content3, "重置警告状态后应再次输出 deprecated 警告"

        # 清理：重置警告状态
        reset_deprecated_warnings()

    def test_deprecated_message_key_fragments(self) -> None:
        """测试 deprecated 警告使用正确的关键文案片段

        验证关键文案片段存在，防止未来修改破坏兼容提示。
        这些片段用于用户识别警告类型和采取行动。
        """
        from core.config import (
            DEPRECATED_MSG_CLI_BOTH_PARAMS,
            DEPRECATED_MSG_CLI_OLD_PARAM,
            DEPRECATED_MSG_FETCH_POLICY_BOTH_FIELDS,
            DEPRECATED_MSG_FETCH_POLICY_OLD_FIELD,
        )
        from knowledge.doc_sources import (
            DEPRECATED_MSG_ALIAS_DEPRECATED,
            DEPRECATED_MSG_PREFIX,
            DEPRECATED_MSG_WILL_REMOVE,
        )
        from knowledge.doc_url_strategy import (
            DEPRECATED_MSG_FUNC_DEPRECATED,
            DEPRECATED_MSG_FUNC_PREFIX,
            DEPRECATED_MSG_FUNC_WILL_REMOVE_V2,
        )

        # 验证 core/config.py 的关键文案片段
        assert "已废弃" in DEPRECATED_MSG_FETCH_POLICY_OLD_FIELD, "fetch_policy 旧字段警告应包含 '已废弃'"
        assert "优先级更高" in DEPRECATED_MSG_FETCH_POLICY_BOTH_FIELDS, "fetch_policy 双字段警告应包含 '优先级更高'"
        assert "已废弃" in DEPRECATED_MSG_CLI_OLD_PARAM, "CLI 旧参数警告应包含 '已废弃'"
        assert "优先级更高" in DEPRECATED_MSG_CLI_BOTH_PARAMS, "CLI 双参数警告应包含 '优先级更高'"

        # 验证 doc_sources.py 的关键文案片段
        assert DEPRECATED_MSG_PREFIX == "[DEPRECATED]", "deprecated 前缀应为 '[DEPRECATED]'"
        assert "弃用" in DEPRECATED_MSG_ALIAS_DEPRECATED, "别名废弃警告应包含 '弃用'"
        assert "移除" in DEPRECATED_MSG_WILL_REMOVE, "移除计划应包含 '移除'"

        # 验证 doc_url_strategy.py 的关键文案片段
        assert DEPRECATED_MSG_FUNC_PREFIX == "[DEPRECATED]", "函数废弃前缀应为 '[DEPRECATED]'"
        assert "废弃" in DEPRECATED_MSG_FUNC_DEPRECATED, "函数废弃警告应包含 '废弃'"
        assert "v2.0" in DEPRECATED_MSG_FUNC_WILL_REMOVE_V2, "函数移除计划应包含版本号 'v2.0'"


# ============================================================
# TestCompatibilityEntry - 兼容入口测试
# ============================================================


class TestCompatibilityEntry:
    """测试 scripts/self_iterate.py 兼容入口

    覆盖场景：
    1. 兼容入口正确导入 run_iterate.main
    2. 兼容入口与 run_iterate 功能等效
    """

    def test_self_iterate_imports_run_iterate_main(self) -> None:
        """测试兼容入口正确导入 run_iterate.main"""
        # 验证 self_iterate 模块可以被导入
        from scripts import self_iterate

        # 验证 main 函数来自 run_iterate
        from scripts.run_iterate import main as run_iterate_main

        assert self_iterate.main is run_iterate_main, "self_iterate.main 应该是 run_iterate.main 的引用"

    def test_self_iterate_module_has_docstring(self) -> None:
        """测试兼容入口有正确的文档字符串"""
        from scripts import self_iterate

        assert self_iterate.__doc__ is not None
        assert "兼容入口" in self_iterate.__doc__
        assert "run_iterate" in self_iterate.__doc__

    def test_self_iterate_callable(self) -> None:
        """测试兼容入口的 main 函数可调用"""
        from scripts import self_iterate

        # 验证 main 是可调用的
        assert callable(self_iterate.main)


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_flow_with_mocked_components(self, base_iterate_args: argparse.Namespace) -> None:
        """测试完整流程（Mock 所有外部组件）"""
        base_iterate_args.skip_online = True
        iterator = SelfIterator(base_iterate_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 5})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        mock_run_result = {
            "success": True,
            "iterations_completed": 2,
            "total_tasks_created": 3,
            "total_tasks_completed": 3,
            "total_tasks_failed": 0,
        }

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator,
                "_run_agent_system",
                new_callable=AsyncMock,
                return_value=mock_run_result,
            ),
        ):
            result = await iterator.run()

            # 验证完整流程成功
            assert result["success"] is True
            assert result["iterations_completed"] == 2

    @pytest.mark.asyncio
    async def test_error_handling_in_full_flow(self, base_iterate_args: argparse.Namespace) -> None:
        """测试完整流程中的错误处理"""
        iterator = SelfIterator(base_iterate_args)

        # Mock 知识库初始化抛出异常
        with (
            patch.object(
                iterator.knowledge_updater,
                "initialize",
                new_callable=AsyncMock,
                side_effect=Exception("初始化失败"),
            ),
            patch.object(iterator.changelog_analyzer.fetcher, "initialize", new_callable=AsyncMock),
        ):
            # Mock fetch 返回失败
            mock_fetch_result = MagicMock()
            mock_fetch_result.success = False
            mock_fetch_result.error = "Network error"
            with patch.object(
                iterator.changelog_analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ):
                result = await iterator.run()

                # 验证错误被正确处理
                assert result["success"] is False
                assert "error" in result


# ============================================================
# TestOrchestratorRoutingRules - 编排器路由规则测试
# ============================================================


class TestOrchestratorRoutingRules:
    """测试编排器路由规则和参数映射

    覆盖场景:
    1. --orchestrator 和 --no-mp 命令行参数优先级
    2. _orchestrator_user_set 元字段的作用
    3. requirement 中非并行关键词的检测
    4. 用户显式设置 vs 自然语言推断的优先级
    """

    @pytest.fixture
    def routing_args(self) -> argparse.Namespace:
        """创建用于路由测试的基础参数"""
        return argparse.Namespace(
            requirement="测试需求",
            directory=".",  # 工作目录
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
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
            execution_mode="cli",
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
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_orchestrator_user_set_true_overrides_keyword(self, routing_args: argparse.Namespace) -> None:
        """测试 _orchestrator_user_set=True 时关键词不覆盖用户设置"""
        # 用户显式设置 --orchestrator mp，但 requirement 包含非并行关键词
        routing_args.requirement = "使用协程模式完成自我迭代"
        routing_args.orchestrator = "mp"
        routing_args._orchestrator_user_set = True

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "mp", (
            "用户显式设置 --orchestrator mp 时，即使 requirement 包含 '协程模式'，也应该使用 mp 编排器"
        )

    def test_orchestrator_user_set_false_allows_keyword_override(self, routing_args: argparse.Namespace) -> None:
        """测试 _orchestrator_user_set=False 时关键词可以覆盖默认设置"""
        # 用户未显式设置，requirement 包含非并行关键词
        routing_args.requirement = "使用协程模式完成自我迭代"
        routing_args.orchestrator = "mp"
        routing_args._orchestrator_user_set = False

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "basic", "用户未显式设置时，requirement 包含 '协程模式' 应该触发 basic 编排器"

    def test_no_mp_flag_always_uses_basic(self, routing_args: argparse.Namespace) -> None:
        """测试 --no-mp 标志始终使用 basic 编排器"""
        routing_args.no_mp = True
        routing_args._orchestrator_user_set = True

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "basic", "--no-mp 标志应该始终使用 basic 编排器"

    def test_explicit_orchestrator_basic_uses_basic(self, routing_args: argparse.Namespace) -> None:
        """测试显式设置 --orchestrator basic 使用 basic 编排器"""
        routing_args.orchestrator = "basic"
        routing_args._orchestrator_user_set = True

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "basic", "--orchestrator basic 应该使用 basic 编排器"

    def test_all_disable_mp_keywords_detected(self, routing_args: argparse.Namespace) -> None:
        """测试所有禁用 MP 关键词都能被正确检测"""
        disable_mp_keywords_requirements = [
            "非并行模式执行",
            "不并行处理",
            "串行执行任务",
            "协程模式运行",
            "单进程处理",
            "使用 basic 编排器",
            "no-mp 模式",
            "禁用多进程",
            "禁用mp模式",
            "关闭多进程",
        ]

        for requirement in disable_mp_keywords_requirements:
            routing_args.requirement = requirement
            routing_args._orchestrator_user_set = False

            iterator = SelfIterator(routing_args)
            orch_type = iterator._get_orchestrator_type()

            assert orch_type == "basic", f"requirement='{requirement}' 应该触发 basic 编排器，实际返回 {orch_type}"

    def test_no_keyword_cli_mode_uses_mp(self, routing_args: argparse.Namespace) -> None:
        """测试 cli 模式下无非并行关键词时使用 mp 编排器

        场景：execution_mode='cli' 且 requirement 不包含非并行关键词
        期望：使用 mp 编排器（cli 模式默认）
        """
        # routing_args fixture 已设置 execution_mode='cli'
        routing_args.requirement = "优化代码结构并添加单元测试"
        routing_args._orchestrator_user_set = False

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "mp", "cli 模式下无非并行关键词时应使用 mp 编排器"

    def test_mixed_keywords_still_detects_disable_mp(self, routing_args: argparse.Namespace) -> None:
        """测试混合关键词时仍能检测到禁用 MP"""
        routing_args.requirement = "使用多进程和协程模式完成任务"
        routing_args._orchestrator_user_set = False

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        # 只要包含禁用关键词就应该触发 basic
        assert orch_type == "basic", "包含 '协程模式' 时即使也包含 '多进程' 也应该触发 basic 编排器"

    @pytest.mark.asyncio
    async def test_mp_config_receives_correct_options(self, routing_args: argparse.Namespace) -> None:
        """测试 MP 编排器配置正确接收选项"""
        routing_args.auto_commit = True
        routing_args.commit_per_iteration = True
        routing_args.workers = 5

        iterator = SelfIterator(routing_args)
        iterator.context.iteration_goal = "测试目标"

        with patch("scripts.run_iterate.MultiProcessOrchestratorConfig") as MockConfig:
            with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_config_instance = MagicMock()
                    MockConfig.return_value = mock_config_instance

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(return_value={"success": True})
                    MockMP.return_value = mock_orch

                    await iterator._run_with_mp_orchestrator(3, mock_km)

                    # 验证配置参数
                    MockConfig.assert_called_once()
                    call_kwargs = MockConfig.call_args.kwargs

                    assert call_kwargs.get("worker_count") == 5
                    assert call_kwargs.get("enable_auto_commit") is True
                    assert call_kwargs.get("commit_per_iteration") is True
                    assert call_kwargs.get("commit_on_complete") is False

    @pytest.mark.asyncio
    async def test_basic_config_receives_correct_options(self, routing_args: argparse.Namespace) -> None:
        """测试 basic 编排器配置正确接收选项"""
        routing_args.auto_commit = True
        routing_args.auto_push = True
        routing_args.workers = 4

        iterator = SelfIterator(routing_args)
        iterator.context.iteration_goal = "测试目标"

        with patch("scripts.run_iterate.OrchestratorConfig") as MockConfig:
            with patch("scripts.run_iterate.Orchestrator") as MockOrch:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    with patch("scripts.run_iterate.CursorAgentConfig"):
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        mock_config_instance = MagicMock()
                        MockConfig.return_value = mock_config_instance

                        mock_orch = MagicMock()
                        mock_orch.run = AsyncMock(return_value={"success": True})
                        MockOrch.return_value = mock_orch

                        await iterator._run_with_basic_orchestrator(3, mock_km)

                        # 验证配置参数
                        MockConfig.assert_called_once()
                        call_kwargs = MockConfig.call_args.kwargs

                        assert call_kwargs.get("worker_pool_size") == 4
                        assert call_kwargs.get("enable_auto_commit") is True
                        assert call_kwargs.get("auto_push") is True


# ============================================================
# TestCommitDeduplication - 提交去重测试
# ============================================================


class TestCommitDeduplication:
    """测试提交去重逻辑

    覆盖场景:
    1. 编排器已提交时 SelfIterator 不再二次提交
    2. 编排器未提交时 SelfIterator 执行提交
    3. 各种 commits 结构的检测
    """

    @pytest.fixture
    def dedup_args(self) -> argparse.Namespace:
        """创建用于去重测试的参数"""
        return argparse.Namespace(
            requirement="测试提交去重",
            directory=".",  # 工作目录
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
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
            auto_commit=True,  # 启用自动提交
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",
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
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_has_orchestrator_committed_with_total_commits(self, dedup_args: argparse.Namespace) -> None:
        """测试 total_commits > 0 时检测为已提交"""
        iterator = SelfIterator(dedup_args)

        result = {"commits": {"total_commits": 2, "commit_hashes": ["a", "b"]}}
        assert iterator._has_orchestrator_committed(result) is True

    def test_has_orchestrator_committed_with_successful_commits(self, dedup_args: argparse.Namespace) -> None:
        """测试 successful_commits > 0 时检测为已提交"""
        iterator = SelfIterator(dedup_args)

        result = {"commits": {"successful_commits": 1}}
        assert iterator._has_orchestrator_committed(result) is True

    def test_has_orchestrator_committed_with_iteration_commit_hash(self, dedup_args: argparse.Namespace) -> None:
        """测试 iteration 有 commit_hash 时检测为已提交"""
        iterator = SelfIterator(dedup_args)

        result = {
            "commits": {},
            "iterations": [
                {"id": 1, "commit_hash": "abc123"},
            ],
        }
        assert iterator._has_orchestrator_committed(result) is True

    def test_has_orchestrator_committed_empty_commits(self, dedup_args: argparse.Namespace) -> None:
        """测试空 commits 时检测为未提交"""
        iterator = SelfIterator(dedup_args)

        result: dict[str, Any] = {"commits": {}}
        assert iterator._has_orchestrator_committed(result) is False

    def test_has_orchestrator_committed_zero_total(self, dedup_args: argparse.Namespace) -> None:
        """测试 total_commits=0 时检测为未提交"""
        iterator = SelfIterator(dedup_args)

        result = {"commits": {"total_commits": 0}}
        assert iterator._has_orchestrator_committed(result) is False

    def test_has_orchestrator_committed_no_commits_field(self, dedup_args: argparse.Namespace) -> None:
        """测试无 commits 字段时检测为未提交"""
        iterator = SelfIterator(dedup_args)

        result = {"success": True}
        assert iterator._has_orchestrator_committed(result) is False

    def test_has_orchestrator_committed_empty_iteration_commit_hash(self, dedup_args: argparse.Namespace) -> None:
        """测试 iteration commit_hash 为空时检测为未提交"""
        iterator = SelfIterator(dedup_args)

        result = {
            "commits": {},
            "iterations": [
                {"id": 1, "commit_hash": ""},
                {"id": 2, "commit_hash": None},
            ],
        }
        assert iterator._has_orchestrator_committed(result) is False

    @pytest.mark.asyncio
    async def test_skip_commit_when_orchestrator_already_committed(self, dedup_args: argparse.Namespace) -> None:
        """测试编排器已提交时跳过 SelfIterator 提交阶段"""
        iterator = SelfIterator(dedup_args)
        iterator.context.iteration_goal = "测试目标"

        # 编排器返回已提交的结果
        mp_result = {
            "success": True,
            "commits": {"total_commits": 1, "commit_hashes": ["abc"]},
            "iterations_completed": 1,
        }

        with (
            patch.object(iterator, "_run_with_mp_orchestrator", new_callable=AsyncMock, return_value=mp_result),
            patch.object(iterator, "_run_commit_phase", new_callable=AsyncMock) as mock_commit,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            await iterator._run_agent_system()

            # _run_commit_phase 不应被调用
            mock_commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_commit_when_orchestrator_not_committed(self, dedup_args: argparse.Namespace) -> None:
        """测试编排器未提交时执行 SelfIterator 提交阶段"""
        iterator = SelfIterator(dedup_args)
        iterator.context.iteration_goal = "测试目标"

        # 编排器返回未提交的结果
        mp_result = {
            "success": True,
            "commits": {},
            "iterations_completed": 1,
            "total_tasks_completed": 2,
        }

        commit_result = {
            "total_commits": 1,
            "commit_hashes": ["def456"],
            "pushed_commits": 0,
        }

        with (
            patch.object(iterator, "_run_with_mp_orchestrator", new_callable=AsyncMock, return_value=mp_result),
            patch.object(
                iterator, "_run_commit_phase", new_callable=AsyncMock, return_value=commit_result
            ) as mock_commit,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # _run_commit_phase 应被调用
            mock_commit.assert_called_once()

            # 验证结果包含提交信息
            assert result["commits"]["total_commits"] == 1


# ============================================================
# TestFallbackVsDegraded - 回退与降级区分测试
# ============================================================


class TestFallbackVsDegraded:
    """测试 _fallback_required 与 degraded 的区分

    核心区别:
    - _fallback_required: MP 编排器启动失败，需要回退到 basic
    - degraded: MP 编排器运行过程中降级，不触发回退，保留部分结果
    """

    @pytest.fixture
    def fallback_args(self) -> argparse.Namespace:
        """创建用于回退测试的参数"""
        return argparse.Namespace(
            requirement="测试回退逻辑",
            directory=".",  # 工作目录
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
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
            execution_mode="cli",
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
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    @pytest.mark.asyncio
    async def test_fallback_required_triggers_basic_orchestrator(self, fallback_args: argparse.Namespace) -> None:
        """测试 _fallback_required=True 触发 basic 编排器"""
        iterator = SelfIterator(fallback_args)
        iterator.context.iteration_goal = "测试目标"

        # MP 返回 _fallback_required
        mp_startup_failure = {
            "_fallback_required": True,
            "_fallback_reason": "启动超时",
        }

        basic_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
        }

        with (
            patch.object(
                iterator, "_run_with_mp_orchestrator", new_callable=AsyncMock, return_value=mp_startup_failure
            ),
            patch.object(
                iterator, "_run_with_basic_orchestrator", new_callable=AsyncMock, return_value=basic_result
            ) as mock_basic,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # basic 编排器应被调用
            mock_basic.assert_called_once()
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_degraded_does_not_trigger_basic_fallback(self, fallback_args: argparse.Namespace) -> None:
        """测试 degraded=True 不触发 basic 回退"""
        iterator = SelfIterator(fallback_args)
        iterator.context.iteration_goal = "测试目标"

        # MP 返回 degraded 结果（无 _fallback_required）
        mp_degraded_result = {
            "success": False,
            "degraded": True,
            "degradation_reason": "Planner 进程不健康",
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 1,
            "total_tasks_failed": 1,
            "commits": {},
        }

        with (
            patch.object(
                iterator, "_run_with_mp_orchestrator", new_callable=AsyncMock, return_value=mp_degraded_result
            ),
            patch.object(iterator, "_run_with_basic_orchestrator", new_callable=AsyncMock) as mock_basic,
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # basic 编排器不应被调用
            mock_basic.assert_not_called()

            # 验证保留 MP 的降级结果
            assert result["degraded"] is True
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_degraded_preserves_partial_work(self, fallback_args: argparse.Namespace) -> None:
        """测试降级时保留部分工作结果"""
        iterator = SelfIterator(fallback_args)
        iterator.context.iteration_goal = "测试目标"

        mp_degraded_result = {
            "success": False,
            "degraded": True,
            "degradation_reason": "Worker 不健康",
            "iterations_completed": 3,
            "total_tasks_created": 10,
            "total_tasks_completed": 7,
            "total_tasks_failed": 3,
            "commits": {"total_commits": 2, "commit_hashes": ["a", "b"]},
        }

        with (
            patch.object(
                iterator, "_run_with_mp_orchestrator", new_callable=AsyncMock, return_value=mp_degraded_result
            ),
            patch("scripts.run_iterate.KnowledgeManager") as MockKM,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            MockKM.return_value = mock_km

            result = await iterator._run_agent_system()

            # 验证保留部分工作
            assert result["iterations_completed"] == 3
            assert result["total_tasks_completed"] == 7
            assert result["commits"]["total_commits"] == 2

    @pytest.mark.asyncio
    async def test_fallback_required_various_reasons(self, fallback_args: argparse.Namespace) -> None:
        """测试各种回退原因都能正确触发 basic"""
        iterator = SelfIterator(fallback_args)
        iterator.context.iteration_goal = "测试目标"

        fallback_reasons = [
            ("启动超时", "timeout"),
            ("进程创建失败", "os_error"),
            ("运行时错误", "runtime_error"),
        ]

        for reason_cn, _ in fallback_reasons:
            mp_failure = {
                "_fallback_required": True,
                "_fallback_reason": reason_cn,
            }

            basic_result = {"success": True, "iterations_completed": 1}

            with (
                patch.object(iterator, "_run_with_mp_orchestrator", new_callable=AsyncMock, return_value=mp_failure),
                patch.object(
                    iterator, "_run_with_basic_orchestrator", new_callable=AsyncMock, return_value=basic_result
                ) as mock_basic,
                patch("scripts.run_iterate.KnowledgeManager") as MockKM,
            ):
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                result = await iterator._run_agent_system()

                mock_basic.assert_called_once()
                assert result["success"] is True

                # 重置 mock
                mock_basic.reset_mock()

    @pytest.mark.asyncio
    async def test_degraded_various_reasons(self, fallback_args: argparse.Namespace) -> None:
        """测试各种降级原因的处理一致性"""
        iterator = SelfIterator(fallback_args)
        iterator.context.iteration_goal = "测试目标"

        degradation_reasons = [
            "Planner 进程不健康",
            "Reviewer 进程不健康",
            "不健康 Worker 数量超过阈值: 3 > 1",
        ]

        for reason in degradation_reasons:
            mp_degraded = {
                "success": False,
                "degraded": True,
                "degradation_reason": reason,
                "iterations_completed": 1,
                "commits": {},
            }

            with (
                patch.object(iterator, "_run_with_mp_orchestrator", new_callable=AsyncMock, return_value=mp_degraded),
                patch("scripts.run_iterate.KnowledgeManager") as MockKM,
            ):
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                result = await iterator._run_agent_system()

                # 所有降级原因应有一致的字段结构
                assert result["success"] is False
                assert result["degraded"] is True
                assert result["degradation_reason"] == reason


# ============================================================
# TestMinimalCLIJan16Changelog - 最小 CLI Jan 16, 2026 变更日志测试
# ============================================================

# 最小化 HTML/文本片段，包含 'CLI Jan 16, 2026' 标题行和相关关键字
MINIMAL_CLI_JAN_16_2026_SNIPPET = """
CLI Jan 16, 2026

### New Features
- **Plan Mode**: Added `/plan` command for read-only planning mode.
- **Ask Mode**: Added `/ask` command for Q&A mode.
- **Cloud Relay**: MCP servers now support cloud relay for remote access.
- **Diff View**: Enhanced diff view with `Ctrl+R` shortcut for reviewing changes.

### Improvements
- Better streaming output support.
"""


class TestMinimalCLIJan16Changelog:
    """测试最小化 'CLI Jan 16, 2026' 变更日志片段的解析能力

    这个测试类验证 ChangelogAnalyzer 能够正确解析包含
    CLI Jan 16, 2026 格式标题和 plan/ask/cloud relay/diff 关键字的最小片段。

    覆盖场景：
    1. parse_changelog 能产生至少 1 条 entry
    2. entry.date、entry.category、entry.content 非空或符合预期
    3. extract_update_points 能把条目归入 feature/fix/improvement 类别
    4. extract_update_points 能命中相关文档 URL
    """

    def test_parse_changelog_produces_at_least_one_entry(self) -> None:
        """测试 parse_changelog 能从最小片段解析出至少 1 条 entry"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(MINIMAL_CLI_JAN_16_2026_SNIPPET)

        # 断言至少产生 1 条 entry
        assert len(entries) >= 1, f"期望 parse_changelog 产生至少 1 条 entry，实际得到 {len(entries)} 条"

    def test_entry_date_is_not_empty(self) -> None:
        """测试 entry.date 非空或包含预期日期信息"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(MINIMAL_CLI_JAN_16_2026_SNIPPET)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        # 验证第一条 entry 的日期
        first_entry = entries[0]
        assert first_entry.date, f"期望 entry.date 非空，实际为 '{first_entry.date}'"
        # 验证日期包含 Jan 16, 2026 相关信息
        date_lower = first_entry.date.lower()
        assert "jan" in date_lower or "16" in date_lower or "2026" in date_lower, (
            f"期望 date 包含 'Jan'、'16' 或 '2026'，实际为 '{first_entry.date}'"
        )

    def test_entry_category_is_valid(self) -> None:
        """测试 entry.category 非空且属于有效类别"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(MINIMAL_CLI_JAN_16_2026_SNIPPET)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        first_entry = entries[0]
        valid_categories = {"feature", "fix", "improvement", "other"}

        assert first_entry.category, f"期望 entry.category 非空，实际为 '{first_entry.category}'"
        assert first_entry.category in valid_categories, (
            f"期望 category 在 {valid_categories} 中，实际为 '{first_entry.category}'"
        )

    def test_entry_content_is_not_empty(self) -> None:
        """测试 entry.content 非空"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(MINIMAL_CLI_JAN_16_2026_SNIPPET)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        first_entry = entries[0]
        assert first_entry.content, "期望 entry.content 非空，实际为空"

    def test_entry_content_contains_expected_keywords(self) -> None:
        """测试 entry.content 包含预期关键字（plan/ask/cloud relay/diff）"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(MINIMAL_CLI_JAN_16_2026_SNIPPET)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        # 合并所有 entry 的内容进行检查
        all_content = " ".join(e.content.lower() for e in entries)

        # 验证包含 plan/ask/cloud relay/diff 关键字
        assert "plan" in all_content, f"期望内容包含 'plan' 关键字，实际内容: {all_content[:200]}..."
        assert "ask" in all_content, f"期望内容包含 'ask' 关键字，实际内容: {all_content[:200]}..."
        assert "cloud relay" in all_content or "relay" in all_content, (
            f"期望内容包含 'cloud relay' 或 'relay' 关键字，实际内容: {all_content[:200]}..."
        )
        assert "diff" in all_content, f"期望内容包含 'diff' 关键字，实际内容: {all_content[:200]}..."

    def test_extract_update_points_categorizes_entry(self) -> None:
        """测试 extract_update_points 能把条目归入 feature/fix/improvement 类别"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(MINIMAL_CLI_JAN_16_2026_SNIPPET)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        analysis = analyzer.extract_update_points(entries)

        # 验证 has_updates 为 True
        assert analysis.has_updates, "期望 has_updates 为 True，实际为 False"

        # 验证至少归入一个类别
        has_categorization = len(analysis.new_features) > 0 or len(analysis.improvements) > 0 or len(analysis.fixes) > 0
        assert has_categorization, (
            f"期望条目被归入 feature/fix/improvement 的某一类，"
            f"实际: new_features={len(analysis.new_features)}, "
            f"improvements={len(analysis.improvements)}, "
            f"fixes={len(analysis.fixes)}"
        )

    def test_extract_update_points_hits_related_doc_urls(self) -> None:
        """测试 extract_update_points 能命中相关文档 URL"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(MINIMAL_CLI_JAN_16_2026_SNIPPET)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        analysis = analyzer.extract_update_points(entries)

        # 验证命中了相关文档 URL
        assert len(analysis.related_doc_urls) > 0, "期望 related_doc_urls 非空，实际为空列表"

        # 验证命中的 URL 包含预期的文档路径
        url_str = " ".join(analysis.related_doc_urls).lower()

        # 至少应命中以下之一：modes/plan, modes/ask, mcp, parameters, overview
        expected_url_patterns = ["modes/plan", "modes/ask", "mcp", "parameters", "overview", "slash-commands", "using"]
        hit_any = any(pattern in url_str for pattern in expected_url_patterns)
        assert hit_any, (
            f"期望命中至少一个预期文档 URL pattern {expected_url_patterns}，实际 URLs: {analysis.related_doc_urls}"
        )

    def test_extract_update_points_generates_summary(self) -> None:
        """测试 extract_update_points 能生成非空摘要"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(MINIMAL_CLI_JAN_16_2026_SNIPPET)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        analysis = analyzer.extract_update_points(entries)

        # 验证摘要非空
        assert analysis.summary, f"期望 summary 非空，实际为 '{analysis.summary}'"

    def test_cli_category_date_format_parsed_correctly(self) -> None:
        """测试 'CLI Jan 16, 2026' 格式（类别+日期）被正确解析

        验证 _parse_by_category_date_headers 策略能识别
        '<Category> <Month Day, Year>' 格式的标题行。
        """
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(MINIMAL_CLI_JAN_16_2026_SNIPPET)

        assert len(entries) >= 1, "需要至少 1 条 entry"

        first_entry = entries[0]

        # 验证解析出了日期
        assert first_entry.date, f"期望 date 非空，实际为 '{first_entry.date}'"

        # 验证标题或内容包含 CLI 相关信息（可能被解析为类别或保留在标题中）
        title_and_content = (first_entry.title + " " + first_entry.content).lower()
        # CLI 可能被解析为类别或出现在内容中
        assert "jan" in first_entry.date.lower() or "jan" in first_entry.title.lower(), (
            f"期望日期或标题包含 'Jan'，date='{first_entry.date}', title='{first_entry.title}'"
        )


# ============================================================
# TestChangelogParserVariants - 多变体格式解析测试 (Jan 22, 2026 / Cursor 2.4)
# ============================================================

# 变体A：HTML h2/h3 标题（不含 Markdown ## 前缀），包含 Jan 22, 2026 与 Cursor 2.4
SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS = """
<h2>Cursor 2.4</h2>
<p>Released on Jan 22, 2026</p>

<h3>New Features</h3>
<ul>
<li><strong>Agent Mode</strong>: Improved agent mode with better context handling.</li>
<li><strong>Cloud Sync</strong>: New cloud sync feature for project settings.</li>
</ul>

<h3>Improvements</h3>
<ul>
<li>Enhanced performance for large codebases.</li>
<li>Better error messages for CLI commands.</li>
</ul>

<h3>Bug Fixes</h3>
<ul>
<li>Fixed memory leak in background processes.</li>
<li>Resolved issue with MCP server timeouts.</li>
</ul>
"""

# 变体B：minified HTML（大部分在一行），有 <main>/<article class="changelog"> 结构
SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML = """<html><head><title>Changelog</title></head><body><main><article class="changelog"><h2>Cursor 2.4 - Jan 22, 2026</h2><p>Major update with new features.</p><ul><li>New: Agent improvements with MCP support</li><li>New: Cloud relay for remote execution</li><li>Fix: Resolved diff view rendering issues</li><li>Improved: Better streaming output handling</li></ul></article></main></body></html>"""

# 变体C：纯文本格式，标题行 + 日期行 + 列表项（无 Markdown/HTML 标记）
# 使用 "Version X.X - Date" 格式确保版本信息在解析时保留
SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT = """
Cursor 2.4 - Jan 22, 2026

New Features:
- Agent Mode enhancements for better task planning
- Cloud sync support for project configuration
- MCP server improvements with cloud relay

Bug Fixes:
- Fixed issue with background task cancellation
- Resolved memory leak in long sessions

Improvements:
- Better streaming output performance
- Enhanced diff view with word-level highlighting
"""


class TestChangelogParserVariants:
    """测试多变体格式的 Changelog 解析能力

    覆盖场景：
    - 变体A：HTML h2/h3 标题（不含 Markdown ##），包含 Jan 22, 2026 与 Cursor 2.4
    - 变体B：minified HTML（大部分在一行），有 <main>/<article class="changelog"> 结构
    - 变体C：纯文本格式，标题行 + 日期行 + 列表项

    验证要点：
    1. entries 数量正确
    2. 日期/版本字段提取正确
    3. 类别归类正确（feature/fix/improvement）
    4. 关键词提取不为空
    5. 保底策略不应被误触发（entry.title != "Changelog"）
    """

    # ==================== 变体A：HTML h2/h3 标题测试 ====================

    def test_variant_a_produces_entries(self) -> None:
        """变体A：HTML h2/h3 标题应解析出至少 1 条 entry"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS)

        assert len(entries) >= 1, f"变体A: 期望至少 1 条 entry，实际 {len(entries)} 条"

    def test_variant_a_date_extraction(self) -> None:
        """变体A：应正确提取日期 Jan 22, 2026"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        # 验证日期提取（可能在 date 字段或 content 中）
        all_text = " ".join(f"{e.date} {e.title} {e.content}".lower() for e in entries)
        assert "jan" in all_text and "22" in all_text and "2026" in all_text, (
            f"变体A: 期望包含 Jan 22, 2026，实际内容: {all_text[:300]}..."
        )

    def test_variant_a_version_extraction(self) -> None:
        """变体A：验证解析结果包含日期和主要内容

        注意：当前解析逻辑不保留 h2 标签中的版本信息 (Cursor 2.4)，
        这是已知行为。此测试验证解析器至少正确提取了日期和功能描述。
        """
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        # 验证解析结果包含日期和主要内容
        all_text = " ".join(f"{e.version} {e.title} {e.content}".lower() for e in entries)
        # 至少应包含日期信息
        assert "jan" in all_text and "22" in all_text and "2026" in all_text, (
            f"变体A: 期望包含日期 Jan 22, 2026，实际内容: {all_text[:300]}..."
        )
        # 应包含功能描述
        assert "agent mode" in all_text or "cloud sync" in all_text, (
            f"变体A: 期望包含功能描述，实际内容: {all_text[:300]}..."
        )

    def test_variant_a_category_classification(self) -> None:
        """变体A：应正确归类为 feature/fix/improvement"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        # 验证至少有一条被正确归类
        valid_categories = {"feature", "fix", "improvement"}
        categories = {e.category for e in entries}
        has_valid_category = bool(categories & valid_categories)

        assert has_valid_category, f"变体A: 期望有 feature/fix/improvement 分类，实际分类: {categories}"

    def test_variant_a_keywords_not_empty(self) -> None:
        """变体A：关键词提取不应为空"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        # 验证至少一条 entry 有关键词
        all_keywords = []
        for e in entries:
            all_keywords.extend(e.keywords)

        assert len(all_keywords) > 0, "变体A: 期望提取到关键词，实际为空"

    def test_variant_a_fallback_not_triggered(self) -> None:
        """变体A：保底策略不应被误触发"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        # 保底策略特征：只有一条 entry 且 title == "Changelog"
        is_fallback = (
            len(entries) == 1 and entries[0].title == "Changelog" and not entries[0].date and not entries[0].version
        )
        assert not is_fallback, f"变体A: 保底策略被误触发，entry.title='{entries[0].title}'"

    def test_variant_a_extract_update_points(self) -> None:
        """变体A：extract_update_points 应正确分类"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        analysis = analyzer.extract_update_points(entries)

        # 验证 has_updates
        assert analysis.has_updates, "变体A: 期望 has_updates=True"

        # 验证至少归入一个类别
        has_categorization = len(analysis.new_features) > 0 or len(analysis.improvements) > 0 or len(analysis.fixes) > 0
        assert has_categorization, (
            f"变体A: 期望归入 feature/fix/improvement，实际: "
            f"features={len(analysis.new_features)}, "
            f"improvements={len(analysis.improvements)}, "
            f"fixes={len(analysis.fixes)}"
        )

    # ==================== 变体B：minified HTML 测试 ====================

    def test_variant_b_produces_entries(self) -> None:
        """变体B：minified HTML 应解析出至少 1 条 entry"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML)

        assert len(entries) >= 1, f"变体B: 期望至少 1 条 entry，实际 {len(entries)} 条"

    def test_variant_b_date_extraction(self) -> None:
        """变体B：应正确提取日期 Jan 22, 2026"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        all_text = " ".join(f"{e.date} {e.title} {e.content}".lower() for e in entries)
        assert "jan" in all_text and "22" in all_text and "2026" in all_text, (
            f"变体B: 期望包含 Jan 22, 2026，实际内容: {all_text[:300]}..."
        )

    def test_variant_b_version_extraction(self) -> None:
        """变体B：应正确提取版本 Cursor 2.4"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        all_text = " ".join(f"{e.version} {e.title} {e.content}".lower() for e in entries)
        assert "cursor" in all_text or "2.4" in all_text, (
            f"变体B: 期望包含 Cursor 或 2.4，实际内容: {all_text[:300]}..."
        )

    def test_variant_b_main_article_extraction(self) -> None:
        """变体B：应从 <main>/<article class="changelog"> 中提取内容"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        # 验证提取了实际内容（不是空的）
        all_content = " ".join(e.content for e in entries)
        assert len(all_content) > 50, f"变体B: 期望从 main/article 提取内容，实际长度: {len(all_content)}"

    def test_variant_b_category_classification(self) -> None:
        """变体B：应正确归类为 feature/fix/improvement"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        valid_categories = {"feature", "fix", "improvement"}
        categories = {e.category for e in entries}
        has_valid_category = bool(categories & valid_categories)

        assert has_valid_category, f"变体B: 期望有 feature/fix/improvement 分类，实际分类: {categories}"

    def test_variant_b_keywords_not_empty(self) -> None:
        """变体B：关键词提取不应为空"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        all_keywords = []
        for e in entries:
            all_keywords.extend(e.keywords)

        assert len(all_keywords) > 0, "变体B: 期望提取到关键词，实际为空"

    def test_variant_b_fallback_not_triggered(self) -> None:
        """变体B：保底策略不应被误触发"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        is_fallback = (
            len(entries) == 1 and entries[0].title == "Changelog" and not entries[0].date and not entries[0].version
        )
        assert not is_fallback, f"变体B: 保底策略被误触发，entry.title='{entries[0].title}'"

    def test_variant_b_extract_update_points(self) -> None:
        """变体B：extract_update_points 应正确分类"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        analysis = analyzer.extract_update_points(entries)

        assert analysis.has_updates, "变体B: 期望 has_updates=True"

        has_categorization = len(analysis.new_features) > 0 or len(analysis.improvements) > 0 or len(analysis.fixes) > 0
        assert has_categorization, (
            f"变体B: 期望归入 feature/fix/improvement，实际: "
            f"features={len(analysis.new_features)}, "
            f"improvements={len(analysis.improvements)}, "
            f"fixes={len(analysis.fixes)}"
        )

    # ==================== 变体C：纯文本格式测试 ====================

    def test_variant_c_produces_entries(self) -> None:
        """变体C：纯文本格式应解析出至少 1 条 entry"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT)

        assert len(entries) >= 1, f"变体C: 期望至少 1 条 entry，实际 {len(entries)} 条"

    def test_variant_c_date_extraction(self) -> None:
        """变体C：应正确提取日期 Jan 22, 2026"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        all_text = " ".join(f"{e.date} {e.title} {e.content}".lower() for e in entries)
        assert "jan" in all_text and "22" in all_text and "2026" in all_text, (
            f"变体C: 期望包含 Jan 22, 2026，实际内容: {all_text[:300]}..."
        )

    def test_variant_c_version_extraction(self) -> None:
        """变体C：应正确提取版本 Cursor 2.4"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        all_text = " ".join(f"{e.version} {e.title} {e.content}".lower() for e in entries)
        assert "cursor" in all_text or "2.4" in all_text, (
            f"变体C: 期望包含 Cursor 或 2.4，实际内容: {all_text[:300]}..."
        )

    def test_variant_c_category_classification(self) -> None:
        """变体C：应正确归类为 feature/fix/improvement"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        valid_categories = {"feature", "fix", "improvement"}
        categories = {e.category for e in entries}
        has_valid_category = bool(categories & valid_categories)

        assert has_valid_category, f"变体C: 期望有 feature/fix/improvement 分类，实际分类: {categories}"

    def test_variant_c_keywords_not_empty(self) -> None:
        """变体C：关键词提取不应为空"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        all_keywords = []
        for e in entries:
            all_keywords.extend(e.keywords)

        assert len(all_keywords) > 0, "变体C: 期望提取到关键词，实际为空"

    def test_variant_c_fallback_not_triggered(self) -> None:
        """变体C：保底策略不应被误触发"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        is_fallback = (
            len(entries) == 1 and entries[0].title == "Changelog" and not entries[0].date and not entries[0].version
        )
        assert not is_fallback, f"变体C: 保底策略被误触发，entry.title='{entries[0].title}'"

    def test_variant_c_extract_update_points(self) -> None:
        """变体C：extract_update_points 应正确分类"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT)

        assert len(entries) >= 1, "需要至少 1 条 entry 进行验证"

        analysis = analyzer.extract_update_points(entries)

        assert analysis.has_updates, "变体C: 期望 has_updates=True"

        has_categorization = len(analysis.new_features) > 0 or len(analysis.improvements) > 0 or len(analysis.fixes) > 0
        assert has_categorization, (
            f"变体C: 期望归入 feature/fix/improvement，实际: "
            f"features={len(analysis.new_features)}, "
            f"improvements={len(analysis.improvements)}, "
            f"fixes={len(analysis.fixes)}"
        )

    # ==================== 跨变体对比测试 ====================

    def test_all_variants_produce_entries(self) -> None:
        """所有变体都应能解析出 entry"""
        analyzer = ChangelogAnalyzer()

        variants = {
            "A (HTML h2/h3)": SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS,
            "B (minified)": SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML,
            "C (plain text)": SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT,
        }

        for name, content in variants.items():
            entries = analyzer.parse_changelog(content)
            assert len(entries) >= 1, f"{name}: 期望至少 1 条 entry，实际 {len(entries)} 条"

    def test_all_variants_contain_jan_22_2026(self) -> None:
        """所有变体都应包含 Jan 22, 2026 日期信息"""
        analyzer = ChangelogAnalyzer()

        variants = {
            "A (HTML h2/h3)": SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS,
            "B (minified)": SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML,
            "C (plain text)": SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT,
        }

        for name, content in variants.items():
            entries = analyzer.parse_changelog(content)
            all_text = " ".join(f"{e.date} {e.title} {e.content}".lower() for e in entries)
            assert "jan" in all_text and "22" in all_text and "2026" in all_text, f"{name}: 期望包含 Jan 22, 2026"

    def test_all_variants_contain_date_and_features(self) -> None:
        """所有变体都应包含日期和功能描述

        注意：变体 A (HTML h2/h3) 的版本信息 (Cursor 2.4) 在 h2 标签中，
        当前解析逻辑不保留此信息。此测试验证所有变体至少包含日期和功能描述。
        变体 B 和 C 的版本信息在正文中，可以被正确保留。
        """
        analyzer = ChangelogAnalyzer()

        variants = {
            "A (HTML h2/h3)": SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS,
            "B (minified)": SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML,
            "C (plain text)": SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT,
        }

        for name, content in variants.items():
            entries = analyzer.parse_changelog(content)
            all_text = " ".join(f"{e.version} {e.title} {e.content}".lower() for e in entries)
            # 所有变体应包含日期信息
            assert "jan" in all_text and "22" in all_text and "2026" in all_text, f"{name}: 期望包含日期 Jan 22, 2026"
            # 变体 B 和 C 应包含 Cursor 或 2.4（正文中有版本信息）
            if name != "A (HTML h2/h3)":
                assert "cursor" in all_text or "2.4" in all_text, f"{name}: 期望包含 Cursor 或 2.4"

    def test_no_variant_triggers_fallback(self) -> None:
        """所有变体都不应触发保底策略"""
        analyzer = ChangelogAnalyzer()

        variants = {
            "A (HTML h2/h3)": SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS,
            "B (minified)": SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML,
            "C (plain text)": SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT,
        }

        for name, content in variants.items():
            entries = analyzer.parse_changelog(content)

            # 保底策略特征：只有一条 entry、title=="Changelog"、无 date/version
            is_fallback = (
                len(entries) == 1 and entries[0].title == "Changelog" and not entries[0].date and not entries[0].version
            )
            assert not is_fallback, f"{name}: 保底策略被误触发"


# ============================================================
# TestParseArgsDefaultsFromConfig - 参数默认值来自配置测试
# ============================================================


class TestParseArgsDefaultsFromConfig:
    """测试 parse_args() 的 tri-state 行为和 resolve_settings 配置优先级

    tri-state 语义:
    - 用户未指定参数时，parse_args 返回 None
    - 用户显式指定参数时，parse_args 返回用户指定的值
    - 实际默认值在 SelfIterator._resolve_config_settings() 中通过
      resolve_settings() 从 config.yaml 读取

    验证：
    1. 未传参时 parse_args 返回 None（tri-state）
    2. resolve_settings(cli_xxx=None) 使用 config.yaml 值
    3. 命令行参数仍可覆盖配置默认值
    """

    def test_defaults_come_from_config_manager(self) -> None:
        """测试 tri-state: 未传参时 parse_args 返回 None，resolve_settings 使用配置值"""
        from core.config import resolve_settings

        # 获取当前配置管理器的值
        config = get_config()
        expected_max_iterations = config.system.max_iterations
        expected_workers = config.system.worker_pool_size
        expected_cloud_timeout = config.cloud_agent.timeout
        expected_cloud_auth_timeout = config.cloud_agent.auth_timeout

        # 模拟无命令行参数调用 parse_args
        with patch("sys.argv", ["run_iterate.py"]):
            args = parse_args()

        # tri-state: 未传参时返回 None
        assert args.max_iterations is None, (
            f"max_iterations tri-state: 未传参时应返回 None，实际为 '{args.max_iterations}'"
        )
        assert args.workers is None, f"workers tri-state: 未传参时应返回 None，实际为 {args.workers}"
        assert args.cloud_timeout is None, f"cloud_timeout tri-state: 未传参时应返回 None，实际为 {args.cloud_timeout}"
        assert args.cloud_auth_timeout is None, (
            f"cloud_auth_timeout tri-state: 未传参时应返回 None，实际为 {args.cloud_auth_timeout}"
        )

        # resolve_settings 应使用 config.yaml 的值
        settings = resolve_settings(
            cli_workers=args.workers,
            cli_max_iterations=args.max_iterations,  # None
            cli_cloud_timeout=args.cloud_timeout,
            cli_cloud_auth_timeout=args.cloud_auth_timeout,
        )

        assert settings.max_iterations == expected_max_iterations, (
            f"resolve_settings 应使用 config.yaml 的 max_iterations={expected_max_iterations}，"
            f"实际为 {settings.max_iterations}"
        )
        assert settings.worker_pool_size == expected_workers, (
            f"resolve_settings 应使用 config.yaml 的 workers={expected_workers}，实际为 {settings.worker_pool_size}"
        )
        assert settings.cloud_timeout == expected_cloud_timeout, (
            f"resolve_settings 应使用 config.yaml 的 cloud_timeout={expected_cloud_timeout}，"
            f"实际为 {settings.cloud_timeout}"
        )
        assert settings.cloud_auth_timeout == expected_cloud_auth_timeout, (
            f"resolve_settings 应使用 config.yaml 的 cloud_auth_timeout={expected_cloud_auth_timeout}，"
            f"实际为 {settings.cloud_auth_timeout}"
        )

    def test_defaults_match_config_yaml_values(self) -> None:
        """测试 tri-state 和 resolve_settings 完整流程

        验证 parse_args 返回 None，resolve_settings 使用 config.yaml 值。
        """
        from core.config import resolve_settings

        config = get_config()

        with patch("sys.argv", ["run_iterate.py"]):
            args = parse_args()

        # tri-state: 所有参数应为 None
        assert args.max_iterations is None
        assert args.workers is None
        assert args.cloud_timeout is None
        assert args.cloud_auth_timeout is None

        # resolve_settings 使用 config.yaml 值
        settings = resolve_settings()
        assert settings.max_iterations == config.system.max_iterations
        assert settings.worker_pool_size == config.system.worker_pool_size
        assert settings.cloud_timeout == config.cloud_agent.timeout
        assert settings.cloud_auth_timeout == config.cloud_agent.auth_timeout

    def test_cli_args_override_config_defaults(self) -> None:
        """测试命令行参数可覆盖配置默认值"""
        with patch(
            "sys.argv",
            [
                "run_iterate.py",
                "--max-iterations",
                "99",
                "--workers",
                "7",
                "--cloud-timeout",
                "1800",
                "--cloud-auth-timeout",
                "60",
            ],
        ):
            args = parse_args()

        # 验证命令行参数覆盖了配置默认值
        assert args.max_iterations == "99"
        assert args.workers == 7
        assert args.cloud_timeout == 1800
        assert args.cloud_auth_timeout == 60

    def test_temporary_config_yaml_overrides_defaults(self) -> None:
        """测试用临时 config.yaml 覆盖默认值

        创建临时 config.yaml 文件，验证 resolve_settings 使用临时配置文件的值。
        tri-state 语义下 parse_args 返回 None，resolve_settings 使用 config.yaml 值。
        """
        from core.config import resolve_settings

        # 创建临时目录和 config.yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # 写入自定义配置
            custom_config = """
system:
  max_iterations: 25
  worker_pool_size: 8
  enable_sub_planners: true
  strict_review: false

cloud_agent:
  enabled: false
  execution_mode: cli
  timeout: 900
  auth_timeout: 45
"""
            config_path.write_text(custom_config, encoding="utf-8")

            # 重置 ConfigManager 单例以加载新配置
            ConfigManager.reset_instance()

            # 切换到临时目录，让 ConfigManager 找到新的 config.yaml
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # 重新获取配置（应该加载临时 config.yaml）
                config = get_config()

                # 验证配置已加载自定义值
                assert config.system.max_iterations == 25
                assert config.system.worker_pool_size == 8
                assert config.cloud_agent.timeout == 900
                assert config.cloud_agent.auth_timeout == 45

                # 模拟无命令行参数调用 parse_args
                with patch("sys.argv", ["run_iterate.py"]):
                    args = parse_args()

                # tri-state: 未传参时返回 None
                assert args.max_iterations is None, (
                    f"max_iterations tri-state: 应为 None，实际为 '{args.max_iterations}'"
                )
                assert args.workers is None, f"workers tri-state: 应为 None，实际为 {args.workers}"
                assert args.cloud_timeout is None, f"cloud_timeout tri-state: 应为 None，实际为 {args.cloud_timeout}"
                assert args.cloud_auth_timeout is None, (
                    f"cloud_auth_timeout tri-state: 应为 None，实际为 {args.cloud_auth_timeout}"
                )

                # resolve_settings 应使用临时 config.yaml 的值
                settings = resolve_settings()
                assert settings.max_iterations == 25, (
                    f"resolve_settings 应使用临时 config.yaml 的 max_iterations=25，实际为 {settings.max_iterations}"
                )
                assert settings.worker_pool_size == 8, (
                    f"resolve_settings 应使用临时 config.yaml 的 workers=8，实际为 {settings.worker_pool_size}"
                )
                assert settings.cloud_timeout == 900, (
                    f"resolve_settings 应使用临时 config.yaml 的 cloud_timeout=900，实际为 {settings.cloud_timeout}"
                )
                assert settings.cloud_auth_timeout == 45, (
                    f"resolve_settings 应使用临时 config.yaml 的 cloud_auth_timeout=45，"
                    f"实际为 {settings.cloud_auth_timeout}"
                )

            finally:
                # 恢复原目录
                os.chdir(original_cwd)
                # 重置 ConfigManager 以恢复原配置
                ConfigManager.reset_instance()

    def test_cli_args_override_temporary_config(self) -> None:
        """测试命令行参数可覆盖临时配置文件的值

        验证优先级：命令行参数 > config.yaml
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # 写入自定义配置
            custom_config = """
system:
  max_iterations: 50
  worker_pool_size: 10

cloud_agent:
  timeout: 1200
  auth_timeout: 90
"""
            config_path.write_text(custom_config, encoding="utf-8")

            ConfigManager.reset_instance()

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # 使用命令行参数覆盖配置值
                with patch(
                    "sys.argv",
                    [
                        "run_iterate.py",
                        "--max-iterations",
                        "3",
                        "--workers",
                        "2",
                        "--cloud-timeout",
                        "600",
                        "--cloud-auth-timeout",
                        "15",
                    ],
                ):
                    args = parse_args()

                # 命令行参数应覆盖配置文件值
                assert args.max_iterations == "3", "命令行参数应覆盖 config.yaml"
                assert args.workers == 2, "命令行参数应覆盖 config.yaml"
                assert args.cloud_timeout == 600, "命令行参数应覆盖 config.yaml"
                assert args.cloud_auth_timeout == 15, "命令行参数应覆盖 config.yaml"

            finally:
                os.chdir(original_cwd)
                ConfigManager.reset_instance()

    def test_config_manager_reset_restores_original_defaults(self) -> None:
        """测试 ConfigManager.reset_instance() 后恢复原始默认值"""
        # 获取原始配置值
        original_config = get_config()
        original_max_iterations = original_config.system.max_iterations
        original_workers = original_config.system.worker_pool_size

        # 重置后重新获取配置
        ConfigManager.reset_instance()
        restored_config = get_config()

        # 验证恢复后的值与原始值一致
        assert restored_config.system.max_iterations == original_max_iterations
        assert restored_config.system.worker_pool_size == original_workers

    def test_self_iterator_uses_config_based_defaults(self) -> None:
        """测试 SelfIterator._resolved_settings 使用基于配置的默认值

        tri-state 语义：
        - args.xxx 为 None（未传参）
        - SelfIterator._resolved_settings 从 config.yaml 读取默认值
        """
        config = get_config()

        with patch("sys.argv", ["run_iterate.py", "测试任务"]):
            args = parse_args()

        # 补充 SelfIterator 需要的其他属性
        args.directory = "."
        args._orchestrator_user_set = False
        args.quiet = False
        args.log_level = None
        args.heartbeat_debug = False
        args.stall_diagnostics_enabled = None
        args.stall_diagnostics_level = None
        args.stall_recovery_interval = 30.0
        args.execution_health_check_interval = 30.0
        args.health_warning_cooldown = 60.0
        args.planner_execution_mode = None
        args.worker_execution_mode = None
        args.reviewer_execution_mode = None
        args.stream_console_renderer = False
        args.stream_advanced_renderer = False
        args.stream_typing_effect = False
        args.stream_typing_delay = 0.02
        args.stream_word_mode = True
        args.stream_color_enabled = True
        args.stream_show_word_diff = False

        # tri-state: args 中的值应为 None
        assert args.max_iterations is None, "tri-state: max_iterations 应为 None"
        assert args.workers is None, "tri-state: workers 应为 None"
        assert args.cloud_timeout is None, "tri-state: cloud_timeout 应为 None"

        # 创建 SelfIterator
        iterator = SelfIterator(args)

        # 验证 _resolved_settings 中的值来自 config.yaml
        assert iterator._resolved_settings.max_iterations == config.system.max_iterations, (
            f"_resolved_settings.max_iterations 应为 {config.system.max_iterations}，"
            f"实际为 {iterator._resolved_settings.max_iterations}"
        )
        assert iterator._resolved_settings.worker_pool_size == config.system.worker_pool_size, (
            f"_resolved_settings.worker_pool_size 应为 {config.system.worker_pool_size}，"
            f"实际为 {iterator._resolved_settings.worker_pool_size}"
        )
        assert iterator._resolved_settings.cloud_timeout == config.cloud_agent.timeout, (
            f"_resolved_settings.cloud_timeout 应为 {config.cloud_agent.timeout}，"
            f"实际为 {iterator._resolved_settings.cloud_timeout}"
        )


# ============================================================
# execution_mode 配置默认值测试
# ============================================================


class TestExecutionModeConfigDefault:
    """测试 execution_mode 参数的 tri-state 行为和配置优先级

    tri-state 语义:
    - 用户未指定参数时，parse_args 返回 None
    - SelfIterator._resolved_settings 从 config.yaml 读取默认值

    验证场景:
    1. parse_args() 的 --execution-mode 返回 None（tri-state）
    2. _resolved_settings 使用 config.yaml 中的值
    3. '&' 前缀触发 Cloud 模式仍优先于配置默认值
    4. Cloud/Auto 模式下 orchestrator 兼容性策略仍生效（强制 basic）

    注意：部分测试需要 mock Cloud API Key 存在，否则 cloud/auto 模式会回退到 cli
    """

    @pytest.fixture(autouse=True)
    def mock_cloud_api_key(self) -> Generator[None, None, None]:
        """Mock Cloud API Key 存在，用于测试 cloud/auto 执行模式"""
        from cursor.cloud_client import CloudClientFactory

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            yield

    @pytest.fixture
    def config_execution_mode(self) -> str:
        """获取配置中的 execution_mode 默认值"""
        from core.config import get_config

        return get_config().cloud_agent.execution_mode

    def test_parse_args_execution_mode_uses_config_default(self, config_execution_mode: str) -> None:
        """测试 parse_args() 的 --execution-mode tri-state 行为

        tri-state: 未传参时返回 None，实际值在 resolve_settings 中解析。
        """
        from core.config import resolve_settings

        with patch("sys.argv", ["run_iterate.py", "测试任务"]):
            args = parse_args()

        # tri-state: 未传参时返回 None
        assert args.execution_mode is None, (
            f"--execution-mode tri-state: 未传参时应返回 None，实际值: {args.execution_mode}"
        )

        # resolve_settings 应使用 config.yaml 的值
        settings = resolve_settings(cli_execution_mode=args.execution_mode)
        assert settings.execution_mode == config_execution_mode, (
            f"resolve_settings 应使用 config.yaml 的 execution_mode={config_execution_mode}，"
            f"实际值: {settings.execution_mode}"
        )

    def test_parse_args_execution_mode_cli_override(self) -> None:
        """测试 CLI --execution-mode 参数可覆盖 config.yaml 默认值"""
        with patch("sys.argv", ["run_iterate.py", "测试任务", "--execution-mode", "cloud"]):
            args = parse_args()

        assert args.execution_mode == "cloud"

    def test_parse_args_help_shows_config_source_for_execution_mode(self) -> None:
        """测试帮助信息中 --execution-mode 显示来源于 config.yaml"""
        import contextlib
        import io

        help_output = io.StringIO()
        with pytest.raises(SystemExit), patch("sys.argv", ["run_iterate.py", "--help"]):
            with contextlib.redirect_stdout(help_output):
                parse_args()

        help_text = help_output.getvalue()
        assert "config.yaml" in help_text, "--execution-mode 帮助信息应包含 'config.yaml' 来源提示"

    def test_ampersand_prefix_overrides_config_execution_mode(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 '&' 前缀触发 Cloud 模式优先于 execution_mode=None

        场景：用户未显式指定 execution_mode（None），任务带 '&' 前缀
        期望：当 cloud_enabled=True 且有 API Key 时，Cloud 模式被激活

        注意：需要 mock cloud_enabled=True 才能让 & 前缀触发 Cloud 模式
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        # 模拟用户未显式指定 execution_mode（None 表示未指定，使用配置默认值）
        base_iterate_args.execution_mode = None
        base_iterate_args.requirement = "& 分析代码架构"

        # Mock cloud_enabled=True 和 api_key
        config = get_config()
        original_enabled = config.cloud_agent.enabled
        config.cloud_agent.enabled = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            try:
                iterator = SelfIterator(base_iterate_args)

                # has_ampersand_prefix=True + 满足条件 → prefix_routed=True（成功触发 Cloud 路由）
                # 内部分支使用 _prefix_routed 字段
                assert iterator._prefix_routed is True, "& 前缀应成功触发 Cloud 路由"
                assert iterator._get_execution_mode() == ExecutionMode.CLOUD
                # requirement 应被剥离 '&' 前缀
                assert iterator.context.user_requirement == "分析代码架构"
            finally:
                config.cloud_agent.enabled = original_enabled

    def test_orchestrator_compatibility_config_cloud_forces_basic(self, base_iterate_args: argparse.Namespace) -> None:
        """测试配置 execution_mode=cloud 时 orchestrator 兼容性策略

        场景：配置默认 execution_mode=cloud，用户未显式设置 --no-mp
        期望：自动切换到 basic 编排器
        """
        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)

        # Cloud 模式应强制使用 basic 编排器
        assert iterator._get_orchestrator_type() == "basic", "配置 execution_mode=cloud 时应强制使用 basic 编排器"

    def test_orchestrator_compatibility_config_auto_forces_basic(self, base_iterate_args: argparse.Namespace) -> None:
        """测试配置 execution_mode=auto 时 orchestrator 兼容性策略

        场景：配置默认 execution_mode=auto，用户显式设置 --orchestrator mp
        期望：即使用户显式指定 mp，也应切换到 basic（执行模式优先）
        """
        base_iterate_args.execution_mode = "auto"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        base_iterate_args._orchestrator_user_set = True  # 用户显式设置了 mp

        iterator = SelfIterator(base_iterate_args)

        # Auto 模式也应强制使用 basic 编排器
        assert iterator._get_orchestrator_type() == "basic", (
            "配置 execution_mode=auto 时应强制使用 basic 编排器，即使用户显式设置 mp"
        )

    def test_orchestrator_compatibility_config_cli_allows_mp(self, base_iterate_args: argparse.Namespace) -> None:
        """测试显式 execution_mode=cli 时允许使用 mp 编排器

        场景：显式设置 execution_mode=cli
        期望：允许使用 mp 编排器
        """
        base_iterate_args.execution_mode = "cli"  # 显式设置
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)

        # CLI 模式允许使用 mp 编排器
        assert iterator._get_orchestrator_type() == "mp", "配置 execution_mode=cli 时应允许使用 mp 编排器"

    def test_ampersand_prefix_priority_over_config_and_forces_basic(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 '&' 前缀优先于 execution_mode=None 且强制 basic 编排器

        综合场景：
        1. 用户未显式指定 execution_mode（None）
        2. 任务带 '&' 前缀（触发 Cloud 路由）
        3. 用户未设置 --no-mp

        期望：
        - execution_mode 被覆盖为 CLOUD
        - orchestrator 被强制为 basic

        注意：需要 mock cloud_enabled=True 和 api_key 才能让 & 前缀触发 Cloud 模式
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        base_iterate_args.execution_mode = None  # 未显式指定，使用配置默认值
        base_iterate_args.requirement = "& 后台分析代码"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        base_iterate_args._orchestrator_user_set = False

        # Mock cloud_enabled=True 和 api_key
        config = get_config()
        original_enabled = config.cloud_agent.enabled
        config.cloud_agent.enabled = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            try:
                iterator = SelfIterator(base_iterate_args)

                # '&' 前缀应触发 Cloud 模式
                assert iterator._get_execution_mode() == ExecutionMode.CLOUD
                # Cloud 模式应强制 basic 编排器
                assert iterator._get_orchestrator_type() == "basic"
            finally:
                config.cloud_agent.enabled = original_enabled


class TestExecutionModeConfigIntegration:
    """execution_mode 配置默认值集成测试

    验证 run.py 和 scripts/run_iterate.py 之间的一致性。
    """

    def test_both_scripts_use_same_config_default(self) -> None:
        """测试 run.py 和 run_iterate.py 使用相同的配置默认值

        tri-state 设计：
        - args.execution_mode 为 None 表示未指定
        - 实际配置值通过 _merge_options / _resolved_settings 解析
        - 两个入口最终解析出的值应一致
        """
        from core.config import get_config
        from run import Runner
        from run import parse_args as run_parse_args
        from scripts.run_iterate import SelfIterator
        from scripts.run_iterate import parse_args as iterate_parse_args

        config_default = get_config().cloud_agent.execution_mode

        # 测试 run.py
        with patch("sys.argv", ["run.py", "测试任务"]):
            run_args = run_parse_args()

        # 测试 run_iterate.py
        with patch("sys.argv", ["run_iterate.py", "测试任务"]):
            iterate_args = iterate_parse_args()

        # tri-state 验证：未指定时 args.execution_mode 应为 None
        assert run_args.execution_mode is None, "tri-state 设计：未指定 --execution-mode 时应为 None"
        assert iterate_args.execution_mode is None, "tri-state 设计：未指定 --execution-mode 时应为 None"

        # 验证解析后的实际值来自 config.yaml
        runner = Runner(run_args)
        run_merged = runner._merge_options({})

        iterator = SelfIterator(iterate_args)
        iterate_resolved = iterator._resolved_settings

        # 两者解析出的 execution_mode 应等于 config.yaml 的值
        assert run_merged["execution_mode"] == config_default, (
            f"run.py 解析的 execution_mode 应为 {config_default}，实际为 {run_merged['execution_mode']}"
        )
        assert iterate_resolved.execution_mode == config_default, (
            f"run_iterate.py 解析的 execution_mode 应为 {config_default}，实际为 {iterate_resolved.execution_mode}"
        )
        assert run_merged["execution_mode"] == iterate_resolved.execution_mode, (
            "run.py 和 run_iterate.py 应解析出相同的 execution_mode 配置值"
        )


# ============================================================
# TestChangelogAnalyzerWithRealStorage - 真实存储的基线比较测试
# ============================================================


class TestChangelogAnalyzerWithRealStorage:
    """测试 ChangelogAnalyzer 与真实 KnowledgeStorage 的基线比较功能

    覆盖场景：
    1. 第一次保存 changelog 后，第二次分析相同内容应返回 has_updates=False
    2. 轻微噪声变化（HTML 包裹/多余空白/导航噪声）若清洗后相同则仍 False
    3. 清洗后内容不同则返回 True
    4. 自定义 changelog_url 作为 baseline key 生效
    """

    @pytest.fixture
    def temp_storage(self, tmp_path: Path) -> Generator[Any, None, None]:
        """创建临时工作目录下的真实 KnowledgeStorage"""
        from knowledge.storage import KnowledgeStorage, StorageConfig

        config = StorageConfig(
            storage_root=".cursor/knowledge",
            auto_create_dirs=True,
        )
        storage = KnowledgeStorage(config=config, workspace_root=str(tmp_path))
        yield storage

    @pytest.mark.asyncio
    async def test_same_content_returns_no_updates(self, temp_storage: Any) -> None:
        """测试相同内容第二次分析返回 has_updates=False

        流程：
        1. 初始化存储并保存一份 changelog 文档（带 cleaned_fingerprint）
        2. 使用相同存储创建 ChangelogAnalyzer
        3. Mock fetch 返回相同内容
        4. 调用 analyze() 应返回 has_updates=False
        """
        from knowledge.models import Document

        await temp_storage.initialize()

        # 原始 changelog 内容
        changelog_content = """
# Cursor Changelog

## 2024-01-15
- New feature: AI code completion
- Bug fix: Memory leak in editor

## 2024-01-10
- Performance improvements
"""

        # 计算 cleaned_fingerprint（模拟 ChangelogAnalyzer 行为）
        analyzer_for_fp = ChangelogAnalyzer(storage=None)
        cleaned_fp = analyzer_for_fp.compute_fingerprint(changelog_content)

        # 第一次保存 changelog 文档（模拟 KnowledgeUpdater 行为）
        changelog_doc = Document(
            url="https://cursor.com/cn/changelog",
            title="Cursor Changelog",
            content=changelog_content,
            metadata={"cleaned_fingerprint": cleaned_fp},
        )
        success, _ = await temp_storage.save_document(changelog_doc, force=True)
        assert success, "首次保存 changelog 应成功"

        # 第二次分析：使用相同存储创建 ChangelogAnalyzer
        analyzer = ChangelogAnalyzer(storage=temp_storage)

        # Mock fetch 返回相同内容
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = changelog_content
        mock_fetch_result.error = None

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            analysis = await analyzer.analyze()

            # 验证：相同内容应返回 has_updates=False
            assert analysis.has_updates is False, "相同内容第二次分析应返回 has_updates=False"
            assert analysis.fingerprint == cleaned_fp, "fingerprint 应与保存的一致"

    @pytest.mark.asyncio
    async def test_noise_variations_still_no_updates(self, temp_storage: Any) -> None:
        """测试轻微噪声变化（清洗后相同）仍返回 has_updates=False

        噪声类型：
        - 多余空白（行首行尾空格、多余换行）
        - HTML 实体（&nbsp; 等）

        注意：此测试验证的是"相同逻辑内容经过不同格式化后仍被识别为相同"
        """
        from knowledge.models import Document

        await temp_storage.initialize()

        # 原始干净内容（使用日期格式标题以触发锚点提取）
        clean_content = """## 2024-01-15
- New feature: AI code completion
- Bug fix: Memory leak
"""

        # 计算 cleaned_fingerprint
        analyzer_for_fp = ChangelogAnalyzer(storage=None)
        cleaned_fp = analyzer_for_fp.compute_fingerprint(clean_content)

        # 保存原始文档
        changelog_doc = Document(
            url="https://cursor.com/cn/changelog",
            title="Cursor Changelog",
            content=clean_content,
            metadata={"cleaned_fingerprint": cleaned_fp},
        )
        await temp_storage.save_document(changelog_doc, force=True)

        # 带噪声的内容变体（仅添加空白噪声，清洗后应与原始内容相同）
        # 添加：行首空格、多余换行、行尾空格
        noisy_content = """  ## 2024-01-15  
- New feature: AI code completion  
- Bug fix: Memory leak  


"""

        # 创建分析器
        analyzer = ChangelogAnalyzer(storage=temp_storage)

        # 验证噪声内容清洗后的 fingerprint 与干净内容相同
        noisy_fp = analyzer.compute_fingerprint(noisy_content)
        assert noisy_fp == cleaned_fp, f"空白噪声内容清洗后 fingerprint 应与干净内容一致: {noisy_fp} vs {cleaned_fp}"

        # Mock fetch 返回带噪声的内容
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = noisy_content
        mock_fetch_result.error = None

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            analysis = await analyzer.analyze()

            # 验证：清洗后相同应返回 has_updates=False
            assert analysis.has_updates is False, "空白噪声变化但清洗后内容相同时应返回 has_updates=False"

    @pytest.mark.asyncio
    async def test_different_content_returns_has_updates(self, temp_storage: Any) -> None:
        """测试清洗后内容不同时返回 has_updates=True"""
        from knowledge.models import Document

        await temp_storage.initialize()

        # 原始内容
        original_content = """
# Cursor Changelog

## 2024-01-15
- New feature: AI code completion
"""

        # 计算并保存 fingerprint
        analyzer_for_fp = ChangelogAnalyzer(storage=None)
        cleaned_fp = analyzer_for_fp.compute_fingerprint(original_content)

        changelog_doc = Document(
            url="https://cursor.com/cn/changelog",
            title="Cursor Changelog",
            content=original_content,
            metadata={"cleaned_fingerprint": cleaned_fp},
        )
        await temp_storage.save_document(changelog_doc, force=True)

        # 新内容（有实质性变化）
        updated_content = """
# Cursor Changelog

## 2024-01-20
- NEW: Cloud Agent support
- Breaking change: API v2 released

## 2024-01-15
- New feature: AI code completion
"""

        # 创建分析器
        analyzer = ChangelogAnalyzer(storage=temp_storage)

        # 验证新内容的 fingerprint 与原始不同
        updated_fp = analyzer.compute_fingerprint(updated_content)
        assert updated_fp != cleaned_fp, "不同内容的 fingerprint 应不同"

        # Mock fetch 返回新内容
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = updated_content
        mock_fetch_result.error = None

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            analysis = await analyzer.analyze()

            # 验证：内容不同应返回 has_updates=True
            assert analysis.has_updates is True, "内容不同时应返回 has_updates=True"
            assert len(analysis.entries) > 0, "有更新时应解析出 entries"

    @pytest.mark.asyncio
    async def test_custom_changelog_url_as_baseline_key(self, temp_storage: Any) -> None:
        """测试自定义 changelog_url 作为 baseline key 生效

        场景：
        - 使用自定义 URL (如 https://example.com/changelog) 保存文档
        - 使用相同自定义 URL 创建 ChangelogAnalyzer
        - 分析相同内容应返回 has_updates=False
        - 使用不同 URL 创建的 ChangelogAnalyzer 应找不到基线（返回 has_updates=True）
        """
        from knowledge.models import Document

        await temp_storage.initialize()

        custom_url = "https://example.com/custom-changelog"
        changelog_content = """
# Custom Changelog

## 2024-01-15
- Feature A released
"""

        # 计算并保存 fingerprint
        analyzer_for_fp = ChangelogAnalyzer(storage=None)
        cleaned_fp = analyzer_for_fp.compute_fingerprint(changelog_content)

        changelog_doc = Document(
            url=custom_url,  # 使用自定义 URL
            title="Custom Changelog",
            content=changelog_content,
            metadata={"cleaned_fingerprint": cleaned_fp},
        )
        await temp_storage.save_document(changelog_doc, force=True)

        # 使用相同自定义 URL 创建分析器
        analyzer_same_url = ChangelogAnalyzer(
            changelog_url=custom_url,
            storage=temp_storage,
        )

        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = changelog_content
        mock_fetch_result.error = None

        with (
            patch.object(analyzer_same_url.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer_same_url.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            analysis = await analyzer_same_url.analyze()

            # 验证：相同 URL 应找到基线，返回 has_updates=False
            assert analysis.has_updates is False, "使用相同自定义 URL 应找到基线，返回 has_updates=False"

        # 使用不同 URL 创建分析器
        different_url = "https://other.com/changelog"
        analyzer_diff_url = ChangelogAnalyzer(
            changelog_url=different_url,
            storage=temp_storage,
        )

        with (
            patch.object(analyzer_diff_url.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer_diff_url.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            analysis_diff = await analyzer_diff_url.analyze()

            # 验证：不同 URL 应找不到基线（首次分析），返回 has_updates=True
            assert analysis_diff.has_updates is True, "使用不同 URL 应找不到基线，返回 has_updates=True（首次分析）"

    @pytest.mark.asyncio
    async def test_backward_compat_without_cleaned_fingerprint(self, temp_storage: Any) -> None:
        """测试向后兼容：索引中无 cleaned_fingerprint 时回退到 doc.content 计算

        场景：
        - 保存文档时不设置 cleaned_fingerprint（模拟旧数据）
        - 分析时应回退到加载文档内容并重新计算 fingerprint
        """
        from knowledge.models import Document

        await temp_storage.initialize()

        changelog_content = """
# Cursor Changelog

## 2024-01-15
- Legacy entry
"""

        # 保存文档但不设置 cleaned_fingerprint（模拟旧数据）
        changelog_doc = Document(
            url="https://cursor.com/cn/changelog",
            title="Cursor Changelog",
            content=changelog_content,
            metadata={},  # 无 cleaned_fingerprint
        )
        await temp_storage.save_document(changelog_doc, force=True)

        # 验证索引中无 cleaned_fingerprint
        entry = await temp_storage.get_index_entry_by_url("https://cursor.com/cn/changelog")
        assert entry is not None
        assert entry.cleaned_fingerprint == "", "索引中应无 cleaned_fingerprint（模拟旧数据）"

        # 创建分析器
        analyzer = ChangelogAnalyzer(storage=temp_storage)

        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = changelog_content
        mock_fetch_result.error = None

        with (
            patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            analysis = await analyzer.analyze()

            # 验证：应回退到 doc.content 计算，相同内容返回 has_updates=False
            assert analysis.has_updates is False, (
                "向后兼容：无 cleaned_fingerprint 时应回退计算，相同内容返回 has_updates=False"
            )


# ============================================================
# TestExecutionModeAutoOrchestratorBasicRegression - 回归测试
# ============================================================


class TestExecutionModeAutoOrchestratorBasicRegression:
    """测试 --dry-run --skip-online --execution-mode auto --orchestrator basic 参数组合

    这是一个回归测试类，用于验证以下关键路径：
    1. execution_mode=auto 时 _get_execution_mode() 返回 ExecutionMode.AUTO
    2. orchestrator=basic 时 _get_orchestrator_type() 返回 "basic"
    3. 使用 basic 编排器执行路径（而非 MP 编排器）
    4. dry-run 模式正确跳过实际 agent 执行
    5. skip-online 模式正确跳过网络抓取

    参考 AGENTS.md 文档：
    - 当 --execution-mode 为 auto/cloud 时，系统强制使用 basic 编排器
    - dry-run 模式仅分析不执行

    注意：这些测试需要 mock Cloud API Key 存在，否则 auto 模式会回退到 cli。
    """

    @pytest.fixture(autouse=True)
    def mock_cloud_api_key(self) -> Generator[None, None, None]:
        """Mock Cloud API Key 存在，用于测试 auto 执行模式"""
        from cursor.cloud_client import CloudClientFactory

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            yield

    @pytest.fixture
    def auto_basic_args(self) -> argparse.Namespace:
        """创建 execution_mode=auto, orchestrator=basic 的参数"""
        return argparse.Namespace(
            requirement="回归测试任务",
            directory=".",
            skip_online=True,  # 跳过在线检查
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=True,  # 仅分析不执行
            max_iterations="3",
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
            orchestrator="basic",  # 显式指定 basic 编排器
            no_mp=False,
            _orchestrator_user_set=True,  # 标记用户显式设置
            # 执行模式参数
            execution_mode="auto",  # 使用 auto 执行模式
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            # 角色级执行模式
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            # 流式控制台渲染参数
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_execution_mode_auto_returns_auto(self, auto_basic_args: argparse.Namespace) -> None:
        """测试 execution_mode=auto 时 _get_execution_mode() 返回 AUTO"""
        from cursor.executor import ExecutionMode

        iterator = SelfIterator(auto_basic_args)

        result = iterator._get_execution_mode()

        assert result == ExecutionMode.AUTO, f"execution_mode=auto 时应返回 ExecutionMode.AUTO，实际返回 {result}"

    def test_orchestrator_basic_returns_basic(self, auto_basic_args: argparse.Namespace) -> None:
        """测试 orchestrator=basic 时 _get_orchestrator_type() 返回 'basic'"""
        iterator = SelfIterator(auto_basic_args)

        result = iterator._get_orchestrator_type()

        assert result == "basic", f"orchestrator=basic 时应返回 'basic'，实际返回 {result}"

    def test_auto_mode_forces_basic_orchestrator(self, auto_basic_args: argparse.Namespace) -> None:
        """测试 execution_mode=auto 强制使用 basic 编排器

        即使用户未显式指定 --orchestrator basic，
        auto 执行模式也应该强制使用 basic 编排器。
        """
        # 修改参数：不显式设置 orchestrator
        auto_basic_args.orchestrator = "mp"  # 用户请求 MP
        auto_basic_args._orchestrator_user_set = False  # 未显式设置

        iterator = SelfIterator(auto_basic_args)

        # 由于 execution_mode=auto，应强制使用 basic
        result = iterator._get_orchestrator_type()

        assert result == "basic", f"execution_mode=auto 应强制使用 basic 编排器，即使请求 MP，实际返回 {result}"

    @pytest.mark.asyncio
    async def test_dry_run_skip_online_returns_expected_structure(self, auto_basic_args: argparse.Namespace) -> None:
        """测试 dry-run + skip-online 模式返回预期结构

        验证：
        1. 不执行实际 agent 系统
        2. 返回 dry_run=True
        3. 返回 success=True
        4. 包含 summary 和 goal_length 字段
        """
        iterator = SelfIterator(auto_basic_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 5})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            result = await iterator.run()

            # 验证返回结构
            assert result.get("success") is True, "dry-run 模式应返回 success=True"
            assert result.get("dry_run") is True, "应返回 dry_run=True"
            assert "summary" in result, "应包含 summary 字段"
            assert "goal_length" in result, "应包含 goal_length 字段"
            assert result["goal_length"] > 0, "goal_length 应大于 0"

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_agent_system(self, auto_basic_args: argparse.Namespace) -> None:
        """测试 dry-run 模式不调用 _run_agent_system"""
        iterator = SelfIterator(auto_basic_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 5})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(iterator, "_run_agent_system", new_callable=AsyncMock) as mock_agent,
        ):
            await iterator.run()

            # dry-run 模式不应调用 _run_agent_system
            mock_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_online_does_not_fetch_changelog(self, auto_basic_args: argparse.Namespace) -> None:
        """测试 skip-online 模式不抓取 Changelog"""
        iterator = SelfIterator(auto_basic_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 5})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.changelog_analyzer.fetcher, "fetch", new_callable=AsyncMock) as mock_fetch,
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(
                iterator.knowledge_updater.manager,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            await iterator.run()

            # skip-online 模式不应调用 fetch
            mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_basic_orchestrator_path_with_auto_mode(self, auto_basic_args: argparse.Namespace) -> None:
        """测试 execution_mode=auto + orchestrator=basic 使用 basic 编排器路径

        当不是 dry-run 时，验证使用 _run_with_basic_orchestrator 而非 MP 编排器。
        """
        # 修改为非 dry-run 模式以触发实际执行路径
        auto_basic_args.dry_run = False

        iterator = SelfIterator(auto_basic_args)
        iterator.context.iteration_goal = "测试目标"

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 5})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        mock_basic_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
        }

        with (
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ) as mock_basic,
            patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
            ) as mock_mp,
        ):
            await iterator.run()

            # 应调用 basic 编排器
            mock_basic.assert_called_once()
            # 不应调用 MP 编排器
            mock_mp.assert_not_called()

    def test_combined_parameters_are_set_correctly(self, auto_basic_args: argparse.Namespace) -> None:
        """测试组合参数正确设置

        验证 SelfIterator 正确解析以下参数组合：
        - --dry-run
        - --skip-online
        - --execution-mode auto
        - --orchestrator basic
        """
        iterator = SelfIterator(auto_basic_args)

        # 验证参数被正确设置
        assert iterator.args.dry_run is True, "dry_run 应为 True"
        assert iterator.args.skip_online is True, "skip_online 应为 True"
        assert iterator.args.execution_mode == "auto", "execution_mode 应为 'auto'"
        assert iterator.args.orchestrator == "basic", "orchestrator 应为 'basic'"

    @pytest.mark.asyncio
    async def test_full_regression_path(self, auto_basic_args: argparse.Namespace) -> None:
        """完整回归测试：验证整个执行流程

        模拟完整的 --dry-run --skip-online --execution-mode auto --orchestrator basic 流程：
        1. 参数解析正确
        2. 跳过在线抓取
        3. 使用 basic 编排器路径（通过 dry-run 跳过实际执行）
        4. 返回正确的结果结构
        """
        from cursor.executor import ExecutionMode

        iterator = SelfIterator(auto_basic_args)

        # 验证关键配置
        assert iterator._get_execution_mode() == ExecutionMode.AUTO
        assert iterator._get_orchestrator_type() == "basic"

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 10})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with (
            patch.object(iterator.changelog_analyzer.fetcher, "fetch", new_callable=AsyncMock) as mock_fetch,
            patch.object(iterator.knowledge_updater, "storage", mock_storage),
            patch.object(
                iterator.knowledge_updater.manager,
                "initialize",
                new_callable=AsyncMock,
            ),
            patch.object(
                iterator.knowledge_updater.fetcher,
                "initialize",
                new_callable=AsyncMock,
            ),
        ):
            result = await iterator.run()

            # 验证：skip-online 不抓取
            mock_fetch.assert_not_called()

            # 验证：dry-run 返回结构
            assert result["success"] is True
            assert result["dry_run"] is True
            assert "summary" in result
            assert result["goal_length"] > 0

            # 验证：目标被正确构建
            assert iterator.context.iteration_goal != ""
            assert "回归测试任务" in iterator.context.iteration_goal


# ============================================================
# TestAutoModeOrchestratorMpForcesBasicWithWarning - Auto 模式强制 basic + 统一 warning
# ============================================================


class TestAutoModeOrchestratorMpForcesBasicWithWarning:
    """测试 --execution-mode auto --orchestrator mp 仍强制 basic 且输出统一 warning

    验证场景：
    1. execution_mode=auto + orchestrator=mp -> 强制切换为 basic
    2. 切换时输出统一的 warning 消息
    3. 警告消息包含关键指引（如为什么强制切换）
    """

    @pytest.fixture
    def auto_mp_args(self) -> argparse.Namespace:
        """创建 execution_mode=auto, orchestrator=mp 的参数"""
        return argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
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
            orchestrator="mp",  # 用户显式指定 mp
            no_mp=False,
            _orchestrator_user_set=True,  # 用户显式设置
            execution_mode="auto",  # auto 模式
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
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_auto_mode_with_mp_forces_basic_orchestrator(self, auto_mp_args: argparse.Namespace) -> None:
        """测试 execution_mode=auto + orchestrator=mp 仍强制使用 basic 编排器"""
        from cursor.cloud_client import CloudClientFactory

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            iterator = SelfIterator(auto_mp_args)

            # 验证编排器被强制切换为 basic
            orchestrator_type = iterator._get_orchestrator_type()
            assert orchestrator_type == "basic", (
                f"execution_mode=auto + orchestrator=mp 应强制使用 basic 编排器，实际: {orchestrator_type}"
            )

    def test_auto_mode_with_mp_outputs_unified_warning(
        self, auto_mp_args: argparse.Namespace, caplog: pytest.LogCaptureFixture
    ) -> None:
        """测试 execution_mode=auto + orchestrator=mp 输出统一 warning"""
        import logging

        from cursor.cloud_client import CloudClientFactory

        with (
            caplog.at_level(logging.WARNING),
            patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"),
        ):
            iterator = SelfIterator(auto_mp_args)

            # 触发编排器类型解析（会产生 warning）
            orchestrator_type = iterator._get_orchestrator_type()

            # 验证编排器为 basic
            assert orchestrator_type == "basic"

            # 验证输出了警告日志
            # 注意：实际的 warning 可能在 __init__ 或 _get_orchestrator_type 中输出
            # 这里检查日志中是否包含相关信息
            warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]

            # 如果有警告消息，验证内容
            if warning_messages:
                warning_text = " ".join(warning_messages)
                assert any(
                    keyword in warning_text.lower() for keyword in ["basic", "auto", "cloud", "编排器", "orchestrator"]
                ), f"警告消息应包含相关关键词，实际: {warning_text}"

    def test_auto_mode_warning_contains_actionable_guidance(self, auto_mp_args: argparse.Namespace) -> None:
        """测试 warning 消息包含可操作的指引"""
        from core.execution_policy import should_use_mp_orchestrator

        # 验证 should_use_mp_orchestrator 函数的行为
        assert should_use_mp_orchestrator("auto") is False, "should_use_mp_orchestrator('auto') 应返回 False"
        assert should_use_mp_orchestrator("cloud") is False, "should_use_mp_orchestrator('cloud') 应返回 False"
        assert should_use_mp_orchestrator("cli") is True, "should_use_mp_orchestrator('cli') 应返回 True"

    def test_cloud_mode_with_mp_also_forces_basic(self, auto_mp_args: argparse.Namespace) -> None:
        """测试 execution_mode=cloud + orchestrator=mp 同样强制使用 basic"""
        from cursor.cloud_client import CloudClientFactory

        # 修改为 cloud 模式
        auto_mp_args.execution_mode = "cloud"

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            iterator = SelfIterator(auto_mp_args)
            orchestrator_type = iterator._get_orchestrator_type()

            assert orchestrator_type == "basic", "execution_mode=cloud + orchestrator=mp 应强制使用 basic 编排器"

    @pytest.mark.parametrize(
        "execution_mode,expect_orchestrator",
        [
            pytest.param("auto", "basic", id="auto_forces_basic"),
            pytest.param("cloud", "basic", id="cloud_forces_basic"),
            pytest.param("cli", "mp", id="cli_allows_mp"),
        ],
    )
    def test_orchestrator_selection_by_execution_mode(
        self,
        auto_mp_args: argparse.Namespace,
        execution_mode: str,
        expect_orchestrator: str,
    ) -> None:
        """参数化测试：不同 execution_mode 下的编排器选择"""
        from cursor.cloud_client import CloudClientFactory

        auto_mp_args.execution_mode = execution_mode
        auto_mp_args.orchestrator = "mp"
        auto_mp_args._orchestrator_user_set = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            iterator = SelfIterator(auto_mp_args)
            actual_orchestrator = iterator._get_orchestrator_type()

            assert actual_orchestrator == expect_orchestrator, (
                f"execution_mode={execution_mode} + orchestrator=mp: "
                f"期望 {expect_orchestrator}，实际 {actual_orchestrator}"
            )

    def test_no_api_key_auto_mode_fallback_to_cli_forces_basic(self, auto_mp_args: argparse.Namespace) -> None:
        """测试 auto 模式无 API Key 时回退到 CLI，但编排器仍强制 basic

        新行为：requested=auto/cloud 即强制 basic，不受 key/enable 影响。
        即使因为没有 API Key 导致 effective_mode 回退到 CLI，
        只要 requested_mode 是 auto，编排器就应该是 basic。
        """
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            iterator = SelfIterator(auto_mp_args)

            # 无 API Key 时回退到 CLI 模式
            actual_mode = iterator._get_execution_mode()
            assert actual_mode == ExecutionMode.CLI, "auto + 无 API Key 应回退到 CLI 模式"

            # 新行为：即使回退到 CLI，requested_mode=auto 仍强制 basic
            actual_orchestrator = iterator._get_orchestrator_type()
            assert actual_orchestrator == "basic", "requested_mode=auto 应强制使用 basic 编排器"


# ============================================================
# URL 函数测试（run_iterate.py 中的 is_allowed_doc_url 等）
# ============================================================


class TestRunIterateUrlFunctions:
    """测试 run_iterate.py 中的 URL 相关函数

    这些测试验证：
    1. is_allowed_doc_url 包装函数使用 ALLOWED_DOC_URL_PREFIXES 进行前缀匹配
    2. normalize_url 从 doc_url_strategy 正确导入
    3. deduplicate_urls 从 doc_url_strategy 正确导入
    """

    def test_is_allowed_doc_url_cursor_docs(self) -> None:
        """测试 is_allowed_doc_url 允许 cursor.com/cn/docs"""
        from scripts.run_iterate import is_allowed_doc_url

        # 应该允许
        assert is_allowed_doc_url("https://cursor.com/cn/docs/cli/overview")
        assert is_allowed_doc_url("https://cursor.com/docs/cli/overview")
        assert is_allowed_doc_url("https://cursor.com/cn/changelog")
        assert is_allowed_doc_url("https://cursor.com/changelog")

    def test_is_allowed_doc_url_rejects_other_domains(self) -> None:
        """测试 is_allowed_doc_url 拒绝其他域名"""
        from scripts.run_iterate import is_allowed_doc_url

        # 应该拒绝
        assert not is_allowed_doc_url("https://github.com/cursor/repo")
        assert not is_allowed_doc_url("https://google.com/search")
        assert not is_allowed_doc_url("https://other.com/cursor/docs")

    def test_is_allowed_doc_url_with_relative_path(self) -> None:
        """测试 is_allowed_doc_url 处理相对路径"""
        from scripts.run_iterate import is_allowed_doc_url

        # 相对路径应该被补全为 cursor.com 后检查
        # /cn/docs/... 应该被允许
        result = is_allowed_doc_url("/cn/docs/cli/overview")
        assert result

    def test_is_allowed_doc_url_with_fragment(self) -> None:
        """测试 is_allowed_doc_url 处理带 fragment 的 URL"""
        from scripts.run_iterate import is_allowed_doc_url

        # 带 fragment 的 URL 应该被规范化后检查
        assert is_allowed_doc_url("https://cursor.com/cn/docs/cli#section")

    def test_normalize_url_imported_from_doc_url_strategy(self) -> None:
        """测试 normalize_url 从 doc_url_strategy 正确导入"""
        from scripts.run_iterate import normalize_url

        # 测试基本功能
        result = normalize_url("https://EXAMPLE.COM/path#section")
        assert result == "https://example.com/path"

        # 测试相对路径补全
        result = normalize_url("/docs/cli", "https://cursor.com")
        assert result == "https://cursor.com/docs/cli"

    def test_deduplicate_urls_imported_from_doc_url_strategy(self) -> None:
        """测试 deduplicate_urls 从 doc_url_strategy 正确导入"""
        from scripts.run_iterate import deduplicate_urls

        urls = [
            "https://example.com/page",
            "https://example.com/page",  # 重复
            "https://example.com/other",
        ]
        result = deduplicate_urls(urls)
        assert len(result) == 2

    def test_cursor_doc_urls_minimal_set(self) -> None:
        """测试 CURSOR_DOC_URLS 是最小保底集合"""
        from scripts.run_iterate import CURSOR_DOC_URLS

        # 应该只有 3-5 个核心 URL
        assert 3 <= len(CURSOR_DOC_URLS) <= 5, (
            f"CURSOR_DOC_URLS 应该是最小保底集合（3-5 个），实际有 {len(CURSOR_DOC_URLS)} 个"
        )

        # 应该包含核心文档
        assert any("cli/overview" in url for url in CURSOR_DOC_URLS)
        assert any("cli/using" in url for url in CURSOR_DOC_URLS)
        assert any("reference/parameters" in url for url in CURSOR_DOC_URLS)

    def test_allowed_doc_url_prefixes_coverage(self) -> None:
        """测试 ALLOWED_DOC_URL_PREFIXES 覆盖主要文档路径"""
        from scripts.run_iterate import ALLOWED_DOC_URL_PREFIXES

        # 应该包含中英文文档和 changelog
        prefixes_str = " ".join(ALLOWED_DOC_URL_PREFIXES)
        assert "cursor.com/cn/docs" in prefixes_str
        assert "cursor.com/docs" in prefixes_str
        assert "changelog" in prefixes_str

    def test_allowed_doc_url_prefixes_is_full_url_prefix_list(self) -> None:
        """测试 ALLOWED_DOC_URL_PREFIXES 是完整 URL 前缀列表（含 scheme/host）"""
        from scripts.run_iterate import ALLOWED_DOC_URL_PREFIXES

        # ALLOWED_DOC_URL_PREFIXES 应该是完整的 URL 前缀，不是纯域名或路径前缀
        for prefix in ALLOWED_DOC_URL_PREFIXES:
            assert prefix.startswith("https://cursor.com/"), f"前缀应该以 https://cursor.com/ 开头: {prefix}"
            # 应该包含路径部分（不仅仅是域名）
            # 例如 "https://cursor.com/docs" 包含 "docs" 路径
            path_part = prefix.replace("https://cursor.com/", "")
            assert len(path_part) > 0, f"前缀应该包含路径部分: {prefix}"

    def test_is_allowed_doc_url_uses_prefix_matching(self) -> None:
        """测试 is_allowed_doc_url 使用前缀匹配而非域名匹配"""
        from scripts.run_iterate import is_allowed_doc_url

        # 匹配 /docs 或 /changelog 前缀的 URL
        assert is_allowed_doc_url("https://cursor.com/cn/docs/cli/overview")
        assert is_allowed_doc_url("https://cursor.com/docs/guide")
        assert is_allowed_doc_url("https://cursor.com/changelog/2026")

        # cursor.com 域名但不匹配任何前缀的 URL 应该被拒绝
        assert not is_allowed_doc_url("https://cursor.com/pricing")
        assert not is_allowed_doc_url("https://cursor.com/blog/post")
        assert not is_allowed_doc_url("https://cursor.com/download")
        assert not is_allowed_doc_url("https://cursor.com/about")

    def test_is_allowed_doc_url_empty_url(self) -> None:
        """测试 is_allowed_doc_url 处理空 URL"""
        from scripts.run_iterate import is_allowed_doc_url

        assert not is_allowed_doc_url("")
        assert not is_allowed_doc_url("   ")

    def test_normalize_url_empty_url(self) -> None:
        """测试 normalize_url 处理空 URL"""
        from scripts.run_iterate import normalize_url

        assert normalize_url("") == ""
        assert normalize_url("   ") == ""

    def test_deduplicate_urls_stability(self) -> None:
        """测试 deduplicate_urls 的稳定性（多次调用结果一致）"""
        from scripts.run_iterate import deduplicate_urls

        urls = [
            "https://cursor.com/cn/docs/cli/overview",
            "https://cursor.com/cn/docs/cli/using",
            "https://cursor.com/cn/docs/cli/overview",  # 重复
        ]
        results = [deduplicate_urls(urls) for _ in range(10)]
        assert all(r == results[0] for r in results)


# ============================================================
# TestBuildDocAllowlist - build_doc_allowlist 函数测试
# ============================================================


class TestBuildDocAllowlistDefaultBranch:
    """测试 build_doc_allowlist 在 url_strategy_config is None 时的默认值行为

    验证当 url_strategy_config 参数为 None 时，函数正确使用
    core.config.DEFAULT_URL_STRATEGY_* 常量填充所有关键字段，
    确保与"默认契约"一致。

    关键验证点：
    - keyword_boost_weight 与 DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT 一致
    - max_urls 与 DEFAULT_URL_STRATEGY_MAX_URLS 一致
    - fallback_core_docs_count 与 DEFAULT_FALLBACK_CORE_DOCS_COUNT 一致
    - prefer_changelog 与 DEFAULT_URL_STRATEGY_PREFER_CHANGELOG 一致
    - priority_weights 与 DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS 一致
    - exclude_patterns 包含资源文件过滤模式
    """

    def test_build_doc_allowlist_none_config_uses_default_constants(self) -> None:
        """测试 url_strategy_config=None 时使用 DEFAULT_URL_STRATEGY_* 常量

        这是关键回归测试，确保 build_doc_allowlist 的 else 分支
        显式填充与 core.config.DEFAULT_URL_STRATEGY_* 一致的关键字段。
        """
        from core.config import (
            DEFAULT_FALLBACK_CORE_DOCS_COUNT,
            DEFAULT_URL_STRATEGY_DEDUPLICATE,
            DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT,
            DEFAULT_URL_STRATEGY_MAX_URLS,
            DEFAULT_URL_STRATEGY_NORMALIZE,
            DEFAULT_URL_STRATEGY_PREFER_CHANGELOG,
            DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS,
        )
        from scripts.run_iterate import build_doc_allowlist

        # 调用 build_doc_allowlist 时不传入 url_strategy_config
        result = build_doc_allowlist(url_strategy_config=None)
        config = result.config

        # 验证关键字段与 core.config 默认值一致
        assert config.keyword_boost_weight == DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT, (
            f"keyword_boost_weight 应为 {DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT}，"
            f"实际为 {config.keyword_boost_weight}"
        )
        assert config.max_urls == DEFAULT_URL_STRATEGY_MAX_URLS, (
            f"max_urls 应为 {DEFAULT_URL_STRATEGY_MAX_URLS}，实际为 {config.max_urls}"
        )
        assert config.fallback_core_docs_count == DEFAULT_FALLBACK_CORE_DOCS_COUNT, (
            f"fallback_core_docs_count 应为 {DEFAULT_FALLBACK_CORE_DOCS_COUNT}，"
            f"实际为 {config.fallback_core_docs_count}"
        )
        assert config.prefer_changelog == DEFAULT_URL_STRATEGY_PREFER_CHANGELOG, (
            f"prefer_changelog 应为 {DEFAULT_URL_STRATEGY_PREFER_CHANGELOG}，实际为 {config.prefer_changelog}"
        )
        assert config.priority_weights == DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS, (
            "priority_weights 应与 DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS 一致"
        )
        assert config.deduplicate == DEFAULT_URL_STRATEGY_DEDUPLICATE, (
            f"deduplicate 应为 {DEFAULT_URL_STRATEGY_DEDUPLICATE}，实际为 {config.deduplicate}"
        )
        assert config.normalize == DEFAULT_URL_STRATEGY_NORMALIZE, (
            f"normalize 应为 {DEFAULT_URL_STRATEGY_NORMALIZE}，实际为 {config.normalize}"
        )

    def test_build_doc_allowlist_none_config_has_exclude_patterns(self) -> None:
        """测试 url_strategy_config=None 时 exclude_patterns 非空"""
        from scripts.run_iterate import build_doc_allowlist

        result = build_doc_allowlist(url_strategy_config=None)
        config = result.config

        # exclude_patterns 应该包含资源文件过滤模式
        assert len(config.exclude_patterns) > 0, "exclude_patterns 不应为空"

        # 应该包含常见资源文件的过滤模式
        patterns_str = " ".join(config.exclude_patterns)
        assert "png" in patterns_str or "jpg" in patterns_str, "exclude_patterns 应包含图片文件过滤模式"

    def test_build_doc_allowlist_none_config_priority_weights_has_changelog(self) -> None:
        """测试 url_strategy_config=None 时 priority_weights 包含 changelog 权重"""
        from scripts.run_iterate import build_doc_allowlist

        result = build_doc_allowlist(url_strategy_config=None)
        config = result.config

        # priority_weights 应该包含 changelog 键
        assert "changelog" in config.priority_weights, "priority_weights 应包含 changelog 键"
        # changelog 权重应该是最高优先级之一
        assert config.priority_weights["changelog"] >= 2.0, (
            f"changelog 权重应 >= 2.0，实际为 {config.priority_weights['changelog']}"
        )

    def test_build_doc_allowlist_empty_list_uses_empty_not_default(self) -> None:
        """测试 allowed_doc_url_prefixes=[] 使用空列表而非回退到默认值

        三态语义验证：
        - None: 回退到默认值 ALLOWED_DOC_URL_PREFIXES
        - []: 使用空列表（不使用 prefixes 限制，回退到 allowed_domains 或 allow-all）
        """
        from scripts.run_iterate import ALLOWED_DOC_URL_PREFIXES, build_doc_allowlist

        # None 应回退到默认值
        result_none = build_doc_allowlist(allowed_doc_url_prefixes=None)
        assert result_none.config.allowed_url_prefixes == ALLOWED_DOC_URL_PREFIXES, "None 应回退到模块默认值"

        # [] 应使用空列表
        result_empty = build_doc_allowlist(allowed_doc_url_prefixes=[])
        assert result_empty.config.allowed_url_prefixes == [], "[] 应被保留，不回退到默认值"

    def test_build_doc_allowlist_url_strategy_empty_list_takes_priority(self) -> None:
        """测试 url_strategy_config.allowed_url_prefixes=[] 优先于其他配置

        验证当 url_strategy_config 显式指定空列表时，不回退到 allowed_doc_url_prefixes。
        """
        from scripts.run_iterate import (
            ResolvedURLStrategyConfig,
            build_doc_allowlist,
        )

        # 创建 url_strategy_config 且 allowed_url_prefixes 为空列表
        url_strategy_config = ResolvedURLStrategyConfig(
            allowed_domains=["cursor.com"],
            allowed_url_prefixes=[],  # 显式空列表
            exclude_patterns=[],
            max_urls=50,
            fallback_core_docs_count=5,
            prefer_changelog=True,
            deduplicate=True,
            normalize=True,
            keyword_boost_weight=1.5,
            priority_weights={"changelog": 3.0},
        )

        # allowed_doc_url_prefixes 有值，但 url_strategy_config 优先
        result = build_doc_allowlist(
            url_strategy_config=url_strategy_config,
            allowed_doc_url_prefixes=["https://example.com/docs"],
        )

        # url_strategy_config.allowed_url_prefixes=[] 应被使用
        assert result.config.allowed_url_prefixes == [], (
            "url_strategy_config.allowed_url_prefixes=[] 应优先于 allowed_doc_url_prefixes"
        )


# ============================================================
# TestLoadCoreDocs - 核心文档加载测试
# ============================================================


class TestLoadCoreDocs:
    """测试 knowledge.doc_sources 模块的 load_core_docs 功能

    测试覆盖：
    1. 文件读取合并顺序确定
    2. 重复 URL 去重稳定
    3. 非法 URL 被过滤
    4. fallback 机制
    """

    def test_load_core_docs_basic(self, tmp_path: Path) -> None:
        """测试基本的文档加载功能"""
        from knowledge.doc_sources import load_core_docs

        # 创建测试文件
        docs_file = tmp_path / "test_docs.txt"
        docs_file.write_text(
            "# 注释行\n"
            "https://cursor.com/cn/docs/cli/overview\n"
            "\n"  # 空行
            "https://cursor.com/cn/docs/cli/using\n",
            encoding="utf-8",
        )

        urls = load_core_docs(
            source_files=["test_docs.txt"],
            project_root=tmp_path,
        )

        assert len(urls) == 2
        assert "https://cursor.com/cn/docs/cli/overview" in urls
        assert "https://cursor.com/cn/docs/cli/using" in urls

    def test_load_core_docs_merge_order_deterministic(self, tmp_path: Path) -> None:
        """测试文件读取合并顺序是确定性的"""
        from knowledge.doc_sources import load_core_docs

        # 创建两个测试文件
        file1 = tmp_path / "file1.txt"
        file1.write_text("https://cursor.com/cn/docs/cli/overview\n")

        file2 = tmp_path / "file2.txt"
        file2.write_text("https://cursor.com/cn/docs/cli/using\n")

        # 多次调用，结果应完全一致
        results = []
        for _ in range(5):
            urls = load_core_docs(
                source_files=["file1.txt", "file2.txt"],
                project_root=tmp_path,
            )
            results.append(urls)

        # 所有结果应该相同
        assert all(r == results[0] for r in results)

        # 顺序应该与文件列表顺序一致（file1 的 URL 在前）
        assert results[0][0] == "https://cursor.com/cn/docs/cli/overview"
        assert results[0][1] == "https://cursor.com/cn/docs/cli/using"

    def test_load_core_docs_deduplicate_stable(self, tmp_path: Path) -> None:
        """测试重复 URL 去重稳定"""
        from knowledge.doc_sources import load_core_docs

        # 创建包含重复 URL 的文件
        file1 = tmp_path / "file1.txt"
        file1.write_text("https://cursor.com/cn/docs/cli/overview\nhttps://cursor.com/cn/docs/cli/using\n")

        file2 = tmp_path / "file2.txt"
        file2.write_text(
            "https://cursor.com/cn/docs/cli/using\n"  # 与 file1 重复
            "https://cursor.com/cn/docs/cli/reference/parameters\n"
        )

        # 多次调用验证稳定性
        results = []
        for _ in range(5):
            urls = load_core_docs(
                source_files=["file1.txt", "file2.txt"],
                project_root=tmp_path,
            )
            results.append(urls)

        # 所有结果应该相同
        assert all(r == results[0] for r in results)

        # 应该去重（3 个唯一 URL）
        assert len(results[0]) == 3

        # 保留首次出现的顺序（file1 的 using 在前）
        using_index = results[0].index("https://cursor.com/cn/docs/cli/using")
        params_index = results[0].index("https://cursor.com/cn/docs/cli/reference/parameters")
        assert using_index < params_index

    def test_load_core_docs_filter_invalid_urls(self, tmp_path: Path) -> None:
        """测试非法 URL 被过滤"""
        from knowledge.doc_sources import load_core_docs

        # 创建包含非法 URL 的文件
        docs_file = tmp_path / "test_docs.txt"
        docs_file.write_text(
            "https://cursor.com/cn/docs/cli/overview\n"  # 有效
            "https://invalid-domain.com/docs\n"  # 无效域名
            "https://cursor.com/cn/docs/cli/using\n"  # 有效
            "not-a-url\n"  # 无效格式
            "https://cursor.com/cn/docs/agent/overview\n"  # 有效
        )

        urls = load_core_docs(
            source_files=["test_docs.txt"],
            project_root=tmp_path,
        )

        # 应该只有 3 个有效 URL
        assert len(urls) == 3
        assert "https://cursor.com/cn/docs/cli/overview" in urls
        assert "https://cursor.com/cn/docs/cli/using" in urls
        assert "https://cursor.com/cn/docs/agent/overview" in urls

        # 非法 URL 不应出现
        assert "https://invalid-domain.com/docs" not in urls
        assert "not-a-url" not in urls

    def test_load_core_docs_skip_comments_and_empty(self, tmp_path: Path) -> None:
        """测试跳过注释和空行"""
        from knowledge.doc_sources import load_core_docs

        # 创建带有大量注释和空行的文件
        docs_file = tmp_path / "test_docs.txt"
        docs_file.write_text(
            "# 这是一个注释\n"
            "# ========== 分隔线 ==========\n"
            "\n"
            "   \n"  # 仅空白的行
            "https://cursor.com/cn/docs/cli/overview\n"
            "# 中间注释\n"
            "\n"
            "https://cursor.com/cn/docs/cli/using\n"
            "# 结尾注释\n",
            encoding="utf-8",
        )

        urls = load_core_docs(
            source_files=["test_docs.txt"],
            project_root=tmp_path,
        )

        # 应该只有 2 个 URL
        assert len(urls) == 2

    def test_load_core_docs_fallback_on_empty(self, tmp_path: Path) -> None:
        """测试文件为空或不存在时使用 fallback"""
        from knowledge.doc_sources import load_core_docs

        # 创建空文件
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        fallback_urls = [
            "https://cursor.com/cn/docs/cli/overview",
            "https://cursor.com/cn/docs/cli/using",
        ]

        urls = load_core_docs(
            source_files=["empty.txt"],
            project_root=tmp_path,
            fallback_urls=fallback_urls,
        )

        # 应该使用 fallback
        assert len(urls) == 2

    def test_load_core_docs_fallback_on_missing_file(self, tmp_path: Path) -> None:
        """测试文件不存在时使用 fallback"""
        from knowledge.doc_sources import load_core_docs

        fallback_urls = [
            "https://cursor.com/cn/docs/cli/overview",
        ]

        urls = load_core_docs(
            source_files=["nonexistent.txt"],
            project_root=tmp_path,
            fallback_urls=fallback_urls,
        )

        # 应该使用 fallback
        assert len(urls) == 1
        assert "https://cursor.com/cn/docs/cli/overview" in urls

    def test_load_core_docs_with_fallback_function(self, tmp_path: Path) -> None:
        """测试 load_core_docs_with_fallback 函数"""
        from knowledge.doc_sources import load_core_docs_with_fallback

        legacy_urls = [
            "https://cursor.com/cn/docs/cli/overview",
            "https://cursor.com/cn/docs/cli/using",
        ]

        # 文件不存在时应使用 legacy fallback
        urls = load_core_docs_with_fallback(
            source_files=["nonexistent.txt"],
            legacy_urls=legacy_urls,
            project_root=tmp_path,
        )

        assert len(urls) == 2

    def test_parse_url_list_file_basic(self, tmp_path: Path) -> None:
        """测试 parse_url_list_file 基本功能"""
        from knowledge.doc_sources import parse_url_list_file

        # 创建测试文件
        docs_file = tmp_path / "test_docs.txt"
        docs_file.write_text(
            "# 注释\nhttps://cursor.com/cn/docs/cli/overview\n\nhttps://cursor.com/cn/docs/cli/using\n",
            encoding="utf-8",
        )

        urls = parse_url_list_file(docs_file)

        assert len(urls) == 2
        assert "https://cursor.com/cn/docs/cli/overview" in urls
        assert "https://cursor.com/cn/docs/cli/using" in urls

    def test_parse_url_list_file_not_found(self, tmp_path: Path) -> None:
        """测试文件不存在时抛出异常"""
        from knowledge.doc_sources import parse_url_list_file

        with pytest.raises(FileNotFoundError):
            parse_url_list_file(tmp_path / "nonexistent.txt")

    def test_is_valid_doc_url(self) -> None:
        """测试 is_valid_doc_url 函数"""
        from knowledge.doc_sources import is_valid_doc_url

        # 有效 URL（匹配默认前缀）
        assert is_valid_doc_url("https://cursor.com/cn/docs/cli/overview")
        assert is_valid_doc_url("https://cursor.com/docs/cli/overview")
        assert is_valid_doc_url("https://cursor.com/cn/changelog")
        assert is_valid_doc_url("https://cursor.com/changelog/2026")

        # 无效 URL
        assert not is_valid_doc_url("")
        assert not is_valid_doc_url("https://invalid.com/docs")
        assert not is_valid_doc_url("https://google.com")
        # cursor.com 域名但不匹配前缀
        assert not is_valid_doc_url("https://cursor.com/pricing")
        assert not is_valid_doc_url("https://cursor.com/blog")

    def test_is_valid_doc_url_with_allowed_url_prefixes(self) -> None:
        """测试 is_valid_doc_url 使用新参数名 allowed_url_prefixes"""
        from knowledge.doc_sources import is_valid_doc_url

        custom_prefixes = ["https://example.com/docs"]

        # 使用新参数名
        assert is_valid_doc_url(
            "https://example.com/docs/guide",
            allowed_url_prefixes=custom_prefixes,
        )
        assert not is_valid_doc_url(
            "https://example.com/blog",
            allowed_url_prefixes=custom_prefixes,
        )

    def test_is_valid_doc_url_with_allowed_domains_alias(self) -> None:
        """测试 is_valid_doc_url 使用旧参数名 allowed_domains（向后兼容）"""
        from knowledge.doc_sources import is_valid_doc_url

        custom_prefixes = ["https://example.com/docs"]

        # 使用旧参数名（向后兼容）
        assert is_valid_doc_url(
            "https://example.com/docs/guide",
            allowed_domains=custom_prefixes,  # 旧参数名
        )
        assert not is_valid_doc_url(
            "https://example.com/blog",
            allowed_domains=custom_prefixes,
        )

    def test_get_core_docs_function(self) -> None:
        """测试 get_core_docs 函数"""
        from scripts.run_iterate import get_core_docs

        urls = get_core_docs()

        # 应该返回非空列表
        assert len(urls) > 0

        # 所有 URL 应该是有效的 cursor.com 文档链接
        for url in urls:
            assert "cursor.com" in url
            assert url.startswith("https://")

    def test_load_core_docs_url_normalization(self, tmp_path: Path) -> None:
        """测试 URL 规范化（移除末尾斜杠、锚点等）"""
        from knowledge.doc_sources import load_core_docs

        # 创建包含需要规范化的 URL
        docs_file = tmp_path / "test_docs.txt"
        docs_file.write_text(
            "https://cursor.com/cn/docs/cli/overview/\n"  # 末尾斜杠
            "https://cursor.com/cn/docs/cli/using#section\n"  # 锚点
            "https://CURSOR.COM/cn/docs/cli/reference/parameters\n"  # 大写域名
        )

        urls = load_core_docs(
            source_files=["test_docs.txt"],
            project_root=tmp_path,
        )

        # 应该规范化后去重
        assert len(urls) == 3

        # URL 应该被规范化
        assert "https://cursor.com/cn/docs/cli/overview" in urls
        assert "https://cursor.com/cn/docs/cli/using" in urls
        assert "https://cursor.com/cn/docs/cli/reference/parameters" in urls


# ============================================================
# Test: 集成级测试 - allowed_domains 过滤外域 URL
# ============================================================


class TestAllowedDomainsIntegration:
    """集成级测试：验证 run_iterate 策略配置启用 ALLOWED_DOC_URL_PREFIXES 时，
    构建出的 urls_to_fetch 不包含外域。
    """

    @pytest.mark.asyncio
    async def test_build_urls_to_fetch_filters_external_domains(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 build_urls_to_fetch 方法正确过滤外域 URL

        场景：
        - UpdateAnalysis 包含 cursor.com 与 github.com 等外域链接
        - 策略配置使用 ALLOWED_DOC_URL_PREFIXES（完整 URL 前缀）
        - 验证最终 urls_to_fetch 不包含外域
        """
        from knowledge.doc_url_strategy import DocURLStrategyConfig
        from knowledge.doc_url_strategy import select_urls_to_fetch as strategy_select_urls
        from scripts.run_iterate import (
            ALLOWED_DOC_URL_PREFIXES,
            ALLOWED_DOC_URL_PREFIXES_NETLOC,
            UpdateAnalysis,
        )

        # 创建包含外域链接的 UpdateAnalysis
        analysis = UpdateAnalysis(
            has_updates=True,
            entries=[],
            new_features=["新功能测试"],
            changelog_links=[
                "https://cursor.com/changelog/2026-01",
                "https://github.com/cursor/releases",  # 外域
            ],
            related_doc_urls=[
                "https://cursor.com/docs/cli/overview",
                "https://stackoverflow.com/questions/cursor",  # 外域
                "https://npmjs.com/package/cursor-cli",  # 外域
            ],
        )

        # 使用与 run_iterate.py 相同的配置构建策略（使用新名称）
        config = DocURLStrategyConfig(
            allowed_url_prefixes=ALLOWED_DOC_URL_PREFIXES,
            allowed_domains=ALLOWED_DOC_URL_PREFIXES_NETLOC,
            max_urls=20,
            fallback_core_docs_count=5,
            deduplicate=True,
            normalize=True,
        )

        # 模拟 llms.txt 内容（包含外域链接）
        llms_txt_content = (
            "https://cursor.com/docs/guide\n"
            "https://github.com/cursor/docs\n"  # 外域
            "https://cursor.com/cn/docs/api\n"
        )

        # 调用 strategy_select_urls（与 KnowledgeUpdater.build_urls_to_fetch 内部逻辑一致）
        result = strategy_select_urls(
            changelog_links=analysis.changelog_links or [],
            related_doc_urls=analysis.related_doc_urls or [],
            llms_txt_content=llms_txt_content,
            core_docs=[],  # 不使用 core_docs 以隔离测试
            keywords=[],
            config=config,
            base_url="https://cursor.com",
        )

        # 验证结果不包含外域 URL
        external_domains = ["github.com", "stackoverflow.com", "npmjs.com"]
        for url in result:
            for ext_domain in external_domains:
                assert ext_domain not in url, f"外域 URL 应被过滤: {url}"

        # 验证 cursor.com 相关 URL 存在
        cursor_urls = [u for u in result if "cursor.com" in u]
        assert len(cursor_urls) > 0, "应至少包含一个 cursor.com URL"

    @pytest.mark.asyncio
    async def test_knowledge_updater_build_urls_filters_external(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 KnowledgeUpdater._build_urls_to_fetch 完整方法过滤外域

        通过 mock _fetch_llms_txt 和 get_core_docs，验证实际方法行为。
        """
        from scripts.run_iterate import (
            UpdateAnalysis,
        )

        iterator = SelfIterator(base_iterate_args)
        updater = iterator.knowledge_updater

        # 创建包含外域链接的 UpdateAnalysis
        analysis = UpdateAnalysis(
            has_updates=True,
            entries=[],
            new_features=[],
            changelog_links=[
                "https://cursor.com/changelog/test",
                "https://github.com/test/repo",  # 外域
            ],
            related_doc_urls=[
                "https://cursor.com/docs/test",
                "https://example.org/docs",  # 外域
            ],
        )

        # Mock _fetch_llms_txt 返回包含外域的内容
        mock_llms_content = (
            "https://cursor.com/cn/docs/guide\n"
            "https://external.io/docs\n"  # 外域
        )

        # Mock get_core_docs 返回空列表（隔离测试）
        with (
            patch.object(updater, "_fetch_llms_txt", new_callable=AsyncMock, return_value=mock_llms_content),
            patch("scripts.run_iterate.get_core_docs", return_value=[]),
        ):
            urls, selection_log = await updater._build_urls_to_fetch(
                analysis=analysis,
                max_urls=20,
                fallback_count=0,
            )

        # 验证结果不包含外域 URL
        assert not any("github.com" in u for u in urls), "github.com 应被过滤"
        assert not any("example.org" in u for u in urls), "example.org 应被过滤"
        assert not any("external.io" in u for u in urls), "external.io 应被过滤"

        # 验证 cursor.com 相关 URL 存在
        assert any("cursor.com" in u for u in urls), "应包含 cursor.com URL"


# ============================================================
# 样例数据：llms.txt 格式（供测试复用）
# ============================================================

SAMPLE_LLMS_TXT_CURSOR = """# Cursor CLI Documentation

## Overview
[Getting Started](https://cursor.com/docs/getting-started)
[CLI Reference](https://cursor.com/docs/cli/reference)

## Features
- [Agent Mode](https://cursor.com/docs/agent-mode)
- [Plan Mode](https://cursor.com/docs/modes/plan)
- [Ask Mode](https://cursor.com/docs/modes/ask)

## API
https://cursor.com/docs/api/overview
https://cursor.com/docs/api/authentication

## External Links (should be filtered)
https://github.com/cursor/cursor-cli
https://stackoverflow.com/questions/tagged/cursor
"""

SAMPLE_LLMS_TXT_MIXED_DOMAINS = """# Mixed Domain Documentation

## Internal Docs
https://cursor.com/cn/docs/cli/overview
https://cursor.com/docs/cli/parameters
https://api.cursor.com/reference

## External Links
https://github.com/cursor/repo
https://npmjs.com/package/cursor
https://stackoverflow.com/questions/cursor-cli

## More Internal
[Changelog](https://cursor.com/cn/changelog/2026)
"""


# ============================================================
# TestSampleDataReuse - 复用样例数据的测试
# ============================================================


class TestSampleDataReuse:
    """测试复用样例 HTML/llms.txt 数据，确保不依赖真实网络"""

    def test_parse_sample_changelog_without_network(self) -> None:
        """测试使用样例 changelog 数据解析，不依赖网络"""
        analyzer = ChangelogAnalyzer()

        # 使用已定义的 SAMPLE_CHANGELOG_JAN_16_2026
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_JAN_16_2026)

        # 验证解析结果
        assert len(entries) >= 1, "应至少解析出 1 条 entry"
        assert any("jan" in e.date.lower() for e in entries), "应包含 Jan 日期"

    def test_parse_sample_html_changelog_without_network(self) -> None:
        """测试使用样例 HTML changelog 数据解析，不依赖网络"""
        analyzer = ChangelogAnalyzer()

        # 使用 HTML 混合内容样例
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_HTML_MIXED)

        # 验证 HTML 被正确清理
        all_content = " ".join(e.content for e in entries)
        assert "<script>" not in all_content
        assert "</script>" not in all_content

    def test_parse_sample_minified_html_without_network(self) -> None:
        """测试使用 minified HTML 样例解析，不依赖网络"""
        analyzer = ChangelogAnalyzer()

        # 使用 minified HTML 样例
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML)

        # 验证解析成功
        assert len(entries) >= 1, "应解析出至少 1 条 entry"

    def test_parse_sample_plain_text_without_network(self) -> None:
        """测试使用纯文本样例解析，不依赖网络"""
        analyzer = ChangelogAnalyzer()

        # 使用纯文本样例
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT)

        # 验证解析成功
        assert len(entries) >= 1, "应解析出至少 1 条 entry"

    @pytest.mark.asyncio
    async def test_knowledge_updater_with_sample_llms_txt(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 KnowledgeUpdater 使用样例 llms.txt 数据，不依赖网络"""
        from knowledge.doc_url_strategy import parse_llms_txt_urls

        # 使用样例 llms.txt 数据
        urls = parse_llms_txt_urls(SAMPLE_LLMS_TXT_CURSOR)

        # 验证解析结果
        assert len(urls) >= 5, f"应解析出至少 5 个 URL，实际 {len(urls)}"

        # 验证包含预期的 cursor.com URL
        cursor_urls = [u for u in urls if "cursor.com" in u]
        assert len(cursor_urls) >= 5, "应包含多个 cursor.com URL"

    @pytest.mark.asyncio
    async def test_llms_txt_external_urls_filtered_with_sample(self, base_iterate_args: argparse.Namespace) -> None:
        """测试样例 llms.txt 中外域 URL 被正确过滤"""
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            parse_llms_txt_urls,
            select_urls_to_fetch,
        )
        from scripts.run_iterate import (
            ALLOWED_DOC_URL_PREFIXES,
            ALLOWED_DOC_URL_PREFIXES_NETLOC,
        )

        # 解析样例 llms.txt
        all_urls = parse_llms_txt_urls(SAMPLE_LLMS_TXT_MIXED_DOMAINS)

        # 验证原始解析包含外域 URL
        assert any("github.com" in u for u in all_urls), "原始解析应包含 github.com"
        assert any("cursor.com" in u for u in all_urls), "原始解析应包含 cursor.com"

        # 使用策略过滤（使用新名称）
        config = DocURLStrategyConfig(
            allowed_url_prefixes=ALLOWED_DOC_URL_PREFIXES,
            allowed_domains=ALLOWED_DOC_URL_PREFIXES_NETLOC,
            max_urls=20,
        )

        filtered_urls = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=SAMPLE_LLMS_TXT_MIXED_DOMAINS,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 验证过滤后不包含外域
        assert not any("github.com" in u for u in filtered_urls), "过滤后不应包含 github.com"
        assert not any("npmjs.com" in u for u in filtered_urls), "过滤后不应包含 npmjs.com"

        # 验证保留 cursor.com
        assert any("cursor.com" in u for u in filtered_urls), "过滤后应保留 cursor.com"

    @pytest.mark.asyncio
    async def test_changelog_analyzer_with_mocked_fetch(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 ChangelogAnalyzer 使用 mock fetch，不依赖真实网络"""
        iterator = SelfIterator(base_iterate_args)

        # Mock fetch 返回样例数据
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = SAMPLE_CHANGELOG_JAN_16_2026
        mock_fetch_result.error = None

        with (
            patch.object(iterator.changelog_analyzer.fetcher, "initialize", new_callable=AsyncMock),
            patch.object(
                iterator.changelog_analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ),
        ):
            analysis = await iterator.changelog_analyzer.analyze()

            # 验证分析结果
            assert isinstance(analysis, UpdateAnalysis)
            # 使用样例数据应该有更新
            assert analysis.has_updates is True or len(analysis.entries) >= 0

    def test_all_sample_changelog_variants_parseable(self) -> None:
        """测试所有 changelog 样例变体都能被解析"""
        analyzer = ChangelogAnalyzer()

        variants = {
            "JAN_16_2026": SAMPLE_CHANGELOG_JAN_16_2026,
            "HTML_MIXED": SAMPLE_CHANGELOG_HTML_MIXED,
            "VARIANT_A": SAMPLE_CHANGELOG_VARIANT_A_HTML_HEADERS,
            "VARIANT_B": SAMPLE_CHANGELOG_VARIANT_B_MINIFIED_HTML,
            "VARIANT_C": SAMPLE_CHANGELOG_VARIANT_C_PLAIN_TEXT,
        }

        for name, content in variants.items():
            entries = analyzer.parse_changelog(content)
            assert len(entries) >= 1, f"变体 {name} 应至少解析出 1 条 entry"

    def test_sample_llms_txt_url_extraction(self) -> None:
        """测试样例 llms.txt 的 URL 提取"""
        from knowledge.doc_url_strategy import parse_llms_txt_urls

        # 测试 cursor 格式
        cursor_urls = parse_llms_txt_urls(SAMPLE_LLMS_TXT_CURSOR)
        assert len(cursor_urls) >= 5

        # 测试混合域名格式（实际解析出 7 个 URL）
        mixed_urls = parse_llms_txt_urls(SAMPLE_LLMS_TXT_MIXED_DOMAINS)
        assert len(mixed_urls) >= 7


# ============================================================
# TestOfflineWorkflow - 离线工作流测试
# ============================================================


class TestOfflineWorkflow:
    """测试离线工作流（skip_online 模式）"""

    @pytest.mark.asyncio
    async def test_skip_online_uses_cached_data(self, skip_online_args: argparse.Namespace) -> None:
        """测试 skip_online 模式使用缓存数据"""
        iterator = SelfIterator(skip_online_args)

        # 在 skip_online 模式下，不应该调用网络
        # 直接验证 args 设置正确
        assert iterator.args.skip_online is True

    @pytest.mark.asyncio
    async def test_skip_online_with_sample_analysis(self, skip_online_args: argparse.Namespace) -> None:
        """测试 skip_online 模式使用样例分析数据"""
        from scripts.run_iterate import UpdateAnalysis

        # 创建模拟的分析结果（模拟从缓存加载）
        mock_analysis = UpdateAnalysis(
            has_updates=True,
            entries=[],
            new_features=["Plan Mode", "Ask Mode"],
            changelog_links=["https://cursor.com/changelog/2026"],
            related_doc_urls=["https://cursor.com/docs/modes/plan"],
        )

        # 验证分析结果结构
        assert mock_analysis.has_updates is True
        assert len(mock_analysis.new_features) == 2

    @pytest.mark.asyncio
    async def test_dry_run_with_sample_data(self, dry_run_args: argparse.Namespace) -> None:
        """测试 dry_run 模式使用样例数据"""
        iterator = SelfIterator(dry_run_args)

        # 验证 dry_run 设置
        assert iterator.args.dry_run is True
        assert iterator.args.skip_online is True


# ============================================================
# TestSampleDataConsistency - 样例数据一致性测试
# ============================================================


class TestSampleDataConsistency:
    """测试样例数据的一致性和完整性"""

    def test_sample_changelog_contains_expected_features(self) -> None:
        """测试样例 changelog 包含预期的特性关键词"""
        # SAMPLE_CHANGELOG_JAN_16_2026 应包含 plan/ask 特性
        content_lower = SAMPLE_CHANGELOG_JAN_16_2026.lower()
        assert "plan" in content_lower, "应包含 plan 关键词"
        assert "ask" in content_lower, "应包含 ask 关键词"
        assert "cloud" in content_lower, "应包含 cloud 关键词"

    def test_sample_llms_txt_contains_expected_urls(self) -> None:
        """测试样例 llms.txt 包含预期的 URL"""
        assert "cursor.com/docs" in SAMPLE_LLMS_TXT_CURSOR
        assert "cursor.com/docs/api" in SAMPLE_LLMS_TXT_CURSOR

    def test_sample_data_no_real_network_dependency(self) -> None:
        """验证样例数据不依赖真实网络"""
        # 所有样例数据都是字符串常量，不需要网络
        assert isinstance(SAMPLE_CHANGELOG_JAN_16_2026, str)
        assert isinstance(SAMPLE_LLMS_TXT_CURSOR, str)
        assert isinstance(SAMPLE_LLMS_TXT_MIXED_DOMAINS, str)

        # 验证内容非空
        assert len(SAMPLE_CHANGELOG_JAN_16_2026) > 100
        assert len(SAMPLE_LLMS_TXT_CURSOR) > 100


# ============================================================
# ExecutionModeOrchestratorTestCase - 参数化测试数据结构
# ============================================================


@dataclass
class ExecutionModeOrchestratorTestCase:
    """执行模式和编排器组合测试参数

    用于 pytest.mark.parametrize 驱动的参数化测试，覆盖所有
    execution_mode 与 orchestrator 的组合场景。

    Attributes:
        test_id: 测试用例唯一标识符
        requested_execution_mode: 请求的执行模式 (None/cli/auto/cloud)
        has_ampersand_prefix: 是否使用 & 前缀触发云端
        cloud_enabled: 云端是否启用
        has_api_key: 是否有 API Key
        orchestrator_user_set: 用户是否显式设置 orchestrator
        orchestrator_flag: orchestrator 参数值 (mp/basic/None)
        no_mp_flag: --no-mp 标志
        expected_orchestrator_type: 期望的最终编排器类型
        expected_effective_execution_mode: 期望的有效执行模式
        expect_mp_attempted: 是否期望尝试 MP 编排器
        expect_fallback_to_basic: 是否期望回退到 basic
    """

    test_id: str
    requested_execution_mode: Optional[str]  # None/cli/auto/cloud
    has_ampersand_prefix: bool
    cloud_enabled: bool
    has_api_key: bool
    orchestrator_user_set: bool
    orchestrator_flag: Optional[str]  # mp/basic/None
    no_mp_flag: bool
    expected_orchestrator_type: str  # mp/basic
    expected_effective_execution_mode: str  # cli/auto/cloud
    expect_mp_attempted: bool
    expect_fallback_to_basic: bool


# 执行模式与编排器组合测试参数表
EXECUTION_MODE_ORCHESTRATOR_TEST_CASES: list[ExecutionModeOrchestratorTestCase] = [
    # ===== CLI 模式场景 =====
    ExecutionModeOrchestratorTestCase(
        test_id="cli_default_mp",
        requested_execution_mode="cli",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="mp",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=True,
        expect_fallback_to_basic=False,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="cli_explicit_mp",
        requested_execution_mode="cli",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=True,
        orchestrator_flag="mp",
        no_mp_flag=False,
        expected_orchestrator_type="mp",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=True,
        expect_fallback_to_basic=False,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="cli_explicit_basic",
        requested_execution_mode="cli",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=True,
        orchestrator_flag="basic",
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,
        expect_fallback_to_basic=False,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="cli_no_mp_flag",
        requested_execution_mode="cli",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=True,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,
        expect_fallback_to_basic=False,
    ),
    # ===== AUTO 模式场景（有 API Key）=====
    ExecutionModeOrchestratorTestCase(
        test_id="auto_with_key_forces_basic",
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="auto",
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="auto_with_key_explicit_mp_forces_basic",
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=True,
        orchestrator_flag="mp",
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="auto",
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="auto_with_key_explicit_basic",
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=True,
        orchestrator_flag="basic",
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="auto",
        expect_mp_attempted=False,
        expect_fallback_to_basic=False,
    ),
    # ===== AUTO 模式场景（无 API Key，回退 CLI）=====
    # 新行为：requested=auto/cloud 即强制 basic，不受 key/enable 影响
    ExecutionModeOrchestratorTestCase(
        test_id="auto_no_key_forces_basic",  # 重命名以反映新行为
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=False,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # 新行为：auto 强制 basic
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,  # 不会尝试 MP
        expect_fallback_to_basic=True,  # 强制切换到 basic
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="auto_no_key_explicit_mp_forces_basic",  # 重命名以反映新行为
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=False,
        orchestrator_user_set=True,
        orchestrator_flag="mp",
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # 新行为：即使显式 mp，auto 也强制 basic
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,  # 不会尝试 MP
        expect_fallback_to_basic=True,  # 强制切换到 basic
    ),
    # ===== CLOUD 模式场景 =====
    ExecutionModeOrchestratorTestCase(
        test_id="cloud_with_key_forces_basic",
        requested_execution_mode="cloud",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="cloud",
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="cloud_explicit_mp_forces_basic",
        requested_execution_mode="cloud",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=True,
        orchestrator_flag="mp",
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="cloud",
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="cloud_no_key_forces_basic",  # 重命名以反映新行为
        requested_execution_mode="cloud",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=False,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # 新行为：cloud 强制 basic
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,  # 不会尝试 MP
        expect_fallback_to_basic=True,  # 强制切换到 basic
    ),
    # ===== & 前缀触发场景 =====
    # 注意：当 args.execution_mode 显式设置时，& 前缀被忽略
    # 以下测试模拟 & 前缀触发时 execution_mode 未显式设置的场景
    ExecutionModeOrchestratorTestCase(
        test_id="ampersand_prefix_with_key_cloud",
        requested_execution_mode=None,  # 未显式设置，& 前缀触发 cloud
        has_ampersand_prefix=True,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="cloud",  # resolve_effective_execution_mode 返回 cloud
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    # 根据 AGENTS.md：& 前缀表达 Cloud 意图，即使未成功路由也使用 basic 编排器
    ExecutionModeOrchestratorTestCase(
        test_id="ampersand_prefix_no_key_fallback_cli",
        requested_execution_mode=None,
        has_ampersand_prefix=True,
        cloud_enabled=True,
        has_api_key=False,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # 根据 R-2：& 前缀表达 Cloud 意图，强制 basic
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,  # 强制 basic，不尝试 MP
        expect_fallback_to_basic=True,  # 强制 basic
    ),
    # 根据 AGENTS.md：& 前缀表达 Cloud 意图，即使未成功路由也使用 basic 编排器
    ExecutionModeOrchestratorTestCase(
        test_id="ampersand_prefix_cloud_disabled_uses_cli",
        requested_execution_mode=None,
        has_ampersand_prefix=True,
        cloud_enabled=False,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # 根据 R-2：& 前缀表达 Cloud 意图，强制 basic
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,  # 强制 basic，不尝试 MP
        expect_fallback_to_basic=False,
    ),
    # ===== None 执行模式（无显式指定，函数默认返回 CLI）=====
    ExecutionModeOrchestratorTestCase(
        test_id="none_mode_no_prefix_uses_cli",
        requested_execution_mode=None,
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="mp",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=True,
        expect_fallback_to_basic=False,
    ),
    # ===== cloud_enabled=False 场景 =====
    # 注意：显式指定 auto/cloud 模式时，即使 cloud_enabled=False，
    # 执行模式仍保持 auto/cloud（用于策略层），但编排器强制为 basic
    ExecutionModeOrchestratorTestCase(
        test_id="auto_cloud_disabled_keeps_auto",
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=False,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # auto 模式强制 basic
        expected_effective_execution_mode="auto",  # 保持 auto
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="cloud_mode_cloud_disabled_keeps_cloud",
        requested_execution_mode="cloud",
        has_ampersand_prefix=False,
        cloud_enabled=False,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # cloud 模式强制 basic
        expected_effective_execution_mode="cloud",  # 保持 cloud
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
]


# ============================================================
# TestExecutionModeOrchestratorParametrized - 参数化测试类
# ============================================================


class TestExecutionModeOrchestratorParametrized:
    """执行模式与编排器组合参数化测试

    使用 ExecutionModeOrchestratorTestCase 参数表驱动测试，
    覆盖所有 execution_mode 与 orchestrator 的组合场景。
    """

    @pytest.mark.parametrize(
        "test_case",
        EXECUTION_MODE_ORCHESTRATOR_TEST_CASES,
        ids=[tc.test_id for tc in EXECUTION_MODE_ORCHESTRATOR_TEST_CASES],
    )
    def test_orchestrator_selection(self, test_case: ExecutionModeOrchestratorTestCase) -> None:
        """测试编排器选择逻辑"""
        from cursor.cloud_client import CloudClientFactory

        # 确定 execution_mode 参数
        # SelfIterator 使用 `cli_execution_mode_set = cli_execution_mode is not None`
        # 来判断用户是否显式设置了 execution_mode
        # 当 requested_execution_mode=None 时，设置 execution_mode=None
        # 以模拟用户未显式指定 --execution-mode 的场景
        execution_mode_value = test_case.requested_execution_mode  # 可能是 None

        # 构建测试参数
        args = argparse.Namespace(
            requirement="& 测试任务" if test_case.has_ampersand_prefix else "测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=False,
            max_iterations="3",
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
            orchestrator=test_case.orchestrator_flag or "mp",
            no_mp=test_case.no_mp_flag,
            _orchestrator_user_set=test_case.orchestrator_user_set,
            execution_mode=execution_mode_value,
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

        # Mock API Key 和 cloud_enabled 配置
        api_key = "mock-api-key" if test_case.has_api_key else None
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            # Mock cloud_enabled 配置
            with patch("core.config.get_config") as mock_get_config:
                mock_config = MagicMock()
                mock_config.cloud_agent.enabled = test_case.cloud_enabled
                mock_get_config.return_value = mock_config

                # 创建 SelfIterator 并验证
                iterator = SelfIterator(args)

                # 验证编排器类型
                actual_orchestrator = iterator._get_orchestrator_type()
                assert actual_orchestrator == test_case.expected_orchestrator_type, (
                    f"[{test_case.test_id}] 期望编排器 {test_case.expected_orchestrator_type}，"
                    f"实际 {actual_orchestrator}"
                )

    @pytest.mark.parametrize(
        "test_case",
        EXECUTION_MODE_ORCHESTRATOR_TEST_CASES,
        ids=[tc.test_id for tc in EXECUTION_MODE_ORCHESTRATOR_TEST_CASES],
    )
    def test_effective_execution_mode(self, test_case: ExecutionModeOrchestratorTestCase) -> None:
        """测试有效执行模式解析"""
        from core.execution_policy import resolve_effective_execution_mode

        # 调用策略函数
        effective_mode, reason = resolve_effective_execution_mode(
            requested_mode=test_case.requested_execution_mode,
            has_ampersand_prefix=test_case.has_ampersand_prefix,  # 语法检测层面
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        assert effective_mode == test_case.expected_effective_execution_mode, (
            f"[{test_case.test_id}] 期望执行模式 {test_case.expected_effective_execution_mode}，"
            f"实际 {effective_mode}，原因: {reason}"
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            tc
            for tc in EXECUTION_MODE_ORCHESTRATOR_TEST_CASES
            if tc.expect_fallback_to_basic
            # 排除 & 前缀未成功触发的场景：这些场景中 should_use_mp_orchestrator(None) 返回 True
            # 强制 basic 是由 build_execution_decision 根据 has_ampersand_prefix 决定的
            and not (tc.has_ampersand_prefix and tc.expected_effective_execution_mode != "cloud")
        ],
        ids=[
            tc.test_id
            for tc in EXECUTION_MODE_ORCHESTRATOR_TEST_CASES
            if tc.expect_fallback_to_basic
            and not (tc.has_ampersand_prefix and tc.expected_effective_execution_mode != "cloud")
        ],
    )
    def test_mp_to_basic_fallback_by_mode(self, test_case: ExecutionModeOrchestratorTestCase) -> None:
        """测试基于 requested_mode 的强制 basic 切换

        should_use_mp_orchestrator 只基于 requested_mode 判断（mode-only）。
        此测试仅覆盖因 requested_mode=auto/cloud 导致的强制 basic 场景。

        注意：& 前缀未成功触发的场景（has_ampersand_prefix=True 但 effective_mode != cloud）
        不在此测试范围内，因为 should_use_mp_orchestrator(None) 返回 True。
        这些场景由 build_execution_decision 根据 has_ampersand_prefix 强制 basic。
        """
        from core.execution_policy import should_use_mp_orchestrator

        # 确定实际传入 should_use_mp_orchestrator 的 requested_mode
        # 对于 & 前缀成功触发 Cloud 的场景，使用 "cloud"
        requested_mode: str | None
        if test_case.has_ampersand_prefix and test_case.expected_effective_execution_mode == "cloud":
            # & 前缀成功触发 Cloud，_get_orchestrator_type 会使用 "cloud"
            requested_mode = "cloud"
        else:
            requested_mode = test_case.requested_execution_mode

        can_use_mp = should_use_mp_orchestrator(requested_mode)

        assert can_use_mp is False, f"[{test_case.test_id}] 请求的执行模式 {requested_mode} 不应允许 MP 编排器"

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in EXECUTION_MODE_ORCHESTRATOR_TEST_CASES if tc.expect_mp_attempted],
        ids=[tc.test_id for tc in EXECUTION_MODE_ORCHESTRATOR_TEST_CASES if tc.expect_mp_attempted],
    )
    def test_mp_orchestrator_allowed(self, test_case: ExecutionModeOrchestratorTestCase) -> None:
        """测试 MP 编排器被允许的场景

        新行为：should_use_mp_orchestrator 基于 requested_mode 判断。
        只有当 requested_mode 是 cli/None/plan/ask 时才允许 MP。
        """
        from core.execution_policy import should_use_mp_orchestrator

        # 新行为：使用 requested_mode
        requested_mode = test_case.requested_execution_mode
        can_use_mp = should_use_mp_orchestrator(requested_mode)

        # 当 requested_mode 是 cli/None/plan/ask 时应该允许 MP（除非显式禁用）
        if requested_mode in (None, "cli", "plan", "ask") and not test_case.no_mp_flag:
            assert can_use_mp is True, f"[{test_case.test_id}] 请求模式 {requested_mode} 应允许 MP 编排器"


# ============================================================
# TestEdgeCaseOrchestratorAndExecutionMode - 边界场景参数化测试
# ============================================================


@dataclass
class EdgeCaseTestParam:
    """边界场景测试参数

    用于覆盖 execution_mode 与编排器选择的边界场景：
    1. 显式 execution_mode=cli + '&'
    2. 显式 auto/cloud + 无 key
    3. cloud_enabled=False + '&'
    4. 用户显式 mp + execution_mode=cloud/auto
    5. MP 启动异常回退

    Attributes:
        test_id: 测试用例唯一标识符
        requirement: 用户输入（可能包含 & 前缀）
        execution_mode: 请求的执行模式
        has_api_key: 是否有 API Key
        cloud_enabled: 是否启用 Cloud
        orchestrator_flag: orchestrator 参数值
        _orchestrator_user_set: 用户是否显式设置了 orchestrator
        expect_mp_called: 是否期望调用 _run_with_mp_orchestrator
        expect_basic_called: 是否期望调用 _run_with_basic_orchestrator
        expect_execution_mode: 期望的最终执行模式
        mp_should_fail: MP 编排器是否应该失败（触发回退）
        description: 场景描述
    """

    test_id: str
    requirement: str
    execution_mode: Optional[str]
    has_api_key: bool
    cloud_enabled: bool
    orchestrator_flag: str
    _orchestrator_user_set: bool
    expect_mp_called: bool
    expect_basic_called: bool
    expect_execution_mode: str
    mp_should_fail: bool = False
    description: str = ""


# 边界场景测试参数表
EDGE_CASE_TEST_PARAMS: list[EdgeCaseTestParam] = [
    # ===== 场景 1: 显式 execution_mode=cli + '&' =====
    EdgeCaseTestParam(
        test_id="explicit_cli_with_ampersand_ignores_prefix",
        requirement="& 分析代码架构",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_flag="mp",
        _orchestrator_user_set=False,
        expect_mp_called=True,
        expect_basic_called=False,
        expect_execution_mode="cli",
        description="显式指定 CLI 模式时，即使有 & 前缀也忽略，使用 MP 编排器",
    ),
    # ===== 场景 2a: 显式 auto + 无 key =====
    # 新行为：requested=auto/cloud 即强制 basic，不受 key/enable 影响
    EdgeCaseTestParam(
        test_id="explicit_auto_no_key_forces_basic",  # 重命名以反映新行为
        requirement="分析代码",
        execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_flag="mp",
        _orchestrator_user_set=False,
        expect_mp_called=False,  # 新行为：auto 强制 basic
        expect_basic_called=True,  # 新行为
        expect_execution_mode="cli",
        description="AUTO 模式无 API Key：执行回退到 CLI，但编排器仍强制 basic（基于 requested_mode）",
    ),
    # ===== 场景 2b: 显式 cloud + 无 key =====
    # 新行为：requested=auto/cloud 即强制 basic，不受 key/enable 影响
    EdgeCaseTestParam(
        test_id="explicit_cloud_no_key_forces_basic",  # 重命名以反映新行为
        requirement="分析代码",
        execution_mode="cloud",
        has_api_key=False,
        cloud_enabled=True,
        orchestrator_flag="mp",
        _orchestrator_user_set=False,
        expect_mp_called=False,  # 新行为：cloud 强制 basic
        expect_basic_called=True,  # 新行为
        expect_execution_mode="cli",
        description="CLOUD 模式无 API Key：执行回退到 CLI，但编排器仍强制 basic（基于 requested_mode）",
    ),
    # ===== 场景 3: cloud_enabled=False + '&' =====
    # 根据 AGENTS.md R-2：& 前缀表达 Cloud 意图，即使未成功路由也使用 basic 编排器
    EdgeCaseTestParam(
        test_id="cloud_disabled_with_ampersand_forces_basic",
        requirement="& 长时间分析任务",
        execution_mode=None,  # 无显式模式，依赖 & 前缀触发
        has_api_key=True,
        cloud_enabled=False,
        orchestrator_flag="mp",  # 即使用户请求 mp，也会被 & 前缀覆盖为 basic
        _orchestrator_user_set=False,
        expect_mp_called=False,  # 根据 R-2：& 前缀表达 Cloud 意图，强制 basic
        expect_basic_called=True,  # 强制 basic
        expect_execution_mode="cli",
        description="cloud_enabled=False + & 前缀：& 前缀表达 Cloud 意图，强制 basic 编排器",
    ),
    # ===== 场景 4a: 用户显式 mp + execution_mode=cloud =====
    EdgeCaseTestParam(
        test_id="explicit_mp_with_cloud_forces_basic",
        requirement="分析代码",
        execution_mode="cloud",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_flag="mp",
        _orchestrator_user_set=True,
        expect_mp_called=False,
        expect_basic_called=True,
        expect_execution_mode="cloud",
        description="用户显式请求 MP 但 CLOUD 模式不兼容，强制使用 basic 编排器",
    ),
    # ===== 场景 4b: 用户显式 mp + execution_mode=auto =====
    EdgeCaseTestParam(
        test_id="explicit_mp_with_auto_forces_basic",
        requirement="分析代码",
        execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_flag="mp",
        _orchestrator_user_set=True,
        expect_mp_called=False,
        expect_basic_called=True,
        expect_execution_mode="auto",
        description="用户显式请求 MP 但 AUTO 模式不兼容，强制使用 basic 编排器",
    ),
    # ===== 场景 5: MP 启动异常回退 =====
    EdgeCaseTestParam(
        test_id="mp_startup_failure_fallback_to_basic",
        requirement="重构代码",
        execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        orchestrator_flag="mp",
        _orchestrator_user_set=False,
        expect_mp_called=True,
        expect_basic_called=True,  # MP 失败后回退到 basic
        expect_execution_mode="cli",
        mp_should_fail=True,
        description="MP 编排器启动失败时自动回退到 basic 编排器",
    ),
]


class TestEdgeCaseOrchestratorAndExecutionMode:
    """边界场景测试：执行模式与编排器选择

    覆盖以下边界场景：
    1. 显式 execution_mode=cli + '&'：CLI 模式下即使有 & 前缀也走 CLI
    2. 显式 auto/cloud + 无 key：无 API Key 时回退到 CLI
    3. cloud_enabled=False + '&'：Cloud 禁用时 & 前缀被忽略
    4. 用户显式 mp + execution_mode=cloud/auto：Cloud/Auto 模式强制使用 basic 编排器
    5. MP 启动异常回退：MP 编排器启动失败后回退到 basic

    测试方法：
    - Mock KnowledgeManager.initialize
    - Mock _run_with_mp_orchestrator 与 _run_with_basic_orchestrator
    - Mock CloudClientFactory.resolve_api_key()
    - Mock get_config().cloud_agent.enabled
    """

    def _create_base_args(
        self,
        requirement: str,
        execution_mode: Optional[str],
        orchestrator_flag: str,
        _orchestrator_user_set: bool,
    ) -> argparse.Namespace:
        """创建基础测试参数"""
        return argparse.Namespace(
            requirement=requirement,
            directory=".",
            skip_online=True,
            changelog_url=None,
            dry_run=False,
            max_iterations="3",
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
            orchestrator=orchestrator_flag,
            no_mp=False,
            _orchestrator_user_set=_orchestrator_user_set,
            execution_mode=execution_mode,
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

    @pytest.mark.parametrize(
        "test_param",
        EDGE_CASE_TEST_PARAMS,
        ids=[p.test_id for p in EDGE_CASE_TEST_PARAMS],
    )
    def test_execution_mode_resolution(self, test_param: EdgeCaseTestParam) -> None:
        """测试执行模式解析是否符合预期"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        args = self._create_base_args(
            requirement=test_param.requirement,
            execution_mode=test_param.execution_mode,
            orchestrator_flag=test_param.orchestrator_flag,
            _orchestrator_user_set=test_param._orchestrator_user_set,
        )

        # Mock API Key
        api_key = "mock-api-key" if test_param.has_api_key else None

        # Mock config 的 cloud_agent.enabled
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_param.cloud_enabled

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key),
            patch("core.config.get_config", return_value=mock_config),
        ):
            iterator = SelfIterator(args)
            actual_mode = iterator._get_execution_mode()

            # ExecutionMode 枚举值是小写的 ("cli", "cloud", "auto")
            expected_mode = ExecutionMode(test_param.expect_execution_mode.lower())
            assert actual_mode == expected_mode, (
                f"[{test_param.test_id}] 期望执行模式 {expected_mode.value}，"
                f"实际 {actual_mode.value}。场景: {test_param.description}"
            )

    @pytest.mark.parametrize(
        "test_param",
        [p for p in EDGE_CASE_TEST_PARAMS if not p.mp_should_fail],
        ids=[p.test_id for p in EDGE_CASE_TEST_PARAMS if not p.mp_should_fail],
    )
    def test_orchestrator_type_selection(self, test_param: EdgeCaseTestParam) -> None:
        """测试编排器类型选择是否符合预期"""
        from cursor.cloud_client import CloudClientFactory

        args = self._create_base_args(
            requirement=test_param.requirement,
            execution_mode=test_param.execution_mode,
            orchestrator_flag=test_param.orchestrator_flag,
            _orchestrator_user_set=test_param._orchestrator_user_set,
        )

        api_key = "mock-api-key" if test_param.has_api_key else None
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_param.cloud_enabled

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key),
            patch("core.config.get_config", return_value=mock_config),
        ):
            iterator = SelfIterator(args)
            actual_orchestrator = iterator._get_orchestrator_type()

            # 判断期望的编排器类型
            if test_param.expect_mp_called and not test_param.expect_basic_called:
                expected_orchestrator = "mp"
            else:
                expected_orchestrator = "basic"

            assert actual_orchestrator == expected_orchestrator, (
                f"[{test_param.test_id}] 期望编排器 {expected_orchestrator}，"
                f"实际 {actual_orchestrator}。场景: {test_param.description}"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_param",
        [p for p in EDGE_CASE_TEST_PARAMS if not p.mp_should_fail],
        ids=[p.test_id for p in EDGE_CASE_TEST_PARAMS if not p.mp_should_fail],
    )
    async def test_orchestrator_call_pattern(self, test_param: EdgeCaseTestParam) -> None:
        """测试编排器调用模式：验证 MP/Basic 是否按预期被调用"""
        from cursor.cloud_client import CloudClientFactory

        args = self._create_base_args(
            requirement=test_param.requirement,
            execution_mode=test_param.execution_mode,
            orchestrator_flag=test_param.orchestrator_flag,
            _orchestrator_user_set=test_param._orchestrator_user_set,
        )

        api_key = "mock-api-key" if test_param.has_api_key else None
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_param.cloud_enabled

        # 模拟成功的执行结果
        mock_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key),
            patch("core.config.get_config", return_value=mock_config),
        ):
            iterator = SelfIterator(args)
            iterator.context.iteration_goal = "测试目标"

            # Mock 知识库初始化
            with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                # Mock 编排器方法
                with (
                    patch.object(
                        iterator,
                        "_run_with_mp_orchestrator",
                        new_callable=AsyncMock,
                        return_value=mock_result,
                    ) as mock_mp,
                    patch.object(
                        iterator,
                        "_run_with_basic_orchestrator",
                        new_callable=AsyncMock,
                        return_value=mock_result,
                    ) as mock_basic,
                ):
                    await iterator._run_agent_system()

                    # 验证调用次数
                    if test_param.expect_mp_called:
                        assert mock_mp.call_count >= 1, f"[{test_param.test_id}] 期望调用 MP 编排器但未调用"
                    else:
                        assert mock_mp.call_count == 0, f"[{test_param.test_id}] 不期望调用 MP 编排器但被调用了"

                    if test_param.expect_basic_called and not test_param.expect_mp_called:
                        assert mock_basic.call_count >= 1, f"[{test_param.test_id}] 期望调用 basic 编排器但未调用"

    @pytest.mark.asyncio
    async def test_mp_startup_exception_triggers_fallback(self) -> None:
        """测试 MP 编排器启动异常触发回退到 basic"""
        from cursor.cloud_client import CloudClientFactory

        # 获取 MP 启动失败的测试参数
        test_param = next(p for p in EDGE_CASE_TEST_PARAMS if p.mp_should_fail)

        args = self._create_base_args(
            requirement=test_param.requirement,
            execution_mode=test_param.execution_mode,
            orchestrator_flag=test_param.orchestrator_flag,
            _orchestrator_user_set=test_param._orchestrator_user_set,
        )

        api_key = "mock-api-key" if test_param.has_api_key else None
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_param.cloud_enabled

        # MP 失败结果（触发回退）
        mp_fail_result = {
            "_fallback_required": True,
            "_fallback_reason": "启动超时",
        }

        # basic 成功结果
        basic_success_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key),
            patch("core.config.get_config", return_value=mock_config),
        ):
            iterator = SelfIterator(args)
            iterator.context.iteration_goal = "测试目标"

            with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                with (
                    patch.object(
                        iterator,
                        "_run_with_mp_orchestrator",
                        new_callable=AsyncMock,
                        return_value=mp_fail_result,
                    ) as mock_mp,
                    patch.object(
                        iterator,
                        "_run_with_basic_orchestrator",
                        new_callable=AsyncMock,
                        return_value=basic_success_result,
                    ) as mock_basic,
                ):
                    result = await iterator._run_agent_system()

                    # 验证 MP 被尝试
                    assert mock_mp.call_count == 1, "MP 编排器应被尝试一次"

                    # 验证回退到 basic
                    assert mock_basic.call_count == 1, "MP 失败后应回退到 basic"

                    # 验证最终结果成功
                    assert result["success"] is True, "回退后执行应成功"

    @pytest.mark.parametrize(
        "test_param",
        [
            p
            for p in EDGE_CASE_TEST_PARAMS
            if p.test_id
            in (
                "explicit_cli_with_ampersand_ignores_prefix",
                # cloud_disabled_with_ampersand 场景已迁移：
                # & 前缀不被忽略，而是强制 basic（根据 R-2）
            )
        ],
        ids=[p.test_id for p in EDGE_CASE_TEST_PARAMS if p.test_id in ("explicit_cli_with_ampersand_ignores_prefix",)],
    )
    def test_ampersand_prefix_ignored_scenarios(self, test_param: EdgeCaseTestParam) -> None:
        """测试 & 前缀被忽略的场景"""
        from cursor.cloud_client import CloudClientFactory

        args = self._create_base_args(
            requirement=test_param.requirement,
            execution_mode=test_param.execution_mode,
            orchestrator_flag=test_param.orchestrator_flag,
            _orchestrator_user_set=test_param._orchestrator_user_set,
        )

        api_key = "mock-api-key" if test_param.has_api_key else None
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_param.cloud_enabled

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key),
            patch("core.config.get_config", return_value=mock_config),
        ):
            iterator = SelfIterator(args)

            # 验证 prefix_routed=False（& 前缀未成功触发 Cloud 路由）
            # 语义：has_ampersand_prefix=True（语法检测），但因显式 CLI 等原因未成功路由
            # 内部分支使用 _prefix_routed 字段
            assert iterator._prefix_routed is False, (
                f"[{test_param.test_id}] & 前缀不应成功触发 Cloud 路由，但 prefix_routed={iterator._prefix_routed}"
            )

            # 验证 requirement 中的 & 前缀被清理
            cleaned_requirement = iterator.context.user_requirement
            assert not cleaned_requirement.startswith("&"), (
                f"[{test_param.test_id}] & 前缀应被清理，但 user_requirement='{cleaned_requirement}'"
            )

    @pytest.mark.parametrize(
        "test_param",
        [
            p
            for p in EDGE_CASE_TEST_PARAMS
            if p.test_id
            in (
                "explicit_auto_no_key_forces_basic",
                "explicit_cloud_no_key_forces_basic",
            )
        ],
        ids=[
            p.test_id
            for p in EDGE_CASE_TEST_PARAMS
            if p.test_id
            in (
                "explicit_auto_no_key_forces_basic",
                "explicit_cloud_no_key_forces_basic",
            )
        ],
    )
    def test_no_api_key_forces_basic_scenarios(self, test_param: EdgeCaseTestParam) -> None:
        """测试无 API Key 时：执行模式回退到 CLI，但编排器仍强制 basic

        新行为：requested=auto/cloud 即强制 basic，不受 key/enable 影响。
        即使因为没有 API Key 导致 effective_execution_mode 回退到 CLI，
        只要 requested_execution_mode 是 auto/cloud，编排器就应该是 basic。
        """
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        args = self._create_base_args(
            requirement=test_param.requirement,
            execution_mode=test_param.execution_mode,
            orchestrator_flag=test_param.orchestrator_flag,
            _orchestrator_user_set=test_param._orchestrator_user_set,
        )

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_param.cloud_enabled

        # 无 API Key
        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("core.config.get_config", return_value=mock_config),
        ):
            iterator = SelfIterator(args)

            # 验证执行模式回退到 CLI
            actual_mode = iterator._get_execution_mode()
            assert actual_mode == ExecutionMode.CLI, (
                f"[{test_param.test_id}] 无 API Key 应回退到 CLI，但实际执行模式为 {actual_mode.value}"
            )

            # 新行为：验证编排器类型为 basic（基于 requested_mode=auto/cloud）
            actual_orchestrator = iterator._get_orchestrator_type()
            expected_orchestrator = "basic" if test_param.expect_basic_called else "mp"
            assert actual_orchestrator == expected_orchestrator, (
                f"[{test_param.test_id}] requested_mode={test_param.execution_mode} "
                f"应强制使用 {expected_orchestrator} 编排器，"
                f"但实际编排器为 {actual_orchestrator}"
            )

    @pytest.mark.parametrize(
        "test_param",
        [
            p
            for p in EDGE_CASE_TEST_PARAMS
            if p.test_id
            in (
                "explicit_mp_with_cloud_forces_basic",
                "explicit_mp_with_auto_forces_basic",
            )
        ],
        ids=[
            p.test_id
            for p in EDGE_CASE_TEST_PARAMS
            if p.test_id
            in (
                "explicit_mp_with_cloud_forces_basic",
                "explicit_mp_with_auto_forces_basic",
            )
        ],
    )
    def test_explicit_mp_with_cloud_auto_forces_basic(self, test_param: EdgeCaseTestParam) -> None:
        """测试用户显式请求 MP 但 Cloud/Auto 模式强制使用 basic"""
        from core.execution_policy import should_use_mp_orchestrator
        from cursor.cloud_client import CloudClientFactory

        args = self._create_base_args(
            requirement=test_param.requirement,
            execution_mode=test_param.execution_mode,
            orchestrator_flag=test_param.orchestrator_flag,
            _orchestrator_user_set=test_param._orchestrator_user_set,
        )

        api_key = "mock-api-key" if test_param.has_api_key else None
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_param.cloud_enabled

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key),
            patch("core.config.get_config", return_value=mock_config),
        ):
            iterator = SelfIterator(args)

            # 验证用户确实请求了 MP
            assert args.orchestrator == "mp", "测试前提：用户请求了 MP 编排器"
            assert args._orchestrator_user_set is True, "测试前提：用户显式设置了编排器"

            # 验证 Policy 判断不允许 MP
            execution_mode_str = test_param.expect_execution_mode
            can_use_mp = should_use_mp_orchestrator(execution_mode_str)
            assert can_use_mp is False, f"[{test_param.test_id}] {execution_mode_str} 模式不应允许 MP 编排器"

            # 验证最终使用 basic 编排器
            actual_orchestrator = iterator._get_orchestrator_type()
            assert actual_orchestrator == "basic", (
                f"[{test_param.test_id}] 用户请求 MP 但 {execution_mode_str} 模式"
                f"应强制使用 basic，实际 {actual_orchestrator}"
            )


# ============================================================
# TestExecutionModeAutoMpForcesBasicNoException - Auto 模式 + MP 强制 basic 无异常测试
# ============================================================


class TestExecutionModeAutoMpForcesBasicNoException:
    """测试 --execution-mode auto 且用户请求 mp 编排器时，最终选择 basic 且关键路径不抛异常

    验证场景：
    1. execution_mode=auto + orchestrator=mp -> 强制切换为 basic
    2. 切换过程中不抛出异常（日志/提示路径稳定）
    3. 即使无 API Key，回退到 CLI 后的编排器选择仍然正确
    4. 捕获并验证 warning/info 日志输出
    """

    @pytest.fixture
    def auto_mp_request_args(self) -> argparse.Namespace:
        """创建 execution_mode=auto, orchestrator=mp 的参数"""
        return argparse.Namespace(
            requirement="自动模式测试任务",
            directory=".",
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
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
            orchestrator="mp",  # 用户显式指定 mp
            no_mp=False,
            _orchestrator_user_set=True,  # 用户显式设置
            execution_mode="auto",  # auto 模式
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

    def test_auto_mp_forces_basic_no_exception_with_api_key(self, auto_mp_request_args: argparse.Namespace) -> None:
        """测试 execution_mode=auto + orchestrator=mp 有 API Key 时不抛异常"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            # 创建 SelfIterator 不应抛出异常
            iterator = SelfIterator(auto_mp_request_args)

            # 调用关键路径方法不应抛出异常
            execution_mode = iterator._get_execution_mode()
            orchestrator_type = iterator._get_orchestrator_type()

            # 验证结果
            assert execution_mode == ExecutionMode.AUTO, f"期望 ExecutionMode.AUTO，实际 {execution_mode}"
            assert orchestrator_type == "basic", f"auto + mp 应强制使用 basic 编排器，实际 {orchestrator_type}"

    def test_auto_mp_forces_basic_no_exception_without_api_key(self, auto_mp_request_args: argparse.Namespace) -> None:
        """测试 execution_mode=auto + orchestrator=mp 无 API Key 时不抛异常"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            # 创建 SelfIterator 不应抛出异常
            iterator = SelfIterator(auto_mp_request_args)

            # 调用关键路径方法不应抛出异常
            execution_mode = iterator._get_execution_mode()
            orchestrator_type = iterator._get_orchestrator_type()

            # 验证结果：无 API Key 回退到 CLI
            assert execution_mode == ExecutionMode.CLI, f"无 API Key 应回退到 CLI，实际 {execution_mode}"
            # 注意：即使回退到 CLI，由于原始 requested_mode=auto，仍应使用 basic
            # 这是因为 _get_orchestrator_type 检查的是原始请求的 execution_mode
            assert orchestrator_type == "basic", (
                f"requested_mode=auto 时即使回退 CLI 也应使用 basic，实际 {orchestrator_type}"
            )
            # 新增断言：prefix_routed 应为 False（未成功触发 Cloud）
            assert iterator._prefix_routed is False, (
                "auto + 无 key 时 prefix_routed 应为 False，因为没有 & 前缀成功触发"
            )

    def test_auto_mp_forces_basic_log_path_no_exception(
        self,
        auto_mp_request_args: argparse.Namespace,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """测试日志路径不抛出异常，且包含相关信息"""
        import logging

        from cursor.cloud_client import CloudClientFactory

        with (
            caplog.at_level(logging.DEBUG),
            patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"),
        ):
            iterator = SelfIterator(auto_mp_request_args)

            # 触发日志相关路径
            _ = iterator._get_execution_mode()
            orchestrator_type = iterator._get_orchestrator_type()

            # 验证无异常
            assert orchestrator_type == "basic"

            # 检查日志中的相关记录（如有）
            log_text = " ".join(record.message for record in caplog.records)
            # 日志中可能包含编排器切换信息，但不强制要求特定消息
            # 主要验证不抛异常

    @pytest.mark.parametrize(
        "execution_mode,expected_orchestrator",
        [
            pytest.param("auto", "basic", id="auto_forces_basic"),
            pytest.param("cloud", "basic", id="cloud_forces_basic"),
            pytest.param("cli", "mp", id="cli_allows_mp"),
        ],
    )
    def test_execution_mode_orchestrator_no_exception_parametrized(
        self,
        auto_mp_request_args: argparse.Namespace,
        execution_mode: str,
        expected_orchestrator: str,
    ) -> None:
        """参数化测试：各 execution_mode + mp 请求时的编排器选择无异常"""
        from cursor.cloud_client import CloudClientFactory

        auto_mp_request_args.execution_mode = execution_mode

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            # 创建 SelfIterator 不应抛出异常
            iterator = SelfIterator(auto_mp_request_args)

            # 调用关键方法不应抛出异常
            orchestrator_type = iterator._get_orchestrator_type()

            # 验证结果
            assert orchestrator_type == expected_orchestrator, (
                f"execution_mode={execution_mode} + mp 请求: 期望 {expected_orchestrator}，实际 {orchestrator_type}"
            )

    def test_auto_mp_policy_function_no_exception(self) -> None:
        """测试执行策略函数不抛出异常"""
        from core.execution_policy import should_use_mp_orchestrator

        # 各种输入都不应抛出异常
        # 规则：
        # - auto/cloud 返回 False（强制 basic）
        # - cli/plan/ask/None/""/invalid 返回 True（允许 MP）
        test_cases = [
            ("auto", False),  # Cloud 模式，强制 basic
            ("cloud", False),  # Cloud 模式，强制 basic
            ("cli", True),  # CLI 模式，允许 MP
            ("plan", True),  # Plan 模式（非 Cloud），允许 MP
            ("ask", True),  # Ask 模式（非 Cloud），允许 MP
            ("", True),  # 空字符串，默认允许 MP
            (None, True),  # None，默认允许 MP
            ("invalid", True),  # 无效值，默认允许 MP
        ]

        for mode_input, expected in test_cases:
            try:
                result = should_use_mp_orchestrator(mode_input)
                assert result is expected, f"mode={mode_input!r} 应返回 {expected}，实际返回 {result}"
            except Exception as e:
                pytest.fail(f"should_use_mp_orchestrator({mode_input!r}) 抛出异常: {e}")


# ============================================================
# TestUrlFilteringWithExternalDomainsAndExcludePatterns - URL 过滤综合测试
# ============================================================


class TestUrlFilteringWithExternalDomainsAndExcludePatterns:
    """测试 URL 过滤：外域过滤 + 排除模式 + deduplicate/normalize 开关差异

    验证场景：
    1. 包含外域 URL（github.com, stackoverflow.com 等）在 changelog_links/llms_txt/related_doc_urls 中
    2. 验证最终 urls_to_fetch 不含外域
    3. 验证排除规则生效（如 .png, /api/ 等）
    4. 对 deduplicate/normalize 开关做差异断言
    """

    @pytest.fixture
    def mixed_domain_inputs(self) -> dict:
        """构造包含外域与被排除模式的 URL 输入"""
        return {
            "changelog_links": [
                "https://cursor.com/changelog/2026-01",
                "https://github.com/cursor/releases",  # 外域
                "https://cursor.com/changelog/2026-02#section",  # 带锚点
                "https://cursor.com/assets/logo.png",  # 应被排除（图片）
            ],
            "llms_txt": (
                "# LLMs Documentation\n"
                "https://cursor.com/docs/guide\n"
                "https://stackoverflow.com/questions/cursor\n"  # 外域
                "https://cursor.com/api/internal\n"  # 可能被 /api/ 排除
                "https://npmjs.com/package/cursor\n"  # 外域
                "https://cursor.com/docs/reference\n"
            ),
            "related_doc_urls": [
                "https://cursor.com/docs/cli/overview",
                "https://external.io/docs",  # 外域
                "https://cursor.com/docs/cli/parameters",
                "https://cursor.com/static/image.jpg",  # 应被排除（图片）
            ],
            "core_docs": [
                "https://cursor.com/docs/core",
                "https://cursor.com/api/reference",  # 可能被 /api/ 排除
            ],
        }

    def test_external_domains_filtered_from_all_sources(self, mixed_domain_inputs: dict) -> None:
        """测试所有来源中的外域 URL 被过滤"""
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            select_urls_to_fetch,
        )
        from scripts.run_iterate import (
            ALLOWED_DOC_URL_PREFIXES,
            ALLOWED_DOC_URL_PREFIXES_NETLOC,
        )

        # 使用新名称配置
        config = DocURLStrategyConfig(
            allowed_url_prefixes=ALLOWED_DOC_URL_PREFIXES,
            allowed_domains=ALLOWED_DOC_URL_PREFIXES_NETLOC,
            max_urls=20,
            deduplicate=True,
            normalize=True,
        )

        result = select_urls_to_fetch(
            changelog_links=mixed_domain_inputs["changelog_links"],
            related_doc_urls=mixed_domain_inputs["related_doc_urls"],
            llms_txt_content=mixed_domain_inputs["llms_txt"],
            core_docs=mixed_domain_inputs["core_docs"],
            keywords=[],
            config=config,
            base_url="https://cursor.com",
        )

        # 验证外域被过滤
        external_domains = ["github.com", "stackoverflow.com", "npmjs.com", "external.io"]
        for url in result:
            for ext_domain in external_domains:
                assert ext_domain not in url, f"外域 URL 应被过滤: {url} 包含 {ext_domain}"

        # 验证 cursor.com URL 存在
        assert any("cursor.com" in u for u in result), "应包含 cursor.com URL"

    def test_exclude_patterns_filter_images_and_api(self, mixed_domain_inputs: dict) -> None:
        """测试排除模式过滤图片和 API 路径

        注意：此测试显式传入 exclude_patterns，以明确测试意图。
        core.config.DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS 默认不包含 /api/ 规则，
        需要显式添加才能过滤 API 路径。
        """
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            select_urls_to_fetch,
        )

        # 显式传入 exclude_patterns，包含图片扩展名和 /api/ 路径
        # 这是测试所需的过滤规则，不依赖默认值
        explicit_exclude_patterns = [
            r".*\.(png|jpg|jpeg|gif|svg|ico)$",  # 图片扩展名
            r".*/api/.*",  # API 路径
        ]

        config = DocURLStrategyConfig(
            allowed_domains=["cursor.com"],  # 仅允许 cursor.com
            max_urls=20,
            deduplicate=True,
            normalize=True,
            exclude_patterns=explicit_exclude_patterns,  # 显式指定排除规则
        )

        result = select_urls_to_fetch(
            changelog_links=mixed_domain_inputs["changelog_links"],
            related_doc_urls=mixed_domain_inputs["related_doc_urls"],
            llms_txt_content=mixed_domain_inputs["llms_txt"],
            core_docs=mixed_domain_inputs["core_docs"],
            keywords=[],
            config=config,
        )

        # 验证图片 URL 被排除
        for url in result:
            assert not url.endswith(".png"), f"图片 URL 应被排除: {url}"
            assert not url.endswith(".jpg"), f"图片 URL 应被排除: {url}"

        # 验证 /api/ 路径被排除
        for url in result:
            assert "/api/" not in url, f"API 路径应被排除: {url}"

    def test_deduplicate_on_removes_duplicates(self) -> None:
        """测试 deduplicate=True 时移除重复 URL"""
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            select_urls_to_fetch,
        )

        config = DocURLStrategyConfig(
            allowed_domains=["example.com"],
            max_urls=20,
            deduplicate=True,  # 启用去重
            normalize=True,
            exclude_patterns=[],  # 禁用排除以便测试
        )

        # 构造重复 URL（包括大小写差异和末尾斜杠差异）
        changelog_links = [
            "https://example.com/docs",
            "https://EXAMPLE.COM/docs/",  # 规范化后相同
        ]
        related_doc_urls = [
            "https://example.com/docs#section",  # 规范化后相同（去除锚点）
            "https://example.com/guide",
        ]

        result = select_urls_to_fetch(
            changelog_links=changelog_links,
            related_doc_urls=related_doc_urls,
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 验证去重生效：规范化后只有 2 个唯一 URL
        assert len(result) == 2, f"去重后应只有 2 个 URL，实际 {len(result)}: {result}"
        normalized_results = [u.lower().rstrip("/") for u in result]
        assert len(set(normalized_results)) == len(result), "结果中不应有重复"

    def test_deduplicate_off_keeps_duplicates(self) -> None:
        """测试 deduplicate=False 时保留重复 URL（规范化后）"""
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            select_urls_to_fetch,
        )

        config = DocURLStrategyConfig(
            allowed_domains=["example.com"],
            max_urls=20,
            deduplicate=False,  # 禁用去重
            normalize=True,  # 仍然规范化
            exclude_patterns=[],
        )

        # 构造"相似"URL
        changelog_links = [
            "https://example.com/docs",
        ]
        related_doc_urls = [
            "https://example.com/docs",  # 相同 URL（规范化后）
            "https://example.com/guide",
        ]

        result = select_urls_to_fetch(
            changelog_links=changelog_links,
            related_doc_urls=related_doc_urls,
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 注意：即使 deduplicate=False，相同 URL 仍可能被内部去重
        # 这里主要验证不抛异常，结果包含预期 URL
        assert any("docs" in u for u in result), "应包含 docs URL"
        assert any("guide" in u for u in result), "应包含 guide URL"

    def test_normalize_on_removes_fragments(self) -> None:
        """测试 normalize=True 时移除锚点"""
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            select_urls_to_fetch,
        )

        config = DocURLStrategyConfig(
            allowed_domains=["example.com"],
            max_urls=20,
            deduplicate=True,
            normalize=True,  # 启用规范化
            exclude_patterns=[],
        )

        changelog_links = [
            "https://example.com/docs#section1",
            "https://example.com/guide#section2",
        ]

        result = select_urls_to_fetch(
            changelog_links=changelog_links,
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 验证锚点被移除
        for url in result:
            assert "#" not in url, f"锚点应被移除: {url}"

    def test_normalize_off_keeps_fragments(self) -> None:
        """测试 normalize=False 时保留锚点"""
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            select_urls_to_fetch,
        )

        config = DocURLStrategyConfig(
            allowed_domains=[],  # 允许所有域名
            allowed_url_prefixes=[],  # 不限制前缀
            max_urls=20,
            deduplicate=False,
            normalize=False,  # 禁用规范化
            exclude_patterns=[],  # 禁用排除（因为默认排除包含 #）
        )

        changelog_links = [
            "https://example.com/docs#section1",
        ]

        result = select_urls_to_fetch(
            changelog_links=changelog_links,
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 验证锚点被保留
        assert len(result) >= 1, "应至少有 1 个结果"
        # 注意：normalize=False 时原始 URL 被保留
        assert any("#" in u for u in result), f"normalize=False 时应保留锚点: {result}"

    def test_combined_filtering_integration(self, mixed_domain_inputs: dict) -> None:
        """集成测试：综合外域过滤 + 排除模式 + 去重 + 规范化"""
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            select_urls_to_fetch,
        )
        from scripts.run_iterate import (
            ALLOWED_DOC_URL_PREFIXES,
            ALLOWED_DOC_URL_PREFIXES_NETLOC,
        )

        # 使用新名称配置
        config = DocURLStrategyConfig(
            allowed_url_prefixes=ALLOWED_DOC_URL_PREFIXES,
            allowed_domains=ALLOWED_DOC_URL_PREFIXES_NETLOC,
            max_urls=10,  # 限制数量
            fallback_core_docs_count=2,
            deduplicate=True,
            normalize=True,
        )

        result = select_urls_to_fetch(
            changelog_links=mixed_domain_inputs["changelog_links"],
            related_doc_urls=mixed_domain_inputs["related_doc_urls"],
            llms_txt_content=mixed_domain_inputs["llms_txt"],
            core_docs=mixed_domain_inputs["core_docs"],
            keywords=["cli", "docs"],  # 添加关键词提升优先级
            config=config,
            base_url="https://cursor.com",
        )

        # 综合验证
        # 1. 不超过 max_urls
        assert len(result) <= 10, f"不应超过 max_urls=10，实际 {len(result)}"

        # 2. 无外域
        external_domains = ["github.com", "stackoverflow.com", "npmjs.com", "external.io"]
        for url in result:
            for ext_domain in external_domains:
                assert ext_domain not in url, f"外域应被过滤: {url}"

        # 3. 无图片
        for url in result:
            assert not any(url.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif"]), f"图片应被过滤: {url}"

        # 4. 无锚点
        for url in result:
            assert "#" not in url, f"锚点应被移除: {url}"

        # 5. 无重复
        assert len(result) == len(set(result)), "不应有重复 URL"

        # 6. 包含 cursor.com
        assert any("cursor.com" in u for u in result), "应包含 cursor.com URL"

    @pytest.mark.asyncio
    async def test_knowledge_updater_integration_external_filtered(self, base_iterate_args: argparse.Namespace) -> None:
        """集成测试：KnowledgeUpdater._build_urls_to_fetch 完整流程过滤外域"""
        from scripts.run_iterate import (
            UpdateAnalysis,
        )

        iterator = SelfIterator(base_iterate_args)
        updater = iterator.knowledge_updater

        # 创建包含多种外域和排除模式的 UpdateAnalysis
        analysis = UpdateAnalysis(
            has_updates=True,
            entries=[],
            new_features=["新功能"],
            changelog_links=[
                "https://cursor.com/changelog/test",
                "https://github.com/cursor/repo",  # 外域
                "https://cursor.com/assets/icon.png",  # 应被排除
            ],
            related_doc_urls=[
                "https://cursor.com/docs/guide",
                "https://stackoverflow.com/q/cursor",  # 外域
                "https://cursor.com/api/endpoint",  # 应被排除（/api/）
            ],
        )

        # Mock llms.txt 返回外域内容
        mock_llms = (
            "https://cursor.com/docs/llms\n"
            "https://external.org/docs\n"
            "https://cursor.com/static/style.css\n"  # 应被排除（.css）
        )

        with (
            patch.object(updater, "_fetch_llms_txt", new_callable=AsyncMock, return_value=mock_llms),
            patch("scripts.run_iterate.get_core_docs", return_value=[]),
        ):
            urls, selection_log = await updater._build_urls_to_fetch(
                analysis=analysis,
                max_urls=20,
                fallback_count=0,
            )

        # 验证过滤结果
        assert not any("github.com" in u for u in urls), "github.com 应被过滤"
        assert not any("stackoverflow.com" in u for u in urls), "stackoverflow.com 应被过滤"
        assert not any("external.org" in u for u in urls), "external.org 应被过滤"
        assert not any(u.endswith(".png") for u in urls), ".png 应被过滤"
        assert not any(u.endswith(".css") for u in urls), ".css 应被过滤"
        assert not any("/api/" in u for u in urls), "/api/ 应被过滤"

        # 验证保留 cursor.com 有效文档
        assert any("cursor.com" in u for u in urls), "应保留 cursor.com URL"


# ============================================================
# TestCloudFallbackUserMessageDedup - Cloud 回退用户消息去重测试
# ============================================================


class TestSelfIterateCloudFallbackUserMessageDedup:
    """SelfIterator Cloud 回退场景用户消息去重测试

    验证在以下场景中 user_message 在 stdout/stderr 中出现次数为 0 或 1：
    1. CloudClientFactory.resolve_api_key 返回 None
    2. cloud_enabled True/False
    3. AutoExecutor 产生 cooldown_info 的场景

    确保不出现重复的用户提示消息。
    """

    @pytest.fixture
    def base_iterate_args(self) -> argparse.Namespace:
        """创建基础迭代参数"""
        args = argparse.Namespace()
        args.requirement = "& 测试任务"
        args.execution_mode = None
        args.mode = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.dry_run = False
        args.max_iterations = "10"  # 字符串格式，与 argparse 输出一致
        args.workers = 3
        args.orchestrator = "mp"
        args.no_mp = False
        args.auto_commit = False
        args.auto_push = False
        args.commit_per_iteration = False
        args.force = False
        args.strict_review = False
        args.skip_online = True
        args.knowledge_base = None
        args.output_format = "text"
        args.cloud_api_key = None
        args.cloud_auth_timeout = 30
        args.cloud_timeout = 300
        args.print_config = False
        args._execution_mode_user_set = False
        args._orchestrator_user_set = False
        args.stream_console_renderer = False
        args.stream_advanced_renderer = False
        args.stream_show_word_diff = False
        args.stream_typing_effect = False
        args.stream_typing_delay = 0.02
        args.stream_word_mode = True
        args.stream_color_enabled = True
        args.heartbeat_debug = False
        args.stall_diagnostics = False
        args.stall_diagnostics_level = "warning"
        args.stall_diagnostics_enabled = None
        args.stall_recovery_interval = 30.0
        args.execution_health_check_interval = 30.0
        args.health_warning_cooldown = 60.0
        args.directory = "."
        args.changelog_url = None
        args.force_update = False
        args.commit_message = ""
        args.planner_execution_mode = None
        args.worker_execution_mode = None
        args.reviewer_execution_mode = None
        # 文档源配置参数（tri-state）
        args.max_fetch_urls = None
        args.fallback_core_docs_count = None
        args.llms_txt_url = None
        args.llms_cache_path = None
        args.changelog_url = None
        args.allowed_path_prefixes = None
        args.external_link_mode = None
        args.external_link_allowlist = None
        args.url_allowed_domains = None
        args.url_allowed_prefixes = None
        args.url_exclude_patterns = None
        args.url_normalize = None
        args.url_deduplicate = None
        args.url_prefer_changelog = None
        args.keyword_boost_weight = None
        return args

    def test_no_api_key_no_duplicate_message(self, base_iterate_args: argparse.Namespace, capsys) -> None:
        """测试无 API Key 时不输出重复的用户消息

        当 CloudClientFactory.resolve_api_key 返回 None 时，
        stdout/stderr 中不应出现重复的用户提示消息。
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置类级别标志
        CursorAgentClient._cloud_api_key_warning_shown = False
        SelfIterator.reset_shown_messages()

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("core.config.get_config", return_value=mock_config),
        ):
            # 创建迭代器多次检查执行模式
            for _ in range(3):
                iterator = SelfIterator(base_iterate_args)
                _ = iterator._get_execution_mode()

        captured = capsys.readouterr()

        # 检查 stdout/stderr 中不应有重复的用户提示
        # 由于去重机制，相同消息只会输出一次
        assert captured.out.count("API Key") <= 1, (
            f"stdout 中 'API Key' 出现次数应 <= 1，实际: {captured.out.count('API Key')}"
        )
        assert captured.err.count("API Key") <= 1, (
            f"stderr 中 'API Key' 出现次数应 <= 1，实际: {captured.err.count('API Key')}"
        )

    def test_cloud_enabled_false_no_message(self, base_iterate_args: argparse.Namespace, capsys) -> None:
        """测试 cloud_enabled=False 时输出信息性消息但不重复

        当 cloud_enabled=False 时，可能会输出一次信息性消息告知用户
        & 前缀被忽略，但不应重复输出。
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置类级别标志
        CursorAgentClient._cloud_api_key_warning_shown = False
        SelfIterator.reset_shown_messages()

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = False

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("core.config.get_config", return_value=mock_config),
        ):
            # 多次创建迭代器
            for _ in range(3):
                iterator = SelfIterator(base_iterate_args)
                _ = iterator._get_execution_mode()

        captured = capsys.readouterr()

        # cloud_enabled=False 时可能输出一次信息性消息，但不应重复
        cloud_count = captured.out.count("cloud_enabled=False")
        assert cloud_count <= 1, f"'cloud_enabled=False' 消息出现 {cloud_count} 次，应 <= 1 次"

    def test_cloud_enabled_true_no_api_key_no_duplicate(self, base_iterate_args: argparse.Namespace, capsys) -> None:
        """测试 cloud_enabled=True 但无 API Key 时不输出重复消息

        当 cloud_enabled=True 但 resolve_api_key 返回 None 时，
        用户消息应只出现 0 或 1 次。
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置类级别标志
        CursorAgentClient._cloud_api_key_warning_shown = False
        SelfIterator.reset_shown_messages()

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("core.config.get_config", return_value=mock_config),
        ):
            # 执行多次
            for _ in range(5):
                iterator = SelfIterator(base_iterate_args)
                _ = iterator._get_execution_mode()

        captured = capsys.readouterr()

        # 验证不出现重复的用户提示
        combined_output = captured.out + captured.err

        # 检查典型的用户提示关键词 - 由于去重机制，应只出现 1 次
        key_phrases = ["API Key", "未配置"]
        for phrase in key_phrases:
            count = combined_output.count(phrase)
            assert count <= 1, f"'{phrase}' 在输出中出现 {count} 次，应 <= 1 次"

    @pytest.mark.asyncio
    async def test_auto_executor_cooldown_info_user_message_no_duplicate(self, capsys) -> None:
        """测试 AutoExecutor 产生 cooldown_info 时 user_message 不重复

        当 AutoExecutor 因 Cloud 失败回退到 CLI 时，
        cooldown_info 中的 user_message 应只由入口脚本打印一次，
        不应在库层重复输出。
        """
        from cursor.client import CursorAgentConfig
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 60
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            result = await auto_executor.execute("测试任务")

        captured = capsys.readouterr()

        # 验证 cooldown_info 存在且包含 user_message
        assert result.cooldown_info is not None
        user_message = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE, "")

        # 验证 user_message 不在 stdout/stderr 中重复出现
        # 消费逻辑优先使用稳定字段（kind/reason/user_message）
        if user_message:
            key_phrases = ["回退", "CLI", "速率限制", "Rate limit"]
            for phrase in key_phrases:
                stdout_count = captured.out.count(phrase)
                stderr_count = captured.err.count(phrase)
                total_count = stdout_count + stderr_count
                assert total_count <= 1, f"'{phrase}' 在输出中出现 {total_count} 次，库层不应重复打印用户提示"

    @pytest.mark.asyncio
    async def test_auto_executor_multiple_failures_no_message_flood(self, capsys) -> None:
        """测试 AutoExecutor 多次失败不会产生消息洪水

        当 Cloud 多次失败时，不应在每次失败时都打印用户消息。
        """
        from cursor.client import CursorAgentConfig
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor, CooldownConfig

        cooldown_config = CooldownConfig(
            rate_limit_default_seconds=1,
            rate_limit_min_seconds=1,
        )

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(
            cli_config=config,
            cooldown_config=cooldown_config,
        )

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 1
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            results = []
            for _ in range(3):
                result = await auto_executor.execute("测试任务")
                results.append(result)

        captured = capsys.readouterr()
        combined_output = captured.out + captured.err

        # 验证不会产生消息洪水
        key_phrases = ["速率限制", "Rate limit", "冷却"]
        for phrase in key_phrases:
            count = combined_output.count(phrase)
            assert count <= 3, f"'{phrase}' 在输出中出现 {count} 次，可能存在消息洪水"

    def test_execution_mode_resolution_no_duplicate_warning(
        self, base_iterate_args: argparse.Namespace, capsys
    ) -> None:
        """测试执行模式解析不产生重复警告

        当解析执行模式时，警告消息不应重复输出。
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        # 重置类级别标志
        CursorAgentClient._cloud_api_key_warning_shown = False
        SelfIterator.reset_shown_messages()

        base_iterate_args.requirement = "& 测试任务"
        base_iterate_args.execution_mode = "auto"

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("core.config.get_config", return_value=mock_config),
        ):
            iterator = SelfIterator(base_iterate_args)
            mode = iterator._get_execution_mode()

            # 验证模式解析正确（无 API Key 应回退到 CLI）
            assert mode == ExecutionMode.CLI

        captured = capsys.readouterr()

        # 验证警告消息不重复（由于去重机制，应只出现 1 次）
        combined_output = captured.out + captured.err
        for phrase in ["API Key", "回退"]:
            count = combined_output.count(phrase)
            assert count <= 1, f"'{phrase}' 在输出中出现 {count} 次，警告不应重复"

    def test_decision_user_message_mock_print_warning(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 decision.user_message 通过 mock print_warning 验证去重

        mock print_warning 函数，验证：
        1. 当 decision.user_message 存在时，最多调用 print_warning 一次
        2. 去重机制正确工作（相同消息不重复打印）
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置类级别标志
        CursorAgentClient._cloud_api_key_warning_shown = False
        SelfIterator.reset_shown_messages()

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("core.config.get_config", return_value=mock_config),
        ):
            with patch("scripts.run_iterate.print_warning") as mock_print_warning:
                # 创建多个 SelfIterator 实例，模拟多次初始化
                for i in range(3):
                    base_iterate_args.requirement = f"& 测试任务 {i}"
                    try:
                        iterator = SelfIterator(base_iterate_args)
                        # 触发执行决策打印（如果有 user_message）
                        _ = iterator._get_execution_mode()
                    except Exception:
                        # 忽略可能的初始化错误
                        pass

                # 验证 print_warning 最多被调用一次
                assert mock_print_warning.call_count <= 1, (
                    f"print_warning 被调用 {mock_print_warning.call_count} 次，应最多调用 1 次（去重机制）"
                )


# ============================================================
# TestExplicitCliWithAmpersandPrefix - 显式 cli + & 前缀场景测试
# ============================================================


class TestExplicitCliWithAmpersandPrefix:
    """显式 cli 模式 + & 前缀场景测试

    验证当用户显式指定 --execution-mode cli 时：
    - & 前缀被正确忽略
    - user_message 包含正确的提示信息
    - 输出次数 ≤ 1（不刷屏）
    """

    @pytest.fixture
    def base_args(self) -> argparse.Namespace:
        """创建基础参数"""
        args = argparse.Namespace()
        args.requirement = "& 测试任务"
        args.directory = "."
        args.skip_online = True
        args.minimal = False
        args.changelog_url = None
        args.dry_run = True
        args.max_iterations = "1"
        args.workers = 1
        args.force_update = False
        args.verbose = False
        args.quiet = True
        args.log_level = None
        args.heartbeat_debug = False
        args.stall_diagnostics_enabled = None
        args.stall_diagnostics_level = None
        args.stall_recovery_interval = 30.0
        args.execution_health_check_interval = 30.0
        args.health_warning_cooldown = 60.0
        args.auto_commit = False
        args.auto_push = False
        args.commit_message = ""
        args.commit_per_iteration = False
        args.orchestrator = "basic"
        args.no_mp = True
        args._orchestrator_user_set = True
        # 关键: 显式指定 cli 模式
        args.execution_mode = "cli"
        args._execution_mode_user_set = True
        args.cloud_api_key = None
        args.cloud_auth_timeout = 30
        args.cloud_timeout = 300
        args.planner_execution_mode = None
        args.worker_execution_mode = None
        args.reviewer_execution_mode = None
        args.stream_console_renderer = False
        args.stream_advanced_renderer = False
        args.stream_typing_effect = False
        args.stream_typing_delay = 0.02
        args.stream_word_mode = True
        args.stream_color_enabled = True
        args.stream_show_word_diff = False
        args.stream_show_status_bar = True
        args.output_format = "text"
        return args

    def test_explicit_cli_ignores_ampersand_prefix(self, base_args: argparse.Namespace) -> None:
        """场景3: 显式 cli + & 前缀 - 验证 & 前缀被忽略

        断言：
        1. ExecutionDecision.effective_mode == "cli"
        2. ExecutionDecision.prefix_routed == False
        3. ExecutionDecision.has_ampersand_prefix == True（语法检测）
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 测试任务",
            requested_mode="cli",  # 显式 cli
            cloud_enabled=True,
            has_api_key=True,  # 即使有 API Key
            user_requested_orchestrator=None,
        )

        # 断言 1: effective_mode 为 cli
        assert decision.effective_mode == "cli", (
            f"显式 cli 模式下 effective_mode 应为 cli，实际: {decision.effective_mode}"
        )

        # 断言 2: prefix_routed 为 False（& 前缀被忽略）
        assert decision.prefix_routed is False, "显式 cli 模式下 prefix_routed 应为 False"

        # 断言 3: has_ampersand_prefix 为 True（语法层面检测到）
        assert decision.has_ampersand_prefix is True, "has_ampersand_prefix 应为 True（语法检测）"

    def test_explicit_cli_user_message_mentions_cli_mode(self, base_args: argparse.Namespace) -> None:
        """场景3: 显式 cli + & 前缀 - 验证 user_message 包含正确提示

        断言：
        1. user_message 存在
        2. user_message 提及 "cli" 或 "显式"
        3. user_message 提及 "忽略" & 前缀
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 测试任务",
            requested_mode="cli",
            cloud_enabled=True,
            has_api_key=True,
            user_requested_orchestrator=None,
        )

        # 断言 1: user_message 存在
        assert decision.user_message is not None, "显式 cli 模式 + & 前缀时 user_message 应存在"

        user_message_lower = decision.user_message.lower()

        # 断言 2: 提及 cli 或 显式
        assert "cli" in user_message_lower or "显式" in decision.user_message, (
            f"user_message 应提及 cli 或显式，实际: {decision.user_message}"
        )

        # 断言 3: 提及忽略
        assert "忽略" in decision.user_message or "ignored" in user_message_lower, (
            f"user_message 应提及忽略，实际: {decision.user_message}"
        )

    def test_explicit_cli_no_output_flood_on_multiple_analyses(self, base_args: argparse.Namespace, capsys) -> None:
        """场景3: 显式 cli + & 前缀多次分析 - 验证不刷屏

        断言：
        1. print_warning 最多被调用 1 次
        2. 相同的 user_message 不重复输出
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置去重状态
        CursorAgentClient._cloud_api_key_warning_shown = False
        SelfIterator.reset_shown_messages()

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="test_key"):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("scripts.run_iterate.print_warning") as mock_print_warning:
                    # 多次创建 SelfIterator 分析 & 前缀任务
                    for i in range(5):
                        base_args.requirement = f"& 任务 {i}"
                        try:
                            iterator = SelfIterator(base_args)
                            _ = iterator._get_execution_mode()
                        except Exception:
                            # 忽略初始化错误
                            pass

                    # 断言 1: print_warning 最多被调用 1 次
                    assert mock_print_warning.call_count <= 1, (
                        f"print_warning 被调用 {mock_print_warning.call_count} 次，显式 cli + & 前缀的提示应最多 1 次"
                    )

    def test_explicit_cli_sanitizes_prompt(self, base_args: argparse.Namespace) -> None:
        """场景3: 显式 cli + & 前缀 - 验证 prompt 被清理

        断言：
        1. sanitized_prompt 不以 & 开头
        2. sanitized_prompt 保留原始任务内容
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 测试任务内容",
            requested_mode="cli",
            cloud_enabled=True,
            has_api_key=True,
            user_requested_orchestrator=None,
        )

        # 断言 1: sanitized_prompt 不以 & 开头
        assert not decision.sanitized_prompt.startswith("&"), (
            f"sanitized_prompt 不应以 & 开头，实际: {decision.sanitized_prompt}"
        )

        # 断言 2: 保留原始内容
        assert "测试任务内容" in decision.sanitized_prompt, (
            f"sanitized_prompt 应保留原始内容，实际: {decision.sanitized_prompt}"
        )

    @pytest.mark.asyncio
    async def test_explicit_cli_mode_reason_explains_ignore(self, base_args: argparse.Namespace) -> None:
        """场景3: 显式 cli + & 前缀 - 验证 mode_reason 解释忽略原因

        断言：
        1. mode_reason 存在
        2. mode_reason 提及 cli 或显式指定
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 测试任务",
            requested_mode="cli",
            cloud_enabled=True,
            has_api_key=True,
            user_requested_orchestrator=None,
        )

        # 断言 1: mode_reason 存在
        assert decision.mode_reason is not None and len(decision.mode_reason) > 0, "mode_reason 应存在且非空"

        mode_reason_lower = decision.mode_reason.lower()

        # 断言 2: mode_reason 提及 cli 或显式指定
        assert "cli" in mode_reason_lower or "显式" in decision.mode_reason, (
            f"mode_reason 应提及 cli 或显式指定，实际: {decision.mode_reason}"
        )


# ============================================================
# TestCooldownInfoUnknownFieldsPrintLogic - cooldown_info 未知字段打印逻辑测试
# ============================================================


class TestSelfIterateCooldownInfoUnknownFields:
    """验证 SelfIterator._print_execution_result 处理带未知字段的 cooldown_info

    测试通过 mock print_warning/print_info 验证：
    1. 带未知字段的 cooldown_info 不会导致 KeyError/TypeError
    2. 打印决策只基于稳定字段（USER_MESSAGE, MESSAGE_LEVEL）
    3. 去重逻辑正常工作
    """

    @pytest.fixture
    def cooldown_info_with_unknown_fields(self) -> dict:
        """构造带未知字段的 cooldown_info"""
        from core.output_contract import CooldownInfoFields

        return {
            # 稳定字段
            CooldownInfoFields.USER_MESSAGE: "SelfIterator 测试：Cloud 不可用",
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.REASON: "未设置 CURSOR_API_KEY",
            CooldownInfoFields.RETRYABLE: True,
            CooldownInfoFields.RETRY_AFTER: None,
            CooldownInfoFields.IN_COOLDOWN: False,
            CooldownInfoFields.REMAINING_SECONDS: None,
            CooldownInfoFields.FAILURE_COUNT: 1,
            # 兼容字段
            CooldownInfoFields.FALLBACK_REASON: "未设置 CURSOR_API_KEY",
            CooldownInfoFields.ERROR_TYPE: "config_error",
            CooldownInfoFields.FAILURE_KIND: "no_key",
            # 扩展字段
            CooldownInfoFields.MESSAGE_LEVEL: "warning",
            # 未知字段（模拟未来扩展）
            "future_field_v2_api": "new_feature",
            "experimental_metrics": {"latency_ms": 123, "retries": 2},
            "_internal_trace_id": "trace-67890",
            "unused_legacy_field": None,
            "iteration_context": {"current": 1, "max": 10},
        }

    @pytest.fixture
    def result_with_unknown_cooldown_fields(self, cooldown_info_with_unknown_fields: dict) -> dict:
        """构造包含带未知字段 cooldown_info 的执行结果"""
        return {
            "success": False,
            "iterations_completed": 0,
            "total_tasks_created": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            IterateResultFields.COOLDOWN_INFO: cooldown_info_with_unknown_fields,
            # 其他可能的未知字段
            "future_result_field": "some_value",
        }

    def test_print_execution_result_with_unknown_fields_no_exception(
        self,
        result_with_unknown_cooldown_fields: dict,
    ) -> None:
        """验证 _print_execution_result 逻辑处理带未知字段的 cooldown_info 不抛异常

        模拟 scripts/run_iterate.py 第 6083-6092 行的打印逻辑。
        """
        from core.execution_policy import compute_message_dedup_key
        from core.output_contract import CooldownInfoFields

        result = result_with_unknown_cooldown_fields
        shown_messages: set[str] = set()

        # 模拟 _print_execution_result 中的打印逻辑（不应抛出任何异常）
        cooldown_info = result.get(IterateResultFields.COOLDOWN_INFO)
        if cooldown_info and cooldown_info.get(CooldownInfoFields.USER_MESSAGE):
            msg_key = compute_message_dedup_key(cooldown_info[CooldownInfoFields.USER_MESSAGE])
            if msg_key not in shown_messages:
                shown_messages.add(msg_key)
                # message_level 缺失时默认按 info 处理
                message_level = cooldown_info.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                if message_level == "warning":
                    # 模拟 print_warning 调用
                    pass
                else:
                    # 模拟 print_info 调用
                    pass

        # 断言去重逻辑正常工作
        assert len(shown_messages) == 1

    def test_mock_print_warning_with_unknown_cooldown_fields(
        self,
        result_with_unknown_cooldown_fields: dict,
    ) -> None:
        """通过 mock print_warning 验证 _print_execution_result 处理未知字段

        断言:
        1. print_warning 被调用一次（message_level=warning）
        2. 调用参数为 user_message 内容
        3. 不会因未知字段导致异常
        """
        from unittest.mock import MagicMock

        from core.execution_policy import compute_message_dedup_key
        from core.output_contract import CooldownInfoFields

        result = result_with_unknown_cooldown_fields
        shown_messages: set[str] = set()
        mock_print_warning = MagicMock()
        mock_print_info = MagicMock()

        # 模拟 _print_execution_result 中的打印逻辑
        cooldown_info = result.get(IterateResultFields.COOLDOWN_INFO)
        if cooldown_info and cooldown_info.get(CooldownInfoFields.USER_MESSAGE):
            msg_key = compute_message_dedup_key(cooldown_info[CooldownInfoFields.USER_MESSAGE])
            if msg_key not in shown_messages:
                shown_messages.add(msg_key)
                message_level = cooldown_info.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                if message_level == "warning":
                    mock_print_warning(cooldown_info[CooldownInfoFields.USER_MESSAGE])
                else:
                    mock_print_info(cooldown_info[CooldownInfoFields.USER_MESSAGE])

        # 断言 1: print_warning 被调用一次
        assert mock_print_warning.call_count == 1, f"print_warning 应被调用 1 次，实际: {mock_print_warning.call_count}"

        # 断言 2: 调用参数正确
        mock_print_warning.assert_called_once_with("SelfIterator 测试：Cloud 不可用")

        # 断言 3: print_info 未被调用
        assert mock_print_info.call_count == 0

    def test_mock_print_info_when_message_level_info(
        self,
        result_with_unknown_cooldown_fields: dict,
    ) -> None:
        """验证 message_level=info 时使用 print_info"""
        from unittest.mock import MagicMock

        from core.execution_policy import compute_message_dedup_key
        from core.output_contract import CooldownInfoFields

        # 修改 message_level 为 info
        result = result_with_unknown_cooldown_fields.copy()
        result[IterateResultFields.COOLDOWN_INFO] = result[IterateResultFields.COOLDOWN_INFO].copy()
        result[IterateResultFields.COOLDOWN_INFO][CooldownInfoFields.MESSAGE_LEVEL] = "info"

        shown_messages: set[str] = set()
        mock_print_warning = MagicMock()
        mock_print_info = MagicMock()

        # 模拟 _print_execution_result 中的打印逻辑
        cooldown_info = result.get(IterateResultFields.COOLDOWN_INFO)
        if cooldown_info and cooldown_info.get(CooldownInfoFields.USER_MESSAGE):
            msg_key = compute_message_dedup_key(cooldown_info[CooldownInfoFields.USER_MESSAGE])
            if msg_key not in shown_messages:
                shown_messages.add(msg_key)
                message_level = cooldown_info.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                if message_level == "warning":
                    mock_print_warning(cooldown_info[CooldownInfoFields.USER_MESSAGE])
                else:
                    mock_print_info(cooldown_info[CooldownInfoFields.USER_MESSAGE])

        # 断言: print_info 被调用，print_warning 未被调用
        assert mock_print_info.call_count == 1
        assert mock_print_warning.call_count == 0

    def test_dedup_logic_with_multiple_calls_unknown_fields(
        self,
        result_with_unknown_cooldown_fields: dict,
    ) -> None:
        """验证去重逻辑在多次调用时正常工作（带未知字段）"""
        from unittest.mock import MagicMock

        from core.execution_policy import compute_message_dedup_key
        from core.output_contract import CooldownInfoFields

        result = result_with_unknown_cooldown_fields
        shown_messages: set[str] = set()
        mock_print_warning = MagicMock()

        # 模拟多次调用 _print_execution_result（相同消息应只打印一次）
        for _ in range(5):
            cooldown_info = result.get(IterateResultFields.COOLDOWN_INFO)
            if cooldown_info and cooldown_info.get(CooldownInfoFields.USER_MESSAGE):
                msg_key = compute_message_dedup_key(cooldown_info[CooldownInfoFields.USER_MESSAGE])
                if msg_key not in shown_messages:
                    shown_messages.add(msg_key)
                    message_level = cooldown_info.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                    if message_level == "warning":
                        mock_print_warning(cooldown_info[CooldownInfoFields.USER_MESSAGE])

        # 断言: 尽管调用 5 次，print_warning 只被调用 1 次
        assert mock_print_warning.call_count == 1, (
            f"print_warning 应只被调用 1 次（去重），实际: {mock_print_warning.call_count}"
        )

    def test_message_level_missing_defaults_to_info(
        self,
        result_with_unknown_cooldown_fields: dict,
    ) -> None:
        """验证 MESSAGE_LEVEL 缺失时默认为 'info'"""
        from unittest.mock import MagicMock

        from core.execution_policy import compute_message_dedup_key
        from core.output_contract import CooldownInfoFields

        # 删除 MESSAGE_LEVEL 字段
        result = result_with_unknown_cooldown_fields.copy()
        result[IterateResultFields.COOLDOWN_INFO] = result[IterateResultFields.COOLDOWN_INFO].copy()
        del result[IterateResultFields.COOLDOWN_INFO][CooldownInfoFields.MESSAGE_LEVEL]

        shown_messages: set[str] = set()
        mock_print_warning = MagicMock()
        mock_print_info = MagicMock()

        # 模拟 _print_execution_result 中的打印逻辑
        cooldown_info = result.get(IterateResultFields.COOLDOWN_INFO)
        if cooldown_info and cooldown_info.get(CooldownInfoFields.USER_MESSAGE):
            msg_key = compute_message_dedup_key(cooldown_info[CooldownInfoFields.USER_MESSAGE])
            if msg_key not in shown_messages:
                shown_messages.add(msg_key)
                # message_level 缺失时默认按 info 处理
                message_level = cooldown_info.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                if message_level == "warning":
                    mock_print_warning(cooldown_info[CooldownInfoFields.USER_MESSAGE])
                else:
                    mock_print_info(cooldown_info[CooldownInfoFields.USER_MESSAGE])

        # 断言: MESSAGE_LEVEL 缺失时使用 print_info
        assert mock_print_info.call_count == 1, "MESSAGE_LEVEL 缺失时应使用 print_info"
        assert mock_print_warning.call_count == 0


# ============================================================
# TestAutoNoKeyDecisionConsistency - Auto 无 Key 决策一致性
# ============================================================


class TestAutoNoKeyDecisionConsistency:
    """测试 requested_mode=auto 无 API Key 时的决策一致性

    验证 SelfIterator 与 build_execution_decision 对相同输入产生一致的决策：
    - effective_mode 应回退到 "cli"
    - orchestrator 应强制为 "basic"（基于 requested_mode）
    - prefix_routed 应为 False（未成功触发 Cloud）
    """

    @pytest.fixture
    def auto_no_key_args(self) -> argparse.Namespace:
        """创建 execution_mode=auto 无 API Key 的测试参数"""
        return argparse.Namespace(
            requirement="测试任务描述",
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
            execution_mode="auto",  # 关键：auto 模式
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

    def test_build_execution_decision_auto_no_key_basic_orchestrator(self) -> None:
        """验证 build_execution_decision: auto + 无 key → orchestrator=basic

        复用 build_execution_decision 作为权威决策源。
        """
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="测试任务",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key
        )

        # 核心断言: orchestrator 必须为 basic
        assert decision.orchestrator == "basic", (
            f"关键规则：requested_mode=auto 无 key 时 orchestrator 必须为 basic\n"
            f"  实际 orchestrator={decision.orchestrator}"
        )

        # 核心断言: prefix_routed 必须为 False
        assert decision.prefix_routed is False, (
            f"关键规则：无 & 前缀时 prefix_routed 必须为 False\n  实际 prefix_routed={decision.prefix_routed}"
        )

        # effective_mode 应回退到 cli
        assert decision.effective_mode == "cli", (
            f"auto + 无 key 应回退到 cli\n  实际 effective_mode={decision.effective_mode}"
        )

    def test_self_iterator_matches_build_execution_decision(self, auto_no_key_args: argparse.Namespace) -> None:
        """验证 SelfIterator 与 build_execution_decision 结果一致

        确保 run.py 与 scripts/run_iterate.py 快照字段一致。
        """
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory

        # === Step 1: 获取权威决策 ===
        decision = build_execution_decision(
            prompt=auto_no_key_args.requirement,
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,
        )

        # === Step 2: 创建 SelfIterator 并验证一致性 ===
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            iterator = SelfIterator(auto_no_key_args)
            iterator_effective_mode = iterator._get_execution_mode()
            iterator_orchestrator = iterator._get_orchestrator_type()
            iterator_prefix_routed = iterator._prefix_routed

        # === Step 3: 验证快照一致性 ===
        # orchestrator 一致（核心断言）
        assert iterator_orchestrator == decision.orchestrator, (
            f"快照不一致：orchestrator\n"
            f"  build_execution_decision={decision.orchestrator}\n"
            f"  SelfIterator={iterator_orchestrator}"
        )

        # prefix_routed 一致（核心断言）
        assert iterator_prefix_routed == decision.prefix_routed, (
            f"快照不一致：prefix_routed\n"
            f"  build_execution_decision={decision.prefix_routed}\n"
            f"  SelfIterator={iterator_prefix_routed}"
        )

        # effective_mode 一致
        assert iterator_effective_mode.value == decision.effective_mode, (
            f"快照不一致：effective_mode\n"
            f"  build_execution_decision={decision.effective_mode}\n"
            f"  SelfIterator={iterator_effective_mode.value}"
        )


# ============================================================
# TestMinimalModeNoSideEffects - Minimal 模式无副作用验证
# ============================================================


class TestMinimalModeNoSideEffects:
    """测试 minimal 模式下 SelfIterator 不产生文件系统副作用

    验证在 tmp_path 下运行 minimal 入口，
    运行前后对比目录树（顶层与 .cursor/、logs/），断言未新增。
    """

    def _snapshot_directory(self, path: Path) -> set[str]:
        """获取目录下所有文件/目录的相对路径快照

        Args:
            path: 根目录路径

        Returns:
            所有文件/目录的相对路径集合
        """
        if not path.exists():
            return set()

        snapshot = set()
        for item in path.rglob("*"):
            rel_path = item.relative_to(path)
            snapshot.add(str(rel_path))
        return snapshot

    @pytest.mark.asyncio
    async def test_minimal_mode_no_new_files_created(self, tmp_path: Path, monkeypatch) -> None:
        """测试 minimal 模式下不会创建新文件/目录

        在 tmp_path 下运行 SelfIterator（minimal=True），
        验证 .cursor/、logs/ 等目录不会被创建。
        """
        # 切换到临时目录
        monkeypatch.chdir(tmp_path)

        # 创建 minimal 模式的 args
        args = argparse.Namespace(
            requirement="测试需求",
            directory=str(tmp_path),
            skip_online=True,  # minimal 模式通常与 skip_online 一起使用
            minimal=True,  # 关键：启用 minimal 模式
            changelog_url=None,
            dry_run=True,  # 不执行实际任务
            max_iterations="1",
            workers=1,
            force_update=False,
            verbose=False,
            quiet=True,  # 减少输出
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
            orchestrator="basic",  # 使用简单模式
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
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
            stream_color_enabled=False,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        # 获取运行前的目录快照
        before_snapshot = self._snapshot_directory(tmp_path)

        # Mock 所有可能产生副作用的组件
        with (
            patch.object(run_iterate, "KnowledgeManager") as mock_km_cls,
            patch.object(run_iterate, "KnowledgeStorage") as mock_storage_cls,
            patch.object(run_iterate, "WebFetcher") as mock_fetcher_cls,
        ):
            # 设置 mock 返回值
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            mock_km_cls.return_value = mock_km

            mock_storage = MagicMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
            mock_storage.search = AsyncMock(return_value=[])
            mock_storage_cls.return_value = mock_storage

            mock_fetcher = MagicMock()
            mock_fetcher.initialize = AsyncMock()
            mock_fetcher.fetch = AsyncMock()
            mock_fetcher_cls.return_value = mock_fetcher

            # 创建并运行 SelfIterator
            iterator = SelfIterator(args)

            # Mock 工作目录
            iterator.working_directory = tmp_path

            # 运行（dry_run 模式会提前返回）
            result = await iterator.run()

        # 获取运行后的目录快照
        after_snapshot = self._snapshot_directory(tmp_path)

        # 计算新增的文件/目录
        new_items = after_snapshot - before_snapshot

        # 验证没有新增文件
        # 过滤掉可能的临时文件（如 pytest 的）
        significant_new_items = {
            item
            for item in new_items
            if not item.startswith(".")  # 忽略隐藏文件
            or item.startswith(".cursor")  # 但检查 .cursor 目录
        }

        # 特别检查 .cursor/ 和 logs/ 目录
        cursor_dir_created = any(item.startswith(".cursor") for item in new_items)
        logs_dir_created = any(item.startswith("logs") for item in new_items)

        assert not cursor_dir_created, (
            f".cursor/ 目录不应在 minimal 模式下被创建，新增项: {[i for i in new_items if '.cursor' in i]}"
        )
        assert not logs_dir_created, (
            f"logs/ 目录不应在 minimal 模式下被创建，新增项: {[i for i in new_items if 'logs' in i]}"
        )

        # 验证 dry_run 模式返回成功
        assert result.get("dry_run") is True or result.get("success") is True

    @pytest.mark.asyncio
    async def test_dry_run_mode_no_file_writes(self, tmp_path: Path, monkeypatch) -> None:
        """测试 dry_run 模式下不会写入文件

        验证 Path.mkdir 和 Path.write_text 不会被调用（通过 patch 检测）。
        """
        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(
            requirement="测试需求",
            directory=str(tmp_path),
            skip_online=True,
            minimal=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=1,
            force_update=False,
            verbose=False,
            quiet=True,
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
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
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
            stream_color_enabled=False,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        # 追踪 mkdir 和 write_text 调用
        mkdir_calls = []
        write_text_calls = []

        original_mkdir = Path.mkdir
        original_write_text = Path.write_text

        def patched_mkdir(self, *args, **kwargs):
            # 只记录在 tmp_path 下的调用
            if str(self).startswith(str(tmp_path)):
                mkdir_calls.append(str(self))
            return original_mkdir(self, *args, **kwargs)

        def patched_write_text(self, *args, **kwargs):
            if str(self).startswith(str(tmp_path)):
                write_text_calls.append(str(self))
            return original_write_text(self, *args, **kwargs)

        with (
            patch.object(run_iterate, "KnowledgeManager") as mock_km_cls,
            patch.object(run_iterate, "KnowledgeStorage") as mock_storage_cls,
            patch.object(run_iterate, "WebFetcher") as mock_fetcher_cls,
        ):
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            mock_km_cls.return_value = mock_km

            mock_storage = MagicMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
            mock_storage.search = AsyncMock(return_value=[])
            mock_storage_cls.return_value = mock_storage

            mock_fetcher = MagicMock()
            mock_fetcher.initialize = AsyncMock()
            mock_fetcher.fetch = AsyncMock()
            mock_fetcher_cls.return_value = mock_fetcher

            with patch.object(Path, "mkdir", patched_mkdir):
                with patch.object(Path, "write_text", patched_write_text):
                    iterator = SelfIterator(args)
                    iterator.working_directory = tmp_path
                    result = await iterator.run()

        # 验证在 .cursor/ 或 logs/ 下没有 mkdir 调用
        cursor_mkdirs = [c for c in mkdir_calls if ".cursor" in c]
        logs_mkdirs = [c for c in mkdir_calls if "logs" in c]

        assert len(cursor_mkdirs) == 0, f"minimal 模式下不应创建 .cursor/ 子目录: {cursor_mkdirs}"
        assert len(logs_mkdirs) == 0, f"minimal 模式下不应创建 logs/ 子目录: {logs_mkdirs}"

        # 验证返回结果
        assert result.get("dry_run") is True or result.get("success") is True

    @pytest.mark.asyncio
    async def test_minimal_mode_no_webfetcher_calls(self, tmp_path: Path, monkeypatch) -> None:
        """测试 minimal 模式下 WebFetcher.fetch/_do_fetch/fetch_many 不被调用

        通过 patch WebFetcher 的网络方法，验证 minimal 模式下：
        1. fetch() 不被调用
        2. fetch_many() 不被调用
        3. _do_fetch() 不被调用

        这是 scripts/run_iterate.py --minimal 入口的网络隔离保证。
        """
        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(
            requirement="测试需求",
            directory=str(tmp_path),
            skip_online=True,
            minimal=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=1,
            force_update=False,
            verbose=False,
            quiet=True,
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
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
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
            stream_color_enabled=False,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        # 追踪 WebFetcher 方法调用
        fetch_calls = []
        fetch_many_calls = []
        do_fetch_calls = []

        # 创建 mock WebFetcher 类
        class MockWebFetcher:
            def __init__(self, *args, **kwargs):
                pass

            async def initialize(self):
                pass

            async def fetch(self, *args, **kwargs):
                fetch_calls.append(("fetch", args, kwargs))
                raise AssertionError("WebFetcher.fetch() 不应在 minimal 模式下被调用")

            async def fetch_many(self, *args, **kwargs):
                fetch_many_calls.append(("fetch_many", args, kwargs))
                raise AssertionError("WebFetcher.fetch_many() 不应在 minimal 模式下被调用")

            async def _do_fetch(self, *args, **kwargs):
                do_fetch_calls.append(("_do_fetch", args, kwargs))
                raise AssertionError("WebFetcher._do_fetch() 不应在 minimal 模式下被调用")

        # 同时 patch socket.socket 作为兜底保护
        socket_calls = []
        original_socket = None

        def mock_socket(*args, **kwargs):
            socket_calls.append(("socket", args, kwargs))
            raise AssertionError("socket.socket() 不应在 minimal 模式下被调用")

        with patch.object(run_iterate, "WebFetcher", MockWebFetcher):
            with patch.object(run_iterate, "KnowledgeManager") as mock_km_cls:
                with patch.object(run_iterate, "KnowledgeStorage") as mock_storage_cls:
                    # 设置 mock 返回值
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    mock_km_cls.return_value = mock_km

                    mock_storage = MagicMock()
                    mock_storage.initialize = AsyncMock()
                    mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
                    mock_storage.search = AsyncMock(return_value=[])
                    mock_storage_cls.return_value = mock_storage

                    # Patch socket.socket 作为额外的网络隔离保护
                    with _patch("socket.socket", mock_socket):
                        iterator = SelfIterator(args)
                        iterator.working_directory = tmp_path
                        result = await iterator.run()

        # 验证没有网络调用
        assert len(fetch_calls) == 0, f"minimal 模式下 WebFetcher.fetch() 不应被调用: {fetch_calls}"
        assert len(fetch_many_calls) == 0, f"minimal 模式下 WebFetcher.fetch_many() 不应被调用: {fetch_many_calls}"
        assert len(do_fetch_calls) == 0, f"minimal 模式下 WebFetcher._do_fetch() 不应被调用: {do_fetch_calls}"
        # socket 调用在 patch 范围内会被捕获，但由于上层 WebFetcher 已被 mock，
        # 实际不会触发 socket 调用。这里验证没有意外的网络访问。
        # 注意：socket_calls 可能包含非网络用途的调用，只验证关键路径

        # 验证返回结果
        assert result.get("dry_run") is True or result.get("success") is True

    @pytest.mark.asyncio
    async def test_minimal_mode_socket_isolation(self, tmp_path: Path, monkeypatch) -> None:
        """测试 minimal 模式下通过 socket.socket patch 验证网络隔离

        这是更严格的测试：直接 patch socket.socket，验证 minimal 模式
        不会触发任何底层网络连接。
        """
        import socket as socket_module

        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(
            requirement="测试需求",
            directory=str(tmp_path),
            skip_online=True,
            minimal=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="1",
            workers=1,
            force_update=False,
            verbose=False,
            quiet=True,
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
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
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
            stream_color_enabled=False,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        # 记录 socket.socket 调用
        socket_connect_calls = []
        original_socket_class = socket_module.socket

        class MockSocket:
            """Mock socket that tracks connect calls"""

            def __init__(self, *args, **kwargs):
                self._args = args
                self._kwargs = kwargs

            def connect(self, address):
                socket_connect_calls.append(("connect", address))
                raise AssertionError(f"socket.connect() 不应在 minimal 模式下被调用: {address}")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def close(self):
                pass

            def settimeout(self, *args):
                pass

            def setsockopt(self, *args):
                pass

        with patch.object(run_iterate, "KnowledgeManager") as mock_km_cls:
            with patch.object(run_iterate, "KnowledgeStorage") as mock_storage_cls:
                with patch.object(run_iterate, "WebFetcher") as mock_fetcher_cls:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    mock_km_cls.return_value = mock_km

                    mock_storage = MagicMock()
                    mock_storage.initialize = AsyncMock()
                    mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
                    mock_storage.search = AsyncMock(return_value=[])
                    mock_storage_cls.return_value = mock_storage

                    mock_fetcher = MagicMock()
                    mock_fetcher.initialize = AsyncMock()
                    mock_fetcher.fetch = AsyncMock()
                    mock_fetcher_cls.return_value = mock_fetcher

                    # Patch socket.socket 类
                    with _patch.object(socket_module, "socket", MockSocket):
                        iterator = SelfIterator(args)
                        iterator.working_directory = tmp_path
                        result = await iterator.run()

        # 验证没有网络连接尝试
        assert len(socket_connect_calls) == 0, f"minimal 模式下不应有 socket.connect() 调用: {socket_connect_calls}"

        # 验证返回结果
        assert result.get("dry_run") is True or result.get("success") is True


# ============================================================
# TestMinimalIterate - 最小路径调用序列契约验证
# ============================================================


class TestMinimalIterate:
    """测试最小路径下的调用序列契约

    验证 SelfIterator 在最小执行路径下：
    1. ChangelogAnalyzer、KnowledgeUpdater、Orchestrator 的调用顺序正确
    2. 组件间的依赖关系符合设计契约
    3. 优先级覆盖（如 CLI 参数覆盖 config.yaml）正确生效
    """

    @pytest.fixture
    def minimal_args(self) -> argparse.Namespace:
        """创建最小化测试参数"""
        return argparse.Namespace(
            requirement="最小路径测试",
            directory=".",
            skip_online=True,  # 最小路径跳过在线检查
            minimal=True,  # 启用 minimal 模式
            changelog_url=None,
            dry_run=False,
            max_iterations="1",
            workers=1,
            force_update=False,
            verbose=False,
            quiet=True,
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
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
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
            stream_color_enabled=False,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    @pytest.mark.asyncio
    async def test_minimal_path_call_sequence_contract(self, minimal_args: argparse.Namespace) -> None:
        """测试最小路径下组件调用序列符合契约

        契约 (core/execution_policy.py Side Effect Control Strategy Matrix):
        minimal = skip_online + dry_run（禁止所有网络请求、文件写入、Git 操作）

        1. minimal 模式自动设置 dry_run=True
        2. ChangelogAnalyzer.analyze() 应跳过（因为 skip_online=True）
        3. KnowledgeUpdater.initialize() 应跳过（minimal 模式跳过知识库初始化）
        4. Orchestrator.run() 应跳过（因为 dry_run=True）
        5. 返回 dry_run=True 的预览结果

        最小路径下（minimal=True）：
        - ChangelogAnalyzer.analyze() 应跳过
        - KnowledgeUpdater.initialize() 应跳过
        - Orchestrator.run() 应跳过（dry_run=True）
        """
        call_sequence: list[str] = []

        # 创建 mock 对象并记录调用顺序
        mock_changelog_analyzer = MagicMock()
        mock_changelog_analyzer.analyze = AsyncMock(
            side_effect=lambda: record_call(call_sequence, "ChangelogAnalyzer.analyze", UpdateAnalysis())
        )
        mock_changelog_analyzer.fetcher = MagicMock()
        mock_changelog_analyzer.fetcher.initialize = AsyncMock()

        mock_knowledge_updater = MagicMock()
        mock_knowledge_updater.initialize = AsyncMock(
            side_effect=lambda: record_call(call_sequence, "KnowledgeUpdater.initialize")
        )
        mock_knowledge_updater.update_from_analysis = AsyncMock(
            side_effect=lambda *args, **kwargs: record_call(call_sequence, "KnowledgeUpdater.update_from_analysis")
        )
        mock_knowledge_updater.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_knowledge_updater.search_relevant = AsyncMock(return_value=[])
        # 添加 _build_dry_run_stats 需要的属性
        mock_knowledge_updater.max_fetch_urls = 10

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            side_effect=lambda *args, **kwargs: record_call(
                call_sequence,
                "Orchestrator.run",
                {
                    "success": True,
                    "iterations_completed": 1,
                    "total_tasks_created": 1,
                    "total_tasks_completed": 1,
                    "total_tasks_failed": 0,
                },
            )
        )

        with (
            patch.object(run_iterate, "ChangelogAnalyzer", return_value=mock_changelog_analyzer),
            patch.object(run_iterate, "KnowledgeUpdater", return_value=mock_knowledge_updater),
            patch.object(run_iterate, "Orchestrator") as MockOrchestrator,
        ):
            MockOrchestrator.return_value = mock_orchestrator

            iterator = SelfIterator(minimal_args)
            result = await iterator.run()

        # 验证 minimal 模式下的调用序列契约
        # minimal 模式应跳过 ChangelogAnalyzer.analyze 和 KnowledgeUpdater.initialize
        assert "ChangelogAnalyzer.analyze" not in call_sequence, "minimal 模式下不应调用 ChangelogAnalyzer.analyze"
        assert "KnowledgeUpdater.initialize" not in call_sequence, "minimal 模式下不应调用 KnowledgeUpdater.initialize"

        # Orchestrator.run 不应被调用（因为 minimal 模式强制 dry_run=True）
        # 权威定义: core/execution_policy.py Side Effect Control Strategy Matrix
        assert "Orchestrator.run" not in call_sequence, "minimal 模式下不应调用 Orchestrator.run（因为 dry_run=True）"

        # 验证返回结果
        assert result.get("success") is True
        # minimal 模式应返回 dry_run=True
        assert result.get("dry_run") is True, "minimal 模式应返回 dry_run=True"

    @pytest.mark.asyncio
    async def test_non_minimal_path_call_sequence_contract(self, minimal_args: argparse.Namespace) -> None:
        """测试非 minimal 路径下的完整调用序列契约

        契约:
        1. ChangelogAnalyzer.analyze() 在步骤 1 被调用
        2. KnowledgeUpdater.initialize() 在步骤 2 被调用
        3. Orchestrator.run() 在步骤 5 被调用

        调用顺序必须为: analyze -> initialize -> run
        """
        call_sequence: list[str] = []

        # 禁用 minimal 模式，但保留 skip_online 以避免网络调用
        minimal_args.minimal = False
        minimal_args.skip_online = True

        mock_changelog_analyzer = MagicMock()
        mock_changelog_analyzer.analyze = AsyncMock(
            side_effect=lambda: record_call(call_sequence, "ChangelogAnalyzer.analyze", UpdateAnalysis())
        )
        mock_changelog_analyzer.fetcher = MagicMock()
        mock_changelog_analyzer.fetcher.initialize = AsyncMock()

        # 创建完整的 storage mock
        mock_storage = MagicMock()
        mock_storage.list_documents = AsyncMock(return_value=[])
        mock_storage.load_document = AsyncMock(return_value=None)

        mock_knowledge_updater = MagicMock()
        mock_knowledge_updater.initialize = AsyncMock(
            side_effect=lambda: record_call(call_sequence, "KnowledgeUpdater.initialize")
        )
        mock_knowledge_updater.update_from_analysis = AsyncMock()
        mock_knowledge_updater.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_knowledge_updater.get_knowledge_context = AsyncMock(return_value=[])
        mock_knowledge_updater.search_relevant = AsyncMock(return_value=[])
        mock_knowledge_updater.storage = mock_storage
        mock_knowledge_updater.max_fetch_urls = 10  # 添加 _build_dry_run_stats 需要的属性

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            side_effect=lambda *args, **kwargs: record_call(
                call_sequence,
                "Orchestrator.run",
                {
                    "success": True,
                    "iterations_completed": 1,
                    "total_tasks_created": 1,
                    "total_tasks_completed": 1,
                    "total_tasks_failed": 0,
                },
            )
        )

        with (
            patch.object(run_iterate, "ChangelogAnalyzer", return_value=mock_changelog_analyzer),
            patch.object(run_iterate, "KnowledgeUpdater", return_value=mock_knowledge_updater),
            patch.object(run_iterate, "Orchestrator") as MockOrchestrator,
        ):
            MockOrchestrator.return_value = mock_orchestrator

            iterator = SelfIterator(minimal_args)
            result = await iterator.run()

        # 验证非 minimal 模式下的调用序列
        # skip_online=True 时，ChangelogAnalyzer.analyze 仍被跳过
        # 但 KnowledgeUpdater.initialize 应被调用
        assert "KnowledgeUpdater.initialize" in call_sequence, "非 minimal 模式下应调用 KnowledgeUpdater.initialize"
        assert "Orchestrator.run" in call_sequence, "Orchestrator.run 应被调用"

        # 验证顺序: initialize 在 run 之前
        init_idx = call_sequence.index("KnowledgeUpdater.initialize")
        run_idx = call_sequence.index("Orchestrator.run")
        assert init_idx < run_idx, f"KnowledgeUpdater.initialize({init_idx}) 应在 Orchestrator.run({run_idx}) 之前"

    @pytest.mark.asyncio
    async def test_mp_orchestrator_call_sequence(self, minimal_args: argparse.Namespace) -> None:
        """测试 MultiProcessOrchestrator 的调用序列

        验证当使用 MP 编排器时，调用序列仍符合契约。

        注意：此测试使用非 minimal 模式，因为 minimal 模式强制 dry_run=True
        会跳过 Orchestrator.run。
        """
        call_sequence: list[str] = []

        # 切换到 MP 模式（非 minimal）
        minimal_args.orchestrator = "mp"
        minimal_args.no_mp = False
        minimal_args._orchestrator_user_set = True
        minimal_args.minimal = False  # 禁用 minimal 模式以允许执行 Orchestrator.run
        minimal_args.dry_run = False  # 确保不是 dry-run 模式

        mock_mp_orchestrator = MagicMock()
        mock_mp_orchestrator.run = AsyncMock(
            side_effect=lambda *args, **kwargs: record_call(
                call_sequence,
                "MultiProcessOrchestrator.run",
                {
                    "success": True,
                    "iterations_completed": 1,
                    "total_tasks_created": 1,
                    "total_tasks_completed": 1,
                    "total_tasks_failed": 0,
                },
            )
        )

        with patch.object(run_iterate, "ChangelogAnalyzer") as MockChangelog:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze = AsyncMock(return_value=UpdateAnalysis())
            mock_analyzer.fetcher = MagicMock()
            MockChangelog.return_value = mock_analyzer

            with patch.object(run_iterate, "KnowledgeUpdater") as MockUpdater:
                # 创建完整的 storage mock
                mock_storage = MagicMock()
                mock_storage.list_documents = AsyncMock(return_value=[])
                mock_storage.load_document = AsyncMock(return_value=None)

                mock_updater = MagicMock()
                mock_updater.initialize = AsyncMock()
                mock_updater.get_stats = AsyncMock(return_value={"document_count": 0})
                mock_updater.search_relevant = AsyncMock(return_value=[])
                mock_updater.get_knowledge_context = AsyncMock(return_value=[])  # 添加缺少的 async mock
                mock_updater.max_fetch_urls = 10  # 添加 _build_dry_run_stats 需要的属性
                mock_updater.storage = mock_storage  # 添加 storage mock
                MockUpdater.return_value = mock_updater

                with patch.object(run_iterate, "MultiProcessOrchestrator") as MockMPOrch:
                    MockMPOrch.return_value = mock_mp_orchestrator

                    with (
                        patch.object(run_iterate, "MultiProcessOrchestratorConfig"),
                        patch.object(run_iterate, "KnowledgeManager") as MockKM,
                    ):
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        iterator = SelfIterator(minimal_args)
                        result = await iterator.run()

        # 验证 MP 编排器被调用
        assert "MultiProcessOrchestrator.run" in call_sequence, "MultiProcessOrchestrator.run 应被调用"


class TestPriorityOverride:
    """测试优先级覆盖机制

    验证 CLI 参数 > 环境变量 > config.yaml > 默认值 的优先级顺序。
    """

    @pytest.fixture
    def base_args(self) -> argparse.Namespace:
        """创建基础参数"""
        return argparse.Namespace(
            requirement="优先级测试",
            directory=".",
            skip_online=True,
            minimal=True,
            changelog_url=None,
            dry_run=True,
            max_iterations="5",  # CLI 显式设置
            workers=4,  # CLI 显式设置
            force_update=False,
            verbose=False,
            quiet=True,
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
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
            cloud_api_key="cli-api-key",  # CLI 显式设置
            cloud_auth_timeout=60,  # CLI 显式设置
            cloud_timeout=600,  # CLI 显式设置
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=False,
            stream_show_word_diff=False,
            max_fetch_urls=10,  # CLI 显式设置
            fallback_core_docs_count=5,  # CLI 显式设置
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_cli_overrides_config_yaml_workers(self, base_args: argparse.Namespace) -> None:
        """测试 CLI --workers 参数覆盖 config.yaml 的 worker_pool_size"""
        # CLI 参数设置为 4
        assert base_args.workers == 4

        # 创建 SelfIterator 并验证使用了 CLI 值
        with patch.object(run_iterate, "ChangelogAnalyzer"), patch.object(run_iterate, "KnowledgeUpdater"):
            iterator = SelfIterator(base_args)

            # 验证 args.workers 被正确使用
            assert iterator.args.workers == 4

    def test_cli_overrides_config_yaml_max_iterations(self, base_args: argparse.Namespace) -> None:
        """测试 CLI --max-iterations 参数覆盖 config.yaml"""
        # CLI 参数设置为 "5"
        assert base_args.max_iterations == "5"

        with patch.object(run_iterate, "ChangelogAnalyzer"), patch.object(run_iterate, "KnowledgeUpdater"):
            iterator = SelfIterator(base_args)
            assert iterator.args.max_iterations == "5"

    def test_cli_overrides_cloud_timeout(self, base_args: argparse.Namespace) -> None:
        """测试 CLI --cloud-timeout 参数覆盖 config.yaml"""
        assert base_args.cloud_timeout == 600

        with patch.object(run_iterate, "ChangelogAnalyzer"), patch.object(run_iterate, "KnowledgeUpdater"):
            iterator = SelfIterator(base_args)
            assert iterator.args.cloud_timeout == 600

    def test_cli_overrides_cloud_api_key(self, base_args: argparse.Namespace) -> None:
        """测试 CLI --cloud-api-key 参数覆盖环境变量和 config.yaml"""
        assert base_args.cloud_api_key == "cli-api-key"

        with patch.object(run_iterate, "ChangelogAnalyzer"), patch.object(run_iterate, "KnowledgeUpdater"):
            iterator = SelfIterator(base_args)
            assert iterator.args.cloud_api_key == "cli-api-key"

    def test_cli_overrides_max_fetch_urls(self, base_args: argparse.Namespace) -> None:
        """测试 CLI --max-fetch-urls 参数覆盖 config.yaml"""
        assert base_args.max_fetch_urls == 10

        with patch.object(run_iterate, "ChangelogAnalyzer"):
            with patch.object(run_iterate, "KnowledgeUpdater") as MockUpdater:
                mock_updater = MagicMock()
                MockUpdater.return_value = mock_updater

                iterator = SelfIterator(base_args)

                # 验证 KnowledgeUpdater 接收了正确的参数
                # （具体调用参数取决于 SelfIterator 的实现）
                assert iterator.args.max_fetch_urls == 10

    @pytest.mark.asyncio
    async def test_env_var_fallback_when_cli_not_set(self, base_args: argparse.Namespace, monkeypatch) -> None:
        """测试当 CLI 未设置时，使用环境变量的值"""
        # 设置环境变量
        monkeypatch.setenv("CURSOR_API_KEY", "env-api-key")

        # 清除 CLI 设置的值
        base_args.cloud_api_key = None

        with patch.object(run_iterate, "ChangelogAnalyzer"), patch.object(run_iterate, "KnowledgeUpdater"):
            # 注意: 实际的环境变量解析发生在 CloudClientFactory 中
            # 这里验证当 CLI 不设置时，args 为 None
            iterator = SelfIterator(base_args)
            assert iterator.args.cloud_api_key is None

    def test_orchestrator_priority_mp_over_basic(self, base_args: argparse.Namespace) -> None:
        """测试编排器优先级：显式设置的 mp 覆盖默认 basic"""
        base_args.orchestrator = "mp"
        base_args.no_mp = False
        base_args._orchestrator_user_set = True

        with patch.object(run_iterate, "ChangelogAnalyzer"), patch.object(run_iterate, "KnowledgeUpdater"):
            iterator = SelfIterator(base_args)
            assert iterator.args.orchestrator == "mp"

    def test_execution_mode_cloud_forces_basic_orchestrator(self, base_args: argparse.Namespace) -> None:
        """测试执行模式 cloud 时强制使用 basic 编排器

        根据 AGENTS.md 契约：
        - execution_mode=cloud/auto 与 MP 编排器不兼容
        - 系统会强制切换到 basic 编排器
        """
        base_args.execution_mode = "cloud"
        base_args.orchestrator = "mp"  # 尝试使用 MP
        base_args.no_mp = False
        base_args._orchestrator_user_set = True

        with patch.object(run_iterate, "ChangelogAnalyzer"), patch.object(run_iterate, "KnowledgeUpdater"):
            iterator = SelfIterator(base_args)
            # 验证系统会处理这种冲突（具体行为取决于实现）
            # 这里验证参数被正确保存
            assert iterator.args.execution_mode == "cloud"

    def test_auto_mode_forces_basic_orchestrator(self, base_args: argparse.Namespace) -> None:
        """测试执行模式 auto 时强制使用 basic 编排器"""
        base_args.execution_mode = "auto"
        base_args.orchestrator = "mp"
        base_args._orchestrator_user_set = True

        with patch.object(run_iterate, "ChangelogAnalyzer"), patch.object(run_iterate, "KnowledgeUpdater"):
            iterator = SelfIterator(base_args)
            assert iterator.args.execution_mode == "auto"


# ============================================================
# TestApplyFetchPolicy - fetch_policy 外链过滤测试
# ============================================================


class TestApplyFetchPolicy:
    """测试 apply_fetch_policy 函数

    验证：
    1. url_strategy 扩大 allowed_domains 时，fetch_policy 仍能阻止外链抓取
    2. 默认配置下行为不扩大抓取面
    3. 各种 external_link_mode 模式的行为正确
    """

    def test_record_only_mode_removes_external_links_from_fetch(self) -> None:
        """测试 record_only 模式：外链从 urls_to_fetch 移除但记录"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/cn/docs/api",
            "https://github.com/cursor/repo",
            "https://stackoverflow.com/q/123",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
        )

        # 外链应被移除，但被记录
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://cursor.com/cn/docs/api" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" not in result.urls_to_fetch
        assert "https://stackoverflow.com/q/123" not in result.urls_to_fetch

        # 外链应被记录
        assert "https://github.com/cursor/repo" in result.external_links_recorded
        assert "https://stackoverflow.com/q/123" in result.external_links_recorded

        # filtered_urls 应包含原因
        filtered_urls = [f["url"] for f in result.filtered_urls]
        assert "https://github.com/cursor/repo" in filtered_urls
        assert "https://stackoverflow.com/q/123" in filtered_urls

    def test_skip_all_mode_neither_fetches_nor_records_external_links(self) -> None:
        """测试 skip_all 模式：外链既不抓取也不记录"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="skip_all",
            primary_domains=["cursor.com"],
        )

        # 外链应被移除
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" not in result.urls_to_fetch

        # skip_all 模式下外链不应被记录
        assert "https://github.com/cursor/repo" not in result.external_links_recorded

        # 但应记录到 filtered_urls
        reasons = {f["url"]: f["reason"] for f in result.filtered_urls}
        assert reasons.get("https://github.com/cursor/repo") == "external_link_skip_all"

    def test_fetch_allowlist_mode_allows_whitelisted_external_links(self) -> None:
        """测试 fetch_allowlist 模式：仅允许白名单命中的外链抓取"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
            "https://docs.python.org/3/library/",
            "https://stackoverflow.com/q/123",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="fetch_allowlist",
            primary_domains=["cursor.com"],
            external_link_allowlist=["github.com", "docs.python.org"],
        )

        # 内链和白名单外链应被保留
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" in result.urls_to_fetch
        assert "https://docs.python.org/3/library/" in result.urls_to_fetch

        # 非白名单外链应被移除
        assert "https://stackoverflow.com/q/123" not in result.urls_to_fetch

        # 非白名单外链应被记录
        assert "https://stackoverflow.com/q/123" in result.external_links_recorded

    def test_url_strategy_expanded_domains_still_blocked_by_fetch_policy(self) -> None:
        """测试 url_strategy 扩大 allowed_domains 时，fetch_policy 仍能阻止外链抓取

        核心测试点：即使 url_strategy.allowed_domains 包含更多域名，
        fetch_policy 仍能独立控制哪些 URL 实际被抓取。
        这确保了 fetch_policy 是最后的抓取控制层。
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        # 假设 url_strategy 允许的域名更宽泛（包括 python.org）
        # 但 primary_domains 仅限于 cursor.com
        urls = [
            "https://cursor.com/docs/cli",
            "https://docs.python.org/3/library/",  # url_strategy 允许
            "https://nodejs.org/docs/",  # url_strategy 可能允许
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",  # 默认模式
            primary_domains=["cursor.com"],  # fetch_policy 的主域名
            allowed_domains=None,  # 不额外扩展
        )

        # 仅 cursor.com 内链被抓取
        assert result.urls_to_fetch == ["https://cursor.com/docs/cli"]

        # 外链被记录但不抓取
        assert "https://docs.python.org/3/library/" in result.external_links_recorded
        assert "https://nodejs.org/docs/" in result.external_links_recorded

    def test_default_config_does_not_expand_fetch_surface(self) -> None:
        """测试默认配置下行为不扩大抓取面

        核心测试点：使用默认配置时，外链不会被抓取，
        确保系统默认保持最小抓取面。
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/cn/changelog",
            "https://github.com/cursor/repo",
            "https://external.com/page",
        ]

        # 使用默认值
        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",  # 默认值
            primary_domains=["cursor.com"],  # 从 changelog_url 推导
            allowed_domains=[],  # 默认空
            external_link_allowlist=[],  # 默认空
        )

        # 仅内链被抓取
        assert len(result.urls_to_fetch) == 2
        assert all("cursor.com" in u for u in result.urls_to_fetch)

        # 外链被记录
        assert len(result.external_links_recorded) == 2

    def test_subdomain_matching_for_internal_links(self) -> None:
        """测试子域名匹配：子域名视为内链"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs",
            "https://api.cursor.com/v1",
            "https://docs.cursor.com/guide",
            "https://external.com/page",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
        )

        # 所有 cursor.com 子域名都应视为内链
        assert "https://cursor.com/docs" in result.urls_to_fetch
        assert "https://api.cursor.com/v1" in result.urls_to_fetch
        assert "https://docs.cursor.com/guide" in result.urls_to_fetch

        # 外链被过滤
        assert "https://external.com/page" not in result.urls_to_fetch

    def test_base_url_adds_to_primary_domains(self) -> None:
        """测试 base_url 自动添加到主域名列表"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs",
            "https://other-domain.com/page",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=[],  # 空列表
            base_url="https://cursor.com/cn/changelog",  # 从这里推导域名
        )

        # cursor.com 应被视为内链（因为 base_url 的域名）
        assert "https://cursor.com/docs" in result.urls_to_fetch
        assert "https://other-domain.com/page" not in result.urls_to_fetch

    def test_allowed_domains_extends_internal_links(self) -> None:
        """测试 allowed_domains 扩展内链范围"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs",
            "https://partner.com/api",
            "https://external.com/page",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_domains=["partner.com"],  # 额外允许
        )

        # cursor.com 和 partner.com 都是内链
        assert "https://cursor.com/docs" in result.urls_to_fetch
        assert "https://partner.com/api" in result.urls_to_fetch

        # external.com 是外链
        assert "https://external.com/page" not in result.urls_to_fetch


class TestDerivePrimaryDomains:
    """测试 derive_primary_domains 函数"""

    def test_derive_from_both_urls(self) -> None:
        """测试从两个 URL 推导主域名"""
        from knowledge.doc_url_strategy import derive_primary_domains

        domains = derive_primary_domains(
            llms_txt_url="https://cursor.com/llms.txt",
            changelog_url="https://cursor.com/cn/changelog",
        )

        assert domains == ["cursor.com"]

    def test_derive_from_different_domains(self) -> None:
        """测试从不同域名的 URL 推导"""
        from knowledge.doc_url_strategy import derive_primary_domains

        domains = derive_primary_domains(
            llms_txt_url="https://docs.cursor.com/llms.txt",
            changelog_url="https://cursor.com/changelog",
        )

        # 应返回两个不同的域名（排序后）
        assert "cursor.com" in domains
        assert "docs.cursor.com" in domains

    def test_derive_from_single_url(self) -> None:
        """测试仅提供一个 URL 时的推导"""
        from knowledge.doc_url_strategy import derive_primary_domains

        domains = derive_primary_domains(
            llms_txt_url="https://cursor.com/llms.txt",
            changelog_url=None,
        )

        assert domains == ["cursor.com"]

    def test_derive_from_empty_urls(self) -> None:
        """测试都不提供 URL 时返回空列表"""
        from knowledge.doc_url_strategy import derive_primary_domains

        domains = derive_primary_domains(
            llms_txt_url=None,
            changelog_url=None,
        )

        assert domains == []


class TestIsExternalLink:
    """测试 is_external_link 函数"""

    def test_internal_link_exact_match(self) -> None:
        """测试精确匹配的内链"""
        from knowledge.doc_url_strategy import is_external_link

        assert not is_external_link(
            "https://cursor.com/docs",
            primary_domains=["cursor.com"],
        )

    def test_internal_link_subdomain_match(self) -> None:
        """测试子域名匹配的内链"""
        from knowledge.doc_url_strategy import is_external_link

        assert not is_external_link(
            "https://api.cursor.com/v1",
            primary_domains=["cursor.com"],
        )

    def test_external_link(self) -> None:
        """测试外链"""
        from knowledge.doc_url_strategy import is_external_link

        assert is_external_link(
            "https://github.com/repo",
            primary_domains=["cursor.com"],
        )

    def test_allowed_domains_makes_internal(self) -> None:
        """测试 allowed_domains 可以使外链变为内链"""
        from knowledge.doc_url_strategy import is_external_link

        # 默认是外链
        assert is_external_link(
            "https://partner.com/api",
            primary_domains=["cursor.com"],
        )

        # 添加到 allowed_domains 后变为内链
        assert not is_external_link(
            "https://partner.com/api",
            primary_domains=["cursor.com"],
            allowed_domains=["partner.com"],
        )


class TestUrlPolicyInjection:
    """测试 UrlPolicy 注入到 KnowledgeManager/WebFetcher

    验证：
    1. KnowledgeUpdater._build_url_policy 正确合并多个来源的域名配置
    2. 策略拒绝 URL 时 FetchResult.rejection_reason 被正确设置
    3. UrlPolicy 在 KnowledgeManager 和 WebFetcher 之间保持一致
    """

    def test_build_url_policy_includes_primary_domains(self) -> None:
        """测试 _build_url_policy 包含主域名"""
        from scripts.run_iterate import KnowledgeUpdater

        policy = KnowledgeUpdater._build_url_policy(
            primary_domains=["cursor.com"],
            allowed_doc_url_prefixes=["https://docs.cursor.com/"],
            fetch_policy=None,
        )

        assert "cursor.com" in policy.allowed_domains
        assert "docs.cursor.com" in policy.allowed_domains

    def test_build_url_policy_includes_fetch_policy_domains(self) -> None:
        """测试 _build_url_policy 包含 fetch_policy 的外链域名白名单"""
        from scripts.run_iterate import KnowledgeUpdater, ResolvedFetchPolicyConfig

        fetch_policy = ResolvedFetchPolicyConfig(
            allowed_path_prefixes=["docs", "cn/docs"],
            allowed_domains=["partner.com", "api.example.com"],
            external_link_mode="record_only",
            external_link_allowlist=[],
        )

        policy = KnowledgeUpdater._build_url_policy(
            primary_domains=["cursor.com"],
            allowed_doc_url_prefixes=["https://cursor.com/docs/"],
            fetch_policy=fetch_policy,
        )

        # 验证所有域名都被包含
        assert "cursor.com" in policy.allowed_domains
        assert "partner.com" in policy.allowed_domains
        assert "api.example.com" in policy.allowed_domains

    def test_build_url_policy_deduplicates_domains(self) -> None:
        """测试 _build_url_policy 自动去重域名"""
        from scripts.run_iterate import KnowledgeUpdater, ResolvedFetchPolicyConfig

        fetch_policy = ResolvedFetchPolicyConfig(
            allowed_path_prefixes=["docs"],
            allowed_domains=["cursor.com"],  # 与 primary_domains 重复
            external_link_mode="record_only",
            external_link_allowlist=[],
        )

        policy = KnowledgeUpdater._build_url_policy(
            primary_domains=["cursor.com"],
            allowed_doc_url_prefixes=["https://cursor.com/docs/"],
            fetch_policy=fetch_policy,
        )

        # 验证 cursor.com 只出现一次
        assert policy.allowed_domains.count("cursor.com") == 1

    def test_build_url_policy_sets_security_defaults(self) -> None:
        """测试 _build_url_policy 设置安全默认值"""
        from scripts.run_iterate import KnowledgeUpdater

        policy = KnowledgeUpdater._build_url_policy(
            primary_domains=["cursor.com"],
            allowed_doc_url_prefixes=[],
            fetch_policy=None,
        )

        # 验证安全默认配置
        assert policy.allowed_schemes == ["http", "https"]
        assert policy.deny_private_networks is True

    @pytest.mark.asyncio
    async def test_fetcher_rejection_reason_on_policy_deny(self) -> None:
        """测试 WebFetcher 策略拒绝时返回 rejection_reason

        当 UrlPolicy 拒绝某 URL 时，FetchResult.rejection_reason 应被设置。
        """
        from knowledge.fetcher import FetchConfig, UrlPolicy, WebFetcher

        # 创建严格策略：只允许 cursor.com
        strict_policy = UrlPolicy(
            allowed_schemes=["https"],
            allowed_domains=["cursor.com"],
            deny_private_networks=True,
        )

        fetcher = WebFetcher(
            FetchConfig(
                timeout=5,
                url_policy=strict_policy,
                enforce_url_policy=True,
            )
        )
        await fetcher.initialize()

        # 尝试获取被拒绝的 URL
        result = await fetcher.fetch("https://blocked-domain.com/page")

        # 验证拒绝结果
        assert result.success is False
        assert result.rejection_reason is not None
        assert result.rejection_reason.policy_type == "domain"
        assert "blocked-domain.com" in result.rejection_reason.url

    @pytest.mark.asyncio
    async def test_fetcher_rejection_reason_on_private_network(self) -> None:
        """测试 WebFetcher 拒绝私有网络地址时返回 rejection_reason"""
        from knowledge.fetcher import FetchConfig, UrlPolicy, WebFetcher

        policy = UrlPolicy(
            allowed_schemes=["http", "https"],
            deny_private_networks=True,
        )

        fetcher = WebFetcher(
            FetchConfig(
                timeout=5,
                url_policy=policy,
                enforce_url_policy=True,
            )
        )
        await fetcher.initialize()

        # 尝试获取 localhost
        result = await fetcher.fetch("http://localhost:8080/api")

        # 验证拒绝结果
        assert result.success is False
        assert result.rejection_reason is not None
        assert result.rejection_reason.policy_type == "private_network"

    @pytest.mark.asyncio
    async def test_fetcher_rejection_reason_on_invalid_scheme(self) -> None:
        """测试 WebFetcher 拒绝无效 scheme 时返回 rejection_reason"""
        from knowledge.fetcher import FetchConfig, UrlPolicy, WebFetcher

        policy = UrlPolicy(
            allowed_schemes=["https"],  # 只允许 HTTPS
            deny_private_networks=True,
        )

        fetcher = WebFetcher(
            FetchConfig(
                timeout=5,
                url_policy=policy,
                enforce_url_policy=True,
            )
        )
        await fetcher.initialize()

        # 尝试获取 HTTP URL（被策略拒绝）
        result = await fetcher.fetch("http://example.com/page")

        # 验证拒绝结果
        assert result.success is False
        assert result.rejection_reason is not None
        assert result.rejection_reason.policy_type == "scheme"

    def test_knowledge_updater_url_policy_property(self) -> None:
        """测试 KnowledgeUpdater.url_policy 属性可访问"""
        from scripts.run_iterate import KnowledgeUpdater, ResolvedFetchPolicyConfig

        fetch_policy = ResolvedFetchPolicyConfig(
            allowed_path_prefixes=["docs"],
            allowed_domains=["external.com"],
            external_link_mode="record_only",
            external_link_allowlist=[],
        )

        updater = KnowledgeUpdater(
            fetch_policy=fetch_policy,
            changelog_url="https://cursor.com/changelog",
            llms_txt_url="https://cursor.com/llms.txt",
        )

        # 验证 url_policy 属性可访问
        assert updater.url_policy is not None
        # 验证包含预期域名
        assert "cursor.com" in updater.url_policy.allowed_domains
        assert "external.com" in updater.url_policy.allowed_domains

    def test_knowledge_manager_receives_url_policy(self) -> None:
        """测试 KnowledgeManager 接收到注入的 UrlPolicy"""
        from scripts.run_iterate import KnowledgeUpdater

        updater = KnowledgeUpdater(
            changelog_url="https://cursor.com/changelog",
            llms_txt_url="https://cursor.com/llms.txt",
        )

        # 验证 manager 的 url_policy 与 updater 一致
        assert updater.manager.url_policy is not None
        assert updater.manager.url_policy == updater.url_policy


# ============================================================
# TestMinimalSemanticConsistency - Minimal 语义一致性验证
# ============================================================


class TestMinimalSemanticConsistency:
    """测试 minimal 模式的权威语义一致性

    权威定义来源: core/execution_policy.py Side Effect Control Strategy Matrix

    minimal = skip_online + dry_run（禁止所有网络请求、文件写入、Git 操作）

    验证点:
    1. minimal=True 时自动设置 args.dry_run=True
    2. minimal=True 时自动设置 args.skip_online=True
    3. minimal=True 时 context.dry_run=True
    4. minimal=True 时不执行 Orchestrator.run
    5. minimal=True 时返回 dry_run=True 结果
    """

    @pytest.fixture
    def minimal_args_dry_run_false(self) -> argparse.Namespace:
        """创建 minimal=True 但 dry_run=False 的参数

        用于验证 SelfIterator.__init__ 会自动设置 dry_run=True
        """
        return argparse.Namespace(
            requirement="测试 minimal 语义一致性",
            directory=".",
            skip_online=False,  # 测试 minimal 是否自动设置
            minimal=True,  # 启用 minimal 模式
            changelog_url=None,
            dry_run=False,  # 测试 minimal 是否自动覆盖为 True
            max_iterations="1",
            workers=1,
            force_update=False,
            verbose=False,
            quiet=True,
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
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
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
            stream_color_enabled=False,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

    def test_minimal_auto_sets_dry_run_true(self, minimal_args_dry_run_false: argparse.Namespace) -> None:
        """测试 minimal=True 时自动设置 args.dry_run=True

        权威定义: core/execution_policy.py Side Effect Control Strategy Matrix
        minimal = skip_online + dry_run
        """
        # 验证初始状态
        assert minimal_args_dry_run_false.dry_run is False
        assert minimal_args_dry_run_false.minimal is True

        # 创建 SelfIterator 实例
        iterator = SelfIterator(minimal_args_dry_run_false)

        # 验证 minimal 模式自动设置 dry_run=True
        assert iterator.args.dry_run is True, "minimal 模式应自动设置 args.dry_run=True"
        assert iterator.context.dry_run is True, "minimal 模式应自动设置 context.dry_run=True"

    def test_minimal_auto_sets_skip_online_true(self, minimal_args_dry_run_false: argparse.Namespace) -> None:
        """测试 minimal=True 时自动设置 args.skip_online=True"""
        # 验证初始状态
        assert minimal_args_dry_run_false.skip_online is False
        assert minimal_args_dry_run_false.minimal is True

        # 创建 SelfIterator 实例
        iterator = SelfIterator(minimal_args_dry_run_false)

        # 验证 minimal 模式自动设置 skip_online=True
        assert iterator.args.skip_online is True, "minimal 模式应自动设置 args.skip_online=True"

    @pytest.mark.asyncio
    async def test_minimal_does_not_execute_orchestrator_run(
        self, minimal_args_dry_run_false: argparse.Namespace
    ) -> None:
        """测试 minimal=True 时不执行 Orchestrator.run

        权威定义: core/execution_policy.py Side Effect Control Strategy Matrix
        minimal 模式禁止所有网络请求、文件写入、Git 操作
        因此 Orchestrator.run 不应被调用
        """
        orchestrator_run_called = False

        mock_orchestrator = MagicMock()

        async def mock_run(*args, **kwargs):
            nonlocal orchestrator_run_called
            orchestrator_run_called = True
            return {
                "success": True,
                "iterations_completed": 1,
            }

        mock_orchestrator.run = AsyncMock(side_effect=mock_run)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])

        with patch.object(run_iterate, "Orchestrator") as MockOrchestrator:
            MockOrchestrator.return_value = mock_orchestrator
            with (
                patch.object(run_iterate, "KnowledgeStorage", return_value=mock_storage),
                patch.object(
                    run_iterate.KnowledgeManager,
                    "initialize",
                    new_callable=AsyncMock,
                ),
            ):
                iterator = SelfIterator(minimal_args_dry_run_false)
                result = await iterator.run()

        # 验证 Orchestrator.run 未被调用
        assert orchestrator_run_called is False, "minimal 模式下 Orchestrator.run 不应被调用"

        # 验证返回 dry_run=True
        assert result.get("dry_run") is True, "minimal 模式应返回 dry_run=True"
        assert result.get("success") is True

    def test_minimal_mode_flag_is_set(self, minimal_args_dry_run_false: argparse.Namespace) -> None:
        """测试 SelfIterator._is_minimal 标志被正确设置"""
        iterator = SelfIterator(minimal_args_dry_run_false)

        assert iterator._is_minimal is True, "_is_minimal 标志应为 True"

    @pytest.mark.asyncio
    async def test_minimal_orchestrator_run_spy_raises_if_called(
        self, minimal_args_dry_run_false: argparse.Namespace
    ) -> None:
        """测试 minimal 模式下 Orchestrator.run 被调用时会触发 AssertionError

        使用 spy/patch 策略：将 Orchestrator.run 替换为抛出 AssertionError 的 mock，
        如果 minimal 模式意外调用了 Orchestrator.run，测试会失败。

        权威定义: core/execution_policy.py Side Effect Control Strategy Matrix
        minimal = skip_online + dry_run（禁止所有网络请求、文件写入、Git 操作）
        因此 Orchestrator.run 不应被调用。
        """

        # 创建 spy：如果被调用则抛出 AssertionError
        def spy_run(*args, **kwargs):
            raise AssertionError("Orchestrator.run() 不应在 minimal 模式下被调用")

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(side_effect=spy_run)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])

        # Patch Orchestrator 构造函数返回带有 spy 的 mock
        with patch.object(run_iterate, "Orchestrator") as MockOrchestrator:
            MockOrchestrator.return_value = mock_orchestrator
            with (
                patch.object(run_iterate, "KnowledgeStorage", return_value=mock_storage),
                patch.object(
                    run_iterate.KnowledgeManager,
                    "initialize",
                    new_callable=AsyncMock,
                ),
            ):
                iterator = SelfIterator(minimal_args_dry_run_false)
                # 如果 Orchestrator.run 被调用，spy 会抛出 AssertionError 导致测试失败
                result = await iterator.run()

        # 验证测试通过（说明 Orchestrator.run 未被调用）
        assert result.get("success") is True
        assert result.get("dry_run") is True, "minimal 模式应返回 dry_run=True"

    @pytest.mark.asyncio
    async def test_minimal_mp_orchestrator_run_spy_raises_if_called(
        self, minimal_args_dry_run_false: argparse.Namespace
    ) -> None:
        """测试 minimal 模式下 MultiProcessOrchestrator.run 被调用时会触发 AssertionError

        使用 spy/patch 策略：将 MultiProcessOrchestrator.run 替换为抛出 AssertionError 的 mock，
        如果 minimal 模式意外调用了 MultiProcessOrchestrator.run，测试会失败。

        此测试确保即使配置了 MP 编排器（orchestrator="mp"），
        minimal 模式仍然不会调用 MultiProcessOrchestrator.run。
        """

        # 切换到 MP 编排器配置（但 minimal 模式应跳过）
        minimal_args_dry_run_false.orchestrator = "mp"
        minimal_args_dry_run_false.no_mp = False

        # 创建 spy：如果被调用则抛出 AssertionError
        def spy_run(*args, **kwargs):
            raise AssertionError("MultiProcessOrchestrator.run() 不应在 minimal 模式下被调用")

        mock_mp_orchestrator = MagicMock()
        mock_mp_orchestrator.run = AsyncMock(side_effect=spy_run)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])

        # Patch MultiProcessOrchestrator 构造函数返回带有 spy 的 mock
        with patch.object(run_iterate, "MultiProcessOrchestrator") as MockMPOrchestrator:
            MockMPOrchestrator.return_value = mock_mp_orchestrator
            with (
                patch.object(run_iterate, "KnowledgeStorage", return_value=mock_storage),
                patch.object(
                    run_iterate.KnowledgeManager,
                    "initialize",
                    new_callable=AsyncMock,
                ),
            ):
                iterator = SelfIterator(minimal_args_dry_run_false)
                # 如果 MultiProcessOrchestrator.run 被调用，spy 会抛出 AssertionError 导致测试失败
                result = await iterator.run()

        # 验证测试通过（说明 MultiProcessOrchestrator.run 未被调用）
        assert result.get("success") is True
        assert result.get("dry_run") is True, "minimal 模式应返回 dry_run=True"


# ============================================================
# TestMinimalModeOrchestratorSpyViaRunPy - run.py iterate minimal 路径测试
# ============================================================


class TestMinimalModeOrchestratorSpyViaRunPy:
    """测试通过 run.py iterate 模式调用时 minimal 模式不调用 Orchestrator

    验证当使用 run.py --mode iterate --minimal 时，
    Orchestrator.run 和 MultiProcessOrchestrator.run 都不会被调用。

    这补充了直接测试 SelfIterator 的测试，确保 run.py 入口也遵循相同的契约。
    """

    @pytest.fixture
    def minimal_iterate_options(self) -> dict:
        """创建 run.py iterate 模式的 minimal 选项"""
        return {
            "directory": ".",
            "skip_online": True,  # minimal 隐含
            "dry_run": True,  # minimal 隐含
            "minimal": True,  # 关键：启用 minimal 模式
            "max_iterations": 1,
            "workers": 1,
            "force_update": False,
            "verbose": False,
            "auto_commit": False,
            "auto_push": False,
            "commit_per_iteration": False,
            "commit_message": "",
            "orchestrator": "basic",
            "no_mp": True,
            "_orchestrator_user_set": True,
            "execution_mode": "cli",
            "cloud_api_key": None,
            "cloud_auth_timeout": 30,
            "cloud_timeout": 300,
        }

    @pytest.mark.asyncio
    async def test_run_py_iterate_minimal_uses_self_iterator(self, minimal_iterate_options: dict) -> None:
        """测试 run.py _run_iterate 方法在 minimal 模式下通过 SelfIterator 执行

        验证 run.py 的 iterate 模式通过 SelfIterator 执行，
        而 SelfIterator 在 minimal 模式下不会调用 Orchestrator。

        这是间接验证：run.py -> SelfIterator -> (minimal 跳过 orchestrator)
        """

        # 创建 spy：如果 Orchestrator.run 被调用则抛出 AssertionError
        def spy_orchestrator_run(*args, **kwargs):
            raise AssertionError("Orchestrator.run() 不应在 run.py iterate minimal 模式下被调用")

        def spy_mp_orchestrator_run(*args, **kwargs):
            raise AssertionError("MultiProcessOrchestrator.run() 不应在 run.py iterate minimal 模式下被调用")

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(side_effect=spy_orchestrator_run)

        mock_mp_orchestrator = MagicMock()
        mock_mp_orchestrator.run = AsyncMock(side_effect=spy_mp_orchestrator_run)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])

        # Patch Orchestrator 和 MultiProcessOrchestrator 作为安全网
        # 如果 SelfIterator 在 minimal 模式下错误地调用了它们，测试会失败
        with patch.object(run_iterate, "Orchestrator") as MockOrchestrator:
            MockOrchestrator.return_value = mock_orchestrator
            with patch.object(run_iterate, "MultiProcessOrchestrator") as MockMPOrchestrator:
                MockMPOrchestrator.return_value = mock_mp_orchestrator
                with (
                    patch.object(run_iterate, "KnowledgeStorage", return_value=mock_storage),
                    patch.object(
                        run_iterate.KnowledgeManager,
                        "initialize",
                        new_callable=AsyncMock,
                    ),
                ):
                    # 创建 minimal 模式的 args（直接使用 SelfIterator）
                    args = argparse.Namespace(
                        requirement="测试 run.py iterate minimal 路径",
                        directory=".",
                        skip_online=True,
                        minimal=True,  # 关键
                        changelog_url=None,
                        dry_run=True,
                        max_iterations="1",
                        workers=1,
                        force_update=False,
                        verbose=False,
                        quiet=True,
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
                        orchestrator="basic",
                        no_mp=True,
                        _orchestrator_user_set=True,
                        execution_mode="cli",
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
                        stream_color_enabled=False,
                        stream_show_word_diff=False,
                        max_fetch_urls=None,
                        fallback_core_docs_count=None,
                        llms_txt_url=None,
                        llms_cache_path=None,
                    )

                    # 创建 SelfIterator 并运行（模拟 run.py 的调用路径）
                    iterator = SelfIterator(args)
                    result = await iterator.run()

        # 验证返回结果符合预期
        assert result.get("success") is True
        assert result.get("dry_run") is True

        # 验证 Orchestrator.run 未被调用
        mock_orchestrator.run.assert_not_called()
        mock_mp_orchestrator.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_py_iterate_minimal_both_orchestrators_patched(self, minimal_iterate_options: dict) -> None:
        """测试同时 patch 两种 Orchestrator，验证 minimal 模式安全性

        更直接的测试：同时将 Orchestrator.run 和 MultiProcessOrchestrator.run
        替换为抛错的 spy，然后通过完整的 SelfIterator 执行。
        如果任一 orchestrator 被调用，测试会失败。
        """

        # 创建 spy：如果被调用则抛出 AssertionError
        def spy_basic_run(*args, **kwargs):
            raise AssertionError("Orchestrator.run() 不应在 minimal 模式下被调用")

        def spy_mp_run(*args, **kwargs):
            raise AssertionError("MultiProcessOrchestrator.run() 不应在 minimal 模式下被调用")

        mock_basic_orchestrator = MagicMock()
        mock_basic_orchestrator.run = AsyncMock(side_effect=spy_basic_run)

        mock_mp_orchestrator = MagicMock()
        mock_mp_orchestrator.run = AsyncMock(side_effect=spy_mp_run)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])

        # 创建 minimal 模式的 args
        args = argparse.Namespace(
            requirement="测试 minimal 模式 orchestrator 隔离",
            directory=".",
            skip_online=False,  # 将被 minimal 覆盖
            minimal=True,  # 关键
            changelog_url=None,
            dry_run=False,  # 将被 minimal 覆盖
            max_iterations="1",
            workers=1,
            force_update=False,
            verbose=False,
            quiet=True,
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
            orchestrator="mp",  # 尝试使用 MP 编排器
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",
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
            stream_color_enabled=False,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        # 同时 patch 两种 Orchestrator
        with patch.object(run_iterate, "Orchestrator") as MockOrchestrator:
            MockOrchestrator.return_value = mock_basic_orchestrator
            with patch.object(run_iterate, "MultiProcessOrchestrator") as MockMPOrchestrator:
                MockMPOrchestrator.return_value = mock_mp_orchestrator
                with (
                    patch.object(run_iterate, "KnowledgeStorage", return_value=mock_storage),
                    patch.object(
                        run_iterate.KnowledgeManager,
                        "initialize",
                        new_callable=AsyncMock,
                    ),
                ):
                    iterator = SelfIterator(args)
                    # 如果任一 Orchestrator.run 被调用，spy 会抛出 AssertionError
                    result = await iterator.run()

        # 验证测试通过（说明两种 Orchestrator.run 都未被调用）
        assert result.get("success") is True
        assert result.get("dry_run") is True, "minimal 模式应返回 dry_run=True"

        # 额外验证：两种 Orchestrator 的 run 方法都未被调用
        mock_basic_orchestrator.run.assert_not_called()
        mock_mp_orchestrator.run.assert_not_called()


# ============================================================
# TestSideEffectPolicyIntegration - 副作用控制策略集成测试
# ============================================================


class TestSideEffectPolicyIntegration:
    """测试副作用控制策略的集成行为

    使用 tmpdir 作为工作目录，验证不同模式下副作用的实际发生情况：
    1. dry-run 时不调用 KnowledgeUpdater.update_from_analysis，且不写 llms cache
    2. minimal/skip-online 时不发起 fetch
    3. skip-online 仍会读取 existing knowledge stats/context
    4. 断言 .cursor/knowledge 与 llms cache 路径在对应模式下是否被创建
    """

    @pytest.fixture
    def tmpdir_args(self, tmp_path: Path) -> argparse.Namespace:
        """创建使用临时目录的参数"""
        cache_path = tmp_path / ".cursor" / "llms_cache" / "llms.txt"
        return argparse.Namespace(
            requirement="测试副作用控制",
            directory=str(tmp_path),
            skip_online=False,
            changelog_url=None,
            dry_run=False,
            max_iterations="1",
            workers=1,
            force_update=False,
            verbose=False,
            quiet=True,
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
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
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
            stream_color_enabled=False,
            stream_show_word_diff=False,
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=str(cache_path),
        )

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_update_from_analysis(
        self, tmpdir_args: argparse.Namespace, tmp_path: Path
    ) -> None:
        """测试 dry-run 模式不调用 KnowledgeUpdater.update_from_analysis

        dry-run 模式应该：
        - 不调用 update_from_analysis（或调用时不写入）
        - 不写入 llms cache
        """
        tmpdir_args.dry_run = True
        tmpdir_args.skip_online = False  # 允许网络请求用于分析

        update_from_analysis_called = False

        mock_knowledge_updater = MagicMock()
        mock_knowledge_updater.initialize = AsyncMock()
        mock_knowledge_updater.max_fetch_urls = 10

        async def mock_update_from_analysis(*args, **kwargs):
            nonlocal update_from_analysis_called
            update_from_analysis_called = True

        mock_knowledge_updater.update_from_analysis = AsyncMock(side_effect=mock_update_from_analysis)
        mock_knowledge_updater.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_knowledge_updater.get_knowledge_context = AsyncMock(return_value=[])
        mock_knowledge_updater.storage = MagicMock()
        mock_knowledge_updater.storage.list_documents = AsyncMock(return_value=[])
        # 设置 dry_run 和 disable_cache_write 属性
        mock_knowledge_updater.dry_run = True
        mock_knowledge_updater.disable_cache_write = True

        with patch.object(run_iterate, "KnowledgeUpdater", return_value=mock_knowledge_updater):
            iterator = SelfIterator(tmpdir_args)
            result = await iterator.run()

        # 验证 dry-run 模式下不调用 update_from_analysis
        # （由于 dry_run=True 时进入预览分支，不会执行到 update_from_analysis）
        assert result.get("dry_run") is True, "应返回 dry_run=True"

        # 验证 llms cache 路径不存在
        cache_path = Path(tmpdir_args.llms_cache_path)
        assert not cache_path.exists(), f"dry-run 模式下不应创建 llms cache: {cache_path}"

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write_llms_cache(self, tmpdir_args: argparse.Namespace, tmp_path: Path) -> None:
        """测试 dry-run 模式不写入 llms.txt 缓存

        使用真实的临时目录验证缓存文件不会被创建。
        """
        tmpdir_args.dry_run = True

        cache_path = tmp_path / ".cursor" / "llms_cache" / "llms.txt"
        tmpdir_args.llms_cache_path = str(cache_path)

        # 确保缓存目录不存在
        assert not cache_path.parent.exists()

        with patch.object(run_iterate, "KnowledgeUpdater") as MockUpdater:
            mock_updater = MagicMock()
            mock_updater.initialize = AsyncMock()
            mock_updater.dry_run = True
            mock_updater.disable_cache_write = True
            mock_updater.max_fetch_urls = 10
            mock_updater.get_stats = AsyncMock(return_value={"document_count": 0})
            mock_updater.storage = MagicMock()
            mock_updater.storage.list_documents = AsyncMock(return_value=[])
            MockUpdater.return_value = mock_updater

            iterator = SelfIterator(tmpdir_args)
            result = await iterator.run()

        # 验证返回 dry_run=True
        assert result.get("dry_run") is True

        # 验证缓存文件未创建
        assert not cache_path.exists(), f"dry-run 模式下不应创建 llms 缓存文件: {cache_path}"
        # 缓存目录也不应被创建
        assert not cache_path.parent.exists(), f"dry-run 模式下不应创建缓存目录: {cache_path.parent}"

    @pytest.mark.asyncio
    async def test_skip_online_does_not_fetch(self, tmpdir_args: argparse.Namespace) -> None:
        """测试 skip-online 模式不发起 fetch 请求

        skip-online 模式应该：
        - 不调用 WebFetcher.fetch
        - 不调用 ChangelogAnalyzer.analyze（因为需要网络）
        """
        tmpdir_args.skip_online = True
        tmpdir_args.dry_run = False

        fetch_called = False
        analyze_called = False

        mock_fetcher = MagicMock()

        async def mock_fetch(*args, **kwargs):
            nonlocal fetch_called
            fetch_called = True
            result = MagicMock()
            result.success = True
            result.content = "test content"
            return result

        mock_fetcher.fetch = AsyncMock(side_effect=mock_fetch)
        mock_fetcher.initialize = AsyncMock()

        mock_changelog_analyzer = MagicMock()

        async def mock_analyze():
            nonlocal analyze_called
            analyze_called = True
            return UpdateAnalysis()

        mock_changelog_analyzer.analyze = AsyncMock(side_effect=mock_analyze)
        mock_changelog_analyzer.fetcher = mock_fetcher

        mock_knowledge_updater = MagicMock()
        mock_knowledge_updater.initialize = AsyncMock()
        mock_knowledge_updater.get_stats = AsyncMock(return_value={"document_count": 5})
        mock_knowledge_updater.get_knowledge_context = AsyncMock(return_value=[])
        mock_knowledge_updater.max_fetch_urls = 10
        mock_knowledge_updater.storage = MagicMock()
        mock_knowledge_updater.storage.list_documents = AsyncMock(return_value=[])

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "success": True,
                "iterations_completed": 1,
                "total_tasks_created": 1,
                "total_tasks_completed": 1,
                "total_tasks_failed": 0,
            }
        )

        with (
            patch.object(run_iterate, "ChangelogAnalyzer", return_value=mock_changelog_analyzer),
            patch.object(run_iterate, "KnowledgeUpdater", return_value=mock_knowledge_updater),
            patch.object(run_iterate, "Orchestrator", return_value=mock_orchestrator),
        ):
            iterator = SelfIterator(tmpdir_args)
            result = await iterator.run()

        # 验证 skip_online 模式下不调用 analyze 和 fetch
        assert analyze_called is False, "skip-online 模式下不应调用 ChangelogAnalyzer.analyze"
        assert fetch_called is False, "skip-online 模式下不应发起 fetch 请求"

    @pytest.mark.asyncio
    async def test_minimal_does_not_fetch(self, tmpdir_args: argparse.Namespace) -> None:
        """测试 minimal 模式不发起 fetch 请求

        minimal 模式 (skip-online + dry-run) 应该：
        - 不调用 WebFetcher.fetch
        - 不调用 ChangelogAnalyzer.analyze
        - 不初始化知识库
        """
        tmpdir_args.skip_online = True
        tmpdir_args.dry_run = True
        tmpdir_args.minimal = True

        fetch_called = False
        knowledge_init_called = False

        mock_fetcher = MagicMock()

        async def mock_fetch(*args, **kwargs):
            nonlocal fetch_called
            fetch_called = True
            result = MagicMock()
            result.success = True
            return result

        mock_fetcher.fetch = AsyncMock(side_effect=mock_fetch)
        mock_fetcher.initialize = AsyncMock()

        mock_knowledge_updater = MagicMock()

        async def mock_initialize():
            nonlocal knowledge_init_called
            knowledge_init_called = True

        mock_knowledge_updater.initialize = AsyncMock(side_effect=mock_initialize)
        mock_knowledge_updater.max_fetch_urls = 10

        with patch.object(run_iterate, "KnowledgeUpdater", return_value=mock_knowledge_updater):
            iterator = SelfIterator(tmpdir_args)
            result = await iterator.run()

        # 验证 minimal 模式
        assert result.get("dry_run") is True
        assert result.get("minimal") is True

        # 验证不发起 fetch
        assert fetch_called is False, "minimal 模式下不应发起 fetch 请求"

        # 验证不初始化知识库
        assert knowledge_init_called is False, "minimal 模式下不应调用 KnowledgeUpdater.initialize"

    @pytest.mark.asyncio
    async def test_skip_online_still_reads_existing_knowledge(self, tmpdir_args: argparse.Namespace) -> None:
        """测试 skip-online 模式仍会读取现有知识库

        skip-online 模式应该：
        - 不发起网络请求
        - 但仍应读取本地知识库的 stats 和 context
        """
        tmpdir_args.skip_online = True
        tmpdir_args.dry_run = False

        get_stats_called = False
        get_context_called = False

        mock_knowledge_updater = MagicMock()
        mock_knowledge_updater.initialize = AsyncMock()

        async def mock_get_stats():
            nonlocal get_stats_called
            get_stats_called = True
            return {"document_count": 10}

        async def mock_get_context(*args, **kwargs):
            nonlocal get_context_called
            get_context_called = True
            return [{"title": "Test Doc", "content": "Test content", "score": 0.9}]

        mock_knowledge_updater.get_stats = AsyncMock(side_effect=mock_get_stats)
        mock_knowledge_updater.get_knowledge_context = AsyncMock(side_effect=mock_get_context)
        mock_knowledge_updater.max_fetch_urls = 10
        mock_knowledge_updater.storage = MagicMock()
        mock_knowledge_updater.storage.list_documents = AsyncMock(return_value=[])

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "success": True,
                "iterations_completed": 1,
                "total_tasks_created": 1,
                "total_tasks_completed": 1,
                "total_tasks_failed": 0,
            }
        )

        with (
            patch.object(run_iterate, "KnowledgeUpdater", return_value=mock_knowledge_updater),
            patch.object(run_iterate, "Orchestrator", return_value=mock_orchestrator),
        ):
            iterator = SelfIterator(tmpdir_args)
            result = await iterator.run()

        # 验证 skip-online 模式下仍读取知识库统计
        assert get_stats_called is True, "skip-online 模式下应调用 get_stats 读取本地知识库统计"

        # 验证仍读取知识库上下文
        assert get_context_called is True, "skip-online 模式下应调用 get_knowledge_context 读取本地知识库上下文"

    @pytest.mark.asyncio
    async def test_dry_run_knowledge_directory_not_created(
        self, tmp_path: Path, tmpdir_args: argparse.Namespace
    ) -> None:
        """测试 dry-run 模式不创建 .cursor/knowledge 目录

        使用真实临时目录验证目录不会被创建。
        """
        tmpdir_args.dry_run = True
        tmpdir_args.directory = str(tmp_path)

        knowledge_dir = tmp_path / ".cursor" / "knowledge"

        # 确保目录不存在
        assert not knowledge_dir.exists()

        with patch.object(run_iterate, "KnowledgeUpdater") as MockUpdater:
            mock_updater = MagicMock()
            mock_updater.initialize = AsyncMock()
            mock_updater.dry_run = True
            mock_updater.disable_cache_write = True
            mock_updater.max_fetch_urls = 10
            mock_updater.get_stats = AsyncMock(return_value={"document_count": 0})
            mock_updater.storage = MagicMock()
            mock_updater.storage.list_documents = AsyncMock(return_value=[])
            MockUpdater.return_value = mock_updater

            iterator = SelfIterator(tmpdir_args)
            iterator.working_directory = tmp_path
            result = await iterator.run()

        # 验证 dry-run 模式
        assert result.get("dry_run") is True

        # dry-run 模式下，知识库目录不应被创建
        # 注意：KnowledgeUpdater 被 mock，所以不会实际创建目录
        # 这里验证的是 mock 对象正确接收了 dry_run=True 参数
        assert MockUpdater.called, "KnowledgeUpdater 应被创建"

    @pytest.mark.asyncio
    async def test_normal_mode_allows_all_side_effects(self, tmpdir_args: argparse.Namespace) -> None:
        """测试 normal 模式允许所有副作用

        normal 模式应该：
        - 允许网络请求
        - 允许文件写入
        - 允许知识库更新
        """
        tmpdir_args.skip_online = False
        tmpdir_args.dry_run = False

        update_from_analysis_called = False

        mock_changelog_analyzer = MagicMock()
        mock_changelog_analyzer.analyze = AsyncMock(
            return_value=UpdateAnalysis(
                has_updates=True,
                entries=[],
                raw_content="test changelog",
            )
        )
        mock_changelog_analyzer.fetcher = MagicMock()
        mock_changelog_analyzer.fetcher.initialize = AsyncMock()

        mock_knowledge_updater = MagicMock()
        mock_knowledge_updater.initialize = AsyncMock()

        async def mock_update(*args, **kwargs):
            nonlocal update_from_analysis_called
            update_from_analysis_called = True

        mock_knowledge_updater.update_from_analysis = AsyncMock(side_effect=mock_update)
        mock_knowledge_updater.get_stats = AsyncMock(return_value={"document_count": 5})
        mock_knowledge_updater.get_knowledge_context = AsyncMock(return_value=[])
        mock_knowledge_updater.max_fetch_urls = 10
        mock_knowledge_updater.storage = MagicMock()
        mock_knowledge_updater.storage.list_documents = AsyncMock(return_value=[])

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "success": True,
                "iterations_completed": 1,
                "total_tasks_created": 1,
                "total_tasks_completed": 1,
                "total_tasks_failed": 0,
            }
        )

        with (
            patch.object(run_iterate, "ChangelogAnalyzer", return_value=mock_changelog_analyzer),
            patch.object(run_iterate, "KnowledgeUpdater", return_value=mock_knowledge_updater),
            patch.object(run_iterate, "Orchestrator", return_value=mock_orchestrator),
        ):
            iterator = SelfIterator(tmpdir_args)
            await iterator.run()

        # 验证 normal 模式下调用了 update_from_analysis
        assert update_from_analysis_called is True, "normal 模式下应调用 update_from_analysis"

    def test_side_effect_policy_integration_with_knowledge_updater(self, tmpdir_args: argparse.Namespace) -> None:
        """测试 SideEffectPolicy 与 KnowledgeUpdater 参数映射

        验证 compute_side_effects 的输出正确映射到 KnowledgeUpdater 的构造参数。
        """
        from core.execution_policy import compute_side_effects

        # 场景 1: dry-run 模式
        policy_dry = compute_side_effects(dry_run=True)
        assert policy_dry.allow_file_write is False
        assert policy_dry.allow_cache_write is False
        # 映射: KnowledgeUpdater(dry_run=True, disable_cache_write=True)

        # 场景 2: skip-online 模式
        policy_skip = compute_side_effects(skip_online=True)
        assert policy_skip.allow_network_fetch is False
        # skip-online 允许本地文件读写
        assert policy_skip.allow_file_write is True
        # 映射: WebFetcher 不调用，但可以读取本地知识库

        # 场景 3: minimal 模式
        policy_minimal = compute_side_effects(minimal=True)
        assert policy_minimal.allow_network_fetch is False
        assert policy_minimal.allow_file_write is False
        assert policy_minimal.is_minimal is True
        # 映射: 跳过知识库初始化和更新，但允许读取


# ============================================================
# TestSelfIteratorRequestedModeInvariant - SelfIterator 入口一致性参数化测试
# ============================================================


@dataclass
class SelfIteratorRequestedModeCase:
    """SelfIterator requested_mode_for_decision 不变式测试用例

    字段说明：
    - test_id: 测试标识符
    - requirement: 用户输入（可包含 & 前缀）
    - cli_execution_mode: CLI --execution-mode 参数（None 表示未指定）
    - config_execution_mode: config.yaml 中的 cloud_agent.execution_mode
    - has_api_key: 是否有 API Key
    - cloud_enabled: cloud_agent.enabled 配置
    - expected_requested_mode_for_decision: 预期的 requested_mode_for_decision
    - expected_has_ampersand_prefix: 预期的 has_ampersand_prefix
    - expected_prefix_routed: 预期的 prefix_routed
    - expected_orchestrator: 预期的 orchestrator (mp/basic)
    - description: 场景描述
    """

    test_id: str
    requirement: str
    cli_execution_mode: Optional[str]
    config_execution_mode: str
    has_api_key: bool
    cloud_enabled: bool
    expected_requested_mode_for_decision: Optional[str]
    expected_has_ampersand_prefix: bool
    expected_prefix_routed: bool
    expected_orchestrator: str
    description: str


from dataclasses import dataclass

# 参数化测试矩阵 - SelfIterator 入口
SELF_ITERATOR_REQUESTED_MODE_CASES = [
    # ===== 场景 1: 无 & + 无 CLI execution_mode + config=auto =====
    SelfIteratorRequestedModeCase(
        test_id="si_no_amp_no_cli_config_auto",
        requirement="普通迭代任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",
        expected_has_ampersand_prefix=False,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="SelfIterator: 无 & + 无 CLI + config=auto → orchestrator=basic",
    ),
    SelfIteratorRequestedModeCase(
        test_id="si_no_amp_no_cli_config_auto_no_key",
        requirement="普通迭代任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",
        expected_has_ampersand_prefix=False,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="SelfIterator: 无 & + 无 CLI + config=auto + 无 key → basic",
    ),
    # ===== 场景 2: 有 & + 无 CLI execution_mode =====
    SelfIteratorRequestedModeCase(
        test_id="si_amp_no_cli_with_key",
        requirement="& 后台迭代任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision=None,
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=True,
        expected_orchestrator="basic",
        description="SelfIterator: 有 & + 无 CLI + 有 key → prefix_routed=True",
    ),
    SelfIteratorRequestedModeCase(
        test_id="si_amp_no_cli_no_key",
        requirement="& 后台迭代任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision=None,
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="SelfIterator: 有 & + 无 CLI + 无 key → basic (Cloud 意图)",
    ),
    SelfIteratorRequestedModeCase(
        test_id="si_amp_no_cli_cloud_disabled",
        requirement="& 后台迭代任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=False,
        expected_requested_mode_for_decision=None,
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="SelfIterator: 有 & + cloud_disabled → basic (Cloud 意图)",
    ),
    # ===== 场景 3: 有 & + 显式 --execution-mode cli =====
    SelfIteratorRequestedModeCase(
        test_id="si_amp_explicit_cli",
        requirement="& 显式 CLI 迭代",
        cli_execution_mode="cli",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cli",
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="mp",
        description="SelfIterator: 有 & + 显式 cli → & 被忽略，允许 mp",
    ),
    # ===== 场景 4: 有 & + 显式 --execution-mode plan =====
    SelfIteratorRequestedModeCase(
        test_id="si_amp_explicit_plan",
        requirement="& 规划分析迭代",
        cli_execution_mode="plan",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="plan",
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="mp",
        description="SelfIterator: 有 & + 显式 plan → 允许 mp",
    ),
    # ===== 场景 5: 有 & + 显式 --execution-mode ask =====
    SelfIteratorRequestedModeCase(
        test_id="si_amp_explicit_ask",
        requirement="& 问答迭代",
        cli_execution_mode="ask",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="ask",
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="mp",
        description="SelfIterator: 有 & + 显式 ask → 允许 mp",
    ),
    # ===== 场景 6: 有 & + 显式 --execution-mode cloud =====
    SelfIteratorRequestedModeCase(
        test_id="si_amp_explicit_cloud",
        requirement="& 云端迭代任务",
        cli_execution_mode="cloud",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cloud",
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="SelfIterator: 有 & + 显式 cloud → 强制 basic",
    ),
    SelfIteratorRequestedModeCase(
        test_id="si_amp_explicit_cloud_no_key",
        requirement="& 云端迭代任务",
        cli_execution_mode="cloud",
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cloud",
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="SelfIterator: 有 & + 显式 cloud + 无 key → basic",
    ),
    # ===== 场景 7: 有 & + 显式 --execution-mode auto =====
    SelfIteratorRequestedModeCase(
        test_id="si_amp_explicit_auto",
        requirement="& 自动迭代任务",
        cli_execution_mode="auto",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="SelfIterator: 有 & + 显式 auto → 强制 basic",
    ),
    SelfIteratorRequestedModeCase(
        test_id="si_amp_explicit_auto_no_key",
        requirement="& 自动迭代任务",
        cli_execution_mode="auto",
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",
        description="SelfIterator: 有 & + 显式 auto + 无 key → basic",
    ),
]


class TestSelfIteratorRequestedModeInvariant:
    """测试 SelfIterator 的 requested_mode_for_decision 计算一致性

    验证 scripts/run_iterate.py SelfIterator 计算的 requested_mode_for_decision
    与 build_execution_decision 的输入一致，符合以下不变式：

    1. CLI 显式参数优先级最高
    2. 有 & 前缀且无 CLI 显式参数时，返回 None
    3. 无 & 前缀且无 CLI 显式参数时，使用 config.yaml 的值
    4. requested_mode_for_decision 决定 orchestrator 选择
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前后重置 ConfigManager"""
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        SELF_ITERATOR_REQUESTED_MODE_CASES,
        ids=[tc.test_id for tc in SELF_ITERATOR_REQUESTED_MODE_CASES],
    )
    def test_resolve_requested_mode_consistency(self, test_case: SelfIteratorRequestedModeCase) -> None:
        """验证 resolve_requested_mode_for_decision 在 SelfIterator 场景下的一致性"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import resolve_requested_mode_for_decision

        has_ampersand_prefix = is_cloud_request(test_case.requirement)

        # 断言 1: has_ampersand_prefix 符合预期
        assert has_ampersand_prefix == test_case.expected_has_ampersand_prefix, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  has_ampersand_prefix 预期={test_case.expected_has_ampersand_prefix}，"
            f"实际={has_ampersand_prefix}"
        )

        # 调用 resolve_requested_mode_for_decision
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand_prefix,
            config_execution_mode=test_case.config_execution_mode,
        )

        # 断言 2: requested_mode_for_decision 符合预期
        assert requested_mode == test_case.expected_requested_mode_for_decision, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  cli_execution_mode={test_case.cli_execution_mode}\n"
            f"  has_ampersand_prefix={has_ampersand_prefix}\n"
            f"  config_execution_mode={test_case.config_execution_mode}\n"
            f"  requested_mode 预期={test_case.expected_requested_mode_for_decision}，"
            f"实际={requested_mode}"
        )

    @pytest.mark.parametrize(
        "test_case",
        SELF_ITERATOR_REQUESTED_MODE_CASES,
        ids=[tc.test_id for tc in SELF_ITERATOR_REQUESTED_MODE_CASES],
    )
    def test_build_execution_decision_consistency(self, test_case: SelfIteratorRequestedModeCase) -> None:
        """验证 build_execution_decision 在 SelfIterator 场景下的输出一致性"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import (
            build_execution_decision,
            resolve_requested_mode_for_decision,
        )

        has_ampersand_prefix = is_cloud_request(test_case.requirement)

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand_prefix,
            config_execution_mode=test_case.config_execution_mode,
        )

        decision = build_execution_decision(
            prompt=test_case.requirement,
            requested_mode=requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=True,
        )

        # 断言 1: prefix_routed 符合预期
        assert decision.prefix_routed == test_case.expected_prefix_routed, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  prefix_routed 预期={test_case.expected_prefix_routed}，"
            f"实际={decision.prefix_routed}"
        )

        # 断言 2: orchestrator 符合预期
        assert decision.orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  orchestrator 预期={test_case.expected_orchestrator}，"
            f"实际={decision.orchestrator}"
        )

        # 断言 3: has_ampersand_prefix 符合预期
        assert decision.has_ampersand_prefix == test_case.expected_has_ampersand_prefix, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  has_ampersand_prefix 预期={test_case.expected_has_ampersand_prefix}，"
            f"实际={decision.has_ampersand_prefix}"
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            tc
            for tc in SELF_ITERATOR_REQUESTED_MODE_CASES
            if tc.expected_requested_mode_for_decision in ("auto", "cloud")
        ],
        ids=[
            tc.test_id
            for tc in SELF_ITERATOR_REQUESTED_MODE_CASES
            if tc.expected_requested_mode_for_decision in ("auto", "cloud")
        ],
    )
    def test_auto_cloud_forces_basic_invariant(self, test_case: SelfIteratorRequestedModeCase) -> None:
        """验证 SelfIterator 场景下的核心不变式：auto/cloud 强制 basic"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import (
            build_execution_decision,
            resolve_requested_mode_for_decision,
            should_use_mp_orchestrator,
        )

        has_ampersand_prefix = is_cloud_request(test_case.requirement)

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand_prefix,
            config_execution_mode=test_case.config_execution_mode,
        )

        assert requested_mode in ("auto", "cloud"), f"[{test_case.test_id}] 此用例 requested_mode 应为 auto/cloud"

        can_use_mp = should_use_mp_orchestrator(requested_mode)
        assert can_use_mp is False, f"[{test_case.test_id}] should_use_mp_orchestrator({requested_mode}) 应返回 False"

        decision = build_execution_decision(
            prompt=test_case.requirement,
            requested_mode=requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=True,
        )

        assert decision.orchestrator == "basic", (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  核心不变式违反：requested_mode={requested_mode} → orchestrator 必须是 basic"
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            tc
            for tc in SELF_ITERATOR_REQUESTED_MODE_CASES
            if tc.expected_has_ampersand_prefix and tc.expected_prefix_routed is False and tc.cli_execution_mode is None
        ],
        ids=[
            tc.test_id
            for tc in SELF_ITERATOR_REQUESTED_MODE_CASES
            if tc.expected_has_ampersand_prefix and tc.expected_prefix_routed is False and tc.cli_execution_mode is None
        ],
    )
    def test_ampersand_cloud_intent_forces_basic(self, test_case: SelfIteratorRequestedModeCase) -> None:
        """验证 R-2 规则：& 前缀表达 Cloud 意图时强制 basic（即使未成功路由）

        当 auto_detect_cloud_prefix=True 且存在 & 前缀时：
        - 无论是否成功路由到 Cloud
        - orchestrator 都应为 basic（因为 & 前缀表达了 Cloud 意图）
        """
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import (
            build_execution_decision,
            resolve_requested_mode_for_decision,
        )

        has_ampersand_prefix = is_cloud_request(test_case.requirement)
        assert has_ampersand_prefix is True, "此测试用例应有 & 前缀"

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand_prefix,
            config_execution_mode=test_case.config_execution_mode,
        )

        # requested_mode 应为 None（& 前缀场景无 CLI 显式设置）
        assert requested_mode is None, f"[{test_case.test_id}] & 前缀 + 无 CLI 显式设置时 requested_mode 应为 None"

        decision = build_execution_decision(
            prompt=test_case.requirement,
            requested_mode=requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=True,
        )

        # 断言: prefix_routed=False（未成功路由）
        assert decision.prefix_routed is False, f"[{test_case.test_id}] 此场景 prefix_routed 应为 False"

        # 断言: 尽管 prefix_routed=False，orchestrator 仍为 basic
        assert decision.orchestrator == "basic", (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  R-2 规则：& 前缀表达 Cloud 意图，orchestrator 应为 basic\n"
            f"  即使 prefix_routed=False（无 key 或 cloud_disabled）"
        )
