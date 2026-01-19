"""测试 scripts/run_iterate.py 自我迭代脚本

测试覆盖：
1. WebFetcher.fetch 失败时流程不崩溃，仍能进入知识库统计/目标构建
2. MultiProcessOrchestrator.run 抛出异常返回 _fallback_required，触发 basic 编排器回退
3. --dry-run 返回结构字段验证
4. --skip-online 分支验证
"""
import argparse
import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 导入被测模块
from scripts.run_iterate import (
    ChangelogAnalyzer,
    IterationContext,
    IterationGoalBuilder,
    KnowledgeUpdater,
    SelfIterator,
    UpdateAnalysis,
    parse_max_iterations,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def base_iterate_args() -> argparse.Namespace:
    """创建基础迭代参数"""
    return argparse.Namespace(
        requirement="测试需求",
        skip_online=False,
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
    async def test_changelog_fetch_failure_does_not_crash(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 Changelog 获取失败时流程继续执行"""
        iterator = SelfIterator(base_iterate_args)

        # Mock WebFetcher.fetch 返回失败
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = False
        mock_fetch_result.content = ""
        mock_fetch_result.error = "Network error"

        with patch.object(
            iterator.changelog_analyzer.fetcher, "initialize", new_callable=AsyncMock
        ):
            with patch.object(
                iterator.changelog_analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ):
                # 分析 changelog 应返回空结果而不是崩溃
                analysis = await iterator.changelog_analyzer.analyze()

                # 验证返回空的 UpdateAnalysis
                assert isinstance(analysis, UpdateAnalysis)
                assert analysis.has_updates is False
                assert len(analysis.entries) == 0

    @pytest.mark.asyncio
    async def test_fetch_failure_still_enters_knowledge_stats(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator.changelog_analyzer.fetcher, "initialize", new_callable=AsyncMock
        ):
            with patch.object(
                iterator.changelog_analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ):
                with patch.object(
                    iterator.knowledge_updater, "storage", mock_storage
                ):
                    with patch.object(
                        iterator.knowledge_updater.manager,
                        "initialize",
                        new_callable=AsyncMock,
                    ):
                        with patch.object(
                            iterator.knowledge_updater.fetcher,
                            "initialize",
                            new_callable=AsyncMock,
                        ):
                            # 执行到知识库统计阶段
                            await iterator.knowledge_updater.initialize()
                            stats = await iterator.knowledge_updater.get_stats()

                            # 验证知识库统计被调用
                            assert stats["document_count"] == 5
                            mock_storage.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_failure_still_builds_iteration_goal(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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
# TestMPOrchestratorFallback - MP 编排器回退测试
# ============================================================


class TestMPOrchestratorFallback:
    """测试 MultiProcessOrchestrator 失败时回退到 basic 编排器"""

    @pytest.mark.asyncio
    async def test_mp_timeout_triggers_fallback(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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
    async def test_mp_oserror_triggers_fallback(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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
    async def test_mp_runtime_error_triggers_fallback(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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
    async def test_fallback_calls_basic_orchestrator(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_mp_result,
        ):
            with patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ) as mock_basic:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
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
    async def test_fallback_result_structure_stable(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试回退后结果结构保持稳定"""
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

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
            return_value={"_fallback_required": True, "_fallback_reason": "测试"},
        ):
            with patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    result = await iterator._run_agent_system()

                    # 验证所有预期字段存在
                    expected_fields = [
                        "success",
                        "iterations_completed",
                        "total_tasks_created",
                        "total_tasks_completed",
                        "total_tasks_failed",
                    ]
                    for field in expected_fields:
                        assert field in result, f"缺少字段: {field}"


# ============================================================
# TestDryRunMode - dry-run 模式测试
# ============================================================


class TestDryRunMode:
    """测试 --dry-run 模式返回结构"""

    @pytest.mark.asyncio
    async def test_dry_run_returns_expected_fields(
        self, dry_run_args: argparse.Namespace
    ) -> None:
        """测试 dry-run 模式返回预期字段"""
        iterator = SelfIterator(dry_run_args)

        # Mock 知识库组件
        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 3})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with patch.object(iterator.knowledge_updater, "storage", mock_storage):
            with patch.object(
                iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock
            ):
                with patch.object(
                    iterator.knowledge_updater.fetcher,
                    "initialize",
                    new_callable=AsyncMock,
                ):
                    result = await iterator.run()

                    # 验证 dry-run 特有字段
                    assert result.get("success") is True
                    assert result.get("dry_run") is True
                    assert "summary" in result
                    assert "goal_length" in result
                    assert result["goal_length"] > 0

    @pytest.mark.asyncio
    async def test_dry_run_does_not_execute_agent(
        self, dry_run_args: argparse.Namespace
    ) -> None:
        """测试 dry-run 模式不执行 Agent 系统"""
        iterator = SelfIterator(dry_run_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with patch.object(iterator.knowledge_updater, "storage", mock_storage):
            with patch.object(
                iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock
            ):
                with patch.object(
                    iterator.knowledge_updater.fetcher,
                    "initialize",
                    new_callable=AsyncMock,
                ):
                    with patch.object(
                        iterator, "_run_agent_system", new_callable=AsyncMock
                    ) as mock_agent:
                        result = await iterator.run()

                        # 验证 Agent 系统未被调用
                        mock_agent.assert_not_called()

                        # 验证仍返回成功
                        assert result["success"] is True
                        assert result["dry_run"] is True

    @pytest.mark.asyncio
    async def test_dry_run_builds_goal_preview(
        self, dry_run_args: argparse.Namespace
    ) -> None:
        """测试 dry-run 模式构建目标预览"""
        dry_run_args.requirement = "添加新功能支持"
        iterator = SelfIterator(dry_run_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with patch.object(iterator.knowledge_updater, "storage", mock_storage):
            with patch.object(
                iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock
            ):
                with patch.object(
                    iterator.knowledge_updater.fetcher,
                    "initialize",
                    new_callable=AsyncMock,
                ):
                    result = await iterator.run()

                    # 验证目标被构建
                    assert iterator.context.iteration_goal != ""
                    assert "添加新功能支持" in iterator.context.iteration_goal

                    # 验证摘要包含用户需求
                    assert "用户需求" in result["summary"]


# ============================================================
# TestSkipOnlineMode - skip-online 模式测试
# ============================================================


class TestSkipOnlineMode:
    """测试 --skip-online 模式"""

    @pytest.mark.asyncio
    async def test_skip_online_does_not_fetch_changelog(
        self, skip_online_args: argparse.Namespace
    ) -> None:
        """测试 skip-online 模式不获取 Changelog"""
        iterator = SelfIterator(skip_online_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 2})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with patch.object(
            iterator.changelog_analyzer.fetcher, "fetch", new_callable=AsyncMock
        ) as mock_fetch:
            with patch.object(iterator.knowledge_updater, "storage", mock_storage):
                with patch.object(
                    iterator.knowledge_updater.manager,
                    "initialize",
                    new_callable=AsyncMock,
                ):
                    with patch.object(
                        iterator.knowledge_updater.fetcher,
                        "initialize",
                        new_callable=AsyncMock,
                    ):
                        with patch.object(
                            iterator, "_run_agent_system", new_callable=AsyncMock,
                            return_value={"success": True, "iterations_completed": 1,
                                         "total_tasks_created": 1, "total_tasks_completed": 1,
                                         "total_tasks_failed": 0}
                        ):
                            await iterator.run()

                            # 验证 fetch 未被调用
                            mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_online_uses_existing_knowledge(
        self, skip_online_args: argparse.Namespace
    ) -> None:
        """测试 skip-online 模式使用现有知识库"""
        iterator = SelfIterator(skip_online_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 10})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with patch.object(iterator.knowledge_updater, "storage", mock_storage):
            with patch.object(
                iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock
            ):
                with patch.object(
                    iterator.knowledge_updater.fetcher,
                    "initialize",
                    new_callable=AsyncMock,
                ):
                    with patch.object(
                        iterator, "_run_agent_system", new_callable=AsyncMock,
                        return_value={"success": True, "iterations_completed": 1,
                                     "total_tasks_created": 1, "total_tasks_completed": 1,
                                     "total_tasks_failed": 0}
                    ):
                        await iterator.run()

                        # 验证知识库统计被调用
                        mock_storage.get_stats.assert_called()

    @pytest.mark.asyncio
    async def test_skip_online_sets_empty_update_analysis(
        self, skip_online_args: argparse.Namespace
    ) -> None:
        """测试 skip-online 模式设置空的更新分析"""
        iterator = SelfIterator(skip_online_args)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with patch.object(iterator.knowledge_updater, "storage", mock_storage):
            with patch.object(
                iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock
            ):
                with patch.object(
                    iterator.knowledge_updater.fetcher,
                    "initialize",
                    new_callable=AsyncMock,
                ):
                    with patch.object(
                        iterator, "_run_agent_system", new_callable=AsyncMock,
                        return_value={"success": True, "iterations_completed": 1,
                                     "total_tasks_created": 1, "total_tasks_completed": 1,
                                     "total_tasks_failed": 0}
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

    def test_get_orchestrator_type_default_mp(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试默认使用 MP 编排器"""
        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "mp"

    def test_get_orchestrator_type_no_mp_flag(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 --no-mp 标志使用 basic 编排器"""
        base_iterate_args.no_mp = True
        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic"

    def test_get_orchestrator_type_explicit_basic(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试显式指定 basic 编排器"""
        base_iterate_args.orchestrator = "basic"
        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic"


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
        """测试无效值"""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_max_iterations("invalid")
        with pytest.raises(argparse.ArgumentTypeError):
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
        from scripts.run_iterate import ChangelogEntry

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
SAMPLE_CHANGELOG_NO_HEADERS = """
Cursor CLI Updates

We've made several improvements:
- New plan/ask mode switching
- Cloud relay for MCP servers
- Enhanced diff view

This update was released on January 16, 2026.
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
        all_keywords = ' '.join(jan_16_entry.keywords).lower()

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
        url_str = ' '.join(related_urls)

        # 应该匹配到 parameters（因为包含 plan/ask/mode）
        assert any('parameters' in url for url in related_urls), \
            f"应包含 parameters 文档，实际: {related_urls}"

        # 应该匹配到 slash-commands（因为包含 /plan /ask）
        assert any('slash-commands' in url for url in related_urls), \
            f"应包含 slash-commands 文档，实际: {related_urls}"

        # 应该匹配到 mcp（因为包含 cloud relay）
        assert any('mcp' in url for url in related_urls), \
            f"应包含 mcp 文档，实际: {related_urls}"

    def test_html_mixed_content_cleanup(self) -> None:
        """测试 HTML 混合内容清理"""
        analyzer = ChangelogAnalyzer()
        entries = analyzer.parse_changelog(SAMPLE_CHANGELOG_HTML_MIXED)

        # 应该能解析出条目
        assert len(entries) >= 1, "HTML 混合内容应该能解析出条目"

        # 验证 HTML 标签和脚本被清理
        all_content = ' '.join(e.content for e in entries)
        assert '<script>' not in all_content
        assert '</script>' not in all_content
        assert '<div' not in all_content
        assert 'console.log' not in all_content

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
        results = [
            analyzer.parse_changelog(SAMPLE_CHANGELOG_JAN_16_2026)
            for _ in range(5)
        ]

        # 验证每次结果一致
        first_count = len(results[0])
        for i, result in enumerate(results[1:], 2):
            assert len(result) == first_count, \
                f"第 {i} 次解析结果数量 ({len(result)}) 与第 1 次 ({first_count}) 不一致"

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
        params_keywords = analyzer._extract_doc_keywords(
            "https://cursor.com/cn/docs/cli/reference/parameters"
        )
        params_str = ' '.join(params_keywords).lower()
        assert 'plan' in params_str, "parameters 应包含 plan 关键词"
        assert 'ask' in params_str, "parameters 应包含 ask 关键词"
        assert 'mode' in params_str or '模式' in params_str

        # 测试 slash-commands URL 的关键词
        slash_keywords = analyzer._extract_doc_keywords(
            "https://cursor.com/cn/docs/cli/reference/slash-commands"
        )
        slash_str = ' '.join(slash_keywords).lower()
        assert '/plan' in slash_str, "slash-commands 应包含 /plan"
        assert '/ask' in slash_str, "slash-commands 应包含 /ask"

        # 测试 mcp URL 的关键词
        mcp_keywords = analyzer._extract_doc_keywords(
            "https://cursor.com/cn/docs/cli/mcp"
        )
        mcp_str = ' '.join(mcp_keywords).lower()
        assert 'relay' in mcp_str or 'cloud relay' in mcp_str, \
            "mcp 应包含 relay 关键词"

        # 测试 overview URL 的关键词
        overview_keywords = analyzer._extract_doc_keywords(
            "https://cursor.com/cn/docs/cli/overview"
        )
        overview_str = ' '.join(overview_keywords).lower()
        assert 'diff' in overview_str, "overview 应包含 diff 关键词"


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
    async def test_mp_config_receives_commit_per_iteration(
        self, commit_per_iteration_args: argparse.Namespace
    ) -> None:
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
    async def test_orchestrator_committed_detection_with_commits(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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
            ]
        }
        assert iterator._has_orchestrator_committed(result_with_iteration_commit) is True

    @pytest.mark.asyncio
    async def test_orchestrator_committed_detection_without_commits(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 _has_orchestrator_committed 检测编排器未提交"""
        iterator = SelfIterator(base_iterate_args)

        # 空 commits
        result_empty = {"commits": {}}
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
            ]
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
            }
        }

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
            return_value=mp_result_with_commits,
        ):
            with patch.object(
                iterator,
                "_run_commit_phase",
                new_callable=AsyncMock,
            ) as mock_commit_phase:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
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

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
            return_value=mp_result_no_commits,
        ):
            with patch.object(
                iterator,
                "_run_commit_phase",
                new_callable=AsyncMock,
                return_value=mock_commit_result,
            ) as mock_commit_phase:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
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
    async def test_degraded_result_fields_stable(
        self, mp_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
            return_value=mp_degraded_result,
        ):
            with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                result = await iterator._run_agent_system()

                # 策略 B 契约断言
                assert result["success"] is False, "降级时 success 应为 False"
                assert result["degraded"] is True, "降级时 degraded 应为 True"
                assert result["degradation_reason"] == "Planner 进程不健康", \
                    "degradation_reason 应显式标注原因"
                assert result["iterations_completed"] == 1, \
                    "iterations_completed 应记录降级前完成的迭代数"

    @pytest.mark.asyncio
    async def test_degraded_no_fallback_to_basic(
        self, mp_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
            return_value=mp_degraded_result,
        ):
            with patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
            ) as mock_basic:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
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
    async def test_degraded_vs_fallback_required_distinction(
        self, mp_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
            return_value=mp_startup_failure,
        ):
            with patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=basic_result,
            ) as mock_basic:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    result = await iterator._run_agent_system()

                    # _fallback_required 应触发 basic 编排器
                    mock_basic.assert_called_once()

    @pytest.mark.asyncio
    async def test_degraded_result_preserves_partial_work(
        self, mp_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
            return_value=mp_degraded_result,
        ):
            with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                result = await iterator._run_agent_system()

                # 验证部分工作被保留
                assert result["iterations_completed"] == 3
                assert result["total_tasks_completed"] == 7
                assert result["commits"]["total_commits"] == 2

    @pytest.mark.asyncio
    async def test_degraded_with_different_reasons(
        self, mp_args: argparse.Namespace
    ) -> None:
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

            with patch.object(
                iterator,
                "_run_with_mp_orchestrator",
                new_callable=AsyncMock,
                return_value=mp_degraded_result,
            ):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    result = await iterator._run_agent_system()

                    # 所有降级原因应有一致的字段结构
                    assert result["success"] is False
                    assert result["degraded"] is True
                    assert result["degradation_reason"] == case["reason"]
                    assert isinstance(result["iterations_completed"], int)


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_flow_with_mocked_components(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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

        with patch.object(iterator.knowledge_updater, "storage", mock_storage):
            with patch.object(
                iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock
            ):
                with patch.object(
                    iterator.knowledge_updater.fetcher,
                    "initialize",
                    new_callable=AsyncMock,
                ):
                    with patch.object(
                        iterator,
                        "_run_agent_system",
                        new_callable=AsyncMock,
                        return_value=mock_run_result,
                    ):
                        result = await iterator.run()

                        # 验证完整流程成功
                        assert result["success"] is True
                        assert result["iterations_completed"] == 2

    @pytest.mark.asyncio
    async def test_error_handling_in_full_flow(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试完整流程中的错误处理"""
        iterator = SelfIterator(base_iterate_args)

        # Mock 知识库初始化抛出异常
        with patch.object(
            iterator.knowledge_updater, "initialize",
            new_callable=AsyncMock,
            side_effect=Exception("初始化失败"),
        ):
            with patch.object(
                iterator.changelog_analyzer.fetcher, "initialize", new_callable=AsyncMock
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
