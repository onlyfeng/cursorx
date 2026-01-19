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
    ChangelogEntry,
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
        # 执行模式参数
        execution_mode="cli",
        cloud_api_key=None,
        cloud_auth_timeout=30,
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

    def test_get_orchestrator_type_from_requirement_non_parallel(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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
            "当 requirement 包含 '非并行模式' 且用户未显式设置编排器时，"
            "应该自动选择 basic 编排器"
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
            assert iterator._get_orchestrator_type() == "basic", (
                f"requirement='{requirement}' 应该触发 basic 编排器"
            )

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
            "当用户显式设置 --orchestrator mp 时，即使 requirement 包含非并行关键词，"
            "也应该尊重用户的显式设置，返回 mp"
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
        assert iterator._get_orchestrator_type() == "basic", (
            "当用户显式设置 --no-mp 时，应该返回 basic"
        )

    def test_get_orchestrator_type_requirement_without_keyword_uses_default(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 requirement 不包含非并行关键词时使用默认 mp 编排器"""
        # requirement 不包含非并行关键词
        base_iterate_args.requirement = "普通的自我迭代任务，优化代码"
        base_iterate_args.no_mp = False
        base_iterate_args.orchestrator = "mp"
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)
        # 没有非并行关键词，应该返回默认的 mp
        assert iterator._get_orchestrator_type() == "mp", (
            "当 requirement 不包含非并行关键词时，应该使用默认的 mp 编排器"
        )

    def test_get_orchestrator_type_empty_requirement(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试空 requirement 时使用默认 mp 编排器"""
        base_iterate_args.requirement = ""
        base_iterate_args.no_mp = False
        base_iterate_args.orchestrator = "mp"
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "mp", (
            "空 requirement 时应该使用默认的 mp 编排器"
        )


# ============================================================
# TestExecutionModeSelection - 执行模式选择测试
# ============================================================


class TestExecutionModeSelection:
    """测试 execution_mode 参数和相关逻辑

    覆盖场景：
    1. execution_mode=cli 时使用默认 mp 编排器
    2. execution_mode=cloud/auto 时强制使用 basic 编排器
    3. OrchestratorConfig 正确接收 execution_mode 和 cloud_auth_config
    """

    def test_get_execution_mode_cli(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 execution_mode=cli 返回 CLI 模式"""
        from cursor.executor import ExecutionMode

        base_iterate_args.execution_mode = "cli"
        iterator = SelfIterator(base_iterate_args)

        assert iterator._get_execution_mode() == ExecutionMode.CLI

    def test_get_execution_mode_cloud(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 execution_mode=cloud 返回 CLOUD 模式"""
        from cursor.executor import ExecutionMode

        base_iterate_args.execution_mode = "cloud"
        iterator = SelfIterator(base_iterate_args)

        assert iterator._get_execution_mode() == ExecutionMode.CLOUD

    def test_get_execution_mode_auto(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 execution_mode=auto 返回 AUTO 模式"""
        from cursor.executor import ExecutionMode

        base_iterate_args.execution_mode = "auto"
        iterator = SelfIterator(base_iterate_args)

        assert iterator._get_execution_mode() == ExecutionMode.AUTO

    def test_execution_mode_cloud_forces_basic_orchestrator(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 execution_mode=cloud 时强制使用 basic 编排器

        场景：用户设置 --execution-mode cloud 但未显式设置 --orchestrator
        期望：SelfIterator 自动选择 basic 编排器
        """
        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.orchestrator = "mp"  # 默认 mp
        base_iterate_args.no_mp = False
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic", (
            "execution_mode=cloud 时应强制使用 basic 编排器"
        )

    def test_execution_mode_auto_forces_basic_orchestrator(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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

    def test_execution_mode_cli_allows_mp_orchestrator(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 execution_mode=cli 时允许使用 mp 编排器"""
        base_iterate_args.execution_mode = "cli"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args.no_mp = False
        base_iterate_args._orchestrator_user_set = False

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "mp", (
            "execution_mode=cli 时应允许使用 mp 编排器"
        )

    def test_get_cloud_auth_config_from_arg(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试从命令行参数获取 cloud_auth_config"""
        base_iterate_args.cloud_api_key = "test-api-key-12345"
        base_iterate_args.cloud_auth_timeout = 60

        iterator = SelfIterator(base_iterate_args)
        config = iterator._get_cloud_auth_config()

        assert config is not None
        assert config.api_key == "test-api-key-12345"
        assert config.auth_timeout == 60

    def test_get_cloud_auth_config_none_when_no_key(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试无 API Key 时返回 None"""
        base_iterate_args.cloud_api_key = None
        # 确保环境变量也没有设置
        import os
        with patch.dict(os.environ, {}, clear=True):
            iterator = SelfIterator(base_iterate_args)
            config = iterator._get_cloud_auth_config()

            assert config is None

    @pytest.mark.asyncio
    async def test_basic_orchestrator_receives_execution_mode(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
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
    async def test_basic_orchestrator_receives_cloud_auth_config(
        self, base_iterate_args: argparse.Namespace
    ) -> None:
        """测试 basic 编排器正确接收 cloud_auth_config 配置"""
        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.cloud_api_key = "test-cloud-key"
        base_iterate_args.cloud_auth_timeout = 45
        base_iterate_args.skip_online = True
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

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
        ) as mock_mp:
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

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
        ) as mock_mp:
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

                    # 验证 MP 编排器未被调用
                    mock_mp.assert_not_called()

                    # 验证 basic 编排器被调用
                    mock_basic.assert_called_once()


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

        # Jan 16 2026: 应该匹配到 modes/plan 专页（因为包含 plan mode）
        assert any('modes/plan' in url for url in related_urls), \
            f"应包含 modes/plan 文档，实际: {related_urls}"

        # Jan 16 2026: 应该匹配到 modes/ask 专页（因为包含 ask mode）
        assert any('modes/ask' in url for url in related_urls), \
            f"应包含 modes/ask 文档，实际: {related_urls}"

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

        # Jan 16 2026: 测试 modes/plan URL 的关键词
        plan_keywords = analyzer._extract_doc_keywords(
            "https://cursor.com/cn/docs/cli/modes/plan"
        )
        plan_str = ' '.join(plan_keywords).lower()
        assert 'plan' in plan_str, "modes/plan 应包含 plan 关键词"
        assert '规划模式' in plan_str or 'plan mode' in plan_str, \
            "modes/plan 应包含规划模式关键词"

        # Jan 16 2026: 测试 modes/ask URL 的关键词
        ask_keywords = analyzer._extract_doc_keywords(
            "https://cursor.com/cn/docs/cli/modes/ask"
        )
        ask_str = ' '.join(ask_keywords).lower()
        assert 'ask' in ask_str, "modes/ask 应包含 ask 关键词"
        assert '问答模式' in ask_str or 'ask mode' in ask_str, \
            "modes/ask 应包含问答模式关键词"


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
        storage.save_document = AsyncMock(return_value=(True, "保存成功"))
        storage.get_stats = AsyncMock(return_value={"document_count": 1})
        storage.search = AsyncMock(return_value=[])
        storage.list_documents = AsyncMock(return_value=[])
        return storage

    @pytest.mark.asyncio
    async def test_same_changelog_second_analysis_no_updates(
        self, mock_storage: MagicMock
    ) -> None:
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

        with patch.object(
            analyzer.fetcher, "initialize", new_callable=AsyncMock
        ):
            with patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ):
                # 第一次分析 - 无基线
                mock_storage.get_content_hash_by_url.return_value = None
                first_result = await analyzer.analyze()

                # 验证第一次有更新
                assert first_result.has_updates is True
                assert len(first_result.entries) > 0

                # 计算第一次的 fingerprint
                first_fingerprint = analyzer.compute_fingerprint(changelog_content)

                # 第二次分析 - 模拟基线存在（使用第一次的 fingerprint）
                mock_storage.get_content_hash_by_url.return_value = first_fingerprint
                second_result = await analyzer.analyze()

                # 验证第二次无更新
                assert second_result.has_updates is False
                assert len(second_result.entries) == 0
                assert second_result.summary == "未检测到新的更新"
                assert second_result.related_doc_urls == []

    @pytest.mark.asyncio
    async def test_different_changelog_has_updates(
        self, mock_storage: MagicMock
    ) -> None:
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

        with patch.object(
            analyzer.fetcher, "initialize", new_callable=AsyncMock
        ):
            with patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ):
                result = await analyzer.analyze()

                # 验证检测到更新
                assert result.has_updates is True
                assert len(result.entries) > 0

    @pytest.mark.asyncio
    async def test_no_baseline_parses_normally(
        self, mock_storage: MagicMock
    ) -> None:
        """测试无基线时应正常解析"""
        changelog_content = "## Jan 16, 2026\n- Feature A"

        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = changelog_content

        analyzer = ChangelogAnalyzer(storage=mock_storage)
        mock_storage.get_content_hash_by_url.return_value = None

        with patch.object(
            analyzer.fetcher, "initialize", new_callable=AsyncMock
        ):
            with patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ):
                result = await analyzer.analyze()

                # 无基线时应正常解析
                assert result.has_updates is True
                assert len(result.entries) >= 1

    @pytest.mark.asyncio
    async def test_no_updates_skips_related_docs_fetch(
        self, mock_storage: MagicMock
    ) -> None:
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
    async def test_has_updates_fetches_related_docs(
        self, mock_storage: MagicMock
    ) -> None:
        """测试有更新时应抓取 related docs"""
        changelog_content = "## Jan 16, 2026\n- MCP support"

        # 创建有更新的 analysis
        analysis = UpdateAnalysis(
            has_updates=True,
            entries=[],
            summary="检测到更新: 新功能 (1项)",
            raw_content=changelog_content,
            related_doc_urls=["https://cursor.com/cn/docs/cli/mcp"],
        )

        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.content = "MCP documentation content"
        mock_fetch_result.method_used = MagicMock()
        mock_fetch_result.method_used.value = "fetch"

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.fetcher.fetch = AsyncMock(return_value=mock_fetch_result)
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()

        await updater.initialize()
        result = await updater.update_from_analysis(analysis)

        # 验证 fetcher.fetch 被调用
        updater.fetcher.fetch.assert_called()

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
    async def test_custom_changelog_url_second_analysis_no_updates(
        self, mock_storage: MagicMock
    ) -> None:
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

        with patch.object(
            analyzer.fetcher, "initialize", new_callable=AsyncMock
        ):
            with patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ):
                # 第一次分析 - 无基线
                mock_storage.get_content_hash_by_url.return_value = None
                first_result = await analyzer.analyze()

                # 验证第一次有更新
                assert first_result.has_updates is True

                # 验证 analyzer 使用的是自定义 URL
                assert analyzer.changelog_url == custom_url

                # 计算 fingerprint
                fingerprint = analyzer.compute_fingerprint(changelog_content)

                # 第二次分析 - 设置基线（使用自定义 URL 作为 key）
                mock_storage.get_content_hash_by_url.return_value = fingerprint
                second_result = await analyzer.analyze()

                # 验证第二次无更新
                assert second_result.has_updates is False
                assert second_result.summary == "未检测到新的更新"

                # 验证 get_content_hash_by_url 使用的是自定义 URL
                calls = mock_storage.get_content_hash_by_url.call_args_list
                for call in calls:
                    assert call.args[0] == custom_url

    @pytest.mark.asyncio
    async def test_update_from_analysis_uses_custom_changelog_url(
        self, mock_storage: MagicMock
    ) -> None:
        """测试 update_from_analysis 保存文档时使用传入的 changelog_url

        验证：
        1. 保存的文档 URL 是自定义 URL 而非 DEFAULT_CHANGELOG_URL
        2. urls_processed 包含实际处理的 URL
        3. 返回结果中 changelog_url 字段正确
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

        # 捕获保存的文档
        saved_docs: list[Any] = []

        async def capture_save_document(doc: Any, force: bool = False) -> tuple[bool, str]:
            saved_docs.append(doc)
            return (True, "保存成功")

        mock_storage.save_document = AsyncMock(side_effect=capture_save_document)

        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.fetcher.fetch = AsyncMock()
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()

        await updater.initialize()
        result = await updater.update_from_analysis(
            analysis,
            force=False,
            changelog_url=custom_url,  # 传入自定义 URL
        )

        # 验证保存的文档使用自定义 URL
        assert len(saved_docs) >= 1
        changelog_doc = saved_docs[0]
        assert changelog_doc.url == custom_url, \
            f"期望 URL={custom_url}，实际={changelog_doc.url}"

        # 验证返回结果包含 changelog_url 字段
        assert result.get("changelog_url") == custom_url

    @pytest.mark.asyncio
    async def test_baseline_key_consistency_with_custom_url(
        self, mock_storage: MagicMock
    ) -> None:
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

        with patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock):
            with patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ):
                # 第一次分析
                first_result = await analyzer.analyze()
                assert first_result.has_updates is True

        # 模拟 update_from_analysis 保存文档
        updater = KnowledgeUpdater()
        updater.storage = mock_storage
        updater.fetcher = MagicMock()
        updater.fetcher.initialize = AsyncMock()
        updater.fetcher.fetch = AsyncMock()
        updater.manager = MagicMock()
        updater.manager.initialize = AsyncMock()

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

        with patch.object(analyzer.fetcher, "initialize", new_callable=AsyncMock):
            with patch.object(
                analyzer.fetcher,
                "fetch",
                new_callable=AsyncMock,
                return_value=mock_fetch_result,
            ):
                # 第二次分析
                second_result = await analyzer.analyze()

                # 验证第二次无更新
                assert second_result.has_updates is False

                # 验证 _get_baseline_fingerprint 使用的是自定义 URL
                # 通过检查 get_content_hash_by_url 的调用参数
                call_args = mock_storage.get_content_hash_by_url.call_args
                assert call_args.args[0] == custom_url


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
        assert self_iterate.main is run_iterate_main, (
            "self_iterate.main 应该是 run_iterate.main 的引用"
        )

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
        )

    def test_orchestrator_user_set_true_overrides_keyword(
        self, routing_args: argparse.Namespace
    ) -> None:
        """测试 _orchestrator_user_set=True 时关键词不覆盖用户设置"""
        # 用户显式设置 --orchestrator mp，但 requirement 包含非并行关键词
        routing_args.requirement = "使用协程模式完成自我迭代"
        routing_args.orchestrator = "mp"
        routing_args._orchestrator_user_set = True

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "mp", (
            "用户显式设置 --orchestrator mp 时，即使 requirement 包含 '协程模式'，"
            "也应该使用 mp 编排器"
        )

    def test_orchestrator_user_set_false_allows_keyword_override(
        self, routing_args: argparse.Namespace
    ) -> None:
        """测试 _orchestrator_user_set=False 时关键词可以覆盖默认设置"""
        # 用户未显式设置，requirement 包含非并行关键词
        routing_args.requirement = "使用协程模式完成自我迭代"
        routing_args.orchestrator = "mp"
        routing_args._orchestrator_user_set = False

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "basic", (
            "用户未显式设置时，requirement 包含 '协程模式' 应该触发 basic 编排器"
        )

    def test_no_mp_flag_always_uses_basic(
        self, routing_args: argparse.Namespace
    ) -> None:
        """测试 --no-mp 标志始终使用 basic 编排器"""
        routing_args.no_mp = True
        routing_args._orchestrator_user_set = True

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "basic", "--no-mp 标志应该始终使用 basic 编排器"

    def test_explicit_orchestrator_basic_uses_basic(
        self, routing_args: argparse.Namespace
    ) -> None:
        """测试显式设置 --orchestrator basic 使用 basic 编排器"""
        routing_args.orchestrator = "basic"
        routing_args._orchestrator_user_set = True

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "basic", "--orchestrator basic 应该使用 basic 编排器"

    def test_all_disable_mp_keywords_detected(
        self, routing_args: argparse.Namespace
    ) -> None:
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

            assert orch_type == "basic", (
                f"requirement='{requirement}' 应该触发 basic 编排器，"
                f"实际返回 {orch_type}"
            )

    def test_no_keyword_uses_default_mp(
        self, routing_args: argparse.Namespace
    ) -> None:
        """测试无关键词时使用默认 mp 编排器"""
        routing_args.requirement = "优化代码结构并添加单元测试"
        routing_args._orchestrator_user_set = False

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        assert orch_type == "mp", "无非并行关键词时应该使用默认 mp 编排器"

    def test_mixed_keywords_still_detects_disable_mp(
        self, routing_args: argparse.Namespace
    ) -> None:
        """测试混合关键词时仍能检测到禁用 MP"""
        routing_args.requirement = "使用多进程和协程模式完成任务"
        routing_args._orchestrator_user_set = False

        iterator = SelfIterator(routing_args)
        orch_type = iterator._get_orchestrator_type()

        # 只要包含禁用关键词就应该触发 basic
        assert orch_type == "basic", (
            "包含 '协程模式' 时即使也包含 '多进程' 也应该触发 basic 编排器"
        )

    @pytest.mark.asyncio
    async def test_mp_config_receives_correct_options(
        self, routing_args: argparse.Namespace
    ) -> None:
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
    async def test_basic_config_receives_correct_options(
        self, routing_args: argparse.Namespace
    ) -> None:
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
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
            workers=2,
            force_update=False,
            verbose=False,
            auto_commit=True,  # 启用自动提交
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
        )

    def test_has_orchestrator_committed_with_total_commits(
        self, dedup_args: argparse.Namespace
    ) -> None:
        """测试 total_commits > 0 时检测为已提交"""
        iterator = SelfIterator(dedup_args)

        result = {"commits": {"total_commits": 2, "commit_hashes": ["a", "b"]}}
        assert iterator._has_orchestrator_committed(result) is True

    def test_has_orchestrator_committed_with_successful_commits(
        self, dedup_args: argparse.Namespace
    ) -> None:
        """测试 successful_commits > 0 时检测为已提交"""
        iterator = SelfIterator(dedup_args)

        result = {"commits": {"successful_commits": 1}}
        assert iterator._has_orchestrator_committed(result) is True

    def test_has_orchestrator_committed_with_iteration_commit_hash(
        self, dedup_args: argparse.Namespace
    ) -> None:
        """测试 iteration 有 commit_hash 时检测为已提交"""
        iterator = SelfIterator(dedup_args)

        result = {
            "commits": {},
            "iterations": [
                {"id": 1, "commit_hash": "abc123"},
            ]
        }
        assert iterator._has_orchestrator_committed(result) is True

    def test_has_orchestrator_committed_empty_commits(
        self, dedup_args: argparse.Namespace
    ) -> None:
        """测试空 commits 时检测为未提交"""
        iterator = SelfIterator(dedup_args)

        result = {"commits": {}}
        assert iterator._has_orchestrator_committed(result) is False

    def test_has_orchestrator_committed_zero_total(
        self, dedup_args: argparse.Namespace
    ) -> None:
        """测试 total_commits=0 时检测为未提交"""
        iterator = SelfIterator(dedup_args)

        result = {"commits": {"total_commits": 0}}
        assert iterator._has_orchestrator_committed(result) is False

    def test_has_orchestrator_committed_no_commits_field(
        self, dedup_args: argparse.Namespace
    ) -> None:
        """测试无 commits 字段时检测为未提交"""
        iterator = SelfIterator(dedup_args)

        result = {"success": True}
        assert iterator._has_orchestrator_committed(result) is False

    def test_has_orchestrator_committed_empty_iteration_commit_hash(
        self, dedup_args: argparse.Namespace
    ) -> None:
        """测试 iteration commit_hash 为空时检测为未提交"""
        iterator = SelfIterator(dedup_args)

        result = {
            "commits": {},
            "iterations": [
                {"id": 1, "commit_hash": ""},
                {"id": 2, "commit_hash": None},
            ]
        }
        assert iterator._has_orchestrator_committed(result) is False

    @pytest.mark.asyncio
    async def test_skip_commit_when_orchestrator_already_committed(
        self, dedup_args: argparse.Namespace
    ) -> None:
        """测试编排器已提交时跳过 SelfIterator 提交阶段"""
        iterator = SelfIterator(dedup_args)
        iterator.context.iteration_goal = "测试目标"

        # 编排器返回已提交的结果
        mp_result = {
            "success": True,
            "commits": {"total_commits": 1, "commit_hashes": ["abc"]},
            "iterations_completed": 1,
        }

        with patch.object(
            iterator, "_run_with_mp_orchestrator",
            new_callable=AsyncMock, return_value=mp_result
        ):
            with patch.object(
                iterator, "_run_commit_phase", new_callable=AsyncMock
            ) as mock_commit:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    await iterator._run_agent_system()

                    # _run_commit_phase 不应被调用
                    mock_commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_commit_when_orchestrator_not_committed(
        self, dedup_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator, "_run_with_mp_orchestrator",
            new_callable=AsyncMock, return_value=mp_result
        ):
            with patch.object(
                iterator, "_run_commit_phase",
                new_callable=AsyncMock, return_value=commit_result
            ) as mock_commit:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
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
        )

    @pytest.mark.asyncio
    async def test_fallback_required_triggers_basic_orchestrator(
        self, fallback_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator, "_run_with_mp_orchestrator",
            new_callable=AsyncMock, return_value=mp_startup_failure
        ):
            with patch.object(
                iterator, "_run_with_basic_orchestrator",
                new_callable=AsyncMock, return_value=basic_result
            ) as mock_basic:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    result = await iterator._run_agent_system()

                    # basic 编排器应被调用
                    mock_basic.assert_called_once()
                    assert result["success"] is True

    @pytest.mark.asyncio
    async def test_degraded_does_not_trigger_basic_fallback(
        self, fallback_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator, "_run_with_mp_orchestrator",
            new_callable=AsyncMock, return_value=mp_degraded_result
        ):
            with patch.object(
                iterator, "_run_with_basic_orchestrator",
                new_callable=AsyncMock
            ) as mock_basic:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
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
    async def test_degraded_preserves_partial_work(
        self, fallback_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            iterator, "_run_with_mp_orchestrator",
            new_callable=AsyncMock, return_value=mp_degraded_result
        ):
            with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                mock_km = MagicMock()
                mock_km.initialize = AsyncMock()
                MockKM.return_value = mock_km

                result = await iterator._run_agent_system()

                # 验证保留部分工作
                assert result["iterations_completed"] == 3
                assert result["total_tasks_completed"] == 7
                assert result["commits"]["total_commits"] == 2

    @pytest.mark.asyncio
    async def test_fallback_required_various_reasons(
        self, fallback_args: argparse.Namespace
    ) -> None:
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

            with patch.object(
                iterator, "_run_with_mp_orchestrator",
                new_callable=AsyncMock, return_value=mp_failure
            ):
                with patch.object(
                    iterator, "_run_with_basic_orchestrator",
                    new_callable=AsyncMock, return_value=basic_result
                ) as mock_basic:
                    with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        result = await iterator._run_agent_system()

                        mock_basic.assert_called_once()
                        assert result["success"] is True

                        # 重置 mock
                        mock_basic.reset_mock()

    @pytest.mark.asyncio
    async def test_degraded_various_reasons(
        self, fallback_args: argparse.Namespace
    ) -> None:
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

            with patch.object(
                iterator, "_run_with_mp_orchestrator",
                new_callable=AsyncMock, return_value=mp_degraded
            ):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    result = await iterator._run_agent_system()

                    # 所有降级原因应有一致的字段结构
                    assert result["success"] is False
                    assert result["degraded"] is True
                    assert result["degradation_reason"] == reason
