"""运行模式端到端测试

测试 run.py 中各种运行模式的功能，包括：
- 基本协程模式 (basic)
- 多进程模式 (mp)
- 知识库增强模式 (knowledge)
- 自我迭代模式 (iterate)
- 自动分析模式 (auto)
- 任务分析器 (TaskAnalyzer)

使用 Mock 替代真实 Cursor CLI 调用
"""
import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run import (
    MODE_ALIASES,
    RunMode,
    Runner,
    TaskAnalysis,
    TaskAnalyzer,
    parse_max_iterations,
)


# ==================== TestBasicMode ====================


class TestBasicMode:
    """基本协程模式测试"""

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        """创建基本模式的命令行参数"""
        return argparse.Namespace(
            task="测试任务",
            mode="basic",
            directory=".",
            workers=2,
            max_iterations="3",
            strict=False,
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="gpt-5.2-codex",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            execution_mode=None,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            _orchestrator_user_set=False,
        )

    @pytest.mark.asyncio
    async def test_basic_mode_execution(self, mock_args: argparse.Namespace) -> None:
        """测试基本协程模式执行"""
        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.BASIC,
            goal="实现一个简单功能",
            reasoning="使用基本模式",
        )

        # Mock Orchestrator
        mock_result = {
            "success": True,
            "goal": "实现一个简单功能",
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
        }

        with patch("coordinator.Orchestrator") as MockOrchestrator:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockOrchestrator.return_value = mock_instance

            result = await runner.run(analysis)

            # 验证 Orchestrator 被正确初始化
            MockOrchestrator.assert_called_once()
            mock_instance.run.assert_called_once_with("实现一个简单功能")

            # 验证结果
            assert result["success"] is True
            assert result["iterations_completed"] == 1
            assert result["total_tasks_completed"] == 1

    @pytest.mark.asyncio
    async def test_basic_mode_with_options(self, mock_args: argparse.Namespace) -> None:
        """测试带参数的基本模式"""
        # 设置额外参数
        mock_args.workers = 5
        mock_args.max_iterations = "10"
        mock_args.strict = True
        mock_args.stream_log = True

        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.BASIC,
            goal="带参数的任务",
            options={
                "workers": 5,
                "max_iterations": 10,
                "strict": True,
            },
            reasoning="自定义参数",
        )

        mock_result = {
            "success": True,
            "goal": "带参数的任务",
            "iterations_completed": 3,
            "total_tasks_created": 5,
            "total_tasks_completed": 5,
            "total_tasks_failed": 0,
        }

        with patch("coordinator.Orchestrator") as MockOrchestrator:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockOrchestrator.return_value = mock_instance

            with patch("coordinator.OrchestratorConfig") as MockConfig:
                MockConfig.return_value = MagicMock()

                result = await runner.run(analysis)

                # 验证配置参数
                config_call = MockConfig.call_args
                assert config_call is not None
                # 验证 worker_pool_size 参数
                assert config_call[1].get("worker_pool_size") == 5 or \
                       config_call.kwargs.get("worker_pool_size") == 5


# ==================== TestMPMode ====================


class TestMPMode:
    """多进程模式测试"""

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        """创建多进程模式的命令行参数"""
        return argparse.Namespace(
            task="并行重构任务",
            mode="mp",
            directory=".",
            workers=4,
            max_iterations="5",
            strict=False,
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="gpt-5.2-codex",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            execution_mode=None,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            _orchestrator_user_set=False,
        )

    @pytest.mark.asyncio
    async def test_mp_mode_execution(self, mock_args: argparse.Namespace) -> None:
        """测试多进程模式执行"""
        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.MP,
            goal="并行重构代码",
            reasoning="使用多进程模式并行执行",
        )

        mock_result = {
            "success": True,
            "goal": "并行重构代码",
            "iterations_completed": 2,
            "total_tasks_created": 6,
            "total_tasks_completed": 6,
            "total_tasks_failed": 0,
        }

        with patch("coordinator.MultiProcessOrchestrator") as MockMPOrchestrator:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockMPOrchestrator.return_value = mock_instance

            result = await runner.run(analysis)

            # 验证 MultiProcessOrchestrator 被正确初始化
            MockMPOrchestrator.assert_called_once()
            mock_instance.run.assert_called_once_with("并行重构代码")

            # 验证结果
            assert result["success"] is True
            assert result["total_tasks_created"] == 6

    @pytest.mark.asyncio
    async def test_mp_mode_worker_count(self, mock_args: argparse.Namespace) -> None:
        """测试 Worker 数量配置"""
        mock_args.workers = 8

        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.MP,
            goal="多 Worker 任务",
            options={"workers": 8},
            reasoning="使用 8 个 Worker",
        )

        mock_result = {
            "success": True,
            "goal": "多 Worker 任务",
            "iterations_completed": 1,
            "total_tasks_created": 8,
            "total_tasks_completed": 8,
            "total_tasks_failed": 0,
        }

        with patch("coordinator.MultiProcessOrchestrator") as MockMPOrchestrator:
            with patch("coordinator.MultiProcessOrchestratorConfig") as MockConfig:
                mock_instance = MagicMock()
                mock_instance.run = AsyncMock(return_value=mock_result)
                MockMPOrchestrator.return_value = mock_instance
                MockConfig.return_value = MagicMock()

                result = await runner.run(analysis)

                # 验证配置中的 worker_count
                config_call = MockConfig.call_args
                assert config_call is not None
                # worker_count 应该为 8
                assert config_call[1].get("worker_count") == 8 or \
                       config_call.kwargs.get("worker_count") == 8

    @pytest.mark.asyncio
    async def test_mp_mode_commit_config_passthrough(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 MP 模式提交配置透传"""
        # 设置提交相关参数
        mock_args.auto_commit = True
        mock_args.auto_push = True
        mock_args.commit_per_iteration = True

        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.MP,
            goal="带提交配置的 MP 任务",
            options={
                "auto_commit": True,
                "auto_push": True,
                "commit_per_iteration": True,
            },
            reasoning="测试提交配置透传",
        )

        mock_result = {
            "success": True,
            "goal": "带提交配置的 MP 任务",
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        with patch("coordinator.MultiProcessOrchestrator") as MockMPOrchestrator:
            with patch("coordinator.MultiProcessOrchestratorConfig") as MockConfig:
                mock_instance = MagicMock()
                mock_instance.run = AsyncMock(return_value=mock_result)
                MockMPOrchestrator.return_value = mock_instance
                MockConfig.return_value = MagicMock()

                result = await runner.run(analysis)

                # 验证配置中的提交相关参数
                config_call = MockConfig.call_args
                assert config_call is not None

                # 验证 enable_auto_commit 透传
                assert config_call.kwargs.get("enable_auto_commit") is True, \
                    "enable_auto_commit 应该透传为 True"

                # 验证 auto_push 透传
                assert config_call.kwargs.get("auto_push") is True, \
                    "auto_push 应该透传为 True"

                # 验证 commit_per_iteration 透传
                assert config_call.kwargs.get("commit_per_iteration") is True, \
                    "commit_per_iteration 应该透传为 True"

                # 验证 commit_on_complete 语义（当 commit_per_iteration=True 时为 False）
                assert config_call.kwargs.get("commit_on_complete") is False, \
                    "commit_on_complete 应该为 False（因为 commit_per_iteration=True）"

    @pytest.mark.asyncio
    async def test_mp_mode_commit_config_defaults(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 MP 模式提交配置默认值"""
        # 使用默认提交参数
        mock_args.auto_commit = False
        mock_args.auto_push = False
        mock_args.commit_per_iteration = False

        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.MP,
            goal="默认提交配置的 MP 任务",
            reasoning="测试默认提交配置",
        )

        mock_result = {
            "success": True,
            "goal": "默认提交配置的 MP 任务",
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
        }

        with patch("coordinator.MultiProcessOrchestrator") as MockMPOrchestrator:
            with patch("coordinator.MultiProcessOrchestratorConfig") as MockConfig:
                mock_instance = MagicMock()
                mock_instance.run = AsyncMock(return_value=mock_result)
                MockMPOrchestrator.return_value = mock_instance
                MockConfig.return_value = MagicMock()

                result = await runner.run(analysis)

                # 验证配置中的默认提交参数
                config_call = MockConfig.call_args
                assert config_call is not None

                # 验证 enable_auto_commit 默认值（从 CLI 传递 False）
                assert config_call.kwargs.get("enable_auto_commit") is False, \
                    "enable_auto_commit 应该透传 CLI 的 False 值"

                # 验证 auto_push 默认值
                assert config_call.kwargs.get("auto_push") is False, \
                    "auto_push 应该为 False"

                # 验证 commit_per_iteration 默认值
                assert config_call.kwargs.get("commit_per_iteration") is False, \
                    "commit_per_iteration 应该为 False"

                # 验证 commit_on_complete 语义（当 commit_per_iteration=False 时为 True）
                assert config_call.kwargs.get("commit_on_complete") is True, \
                    "commit_on_complete 应该为 True（因为 commit_per_iteration=False）"


# ==================== TestKnowledgeMode ====================


class TestKnowledgeMode:
    """知识库增强模式测试"""

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        """创建知识库模式的命令行参数"""
        return argparse.Namespace(
            task="知识库增强任务",
            mode="knowledge",
            directory=".",
            workers=3,
            max_iterations="5",
            strict=False,
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=True,
            search_knowledge="CLI 参数",
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="gpt-5.2-codex",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            execution_mode=None,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            _orchestrator_user_set=False,
        )

    @pytest.mark.asyncio
    async def test_knowledge_mode_with_manager(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试知识库增强模式与 KnowledgeManager"""
        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.KNOWLEDGE,
            goal="基于知识库的任务",
            reasoning="使用知识库上下文",
        )

        mock_result = {
            "success": True,
            "goal": "基于知识库的任务",
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        with patch("knowledge.KnowledgeManager") as MockKM:
            with patch("knowledge.KnowledgeStorage") as MockKS:
                with patch("coordinator.Orchestrator") as MockOrchestrator:
                    # Mock KnowledgeManager
                    mock_km_instance = MagicMock()
                    mock_km_instance.initialize = AsyncMock()
                    MockKM.return_value = mock_km_instance

                    # Mock KnowledgeStorage
                    mock_ks_instance = MagicMock()
                    mock_ks_instance.initialize = AsyncMock()
                    mock_ks_instance.search = AsyncMock(return_value=[])
                    MockKS.return_value = mock_ks_instance

                    # Mock Orchestrator
                    mock_orch_instance = MagicMock()
                    mock_orch_instance.run = AsyncMock(return_value=mock_result)
                    MockOrchestrator.return_value = mock_orch_instance

                    result = await runner.run(analysis)

                    # 验证 KnowledgeManager 被初始化
                    MockKM.assert_called_once()
                    mock_km_instance.initialize.assert_called_once()

                    # 验证结果
                    assert result["success"] is True

    @pytest.mark.asyncio
    async def test_knowledge_search_integration(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试知识搜索集成"""
        mock_args.search_knowledge = "CLI 参数处理"

        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.KNOWLEDGE,
            goal="搜索知识库任务",
            options={"search_knowledge": "CLI 参数处理"},
            reasoning="搜索相关文档",
        )

        # 模拟搜索结果
        mock_search_result = MagicMock()
        mock_search_result.doc_id = "doc-123"

        mock_doc = MagicMock()
        mock_doc.title = "CLI 参数文档"
        mock_doc.url = "https://example.com/cli"
        mock_doc.content = "CLI 参数处理说明..."

        mock_result = {
            "success": True,
            "goal": "搜索知识库任务",
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
        }

        with patch("knowledge.KnowledgeManager") as MockKM:
            with patch("knowledge.KnowledgeStorage") as MockKS:
                with patch("coordinator.Orchestrator") as MockOrchestrator:
                    # Mock KnowledgeManager
                    mock_km_instance = MagicMock()
                    mock_km_instance.initialize = AsyncMock()
                    MockKM.return_value = mock_km_instance

                    # Mock KnowledgeStorage - 返回搜索结果
                    mock_ks_instance = MagicMock()
                    mock_ks_instance.initialize = AsyncMock()
                    mock_ks_instance.search = AsyncMock(return_value=[mock_search_result])
                    mock_ks_instance.load_document = AsyncMock(return_value=mock_doc)
                    MockKS.return_value = mock_ks_instance

                    # Mock Orchestrator
                    mock_orch_instance = MagicMock()
                    mock_orch_instance.run = AsyncMock(return_value=mock_result)
                    MockOrchestrator.return_value = mock_orch_instance

                    result = await runner.run(analysis)

                    # 验证搜索被调用
                    mock_ks_instance.search.assert_called()

                    # 验证结果
                    assert result["success"] is True


# ==================== TestIterateMode ====================


class TestIterateMode:
    """自我迭代模式测试"""

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        """创建迭代模式的命令行参数"""
        return argparse.Namespace(
            task="自我迭代任务",
            mode="iterate",
            directory=".",
            workers=3,
            max_iterations="5",
            strict=False,
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=True,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="gpt-5.2-codex",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            execution_mode=None,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            _orchestrator_user_set=False,
        )

    @pytest.mark.asyncio
    async def test_iterate_mode_basic(self, mock_args: argparse.Namespace) -> None:
        """测试自我迭代模式基本流程"""
        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.ITERATE,
            goal="更新 CLI 支持",
            options={"skip_online": True},
            reasoning="自我迭代更新",
        )

        mock_result = {
            "success": True,
            "goal": "更新 CLI 支持",
            "dry_run": False,
            "iterations_completed": 2,
            "total_tasks_created": 3,
            "total_tasks_completed": 3,
            "total_tasks_failed": 0,
        }

        with patch("scripts.run_iterate.SelfIterator") as MockIterator:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockIterator.return_value = mock_instance

            result = await runner.run(analysis)

            # 验证 SelfIterator 被调用
            MockIterator.assert_called_once()
            mock_instance.run.assert_called_once()

            # 验证结果
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_iterate_mode_options(self, mock_args: argparse.Namespace) -> None:
        """测试 IterateArgs 参数传递"""
        mock_args.skip_online = True
        mock_args.force_update = True
        mock_args.dry_run = False

        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.ITERATE,
            goal="带参数的迭代",
            options={
                "skip_online": True,
                "force_update": True,
            },
            reasoning="自定义迭代参数",
        )

        mock_result = {
            "success": True,
            "goal": "带参数的迭代",
            "dry_run": False,
        }

        with patch("scripts.run_iterate.SelfIterator") as MockIterator:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockIterator.return_value = mock_instance

            result = await runner.run(analysis)

            # 验证参数被正确传递
            call_args = MockIterator.call_args
            assert call_args is not None
            iterate_args = call_args[0][0]
            assert iterate_args.skip_online is True
            assert iterate_args.force_update is True

    @pytest.mark.asyncio
    async def test_iterate_mode_commit_options(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试提交选项"""
        mock_args.auto_commit = True
        mock_args.auto_push = True
        mock_args.commit_per_iteration = False

        runner = Runner(mock_args)
        analysis = TaskAnalysis(
            mode=RunMode.ITERATE,
            goal="带提交选项的迭代",
            options={
                "auto_commit": True,
                "auto_push": True,
            },
            reasoning="启用自动提交",
        )

        mock_result = {
            "success": True,
            "goal": "带提交选项的迭代",
            "commits": {
                "total_commits": 1,
                "commit_hashes": ["abc123"],
                "pushed_commits": 1,
            },
        }

        with patch("scripts.run_iterate.SelfIterator") as MockIterator:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockIterator.return_value = mock_instance

            result = await runner.run(analysis)

            # 验证提交参数
            call_args = MockIterator.call_args
            assert call_args is not None
            iterate_args = call_args[0][0]
            assert iterate_args.auto_commit is True
            assert iterate_args.auto_push is True

    @pytest.mark.asyncio
    async def test_iterate_mode_mp_orchestrator_called(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 iterate 模式内部调用 MultiProcessOrchestrator"""
        from scripts.run_iterate import SelfIterator

        # 创建 IterateArgs 模拟
        class IterateArgs:
            def __init__(self):
                self.requirement = "测试 MP 编排器"
                self.directory = "."
                self.skip_online = True
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = False
                self.max_iterations = "1"
                self.workers = 2
                self.force_update = False
                self.verbose = False
                self.auto_commit = False
                self.auto_push = False
                self.commit_per_iteration = False
                self.commit_message = ""

        iterate_args = IterateArgs()
        iterator = SelfIterator(iterate_args)
        iterator.context.iteration_goal = "测试 MP 编排器调用"

        mock_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
        }

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig") as MockConfig:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    # 设置 KnowledgeManager mock
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(return_value=mock_result)
                    MockMP.return_value = mock_orch
                    MockConfig.return_value = MagicMock()

                    result = await iterator._run_agent_system()

                    # 断言 MultiProcessOrchestrator 被调用
                    MockMP.assert_called_once()

                    # 验证 run 方法被调用
                    mock_orch.run.assert_called_once()
                    assert result["success"] is True

    @pytest.mark.asyncio
    async def test_iterate_mode_commits_written_to_result(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 iterate 模式 auto_commit=True 时 commits 写入结果"""
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self):
                self.requirement = "测试提交结果"
                self.directory = "."
                self.skip_online = True
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = False
                self.max_iterations = "1"
                self.workers = 2
                self.force_update = False
                self.verbose = False
                self.auto_commit = True  # 启用自动提交
                self.auto_push = False
                self.commit_per_iteration = False
                self.commit_message = ""

        iterate_args = IterateArgs()
        iterator = SelfIterator(iterate_args)
        iterator.context.iteration_goal = "测试提交结果目标"

        mock_run_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        mock_commits = {
            "total_commits": 1,
            "commit_hashes": ["deadbeef123"],
            "commit_messages": ["feat: 测试提交"],
            "pushed_commits": 0,
            "files_changed": ["src/main.py"],
        }

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    # 设置 KnowledgeManager mock
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(return_value=mock_run_result)
                    MockMP.return_value = mock_orch

                    # Mock _run_commit_phase
                    with patch.object(
                        iterator, "_run_commit_phase", new_callable=AsyncMock
                    ) as mock_commit:
                        mock_commit.return_value = mock_commits

                        result = await iterator._run_agent_system()

                        # 验证 _run_commit_phase 被调用
                        mock_commit.assert_called_once()

                        # 验证 commits 写入结果
                        assert "commits" in result
                        assert result["commits"]["total_commits"] == 1
                        assert result["commits"]["commit_hashes"] == ["deadbeef123"]

    @pytest.mark.asyncio
    async def test_iterate_mode_committer_agent_integration(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 iterate 模式 CommitterAgent 集成"""
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self):
                self.requirement = "测试 CommitterAgent"
                self.directory = "."
                self.skip_online = True
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = False
                self.max_iterations = "1"
                self.workers = 2
                self.force_update = False
                self.verbose = False
                self.auto_commit = True
                self.auto_push = True  # 也测试推送
                self.commit_per_iteration = False
                self.commit_message = "自定义提交信息"

        iterate_args = IterateArgs()
        iterator = SelfIterator(iterate_args)

        with patch("scripts.run_iterate.CommitterAgent") as MockCommitter:
            with patch("scripts.run_iterate.CommitterConfig"):
                mock_agent = MagicMock()

                # 模拟有变更
                mock_agent.check_status.return_value = {
                    "is_repo": True,
                    "has_changes": True,
                }

                # 模拟提交成功
                mock_commit_result = MagicMock()
                mock_commit_result.success = True
                mock_commit_result.commit_hash = "commit123"
                mock_commit_result.files_changed = ["file.py"]
                mock_commit_result.error = None
                mock_agent.commit.return_value = mock_commit_result

                # 模拟推送成功
                mock_push_result = MagicMock()
                mock_push_result.success = True
                mock_push_result.error = None
                mock_agent.push.return_value = mock_push_result

                # 模拟 get_commit_summary
                mock_agent.get_commit_summary.return_value = {
                    "successful_commits": 1,
                    "commit_hashes": ["commit123"],
                    "files_changed": ["file.py"],
                }

                MockCommitter.return_value = mock_agent

                result = await iterator._run_commit_phase(1, 2)

                # 验证 CommitterAgent 被创建
                MockCommitter.assert_called_once()

                # 验证 commit 被调用
                mock_agent.commit.assert_called_once()

                # 验证使用了自定义提交信息
                commit_call_args = mock_agent.commit.call_args
                assert "自定义提交信息" in commit_call_args[0][0]

                # 验证 push 被调用
                mock_agent.push.assert_called_once()

                # 验证返回结果
                assert result["total_commits"] == 1
                assert result["pushed_commits"] == 1

    @pytest.mark.asyncio
    async def test_iterate_mode_skip_commit_when_orchestrator_already_committed(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试提交去重：当编排器已提交时，SelfIterator 不会再次调用 CommitterAgent"""
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self):
                self.requirement = "测试提交去重"
                self.directory = "."
                self.skip_online = True
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = False
                self.max_iterations = "1"
                self.workers = 2
                self.force_update = False
                self.verbose = False
                self.auto_commit = True  # 启用自动提交
                self.auto_push = False
                self.commit_per_iteration = False
                self.commit_message = ""
                self.orchestrator = "mp"
                self.no_mp = False

        iterate_args = IterateArgs()
        iterator = SelfIterator(iterate_args)
        iterator.context.iteration_goal = "测试提交去重目标"

        # 模拟编排器返回已提交的结果
        mock_run_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
            # 关键：编排器已完成提交
            "commits": {
                "total_commits": 1,
                "commit_hashes": ["orchestrator_commit_abc123"],
                "pushed_commits": 0,
                "files_changed": ["src/main.py"],
            },
            "iterations": [
                {
                    "id": 1,
                    "commit_hash": "orchestrator_commit_abc123",
                    "commit_pushed": False,
                }
            ],
        }

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    # 设置 KnowledgeManager mock
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(return_value=mock_run_result)
                    MockMP.return_value = mock_orch

                    # Mock _run_commit_phase - 不应该被调用
                    with patch.object(
                        iterator, "_run_commit_phase", new_callable=AsyncMock
                    ) as mock_commit:
                        result = await iterator._run_agent_system()

                        # 关键断言：_run_commit_phase 不应该被调用
                        mock_commit.assert_not_called()

                        # 验证结果保留了编排器的 commits 信息
                        assert "commits" in result
                        assert result["commits"]["total_commits"] == 1
                        assert result["commits"]["commit_hashes"] == ["orchestrator_commit_abc123"]

    @pytest.mark.asyncio
    async def test_iterate_mode_commit_when_orchestrator_not_committed(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试当编排器未提交时，SelfIterator 会执行提交"""
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self):
                self.requirement = "测试后备提交"
                self.directory = "."
                self.skip_online = True
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = False
                self.max_iterations = "1"
                self.workers = 2
                self.force_update = False
                self.verbose = False
                self.auto_commit = True  # 启用自动提交
                self.auto_push = False
                self.commit_per_iteration = False
                self.commit_message = ""
                self.orchestrator = "mp"
                self.no_mp = False

        iterate_args = IterateArgs()
        iterator = SelfIterator(iterate_args)
        iterator.context.iteration_goal = "测试后备提交目标"

        # 模拟编排器返回未提交的结果（commits 为空或不存在）
        mock_run_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
            # 编排器未提交（commits 为空）
            "commits": {
                "total_commits": 0,
                "commit_hashes": [],
                "pushed_commits": 0,
            },
            "iterations": [
                {
                    "id": 1,
                    "commit_hash": "",  # 无提交
                    "commit_pushed": False,
                }
            ],
        }

        mock_commit_result = {
            "total_commits": 1,
            "commit_hashes": ["selfiterator_commit_xyz789"],
            "commit_messages": ["feat: SelfIterator 后备提交"],
            "pushed_commits": 0,
            "files_changed": ["backup.py"],
        }

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    # 设置 KnowledgeManager mock
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(return_value=mock_run_result)
                    MockMP.return_value = mock_orch

                    # Mock _run_commit_phase - 应该被调用
                    with patch.object(
                        iterator, "_run_commit_phase", new_callable=AsyncMock
                    ) as mock_commit:
                        mock_commit.return_value = mock_commit_result

                        result = await iterator._run_agent_system()

                        # 关键断言：_run_commit_phase 应该被调用
                        mock_commit.assert_called_once()

                        # 验证结果使用了 SelfIterator 的 commits
                        assert "commits" in result
                        assert result["commits"]["total_commits"] == 1
                        assert result["commits"]["commit_hashes"] == ["selfiterator_commit_xyz789"]

    def test_has_orchestrator_committed_detection(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 _has_orchestrator_committed 检测逻辑"""
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self):
                self.requirement = ""
                self.directory = "."
                self.skip_online = True
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = False
                self.max_iterations = "1"
                self.workers = 2
                self.force_update = False
                self.verbose = False
                self.auto_commit = True
                self.auto_push = False
                self.commit_per_iteration = False
                self.commit_message = ""

        iterator = SelfIterator(IterateArgs())

        # 测试场景 1: 有 total_commits > 0
        result1 = {
            "commits": {"total_commits": 1, "commit_hashes": []},
            "iterations": [],
        }
        assert iterator._has_orchestrator_committed(result1) is True

        # 测试场景 2: 有有效的 commit_hashes
        result2 = {
            "commits": {"total_commits": 0, "commit_hashes": ["abc123"]},
            "iterations": [],
        }
        assert iterator._has_orchestrator_committed(result2) is True

        # 测试场景 3: 有 successful_commits > 0
        result3 = {
            "commits": {"successful_commits": 1},
            "iterations": [],
        }
        assert iterator._has_orchestrator_committed(result3) is True

        # 测试场景 4: 迭代级别有 commit_hash
        result4 = {
            "commits": {},
            "iterations": [{"id": 1, "commit_hash": "iter_commit_123"}],
        }
        assert iterator._has_orchestrator_committed(result4) is True

        # 测试场景 5: 完全无提交
        result5 = {
            "commits": {"total_commits": 0, "commit_hashes": []},
            "iterations": [{"id": 1, "commit_hash": ""}],
        }
        assert iterator._has_orchestrator_committed(result5) is False

        # 测试场景 6: 空结果
        result6: dict = {"commits": {}, "iterations": []}
        assert iterator._has_orchestrator_committed(result6) is False

        # 测试场景 7: 完全空
        result7: dict = {}
        assert iterator._has_orchestrator_committed(result7) is False

    @pytest.mark.asyncio
    async def test_iterate_mode_no_commit_without_auto_commit_flag(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 iterate 模式默认不带 --auto-commit 时不触发提交

        这个测试验证了默认提交策略：默认不自动提交，需显式开启。
        当用户不指定 --auto-commit 时，iterate 模式应该完成任务但不会触发提交。
        """
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self):
                self.requirement = "测试默认不提交"
                self.directory = "."
                self.skip_online = True
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = False
                self.max_iterations = "1"
                self.workers = 2
                self.force_update = False
                self.verbose = False
                # 关键：默认 auto_commit=False（与命令行默认值一致）
                self.auto_commit = False
                self.auto_push = False
                self.commit_per_iteration = False
                self.commit_message = ""
                self.orchestrator = "mp"
                self.no_mp = False

        iterate_args = IterateArgs()
        iterator = SelfIterator(iterate_args)
        iterator.context.iteration_goal = "测试默认不提交目标"

        mock_run_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    # 设置 KnowledgeManager mock
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(return_value=mock_run_result)
                    MockMP.return_value = mock_orch

                    # Mock _run_commit_phase - 不应该被调用
                    with patch.object(
                        iterator, "_run_commit_phase", new_callable=AsyncMock
                    ) as mock_commit:
                        result = await iterator._run_agent_system()

                        # 关键断言：_run_commit_phase 不应该被调用
                        # 因为 auto_commit=False（默认值）
                        mock_commit.assert_not_called()

                        # 验证结果中没有 commits 字段
                        assert "commits" not in result or result.get("commits") is None

                        # 验证执行成功
                        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_iterate_mode_commit_only_with_explicit_auto_commit(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 iterate 模式仅在显式指定 --auto-commit 时才触发提交

        这个测试验证了默认提交策略：只有显式开启 auto_commit 时才会提交。
        """
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self):
                self.requirement = "测试显式提交"
                self.directory = "."
                self.skip_online = True
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = False
                self.max_iterations = "1"
                self.workers = 2
                self.force_update = False
                self.verbose = False
                # 关键：显式启用 auto_commit
                self.auto_commit = True
                self.auto_push = False
                self.commit_per_iteration = False
                self.commit_message = ""
                self.orchestrator = "mp"
                self.no_mp = False

        iterate_args = IterateArgs()
        iterator = SelfIterator(iterate_args)
        iterator.context.iteration_goal = "测试显式提交目标"

        mock_run_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        mock_commit_result = {
            "total_commits": 1,
            "commit_hashes": ["explicit_commit_abc123"],
            "commit_messages": ["feat: 测试显式提交"],
            "pushed_commits": 0,
            "files_changed": ["src/main.py"],
        }

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMP:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    mock_orch = MagicMock()
                    mock_orch.run = AsyncMock(return_value=mock_run_result)
                    MockMP.return_value = mock_orch

                    # Mock _run_commit_phase - 应该被调用
                    with patch.object(
                        iterator, "_run_commit_phase", new_callable=AsyncMock
                    ) as mock_commit:
                        mock_commit.return_value = mock_commit_result

                        result = await iterator._run_agent_system()

                        # 关键断言：_run_commit_phase 应该被调用
                        # 因为 auto_commit=True（显式开启）
                        mock_commit.assert_called_once()

                        # 验证结果包含 commits 字段
                        assert "commits" in result
                        assert result["commits"]["total_commits"] == 1


# ==================== TestAutoMode ====================


class TestAutoMode:
    """自动模式测试"""

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        """创建自动模式的命令行参数"""
        return argparse.Namespace(
            task="自动分析任务",
            mode="auto",
            directory=".",
            workers=3,
            max_iterations="5",
            strict=False,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
        )

    def test_auto_mode_detection(self, mock_args: argparse.Namespace) -> None:
        """测试自动模式识别"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试识别为 iterate 模式
        analysis = analyzer.analyze("启动自我迭代更新知识库", mock_args)
        assert analysis.mode == RunMode.ITERATE

        # 测试识别为 mp 模式
        analysis = analyzer.analyze("使用多进程并行重构代码", mock_args)
        assert analysis.mode == RunMode.MP

        # 测试识别为 knowledge 模式
        analysis = analyzer.analyze("搜索知识库文档并实现功能", mock_args)
        assert analysis.mode == RunMode.KNOWLEDGE

        # 测试识别为 plan 模式
        analysis = analyzer.analyze("仅规划任务不执行", mock_args)
        assert analysis.mode == RunMode.PLAN

        # 测试识别为 ask 模式
        analysis = analyzer.analyze("直接问答模式回答问题", mock_args)
        assert analysis.mode == RunMode.ASK

    def test_auto_mode_keyword_matching(self, mock_args: argparse.Namespace) -> None:
        """测试关键词匹配"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试 iterate 关键词
        for keyword in ["自我迭代", "self-iterate", "更新知识库"]:
            analysis = analyzer.analyze(f"使用 {keyword} 模式", mock_args)
            assert analysis.mode == RunMode.ITERATE, f"关键词 '{keyword}' 未匹配到 ITERATE"

        # 测试 mp 关键词
        for keyword in ["多进程", "parallel", "并行"]:
            analysis = analyzer.analyze(f"使用 {keyword} 执行", mock_args)
            assert analysis.mode == RunMode.MP, f"关键词 '{keyword}' 未匹配到 MP"

        # 测试 knowledge 关键词
        for keyword in ["知识库", "文档搜索", "docs"]:
            analysis = analyzer.analyze(f"搜索 {keyword}", mock_args)
            assert analysis.mode == RunMode.KNOWLEDGE, f"关键词 '{keyword}' 未匹配到 KNOWLEDGE"


# ==================== TestTaskAnalyzer ====================


class TestTaskAnalyzer:
    """任务分析器测试"""

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        """创建通用命令行参数"""
        return argparse.Namespace(
            task="分析任务",
            mode="auto",
            directory=".",
            workers=3,
            max_iterations="5",
            strict=False,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
        )

    def test_rule_based_analysis(self, mock_args: argparse.Namespace) -> None:
        """测试规则分析"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试模式识别
        analysis = analyzer._rule_based_analysis("使用多进程并行执行任务", mock_args)
        assert analysis.mode == RunMode.MP
        assert "检测到 mp 模式关键词" in analysis.reasoning

        # 测试选项识别
        analysis = analyzer._rule_based_analysis("跳过在线检查更新任务", mock_args)
        assert analysis.options.get("skip_online") is True
        assert "检测到启用选项 skip_online" in analysis.reasoning

        # 测试禁用选项
        analysis = analyzer._rule_based_analysis("禁用提交执行任务", mock_args)
        assert analysis.options.get("auto_commit") is False
        assert "检测到禁用选项 auto_commit" in analysis.reasoning

        # 测试 Worker 数量提取
        analysis = analyzer._rule_based_analysis("使用 5 个 worker 执行", mock_args)
        assert analysis.options.get("workers") == 5
        assert "提取 Worker 数量: 5" in analysis.reasoning

        # 测试无限迭代
        analysis = analyzer._rule_based_analysis("无限迭代直到完成", mock_args)
        assert analysis.options.get("max_iterations") == -1
        assert "检测到无限迭代关键词" in analysis.reasoning

    def test_mode_keyword_matching(self, mock_args: argparse.Namespace) -> None:
        """测试模式关键词匹配"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试各种模式关键词
        test_cases = [
            # (任务描述, 期望模式)
            ("启动自我迭代流程", RunMode.ITERATE),
            ("使用 self-iterate 更新", RunMode.ITERATE),
            ("检查更新 changelog", RunMode.ITERATE),
            ("多进程并行处理", RunMode.MP),
            ("使用 parallel 执行任务", RunMode.MP),
            ("搜索知识库获取信息", RunMode.KNOWLEDGE),
            ("参考 cursor 文档实现", RunMode.KNOWLEDGE),
            ("规划任务分解步骤", RunMode.PLAN),
            ("仅规划不执行", RunMode.PLAN),
            ("问答模式直接对话", RunMode.ASK),
            ("直接提问 question", RunMode.ASK),
            ("简单任务不需要特殊模式", RunMode.BASIC),  # 默认模式
        ]

        for task_desc, expected_mode in test_cases:
            analysis = analyzer.analyze(task_desc, mock_args)
            assert analysis.mode == expected_mode, (
                f"任务 '{task_desc}' 期望模式 {expected_mode.value}，"
                f"实际 {analysis.mode.value}"
            )

    def test_empty_task_analysis(self, mock_args: argparse.Namespace) -> None:
        """测试空任务分析"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 空任务应使用默认模式
        analysis = analyzer.analyze("", mock_args)
        assert analysis.mode == MODE_ALIASES.get(mock_args.mode, RunMode.BASIC)
        assert analysis.goal == ""
        assert "无任务描述" in analysis.reasoning

    def test_option_extraction(self, mock_args: argparse.Namespace) -> None:
        """测试选项提取"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试多个选项同时存在
        analysis = analyzer._rule_based_analysis(
            "跳过在线检查，启用严格模式，使用 3 个 worker 无限迭代",
            mock_args
        )

        assert analysis.options.get("skip_online") is True
        assert analysis.options.get("strict") is True
        assert analysis.options.get("workers") == 3
        assert analysis.options.get("max_iterations") == -1

    def test_combined_analysis(self, mock_args: argparse.Namespace) -> None:
        """测试组合分析"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 组合模式和选项
        analysis = analyzer.analyze(
            "使用多进程并行，5 个 worker，无限迭代重构代码",
            mock_args
        )

        assert analysis.mode == RunMode.MP
        assert analysis.options.get("workers") == 5
        assert analysis.options.get("max_iterations") == -1
        assert analysis.goal == "使用多进程并行，5 个 worker，无限迭代重构代码"


# ==================== 辅助测试 ====================


class TestParseMaxIterations:
    """parse_max_iterations 函数测试"""

    def test_numeric_values(self) -> None:
        """测试数字值"""
        assert parse_max_iterations("5") == 5
        assert parse_max_iterations("10") == 10
        assert parse_max_iterations("100") == 100

    def test_unlimited_values(self) -> None:
        """测试无限迭代值"""
        assert parse_max_iterations("MAX") == -1
        assert parse_max_iterations("max") == -1
        assert parse_max_iterations("UNLIMITED") == -1
        assert parse_max_iterations("INF") == -1
        assert parse_max_iterations("INFINITE") == -1
        assert parse_max_iterations("-1") == -1
        assert parse_max_iterations("0") == -1

    def test_invalid_values(self) -> None:
        """测试无效值

        注意：parse_max_iterations 抛出 MaxIterationsParseError（来自 core.config）
        如需用于 argparse 类型转换，使用 parse_max_iterations_for_argparse
        """
        from core.config import MaxIterationsParseError

        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("invalid")

        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("abc")


class TestRunModeAliases:
    """运行模式别名测试"""

    def test_basic_aliases(self) -> None:
        """测试基本模式别名"""
        assert MODE_ALIASES["default"] == RunMode.BASIC
        assert MODE_ALIASES["basic"] == RunMode.BASIC
        assert MODE_ALIASES["simple"] == RunMode.BASIC

    def test_mp_aliases(self) -> None:
        """测试多进程模式别名"""
        assert MODE_ALIASES["mp"] == RunMode.MP
        assert MODE_ALIASES["multiprocess"] == RunMode.MP
        assert MODE_ALIASES["parallel"] == RunMode.MP

    def test_knowledge_aliases(self) -> None:
        """测试知识库模式别名"""
        assert MODE_ALIASES["knowledge"] == RunMode.KNOWLEDGE
        assert MODE_ALIASES["kb"] == RunMode.KNOWLEDGE
        assert MODE_ALIASES["docs"] == RunMode.KNOWLEDGE

    def test_iterate_aliases(self) -> None:
        """测试迭代模式别名"""
        assert MODE_ALIASES["iterate"] == RunMode.ITERATE
        assert MODE_ALIASES["self-iterate"] == RunMode.ITERATE
        assert MODE_ALIASES["self"] == RunMode.ITERATE
        assert MODE_ALIASES["update"] == RunMode.ITERATE

    def test_plan_aliases(self) -> None:
        """测试规划模式别名"""
        assert MODE_ALIASES["plan"] == RunMode.PLAN
        assert MODE_ALIASES["planning"] == RunMode.PLAN
        assert MODE_ALIASES["analyze"] == RunMode.PLAN

    def test_ask_aliases(self) -> None:
        """测试问答模式别名"""
        assert MODE_ALIASES["ask"] == RunMode.ASK
        assert MODE_ALIASES["chat"] == RunMode.ASK
        assert MODE_ALIASES["question"] == RunMode.ASK
        assert MODE_ALIASES["q"] == RunMode.ASK


# ==================== TestRunPlanAskCliIntegration ====================


class TestRunPlanAskCliIntegration:
    """真正的端到端：子进程运行 run.py，注入 mock agent CLI。

    目标：固化“run.py --mode plan/ask”会向 agent CLI 传递正确的 --mode 参数，
    且 plan/ask 下不会携带 --force（只读语义）。
    """

    def _repo_root(self) -> Path:
        # tests/.. -> 仓库根目录
        return Path(__file__).resolve().parents[1]

    def _mock_agent_cli_path(self) -> Path:
        return self._repo_root() / "tests" / "mock_agent_cli.py"

    def _run_py_path(self) -> Path:
        return self._repo_root() / "run.py"

    def test_run_py_plan_mode_invokes_agent_cli_with_plan(self) -> None:
        env = os.environ.copy()
        env["AGENT_CLI_PATH"] = str(self._mock_agent_cli_path())

        proc = subprocess.run(
            [sys.executable, str(self._run_py_path()), "--mode", "plan", "测试规划任务"],
            cwd=str(self._repo_root()),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
        assert "执行计划" in proc.stdout
        assert "MOCK_PLAN_OUTPUT: ok" in proc.stdout

    def test_run_py_ask_mode_invokes_agent_cli_with_ask(self) -> None:
        env = os.environ.copy()
        env["AGENT_CLI_PATH"] = str(self._mock_agent_cli_path())

        proc = subprocess.run(
            [sys.executable, str(self._run_py_path()), "--mode", "ask", "测试问答问题"],
            cwd=str(self._repo_root()),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
        assert "回答" in proc.stdout
        assert "MOCK_ASK_OUTPUT: ok" in proc.stdout
