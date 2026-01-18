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
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
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
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
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
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=True,
            search_knowledge="CLI 参数",
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            stream_log=True,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
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
            verbose=False,
            skip_online=True,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            stream_log=True,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
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
        """测试无效值"""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_max_iterations("invalid")

        with pytest.raises(argparse.ArgumentTypeError):
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
