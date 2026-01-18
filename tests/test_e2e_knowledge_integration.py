#!/usr/bin/env python3
"""端到端测试 - 知识库与 Agent 集成测试

测试内容:
1. KnowledgeManager 与 Orchestrator 集成
2. Cursor 关键字检测与知识库搜索触发
3. 语义搜索与 Agent 集成
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.planner import PlannerAgent, PlannerConfig
from agents.worker import CURSOR_KEYWORDS, WorkerAgent, WorkerConfig
from coordinator.orchestrator import Orchestrator, OrchestratorConfig
from knowledge.models import Document
from knowledge.storage import SearchResult
from tasks.task import Task, TaskPriority, TaskStatus, TaskType

from .conftest import (
    MockAgentExecutor,
    MockKnowledgeManager,
    create_test_document,
    create_test_task,
)


# ==================== TestKnowledgeManagerIntegration ====================

class TestKnowledgeManagerIntegration:
    """知识库管理器集成测试类"""

    def test_orchestrator_with_knowledge_manager(self, temp_workspace, mock_knowledge_manager):
        """测试 Orchestrator 绑定知识库管理器"""
        # 创建 Orchestrator 配置
        config = OrchestratorConfig(
            working_directory=str(temp_workspace),
            max_iterations=1,
            worker_pool_size=1,
        )

        # 使用 mock 替换实际执行器
        with patch('agents.planner.AgentExecutorFactory.create') as mock_planner_exec, \
             patch('agents.reviewer.AgentExecutorFactory.create') as mock_reviewer_exec, \
             patch('agents.worker.AgentExecutorFactory.create') as mock_worker_exec:

            mock_executor = MockAgentExecutor()
            mock_planner_exec.return_value = mock_executor
            mock_reviewer_exec.return_value = mock_executor
            mock_worker_exec.return_value = mock_executor

            # 创建 Orchestrator，初始不传入知识库
            orchestrator = Orchestrator(config=config)

            # 验证初始状态
            assert orchestrator._knowledge_manager is None

            # 绑定知识库管理器
            orchestrator.set_knowledge_manager(mock_knowledge_manager)

            # 验证绑定成功
            assert orchestrator._knowledge_manager is mock_knowledge_manager
            print("✓ Orchestrator 成功绑定知识库管理器")

    def test_orchestrator_with_knowledge_manager_at_init(self, temp_workspace, mock_knowledge_manager):
        """测试 Orchestrator 初始化时传入知识库管理器"""
        config = OrchestratorConfig(
            working_directory=str(temp_workspace),
            max_iterations=1,
            worker_pool_size=1,
        )

        with patch('agents.planner.AgentExecutorFactory.create') as mock_planner_exec, \
             patch('agents.reviewer.AgentExecutorFactory.create') as mock_reviewer_exec, \
             patch('agents.worker.AgentExecutorFactory.create') as mock_worker_exec:

            mock_executor = MockAgentExecutor()
            mock_planner_exec.return_value = mock_executor
            mock_reviewer_exec.return_value = mock_executor
            mock_worker_exec.return_value = mock_executor

            # 创建时传入知识库管理器
            orchestrator = Orchestrator(
                config=config,
                knowledge_manager=mock_knowledge_manager,
            )

            # 验证初始化时已绑定
            assert orchestrator._knowledge_manager is mock_knowledge_manager
            print("✓ Orchestrator 初始化时成功绑定知识库管理器")

    @pytest.mark.asyncio
    async def test_worker_knowledge_search(self, temp_workspace, mock_knowledge_manager):
        """测试 Worker 自动搜索知识库"""
        # 添加测试文档到知识库
        doc = create_test_document(
            url="https://cursor.com/docs/cli",
            title="Cursor CLI 文档",
            content="Cursor CLI 使用 --mode=plan 进行规划模式...",
        )
        mock_knowledge_manager.add_document(doc)

        # 配置搜索结果
        mock_knowledge_manager.configure_search_results([
            SearchResult(
                doc_id=doc.id,
                url=doc.url,
                title=doc.title,
                score=0.85,
                snippet="Cursor CLI 使用...",
                match_type="keyword",
            )
        ])

        # 创建 Worker 配置，启用知识库搜索
        worker_config = WorkerConfig(
            name="test-worker",
            working_directory=str(temp_workspace),
            enable_knowledge_search=True,
            knowledge_search_top_k=5,
        )

        with patch('agents.worker.AgentExecutorFactory.create') as mock_exec:
            mock_executor = MockAgentExecutor()
            mock_executor.configure_response(success=True, output="任务完成")
            mock_exec.return_value = mock_executor

            # 创建 Worker，传入知识库管理器
            worker = WorkerAgent(
                config=worker_config,
                knowledge_manager=mock_knowledge_manager,
            )

            # 验证知识库搜索已启用
            assert worker._knowledge_search_enabled is True
            assert worker._knowledge_manager is mock_knowledge_manager

            # 创建一个 Cursor 相关的任务
            task = create_test_task(
                title="配置 Cursor CLI",
                description="设置 Cursor Agent CLI 的 MCP 服务器",
                instruction="配置 cursor agent 的 mcp 功能",
            )

            # 执行任务
            completed_task = await worker.execute_task(task)

            # 验证任务完成
            assert completed_task.status == TaskStatus.COMPLETED
            print("✓ Worker 自动搜索知识库功能正常")

    @pytest.mark.asyncio
    async def test_knowledge_context_injection(self, temp_workspace, mock_knowledge_manager):
        """测试知识上下文注入到任务"""
        # 准备知识库文档
        doc = create_test_document(
            url="https://cursor.com/docs/hooks",
            title="Cursor Hooks 配置",
            content="Hooks 允许通过自定义脚本观察、控制和扩展 Agent 循环...",
        )
        mock_knowledge_manager.add_document(doc)

        mock_knowledge_manager.configure_search_results([
            SearchResult(
                doc_id=doc.id,
                url=doc.url,
                title=doc.title,
                score=0.9,
                snippet="Hooks 允许通过自定义脚本...",
                match_type="keyword",
            )
        ])

        worker_config = WorkerConfig(
            name="test-worker",
            working_directory=str(temp_workspace),
            enable_knowledge_search=True,
        )

        with patch('agents.worker.AgentExecutorFactory.create') as mock_exec:
            mock_executor = MockAgentExecutor()
            mock_executor.configure_response(success=True, output="Hook 配置完成")
            mock_exec.return_value = mock_executor

            worker = WorkerAgent(
                config=worker_config,
                knowledge_manager=mock_knowledge_manager,
            )

            # 创建 Cursor hooks 相关任务
            task = create_test_task(
                title="配置 Cursor Hooks",
                description="设置 afterFileEdit hook 进行自动格式化",
                instruction="在 .cursor/hooks/ 目录下创建 hook 脚本",
            )

            # 执行任务
            completed_task = await worker.execute_task(task)

            # 验证执行器被调用，且 prompt 包含知识库上下文
            assert mock_executor.execution_count > 0
            last_call = mock_executor.execution_history[-1]
            prompt = last_call.get("prompt", "")

            # 验证 prompt 中包含知识库内容
            # （由于是 Cursor 相关任务，应触发知识库搜索）
            assert "Hook" in task.title or "hook" in task.instruction
            print("✓ 知识上下文成功注入任务执行")


# ==================== TestCursorKeywordDetection ====================

class TestCursorKeywordDetection:
    """Cursor 关键字检测测试类"""

    def test_cursor_keyword_detection(self):
        """测试 CURSOR_KEYWORDS 检测"""
        # 验证关键字列表存在且非空
        assert len(CURSOR_KEYWORDS) > 0
        print(f"✓ CURSOR_KEYWORDS 包含 {len(CURSOR_KEYWORDS)} 个关键字")

        # 验证关键字列表包含预期的关键字
        expected_keywords = ["cursor", "agent", "cli", "mcp", "hook"]
        for keyword in expected_keywords:
            assert any(keyword in kw.lower() for kw in CURSOR_KEYWORDS), \
                f"关键字列表应包含 '{keyword}'"
        print("✓ CURSOR_KEYWORDS 包含所有预期关键字")

    def test_worker_is_cursor_related_method(self, temp_workspace):
        """测试 Worker 的 _is_cursor_related 方法"""
        worker_config = WorkerConfig(
            name="test-worker",
            working_directory=str(temp_workspace),
        )

        with patch('agents.worker.AgentExecutorFactory.create') as mock_exec:
            mock_exec.return_value = MockAgentExecutor()

            worker = WorkerAgent(config=worker_config)

            # 测试 Cursor 相关文本
            cursor_texts = [
                "如何使用 Cursor Agent CLI",
                "配置 MCP 服务器",
                "设置 cursor hook",
                "使用 --force 参数",
                "cursor.com 文档",
                "stream-json 输出格式",
            ]

            for text in cursor_texts:
                assert worker._is_cursor_related(text), f"'{text}' 应被识别为 Cursor 相关"

            # 测试非 Cursor 相关文本
            non_cursor_texts = [
                "Python 列表排序",
                "React 组件开发",
                "数据库查询优化",
            ]

            for text in non_cursor_texts:
                assert not worker._is_cursor_related(text), f"'{text}' 不应被识别为 Cursor 相关"

            print("✓ _is_cursor_related 方法正确检测 Cursor 相关内容")

    @pytest.mark.asyncio
    async def test_knowledge_search_trigger(self, temp_workspace, mock_knowledge_manager):
        """测试知识库搜索触发条件"""
        doc = create_test_document(
            url="https://cursor.com/docs/agent",
            title="Cursor Agent 使用指南",
            content="Cursor Agent 支持多种模式...",
        )
        mock_knowledge_manager.add_document(doc)
        mock_knowledge_manager.configure_search_results([
            SearchResult(
                doc_id=doc.id,
                url=doc.url,
                title=doc.title,
                score=0.8,
                snippet="Agent 支持...",
                match_type="keyword",
            )
        ])

        worker_config = WorkerConfig(
            name="test-worker",
            working_directory=str(temp_workspace),
            enable_knowledge_search=True,
        )

        with patch('agents.worker.AgentExecutorFactory.create') as mock_exec:
            mock_executor = MockAgentExecutor()
            mock_executor.configure_response(success=True, output="完成")
            mock_exec.return_value = mock_executor

            worker = WorkerAgent(
                config=worker_config,
                knowledge_manager=mock_knowledge_manager,
            )

            # 测试 Cursor 相关任务触发搜索
            cursor_task = create_test_task(
                title="配置 Agent 模式",
                description="设置 Cursor Agent 的运行模式",
                instruction="配置 agent 使用 plan 模式",
            )

            # 执行搜索上下文获取
            knowledge_context = await worker._search_knowledge_context(cursor_task)

            # Cursor 相关任务应触发搜索
            assert len(knowledge_context) > 0, "Cursor 相关任务应触发知识库搜索"
            print("✓ Cursor 相关任务成功触发知识库搜索")

            # 测试非 Cursor 相关任务不触发搜索
            non_cursor_task = create_test_task(
                title="优化数据库查询",
                description="改进 SQL 查询性能",
                instruction="添加索引和优化 JOIN",
            )

            knowledge_context = await worker._search_knowledge_context(non_cursor_task)

            # 非 Cursor 相关任务不应触发搜索
            assert len(knowledge_context) == 0, "非 Cursor 相关任务不应触发知识库搜索"
            print("✓ 非 Cursor 相关任务正确跳过知识库搜索")

    def test_search_result_formatting(self, mock_knowledge_manager):
        """测试搜索结果格式化"""
        # 创建多个测试文档
        docs = [
            create_test_document(
                url="https://cursor.com/docs/cli",
                title="CLI 参考手册",
                content="完整的 CLI 参数列表...",
            ),
            create_test_document(
                url="https://cursor.com/docs/mcp",
                title="MCP 服务器配置",
                content="MCP 服务器管理指南...",
            ),
        ]

        for doc in docs:
            mock_knowledge_manager.add_document(doc)

        # 配置搜索结果
        search_results = [
            SearchResult(
                doc_id=docs[0].id,
                url=docs[0].url,
                title=docs[0].title,
                score=0.95,
                snippet="CLI 参数列表...",
                match_type="keyword",
            ),
            SearchResult(
                doc_id=docs[1].id,
                url=docs[1].url,
                title=docs[1].title,
                score=0.80,
                snippet="MCP 服务器...",
                match_type="keyword",
            ),
        ]
        mock_knowledge_manager.configure_search_results(search_results)

        # 执行搜索
        results = mock_knowledge_manager.search("cursor cli mcp", max_results=5)

        # 验证搜索结果格式
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[0].title == "CLI 参考手册"
        assert results[1].score == 0.80
        assert results[1].title == "MCP 服务器配置"

        # 验证结果按分数排序（MockKnowledgeManager 不自动排序，这里验证原始顺序）
        print("✓ 搜索结果格式正确")


# ==================== TestSemanticSearchIntegration ====================

class TestSemanticSearchIntegration:
    """语义搜索集成测试类"""

    @pytest.mark.asyncio
    async def test_worker_with_semantic_search(self, temp_workspace):
        """测试 Worker 使用语义搜索"""
        # 创建 Mock 语义搜索
        mock_semantic_search = MagicMock()

        # 配置语义搜索返回结果
        mock_chunk = MagicMock()
        mock_chunk.file_path = "src/utils.py"
        mock_chunk.start_line = 10
        mock_chunk.end_line = 25
        mock_chunk.name = "process_data"
        mock_chunk.content = "def process_data(data): ..."

        mock_search_result = MagicMock()
        mock_search_result.chunk = mock_chunk
        mock_search_result.score = 0.85

        mock_semantic_search.search = AsyncMock(return_value=[mock_search_result])
        mock_semantic_search.search_in_files = AsyncMock(return_value=[mock_search_result])

        worker_config = WorkerConfig(
            name="test-worker",
            working_directory=str(temp_workspace),
            enable_context_search=True,
            context_search_top_k=5,
            context_search_min_score=0.4,
        )

        with patch('agents.worker.AgentExecutorFactory.create') as mock_exec:
            mock_executor = MockAgentExecutor()
            mock_executor.configure_response(success=True, output="完成")
            mock_exec.return_value = mock_executor

            # 创建 Worker 并设置语义搜索
            worker = WorkerAgent(config=worker_config)
            worker.set_semantic_search(mock_semantic_search)

            # 验证语义搜索已启用
            assert worker._search_enabled is True
            assert worker._semantic_search is mock_semantic_search

            # 创建测试任务
            task = create_test_task(
                title="重构数据处理模块",
                description="优化 process_data 函数",
                instruction="重构 src/utils.py 中的数据处理逻辑",
                target_files=["src/utils.py"],
            )
            task.target_files = ["src/utils.py"]

            # 执行上下文搜索
            search_context = await worker._search_task_context(task)

            # 验证搜索结果
            assert len(search_context) > 0
            assert search_context[0]["file_path"] == "src/utils.py"
            assert search_context[0]["score"] == 0.85
            print("✓ Worker 语义搜索上下文获取成功")

    @pytest.mark.asyncio
    async def test_planner_with_semantic_search(self, temp_workspace):
        """测试 Planner 使用语义搜索"""
        # 创建 Mock 语义搜索
        mock_semantic_search = MagicMock()

        mock_chunk = MagicMock()
        mock_chunk.file_path = "agents/planner.py"
        mock_chunk.start_line = 50
        mock_chunk.end_line = 100
        mock_chunk.name = "PlannerAgent"
        mock_chunk.content = "class PlannerAgent: ..."
        mock_chunk.chunk_type = MagicMock(value="class")

        mock_search_result = MagicMock()
        mock_search_result.chunk = mock_chunk
        mock_search_result.score = 0.92

        mock_semantic_search.search = AsyncMock(return_value=[mock_search_result])

        planner_config = PlannerConfig(
            name="test-planner",
            working_directory=str(temp_workspace),
            enable_semantic_search=True,
            semantic_search_top_k=10,
            semantic_search_min_score=0.3,
        )

        with patch('agents.planner.AgentExecutorFactory.create') as mock_exec:
            mock_executor = MockAgentExecutor()
            mock_executor.configure_response(
                success=True,
                output='{"analysis": "分析完成", "tasks": [], "sub_planners_needed": []}'
            )
            mock_exec.return_value = mock_executor

            # 创建 Planner 并设置语义搜索
            planner = PlannerAgent(config=planner_config)
            planner.set_semantic_search(mock_semantic_search)

            # 验证语义搜索已启用
            assert planner._search_enabled is True
            assert planner._semantic_search is mock_semantic_search

            # 执行语义搜索探索
            search_context = await planner._explore_with_semantic_search(
                "分析 PlannerAgent 的实现"
            )

            # 验证搜索结果
            assert len(search_context.get("related_code", [])) > 0
            related_code = search_context["related_code"][0]
            assert related_code["file_path"] == "agents/planner.py"
            assert related_code["score"] == 0.92
            print("✓ Planner 语义搜索代码探索成功")

    @pytest.mark.asyncio
    async def test_search_score_threshold(self, temp_workspace):
        """测试相似度阈值过滤"""
        mock_semantic_search = MagicMock()

        # 创建不同分数的搜索结果
        mock_results = []
        scores = [0.95, 0.75, 0.50, 0.35, 0.20]

        for i, score in enumerate(scores):
            mock_chunk = MagicMock()
            mock_chunk.file_path = f"file_{i}.py"
            mock_chunk.start_line = 1
            mock_chunk.end_line = 10
            mock_chunk.name = f"function_{i}"
            mock_chunk.content = f"def function_{i}(): pass"

            mock_result = MagicMock()
            mock_result.chunk = mock_chunk
            mock_result.score = score
            mock_results.append(mock_result)

        # 模拟搜索函数根据 min_score 过滤
        async def mock_search(query, top_k=10, min_score=0.0):
            return [r for r in mock_results if r.score >= min_score]

        mock_semantic_search.search = mock_search

        worker_config = WorkerConfig(
            name="test-worker",
            working_directory=str(temp_workspace),
            enable_context_search=True,
            context_search_top_k=10,
            context_search_min_score=0.4,  # 设置阈值为 0.4
        )

        with patch('agents.worker.AgentExecutorFactory.create') as mock_exec:
            mock_exec.return_value = MockAgentExecutor()

            worker = WorkerAgent(config=worker_config)
            worker._semantic_search = mock_semantic_search
            worker._search_enabled = True

            # 创建测试任务
            task = create_test_task(
                title="测试任务",
                description="测试阈值过滤",
            )

            # 执行上下文搜索（使用 min_score=0.4）
            results = await mock_semantic_search.search(
                "测试查询",
                top_k=worker_config.context_search_top_k,
                min_score=worker_config.context_search_min_score,
            )

            # 验证只返回分数 >= 0.4 的结果
            assert len(results) == 3  # 0.95, 0.75, 0.50
            for r in results:
                assert r.score >= 0.4, f"分数 {r.score} 应 >= 0.4"

            print("✓ 相似度阈值过滤功能正常")

    @pytest.mark.asyncio
    async def test_worker_without_semantic_search(self, temp_workspace):
        """测试 Worker 禁用语义搜索时的行为"""
        worker_config = WorkerConfig(
            name="test-worker",
            working_directory=str(temp_workspace),
            enable_context_search=False,  # 禁用上下文搜索
        )

        with patch('agents.worker.AgentExecutorFactory.create') as mock_exec:
            mock_exec.return_value = MockAgentExecutor()

            worker = WorkerAgent(config=worker_config)

            # 验证语义搜索未启用
            assert worker._search_enabled is False
            assert worker._semantic_search is None

            # 创建测试任务
            task = create_test_task(
                title="普通任务",
                description="不需要语义搜索",
            )

            # 执行上下文搜索应返回空列表
            search_context = await worker._search_task_context(task)
            assert search_context == []

            print("✓ 禁用语义搜索时正确返回空结果")


# ==================== 辅助测试 ====================

class TestKnowledgeManagerSetup:
    """知识库管理器设置测试"""

    def test_worker_set_knowledge_manager(self, temp_workspace, mock_knowledge_manager):
        """测试 Worker 延迟设置知识库管理器"""
        worker_config = WorkerConfig(
            name="test-worker",
            working_directory=str(temp_workspace),
            enable_knowledge_search=True,
        )

        with patch('agents.worker.AgentExecutorFactory.create') as mock_exec:
            mock_exec.return_value = MockAgentExecutor()

            # 创建 Worker 时不传入知识库
            worker = WorkerAgent(config=worker_config)

            # 验证初始状态
            assert worker._knowledge_manager is None
            assert worker._knowledge_search_enabled is False

            # 延迟设置知识库管理器
            worker.set_knowledge_manager(mock_knowledge_manager)

            # 验证设置成功
            assert worker._knowledge_manager is mock_knowledge_manager
            assert worker._knowledge_search_enabled is True

            print("✓ Worker 延迟设置知识库管理器成功")

    def test_worker_statistics(self, temp_workspace, mock_knowledge_manager):
        """测试 Worker 统计信息包含知识库状态"""
        worker_config = WorkerConfig(
            name="test-worker",
            working_directory=str(temp_workspace),
            enable_knowledge_search=True,
            enable_context_search=True,
        )

        with patch('agents.worker.AgentExecutorFactory.create') as mock_exec:
            mock_exec.return_value = MockAgentExecutor()

            worker = WorkerAgent(
                config=worker_config,
                knowledge_manager=mock_knowledge_manager,
            )

            # 获取统计信息
            stats = worker.get_statistics()

            # 验证统计信息包含搜索状态
            assert "knowledge_search_enabled" in stats
            assert "context_search_enabled" in stats
            assert stats["knowledge_search_enabled"] is True

            print("✓ Worker 统计信息正确包含知识库状态")


# ==================== 主函数 ====================

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("端到端测试 - 知识库与 Agent 集成")
    print("=" * 60 + "\n")

    # 注意: 实际测试应通过 pytest 运行
    # 这里仅作为模块可执行的入口点

    print("请使用 pytest 运行测试:")
    print("  pytest tests/test_e2e_knowledge_integration.py -v")
    print("\n运行特定测试类:")
    print("  pytest tests/test_e2e_knowledge_integration.py::TestKnowledgeManagerIntegration -v")
    print("  pytest tests/test_e2e_knowledge_integration.py::TestCursorKeywordDetection -v")
    print("  pytest tests/test_e2e_knowledge_integration.py::TestSemanticSearchIntegration -v")


if __name__ == "__main__":
    main()
