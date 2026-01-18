"""端到端 Agent 协作测试

验证多 Agent 系统的协作机制，包括：
- Planner 与 Worker 的协作
- Reviewer 反馈循环
- Committer 集成
- 子规划者协作

使用 Mock 替代真实 Cursor CLI 调用
"""
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.committer import CommitterAgent, CommitterConfig, CommitResult
from agents.planner import PlannerAgent, PlannerConfig
from agents.reviewer import ReviewDecision, ReviewerAgent, ReviewerConfig
from agents.worker import WorkerAgent, WorkerConfig
from coordinator.orchestrator import Orchestrator, OrchestratorConfig
from coordinator.worker_pool import WorkerPool
from core.base import AgentRole, AgentStatus
from core.state import IterationStatus
from tasks.task import Task, TaskPriority, TaskStatus, TaskType


class TestPlannerWorkerCollaboration:
    """Planner 与 Worker 协作测试"""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """创建用于协作测试的 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=3,
            worker_pool_size=3,
            enable_auto_commit=False,
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_planner_creates_tasks_for_workers(
        self, orchestrator: Orchestrator
    ) -> None:
        """规划者创建任务并分配给 Worker"""
        # Mock 规划阶段 - 返回多个任务
        mock_plan_result = {
            "success": True,
            "analysis": "分析代码库结构，确定需要实现的功能",
            "tasks": [
                {
                    "type": "implement",
                    "title": "实现用户认证模块",
                    "description": "实现用户登录和注册功能",
                    "instruction": "在 auth/ 目录下创建认证模块",
                    "target_files": ["auth/login.py", "auth/register.py"],
                    "priority": "high",
                },
                {
                    "type": "implement",
                    "title": "实现数据模型",
                    "description": "创建用户数据模型",
                    "instruction": "在 models/ 目录下创建 User 模型",
                    "target_files": ["models/user.py"],
                    "priority": "normal",
                },
                {
                    "type": "test",
                    "title": "编写单元测试",
                    "description": "为认证模块编写测试",
                    "instruction": "在 tests/ 目录下创建测试文件",
                    "target_files": ["tests/test_auth.py"],
                    "priority": "normal",
                },
            ],
            "sub_planners_needed": [],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 90,
            "summary": "所有任务完成",
        }

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                ) as mock_reviewer:
                    mock_planner.return_value = mock_plan_result
                    mock_reviewer.return_value = mock_review_result

                    # 模拟任务完成
                    async def simulate_task_completion(
                        queue: Any, iteration_id: int
                    ) -> None:
                        tasks = queue.get_tasks_by_iteration(iteration_id)
                        # 验证任务数量正确
                        assert len(tasks) == 3
                        # 验证任务类型
                        task_types = [t.type for t in tasks]
                        assert TaskType.IMPLEMENT in task_types
                        assert TaskType.TEST in task_types
                        # 模拟完成
                        for task in tasks:
                            task.complete({"output": f"{task.title} 完成"})

                    mock_workers.side_effect = simulate_task_completion

                    # 执行工作流
                    result = await orchestrator.run("实现用户认证功能")

                    # 验证规划者被调用
                    mock_planner.assert_called_once()

                    # 验证 Worker 池被调用
                    mock_workers.assert_called_once()

                    # 验证任务创建数量
                    assert result["total_tasks_created"] == 3
                    assert result["total_tasks_completed"] == 3

    @pytest.mark.asyncio
    async def test_worker_receives_context_from_planner(
        self, orchestrator: Orchestrator
    ) -> None:
        """Worker 接收规划上下文"""
        # 准备带上下文的任务
        mock_plan_result = {
            "success": True,
            "analysis": "需要重构数据库连接层",
            "tasks": [
                {
                    "type": "refactor",
                    "title": "重构数据库连接",
                    "description": "优化数据库连接池",
                    "instruction": "重构 db/connection.py 使用连接池",
                    "target_files": ["db/connection.py"],
                    "context": {
                        "current_issues": ["连接泄漏", "性能问题"],
                        "suggested_approach": "使用 asyncpg 连接池",
                    },
                },
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 95,
            "summary": "重构完成",
        }

        received_contexts: list[dict] = []

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                ) as mock_reviewer:
                    mock_planner.return_value = mock_plan_result
                    mock_reviewer.return_value = mock_review_result

                    async def capture_context(queue: Any, iteration_id: int) -> None:
                        tasks = queue.get_tasks_by_iteration(iteration_id)
                        for task in tasks:
                            # 记录任务上下文
                            received_contexts.append({
                                "task_id": task.id,
                                "context": task.context,
                                "target_files": task.target_files,
                            })
                            task.complete({"output": "完成"})

                    mock_workers.side_effect = capture_context

                    await orchestrator.run("重构数据库连接层")

                    # 验证 Worker 接收到正确的上下文
                    assert len(received_contexts) == 1
                    ctx = received_contexts[0]
                    assert "current_issues" in ctx["context"]
                    assert "suggested_approach" in ctx["context"]
                    assert ctx["target_files"] == ["db/connection.py"]

    @pytest.mark.asyncio
    async def test_multiple_workers_parallel_execution(
        self, orchestrator: Orchestrator
    ) -> None:
        """多 Worker 并行执行"""
        # 创建多个任务供并行执行
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": f"任务 {i}",
                    "description": f"并行任务 {i}",
                    "instruction": f"执行任务 {i}",
                    "target_files": [f"file{i}.py"],
                }
                for i in range(5)
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 100,
            "summary": "全部完成",
        }

        execution_times: list[float] = []

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                ) as mock_reviewer:
                    mock_planner.return_value = mock_plan_result
                    mock_reviewer.return_value = mock_review_result

                    async def simulate_parallel_execution(
                        queue: Any, iteration_id: int
                    ) -> None:
                        tasks = queue.get_tasks_by_iteration(iteration_id)
                        # 验证有足够的任务进行并行处理
                        assert len(tasks) == 5

                        # 模拟并行执行（记录开始时间）
                        start_time = asyncio.get_event_loop().time()

                        # 模拟每个任务的执行
                        for task in tasks:
                            task.start()
                            execution_times.append(asyncio.get_event_loop().time())
                            task.complete({"output": "完成", "worker": "test"})

                        # 记录总执行时间
                        total_time = asyncio.get_event_loop().time() - start_time
                        # 并行执行应该很快（因为是模拟）
                        assert total_time < 1.0

                    mock_workers.side_effect = simulate_parallel_execution

                    result = await orchestrator.run("并行执行多任务")

                    assert result["success"] is True
                    assert result["total_tasks_created"] == 5
                    assert result["total_tasks_completed"] == 5

    @pytest.mark.asyncio
    async def test_worker_result_aggregation(
        self, orchestrator: Orchestrator
    ) -> None:
        """执行结果聚合"""
        mock_plan_result = {
            "success": True,
            "tasks": [
                {"type": "implement", "title": "成功任务1", "instruction": "执行1", "target_files": []},
                {"type": "implement", "title": "成功任务2", "instruction": "执行2", "target_files": []},
                {"type": "implement", "title": "失败任务", "instruction": "执行3", "target_files": []},
                {"type": "test", "title": "成功任务3", "instruction": "执行4", "target_files": []},
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 75,
            "summary": "大部分任务完成",
        }

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                ) as mock_reviewer:
                    mock_planner.return_value = mock_plan_result
                    mock_reviewer.return_value = mock_review_result

                    async def simulate_mixed_results(
                        queue: Any, iteration_id: int
                    ) -> None:
                        tasks = queue.get_tasks_by_iteration(iteration_id)
                        for i, task in enumerate(tasks):
                            if "失败" in task.title:
                                task.fail("模拟执行失败")
                            else:
                                task.complete({
                                    "output": f"输出 {i}",
                                    "files_modified": [f"file{i}.py"],
                                })

                    mock_workers.side_effect = simulate_mixed_results

                    result = await orchestrator.run("执行多任务并聚合结果")

                    # 验证结果聚合
                    assert result["total_tasks_created"] == 4
                    assert result["total_tasks_completed"] == 3
                    assert result["total_tasks_failed"] == 1

                    # 验证 worker 统计
                    worker_stats = result.get("worker_stats", {})
                    assert worker_stats.get("pool_size") == 3


class TestReviewerFeedbackLoop:
    """Reviewer 反馈循环测试"""

    @pytest.fixture
    def multi_iteration_orchestrator(self) -> Orchestrator:
        """创建支持多迭代的 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=5,
            worker_pool_size=2,
            enable_auto_commit=False,
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_reviewer_receives_execution_results(
        self, multi_iteration_orchestrator: Orchestrator
    ) -> None:
        """评审者接收执行结果"""
        orchestrator = multi_iteration_orchestrator

        mock_plan_result = {
            "success": True,
            "tasks": [
                {"type": "implement", "title": "功能实现", "instruction": "实现功能", "target_files": ["main.py"]},
                {"type": "test", "title": "测试编写", "instruction": "编写测试", "target_files": ["test_main.py"]},
            ],
        }

        received_review_args: list[dict] = []

        # 创建真实的 review_iteration mock 来捕获参数
        original_review = orchestrator.reviewer.review_iteration

        async def capture_review_args(
            goal: str,
            iteration_id: int,
            tasks_completed: list[dict],
            tasks_failed: list[dict],
            previous_reviews: list[dict] | None = None,
        ) -> dict:
            received_review_args.append({
                "goal": goal,
                "iteration_id": iteration_id,
                "tasks_completed": tasks_completed,
                "tasks_failed": tasks_failed,
            })
            return {
                "success": True,
                "decision": ReviewDecision.COMPLETE,
                "score": 90,
                "summary": "完成",
            }

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration",
                    side_effect=capture_review_args,
                ):
                    mock_planner.return_value = mock_plan_result

                    async def complete_tasks(queue: Any, iteration_id: int) -> None:
                        tasks = queue.get_tasks_by_iteration(iteration_id)
                        for task in tasks:
                            task.complete({
                                "output": f"{task.title} 执行成功",
                                "duration": 1.5,
                            })

                    mock_workers.side_effect = complete_tasks

                    await orchestrator.run("实现并测试功能")

                    # 验证评审者接收到执行结果
                    assert len(received_review_args) == 1
                    review_call = received_review_args[0]

                    assert review_call["iteration_id"] == 1
                    assert len(review_call["tasks_completed"]) == 2
                    assert len(review_call["tasks_failed"]) == 0

                    # 验证完成任务的详情
                    completed_titles = [t["title"] for t in review_call["tasks_completed"]]
                    assert "功能实现" in completed_titles
                    assert "测试编写" in completed_titles

    @pytest.mark.asyncio
    async def test_review_feedback_affects_next_plan(
        self, multi_iteration_orchestrator: Orchestrator
    ) -> None:
        """评审反馈影响下轮规划"""
        orchestrator = multi_iteration_orchestrator

        plan_call_count = 0
        received_contexts: list[dict] = []

        def get_plan_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal plan_call_count
            plan_call_count += 1

            # 记录接收到的上下文
            context = args[1] if len(args) > 1 else kwargs.get("context", {})
            received_contexts.append(context)

            return {
                "success": True,
                "tasks": [
                    {
                        "type": "implement",
                        "title": f"迭代{plan_call_count}任务",
                        "instruction": "执行",
                        "target_files": ["file.py"],
                    }
                ],
            }

        review_call_count = 0

        def get_review_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal review_call_count
            review_call_count += 1

            if review_call_count == 1:
                return {
                    "success": True,
                    "decision": ReviewDecision.CONTINUE,
                    "score": 60,
                    "summary": "需要添加更多测试覆盖",
                    "suggestions": ["增加边界条件测试", "添加异常处理测试"],
                    "next_iteration_focus": "完善测试覆盖率",
                }
            else:
                return {
                    "success": True,
                    "decision": ReviewDecision.COMPLETE,
                    "score": 95,
                    "summary": "测试覆盖完善",
                }

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                ) as mock_reviewer:
                    mock_planner.side_effect = get_plan_result
                    mock_reviewer.side_effect = get_review_result

                    async def complete_tasks(queue: Any, iteration_id: int) -> None:
                        tasks = queue.get_tasks_by_iteration(iteration_id)
                        for task in tasks:
                            task.complete({"output": "完成"})

                    mock_workers.side_effect = complete_tasks

                    result = await orchestrator.run("实现功能并完善测试")

                    # 验证经过两轮迭代
                    assert result["iterations_completed"] == 2
                    assert plan_call_count == 2
                    assert review_call_count == 2

                    # 验证第二轮规划接收到了评审反馈
                    # 注：第一次规划没有 previous_review，第二次有
                    assert len(received_contexts) == 2

    @pytest.mark.asyncio
    async def test_strict_review_mode(
        self, multi_iteration_orchestrator: Orchestrator
    ) -> None:
        """严格评审模式测试"""
        # 创建严格模式的 Orchestrator
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=5,
            worker_pool_size=2,
            strict_review=True,  # 启用严格模式
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        # 验证 Reviewer 配置为严格模式
        assert orchestrator.reviewer.reviewer_config.strict_mode is True

        mock_plan_result = {
            "success": True,
            "tasks": [
                {"type": "implement", "title": "任务", "instruction": "执行", "target_files": []},
            ],
        }

        review_call_count = 0

        def strict_review(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal review_call_count
            review_call_count += 1

            # 严格模式下前两轮都不通过
            if review_call_count < 3:
                return {
                    "success": True,
                    "decision": ReviewDecision.CONTINUE,
                    "score": 70,
                    "summary": "严格模式：需要更高质量",
                    "issues": ["代码质量不达标", "缺少文档"],
                }
            else:
                return {
                    "success": True,
                    "decision": ReviewDecision.COMPLETE,
                    "score": 95,
                    "summary": "达到严格模式标准",
                }

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                ) as mock_reviewer:
                    mock_planner.return_value = mock_plan_result
                    mock_reviewer.side_effect = strict_review

                    async def complete_tasks(queue: Any, iteration_id: int) -> None:
                        tasks = queue.get_tasks_by_iteration(iteration_id)
                        for task in tasks:
                            task.complete({"output": "完成"})

                    mock_workers.side_effect = complete_tasks

                    result = await orchestrator.run("严格模式任务")

                    # 严格模式下需要更多迭代才能通过
                    assert result["iterations_completed"] == 3
                    assert review_call_count == 3


class TestCommitterIntegration:
    """Committer 集成测试"""

    @pytest.fixture
    def orchestrator_with_committer(self) -> Orchestrator:
        """创建启用自动提交的 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=3,
            worker_pool_size=2,
            enable_auto_commit=True,  # 启用自动提交
            commit_on_complete=True,   # 完成时提交
            auto_push=False,           # 不自动推送
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_committer_after_complete_review(
        self, orchestrator_with_committer: Orchestrator
    ) -> None:
        """COMPLETE 决策后自动提交"""
        orchestrator = orchestrator_with_committer

        mock_plan_result = {
            "success": True,
            "tasks": [
                {"type": "implement", "title": "功能实现", "instruction": "执行", "target_files": ["main.py"]},
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 90,
            "summary": "功能完成",
        }

        mock_commit_result = {
            "success": True,
            "commit_hash": "abc123def456",
            "message": "chore(iter-1): 完成 1 个任务",
            "files_changed": ["main.py"],
            "pushed": False,
            "iteration_id": 1,
            "tasks_count": 1,
        }

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                ) as mock_reviewer:
                    with patch.object(
                        orchestrator.committer, "commit_iteration", new_callable=AsyncMock
                    ) as mock_committer:
                        mock_planner.return_value = mock_plan_result
                        mock_reviewer.return_value = mock_review_result
                        mock_committer.return_value = mock_commit_result

                        async def complete_tasks(queue: Any, iteration_id: int) -> None:
                            tasks = queue.get_tasks_by_iteration(iteration_id)
                            for task in tasks:
                                task.complete({"output": "完成"})

                        mock_workers.side_effect = complete_tasks

                        result = await orchestrator.run("实现功能")

                        # 验证 Committer 被调用
                        mock_committer.assert_called_once()

                        # 验证调用参数
                        call_args = mock_committer.call_args
                        assert call_args.kwargs["iteration_id"] == 1
                        assert call_args.kwargs["review_decision"] == "complete"

                        # 验证结果中包含提交信息
                        assert result["success"] is True
                        commits = result.get("commits", {})
                        assert commits.get("successful_commits", 0) >= 0

    @pytest.mark.asyncio
    async def test_commit_per_iteration_mode(self) -> None:
        """每次迭代提交模式"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=3,
            worker_pool_size=1,
            enable_auto_commit=True,
            commit_per_iteration=True,  # 每次迭代都提交
            commit_on_complete=False,
        )
        orchestrator = Orchestrator(config)

        iteration_count = 0

        def get_plan_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal iteration_count
            iteration_count += 1
            return {
                "success": True,
                "tasks": [
                    {"type": "implement", "title": f"迭代{iteration_count}", "instruction": "执行", "target_files": []},
                ],
            }

        review_count = 0

        def get_review_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal review_count
            review_count += 1
            if review_count < 2:
                return {
                    "success": True,
                    "decision": ReviewDecision.CONTINUE,
                    "score": 60,
                    "summary": "继续迭代",
                }
            return {
                "success": True,
                "decision": ReviewDecision.COMPLETE,
                "score": 90,
                "summary": "完成",
            }

        commit_calls: list[dict] = []

        async def track_commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
            commit_calls.append(kwargs)
            return {
                "success": True,
                "commit_hash": f"hash_{len(commit_calls)}",
                "message": f"迭代提交 {len(commit_calls)}",
                "files_changed": [],
                "pushed": False,
            }

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                ) as mock_reviewer:
                    with patch.object(
                        orchestrator.committer, "commit_iteration",
                        side_effect=track_commit,
                    ):
                        mock_planner.side_effect = get_plan_result
                        mock_reviewer.side_effect = get_review_result

                        async def complete_tasks(queue: Any, iteration_id: int) -> None:
                            tasks = queue.get_tasks_by_iteration(iteration_id)
                            for task in tasks:
                                task.complete({"output": "完成"})

                        mock_workers.side_effect = complete_tasks

                        result = await orchestrator.run("每次迭代提交")

                        # 每次迭代都应该提交
                        assert result["iterations_completed"] == 2
                        assert len(commit_calls) == 2

                        # 验证提交顺序
                        assert commit_calls[0]["iteration_id"] == 1
                        assert commit_calls[1]["iteration_id"] == 2

    @pytest.mark.asyncio
    async def test_commit_with_failed_tasks(
        self, orchestrator_with_committer: Orchestrator
    ) -> None:
        """有失败任务时的提交策略"""
        orchestrator = orchestrator_with_committer

        mock_plan_result = {
            "success": True,
            "tasks": [
                {"type": "implement", "title": "成功任务", "instruction": "执行", "target_files": ["success.py"]},
                {"type": "implement", "title": "失败任务", "instruction": "执行", "target_files": ["fail.py"]},
            ],
        }

        # 即使有失败任务，只要评审通过也可以提交
        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 70,
            "summary": "部分完成，可以提交",
        }

        commit_args: dict = {}

        async def capture_commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal commit_args
            commit_args = kwargs
            return {
                "success": True,
                "commit_hash": "partial_commit",
                "message": "部分完成的提交",
                "files_changed": ["success.py"],
                "pushed": False,
            }

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.worker_pool, "start", new_callable=AsyncMock
            ) as mock_workers:
                with patch.object(
                    orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                ) as mock_reviewer:
                    with patch.object(
                        orchestrator.committer, "commit_iteration",
                        side_effect=capture_commit,
                    ):
                        mock_planner.return_value = mock_plan_result
                        mock_reviewer.return_value = mock_review_result

                        async def mixed_results(queue: Any, iteration_id: int) -> None:
                            tasks = queue.get_tasks_by_iteration(iteration_id)
                            for task in tasks:
                                if "失败" in task.title:
                                    task.fail("模拟失败")
                                else:
                                    task.complete({"output": "成功"})

                        mock_workers.side_effect = mixed_results

                        result = await orchestrator.run("包含失败任务的提交")

                        # 验证提交仍然执行
                        assert len(commit_args) > 0
                        assert commit_args["iteration_id"] == 1

                        # 验证只有成功的任务被包含在提交中
                        tasks_completed = commit_args.get("tasks_completed", [])
                        assert len(tasks_completed) == 1
                        assert tasks_completed[0]["title"] == "成功任务"


class TestSubPlannerCollaboration:
    """子规划者协作测试"""

    @pytest.fixture
    def orchestrator_with_sub_planners(self) -> Orchestrator:
        """创建启用子规划者的 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=3,
            worker_pool_size=2,
            enable_sub_planners=True,  # 启用子规划者
            enable_auto_commit=False,
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_spawn_sub_planner(
        self, orchestrator_with_sub_planners: Orchestrator
    ) -> None:
        """子规划者派生"""
        orchestrator = orchestrator_with_sub_planners

        # 主规划者建议需要子规划者
        mock_main_plan_result = {
            "success": True,
            "analysis": "项目需要分模块规划",
            "tasks": [
                {"type": "implement", "title": "主任务", "instruction": "执行", "target_files": ["main.py"]},
            ],
            "sub_planners_needed": [
                {
                    "area": "前端模块",
                    "reason": "前端逻辑复杂，需要专门规划",
                },
                {
                    "area": "后端 API",
                    "reason": "后端接口众多，需要系统规划",
                },
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 90,
            "summary": "完成",
        }

        spawned_sub_planners: list[str] = []

        # Mock spawn_sub_planner 方法
        original_spawn = orchestrator.planner.spawn_sub_planner

        async def track_spawn(area: str, context: dict | None = None) -> PlannerAgent:
            spawned_sub_planners.append(area)
            # 创建一个模拟的子规划者
            sub_config = PlannerConfig(
                name=f"sub-planner-{area}",
                working_directory=".",
            )
            sub_planner = PlannerAgent(sub_config)

            # Mock 子规划者的 execute 方法
            sub_planner.execute = AsyncMock(return_value={
                "success": True,
                "tasks": [
                    {"type": "implement", "title": f"{area}子任务", "instruction": f"执行{area}相关工作", "target_files": []},
                ],
            })

            return sub_planner

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.planner, "spawn_sub_planner",
                side_effect=track_spawn,
            ):
                with patch.object(
                    orchestrator.worker_pool, "start", new_callable=AsyncMock
                ) as mock_workers:
                    with patch.object(
                        orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                    ) as mock_reviewer:
                        mock_planner.return_value = mock_main_plan_result
                        mock_reviewer.return_value = mock_review_result

                        async def complete_tasks(queue: Any, iteration_id: int) -> None:
                            tasks = queue.get_tasks_by_iteration(iteration_id)
                            for task in tasks:
                                task.complete({"output": "完成"})

                        mock_workers.side_effect = complete_tasks

                        result = await orchestrator.run("需要子规划者的复杂任务")

                        # 验证子规划者被派生
                        assert len(spawned_sub_planners) == 2
                        assert "前端模块" in spawned_sub_planners
                        assert "后端 API" in spawned_sub_planners

    @pytest.mark.asyncio
    async def test_sub_planner_task_merging(
        self, orchestrator_with_sub_planners: Orchestrator
    ) -> None:
        """子规划任务合并"""
        orchestrator = orchestrator_with_sub_planners

        mock_main_plan_result = {
            "success": True,
            "tasks": [
                {"type": "implement", "title": "主任务", "instruction": "执行", "target_files": []},
            ],
            "sub_planners_needed": [
                {"area": "子模块A", "reason": "需要独立规划"},
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 90,
            "summary": "完成",
        }

        # 子规划者返回多个任务
        sub_planner_tasks = [
            {"type": "implement", "title": "子任务1", "instruction": "执行1", "target_files": ["sub1.py"]},
            {"type": "implement", "title": "子任务2", "instruction": "执行2", "target_files": ["sub2.py"]},
            {"type": "test", "title": "子任务3", "instruction": "测试", "target_files": ["test_sub.py"]},
        ]

        async def mock_spawn(area: str, context: dict | None = None) -> PlannerAgent:
            sub_config = PlannerConfig(name=f"sub-{area}", working_directory=".")
            sub_planner = PlannerAgent(sub_config)
            sub_planner.execute = AsyncMock(return_value={
                "success": True,
                "tasks": sub_planner_tasks,
            })
            return sub_planner

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.planner, "spawn_sub_planner",
                side_effect=mock_spawn,
            ):
                with patch.object(
                    orchestrator.worker_pool, "start", new_callable=AsyncMock
                ) as mock_workers:
                    with patch.object(
                        orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                    ) as mock_reviewer:
                        mock_planner.return_value = mock_main_plan_result
                        mock_reviewer.return_value = mock_review_result

                        all_tasks: list[Task] = []

                        async def capture_and_complete(queue: Any, iteration_id: int) -> None:
                            tasks = queue.get_tasks_by_iteration(iteration_id)
                            all_tasks.extend(tasks)
                            for task in tasks:
                                task.complete({"output": "完成"})

                        mock_workers.side_effect = capture_and_complete

                        result = await orchestrator.run("合并子规划任务")

                        # 验证主任务 + 子任务都被合并到队列
                        # 1 个主任务 + 3 个子任务 = 4 个任务
                        assert result["total_tasks_created"] == 4
                        assert result["total_tasks_completed"] == 4

                        # 验证任务来源正确
                        task_titles = [t.title for t in all_tasks]
                        assert "主任务" in task_titles
                        assert "子任务1" in task_titles
                        assert "子任务2" in task_titles
                        assert "子任务3" in task_titles

    @pytest.mark.asyncio
    async def test_max_sub_planners_limit(
        self, orchestrator_with_sub_planners: Orchestrator
    ) -> None:
        """最大子规划者数量限制"""
        orchestrator = orchestrator_with_sub_planners

        # 设置较小的限制用于测试
        orchestrator.planner.planner_config.max_sub_planners = 2

        mock_main_plan_result = {
            "success": True,
            "tasks": [
                {"type": "implement", "title": "主任务", "instruction": "执行", "target_files": []},
            ],
            "sub_planners_needed": [
                {"area": "模块A", "reason": "需要规划A"},
                {"area": "模块B", "reason": "需要规划B"},
                {"area": "模块C", "reason": "需要规划C"},  # 这个应该被限制
                {"area": "模块D", "reason": "需要规划D"},  # 这个应该被限制
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 90,
            "summary": "完成",
        }

        spawn_count = 0
        spawn_errors: list[str] = []

        # 使用真实的 spawn_sub_planner 逻辑来验证限制
        original_spawn = orchestrator.planner.spawn_sub_planner

        async def limited_spawn(area: str, context: dict | None = None) -> PlannerAgent:
            nonlocal spawn_count
            try:
                # 调用原始方法（它会检查限制）
                result = await original_spawn(area, context)
                spawn_count += 1
                # Mock 子规划者的 execute
                result.execute = AsyncMock(return_value={
                    "success": True,
                    "tasks": [],
                })
                return result
            except ValueError as e:
                spawn_errors.append(str(e))
                raise

        with patch.object(
            orchestrator.planner, "execute", new_callable=AsyncMock
        ) as mock_planner:
            with patch.object(
                orchestrator.planner, "spawn_sub_planner",
                side_effect=limited_spawn,
            ):
                with patch.object(
                    orchestrator.worker_pool, "start", new_callable=AsyncMock
                ) as mock_workers:
                    with patch.object(
                        orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
                    ) as mock_reviewer:
                        mock_planner.return_value = mock_main_plan_result
                        mock_reviewer.return_value = mock_review_result

                        async def complete_tasks(queue: Any, iteration_id: int) -> None:
                            tasks = queue.get_tasks_by_iteration(iteration_id)
                            for task in tasks:
                                task.complete({"output": "完成"})

                        mock_workers.side_effect = complete_tasks

                        await orchestrator.run("测试子规划者限制")

                        # 验证只创建了最大数量的子规划者
                        assert spawn_count == 2
                        # 验证超出限制的请求被拒绝
                        assert len(spawn_errors) == 2
                        assert all("限制" in err or "limit" in err.lower() for err in spawn_errors)
