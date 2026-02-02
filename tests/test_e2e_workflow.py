"""端到端工作流测试

验证 Orchestrator 的完整工作流逻辑，包括：
- 单迭代工作流
- 多迭代工作流
- 工作流状态转换

使用 Mock 替代真实 Cursor CLI 调用
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.reviewer import ReviewDecision
from coordinator.orchestrator import Orchestrator, OrchestratorConfig
from core.base import AgentRole, AgentStatus
from core.state import IterationStatus
from tasks.task import TaskStatus


class TestSingleIterationWorkflow:
    """单迭代工作流测试"""

    @pytest.fixture
    def mock_orchestrator(self) -> Orchestrator:
        """创建带 Mock 执行器的 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_pool_size=2,
            enable_auto_commit=False,  # 禁用自动提交简化测试
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_simple_task_completion(self, mock_orchestrator: Orchestrator) -> None:
        """单任务单迭代成功完成"""
        orchestrator = mock_orchestrator

        # Mock 规划阶段 - 返回单个任务
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "实现简单功能",
                    "description": "实现一个简单的功能",
                    "instruction": "创建 hello.py 文件",
                    "target_files": ["hello.py"],
                }
            ],
        }

        # Mock 执行阶段 - 任务成功
        mock_execute_result = MagicMock()
        mock_execute_result.success = True
        mock_execute_result.output = "文件创建成功"
        mock_execute_result.error = None

        # Mock 评审阶段 - 评审通过
        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 95,
            "summary": "任务完成良好",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            # 模拟任务完成
            async def simulate_task_completion(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "任务完成"})

            mock_workers.side_effect = simulate_task_completion

            # 执行工作流
            result = await orchestrator.run("实现一个简单功能")

            # 验证结果
            assert result["success"] is True
            assert result["iterations_completed"] == 1
            assert result["total_tasks_created"] == 1
            assert result["total_tasks_completed"] == 1
            assert result["total_tasks_failed"] == 0

            # 验证各阶段被调用
            mock_planner.assert_called_once()
            mock_workers.assert_called_once()
            mock_reviewer.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_tasks_single_iteration(self, mock_orchestrator: Orchestrator) -> None:
        """多任务单迭代"""
        orchestrator = mock_orchestrator

        # Mock 规划阶段 - 返回多个任务
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "任务1",
                    "description": "实现功能1",
                    "instruction": "创建 module1.py",
                    "target_files": ["module1.py"],
                },
                {
                    "type": "implement",
                    "title": "任务2",
                    "description": "实现功能2",
                    "instruction": "创建 module2.py",
                    "target_files": ["module2.py"],
                },
                {
                    "type": "test",
                    "title": "任务3",
                    "description": "编写测试",
                    "instruction": "创建 test_module.py",
                    "target_files": ["test_module.py"],
                },
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 90,
            "summary": "所有任务完成",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            # 模拟所有任务完成
            async def simulate_all_tasks_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": f"{task.title} 完成"})

            mock_workers.side_effect = simulate_all_tasks_complete

            result = await orchestrator.run("实现多个功能模块")

            assert result["success"] is True
            assert result["total_tasks_created"] == 3
            assert result["total_tasks_completed"] == 3
            assert result["total_tasks_failed"] == 0

    @pytest.mark.asyncio
    async def test_task_failure_handling(self, mock_orchestrator: Orchestrator) -> None:
        """任务失败时的处理"""
        orchestrator = mock_orchestrator

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "成功任务",
                    "description": "会成功的任务",
                    "instruction": "创建 success.py",
                    "target_files": ["success.py"],
                },
                {
                    "type": "implement",
                    "title": "失败任务",
                    "description": "会失败的任务",
                    "instruction": "创建 fail.py",
                    "target_files": ["fail.py"],
                },
            ],
        }

        # 评审决定继续（因为有任务失败）
        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,  # 单迭代模式下直接完成
            "score": 50,
            "summary": "部分任务失败",
            "issues": ["fail.py 创建失败"],
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            # 模拟一个任务成功，一个任务失败
            async def simulate_partial_failure(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for i, task in enumerate(tasks):
                    if i == 0:
                        task.complete({"output": "成功"})
                    else:
                        task.fail("模拟执行失败")

            mock_workers.side_effect = simulate_partial_failure

            result = await orchestrator.run("实现功能")

            # 验证失败任务被正确统计
            assert result["total_tasks_created"] == 2
            assert result["total_tasks_completed"] == 1
            assert result["total_tasks_failed"] == 1


class TestMultiIterationWorkflow:
    """多迭代工作流测试"""

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
    async def test_two_iteration_completion(self, multi_iteration_orchestrator: Orchestrator) -> None:
        """两轮迭代完成目标"""
        orchestrator = multi_iteration_orchestrator

        iteration_count = 0

        # 第一轮规划：部分任务
        # 第二轮规划：补充任务
        def get_plan_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count == 1:
                return {
                    "success": True,
                    "tasks": [
                        {
                            "type": "implement",
                            "title": "基础实现",
                            "description": "实现基础功能",
                            "instruction": "创建基础模块",
                            "target_files": ["base.py"],
                        }
                    ],
                }
            else:
                return {
                    "success": True,
                    "tasks": [
                        {
                            "type": "test",
                            "title": "补充测试",
                            "description": "补充测试用例",
                            "instruction": "创建测试",
                            "target_files": ["test_base.py"],
                        }
                    ],
                }

        # 第一轮评审：需要继续
        # 第二轮评审：完成
        review_call_count = 0

        def get_review_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal review_call_count
            review_call_count += 1
            if review_call_count == 1:
                return {
                    "success": True,
                    "decision": ReviewDecision.CONTINUE,
                    "score": 70,
                    "summary": "需要补充测试",
                    "next_iteration_focus": "添加测试用例",
                }
            else:
                return {
                    "success": True,
                    "decision": ReviewDecision.COMPLETE,
                    "score": 95,
                    "summary": "目标完成",
                }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.side_effect = get_plan_result
            mock_reviewer.side_effect = get_review_result

            # 模拟任务完成
            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            result = await orchestrator.run("实现功能并添加测试")

            assert result["success"] is True
            assert result["iterations_completed"] == 2
            assert result["total_tasks_created"] == 2
            assert mock_planner.call_count == 2
            assert mock_reviewer.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iteration_limit(self, multi_iteration_orchestrator: Orchestrator) -> None:
        """达到最大迭代次数"""
        orchestrator = multi_iteration_orchestrator

        # 每次规划都返回任务
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "持续任务",
                    "description": "持续执行的任务",
                    "instruction": "执行操作",
                    "target_files": ["file.py"],
                }
            ],
        }

        # 评审始终返回继续
        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.CONTINUE,
            "score": 60,
            "summary": "需要继续",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            result = await orchestrator.run("无限任务")

            # 达到最大迭代次数后停止
            assert result["iterations_completed"] == 5
            assert result["success"] is False  # 未完成目标
            assert mock_planner.call_count == 5
            assert mock_reviewer.call_count == 5

    @pytest.mark.asyncio
    async def test_infinite_iteration_mode(self) -> None:
        """max_iterations=-1 无限迭代模式"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=-1,  # 无限迭代
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        iteration_count = 0
        max_test_iterations = 3  # 测试用，3轮后完成

        def get_plan_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal iteration_count
            iteration_count += 1
            return {
                "success": True,
                "tasks": [
                    {
                        "type": "implement",
                        "title": f"迭代{iteration_count}任务",
                        "description": "执行任务",
                        "instruction": "执行",
                        "target_files": ["file.py"],
                    }
                ],
            }

        def get_review_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            if iteration_count >= max_test_iterations:
                return {
                    "success": True,
                    "decision": ReviewDecision.COMPLETE,
                    "score": 100,
                    "summary": "目标完成",
                }
            return {
                "success": True,
                "decision": ReviewDecision.CONTINUE,
                "score": 50,
                "summary": "继续迭代",
            }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.side_effect = get_plan_result
            mock_reviewer.side_effect = get_review_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            result = await orchestrator.run("无限迭代测试")

            assert result["success"] is True
            assert result["iterations_completed"] == 3
            # 验证无限迭代模式下能正常完成
            assert mock_planner.call_count == 3

    @pytest.mark.asyncio
    async def test_review_adjust_decision(self, multi_iteration_orchestrator: Orchestrator) -> None:
        """评审返回 ADJUST 的迭代"""
        orchestrator = multi_iteration_orchestrator

        iteration_count = 0

        def get_plan_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal iteration_count
            iteration_count += 1
            return {
                "success": True,
                "tasks": [
                    {
                        "type": "implement",
                        "title": f"调整后任务{iteration_count}",
                        "description": "调整后的实现",
                        "instruction": "执行调整后的方案",
                        "target_files": ["adjusted.py"],
                    }
                ],
            }

        review_count = 0

        def get_review_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal review_count
            review_count += 1
            if review_count == 1:
                # 第一轮：需要调整方向
                return {
                    "success": True,
                    "decision": ReviewDecision.ADJUST,
                    "score": 40,
                    "summary": "方向错误，需要调整",
                    "suggestions": ["改用其他实现方案"],
                    "next_iteration_focus": "采用新方案",
                }
            else:
                # 第二轮：完成
                return {
                    "success": True,
                    "decision": ReviewDecision.COMPLETE,
                    "score": 90,
                    "summary": "调整后完成",
                }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.side_effect = get_plan_result
            mock_reviewer.side_effect = get_review_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            result = await orchestrator.run("需要调整的任务")

            assert result["success"] is True
            assert result["iterations_completed"] == 2
            # ADJUST 会触发新一轮迭代
            assert mock_planner.call_count == 2

    @pytest.mark.asyncio
    async def test_review_abort_decision(self, multi_iteration_orchestrator: Orchestrator) -> None:
        """评审返回 ABORT 终止"""
        orchestrator = multi_iteration_orchestrator

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "不可行任务",
                    "description": "无法完成的任务",
                    "instruction": "尝试执行",
                    "target_files": ["impossible.py"],
                }
            ],
        }

        # 评审决定终止
        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.ABORT,
            "score": 10,
            "summary": "任务无法完成，建议终止",
            "issues": ["技术限制", "资源不足"],
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            async def simulate_fail(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.fail("无法完成")

            mock_workers.side_effect = simulate_fail

            result = await orchestrator.run("不可行的任务")

            # ABORT 后立即停止
            assert result["success"] is False
            assert result["iterations_completed"] == 1
            assert mock_planner.call_count == 1
            assert mock_reviewer.call_count == 1


class TestWorkflowStateTransition:
    """工作流状态转换测试"""

    @pytest.fixture
    def state_tracking_orchestrator(self) -> Orchestrator:
        """创建用于状态跟踪的 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=3,
            worker_pool_size=2,
            enable_auto_commit=False,
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_iteration_status_transitions(self, state_tracking_orchestrator: Orchestrator) -> None:
        """验证迭代状态变化"""
        orchestrator = state_tracking_orchestrator
        status_history: list[IterationStatus] = []

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "状态测试任务",
                    "description": "用于测试状态转换",
                    "instruction": "执行",
                    "target_files": ["state.py"],
                }
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 100,
            "summary": "完成",
        }

        # 原始的 _planning_phase 方法
        original_planning = orchestrator._planning_phase

        async def track_planning_phase(goal: str, iteration_id: int) -> None:
            # 记录规划阶段的状态
            iteration = orchestrator.state.get_current_iteration()
            if iteration:
                status_history.append(iteration.status)
            await original_planning(goal, iteration_id)

        # 原始的 _execution_phase 方法
        original_execution = orchestrator._execution_phase

        async def track_execution_phase(iteration_id: int) -> None:
            # 记录执行阶段前的状态
            iteration = orchestrator.state.get_current_iteration()
            if iteration:
                status_history.append(iteration.status)
            await original_execution(iteration_id)

        # 原始的 _review_phase 方法
        original_review = orchestrator._review_phase

        async def track_review_phase(goal: str, iteration_id: int) -> ReviewDecision:
            # 记录评审阶段前的状态
            iteration = orchestrator.state.get_current_iteration()
            if iteration:
                status_history.append(iteration.status)
            return await original_review(goal, iteration_id)

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
            patch.object(
                orchestrator,
                "_planning_phase",
                side_effect=track_planning_phase,
            ),
            patch.object(
                orchestrator,
                "_execution_phase",
                side_effect=track_execution_phase,
            ),
            patch.object(
                orchestrator,
                "_review_phase",
                side_effect=track_review_phase,
            ),
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            await orchestrator.run("状态测试")

            # 验证状态转换顺序
            # 状态应该是: PLANNING -> EXECUTING -> REVIEWING
            assert IterationStatus.PLANNING in status_history

    @pytest.mark.asyncio
    async def test_task_status_transitions(self, state_tracking_orchestrator: Orchestrator) -> None:
        """验证任务状态变化"""
        orchestrator = state_tracking_orchestrator

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "任务状态测试",
                    "description": "测试任务状态转换",
                    "instruction": "执行任务",
                    "target_files": ["task_state.py"],
                }
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 100,
            "summary": "完成",
        }

        task_status_history: list[TaskStatus] = []

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            async def track_and_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    # 记录初始状态
                    task_status_history.append(task.status)

                    # 模拟开始执行
                    task.start()
                    task_status_history.append(task.status)

                    # 模拟完成
                    task.complete({"output": "完成"})
                    task_status_history.append(task.status)

            mock_workers.side_effect = track_and_complete

            await orchestrator.run("任务状态测试")

            # 验证任务状态转换：PENDING/QUEUED -> IN_PROGRESS -> COMPLETED
            assert TaskStatus.IN_PROGRESS in task_status_history
            assert TaskStatus.COMPLETED in task_status_history
            # 确保最终状态是 COMPLETED
            assert task_status_history[-1] == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_agent_status_during_execution(self, state_tracking_orchestrator: Orchestrator) -> None:
        """Agent 状态跟踪"""
        orchestrator = state_tracking_orchestrator

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "Agent状态测试",
                    "description": "测试Agent状态",
                    "instruction": "执行",
                    "target_files": ["agent_state.py"],
                }
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 100,
            "summary": "完成",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            # 验证初始状态
            planner_id = orchestrator.planner.id
            reviewer_id = orchestrator.reviewer.id

            assert planner_id in orchestrator.state.agents
            assert reviewer_id in orchestrator.state.agents

            # 验证初始 Agent 状态为 IDLE
            assert orchestrator.state.agents[planner_id].status == AgentStatus.IDLE
            assert orchestrator.state.agents[reviewer_id].status == AgentStatus.IDLE

            # 验证 Worker 数量和状态
            worker_count = 0
            for agent_state in orchestrator.state.agents.values():
                if agent_state.role == AgentRole.WORKER:
                    worker_count += 1
                    assert agent_state.status == AgentStatus.IDLE

            assert worker_count == 2  # worker_pool_size=2

            await orchestrator.run("Agent状态测试")

            # 执行后验证系统状态
            assert orchestrator.state.is_completed is True
            assert orchestrator.state.is_running is False


class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_empty_task_plan(self) -> None:
        """规划阶段返回空任务列表"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=2,
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        # 第一轮返回空任务，第二轮返回任务
        plan_call_count = 0

        def get_plan_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal plan_call_count
            plan_call_count += 1
            if plan_call_count == 1:
                return {"success": True, "tasks": []}  # 空任务列表
            return {
                "success": True,
                "tasks": [
                    {
                        "type": "implement",
                        "title": "实际任务",
                        "description": "有任务的迭代",
                        "instruction": "执行",
                        "target_files": ["file.py"],
                    }
                ],
            }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 100,
            "summary": "完成",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.side_effect = get_plan_result
            mock_reviewer.return_value = mock_review_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            result = await orchestrator.run("空任务测试")

            # 第一轮跳过执行阶段（无任务）
            # 第二轮正常执行
            assert result["iterations_completed"] == 2
            assert result["total_tasks_created"] == 1

    @pytest.mark.asyncio
    async def test_planning_failure(self) -> None:
        """规划阶段失败"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        # 规划失败
        mock_plan_result = {
            "success": False,
            "error": "规划失败：无法分析需求",
            "tasks": [],
        }

        with patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner:
            mock_planner.return_value = mock_plan_result

            result = await orchestrator.run("会失败的规划")

            # 规划失败后应该跳过执行阶段
            assert result["iterations_completed"] == 1
            assert result["total_tasks_created"] == 0

    @pytest.mark.asyncio
    async def test_system_state_persistence(self) -> None:
        """系统状态在迭代间保持"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=3,
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        iteration_count = 0

        def get_plan_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal iteration_count
            iteration_count += 1
            return {
                "success": True,
                "tasks": [
                    {
                        "type": "implement",
                        "title": f"迭代{iteration_count}",
                        "description": "任务",
                        "instruction": "执行",
                        "target_files": ["file.py"],
                    }
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
                    "score": 50,
                    "summary": "继续",
                }
            return {
                "success": True,
                "decision": ReviewDecision.COMPLETE,
                "score": 100,
                "summary": "完成",
            }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.side_effect = get_plan_result
            mock_reviewer.side_effect = get_review_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            result = await orchestrator.run("状态持久化测试")

            # 验证所有迭代都被记录
            assert len(result["iterations"]) == 2
            assert result["iterations"][0]["id"] == 1
            assert result["iterations"][1]["id"] == 2

            # 验证统计数据累积
            assert result["total_tasks_created"] == 2
            assert result["total_tasks_completed"] == 2
