"""端到端错误处理测试

验证系统在各种错误场景下的行为，包括：
- 规划阶段错误
- 执行阶段错误
- 评审阶段错误
- 提交阶段错误
- 边界条件

使用 Mock 替代真实 Cursor CLI 调用
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agents.reviewer import ReviewDecision
from coordinator.orchestrator import Orchestrator, OrchestratorConfig


class TestPlanningErrors:
    """规划阶段错误测试"""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """创建测试用 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=3,
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_empty_plan_result(self, orchestrator: Orchestrator) -> None:
        """规划返回空任务列表"""
        # 规划返回空任务列表
        mock_plan_result = {
            "success": True,
            "tasks": [],
        }

        # 评审应该也被调用（即使没有任务）
        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 100,
            "summary": "无任务需要执行",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            result = await orchestrator.run("空任务测试")

            # 验证结果
            assert result["total_tasks_created"] == 0
            # 规划成功但没有任务
            mock_planner.assert_called()

    @pytest.mark.asyncio
    async def test_invalid_plan_json(self, orchestrator: Orchestrator) -> None:
        """规划返回无效 JSON 结构"""
        # 返回不符合预期结构的结果
        mock_plan_result = {
            "success": True,
            "tasks": "这不是一个列表",  # 错误: tasks 应该是列表
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 0,
            "summary": "规划格式错误",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            # 应该能够处理无效格式而不崩溃
            result = await orchestrator.run("无效JSON测试")

            # 验证系统能够容错处理
            assert "error" in result or result["total_tasks_created"] == 0

    @pytest.mark.asyncio
    async def test_planner_timeout(self, orchestrator: Orchestrator) -> None:
        """规划器超时"""

        async def slow_planner(*args: Any, **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(10)  # 模拟超时
            return {"success": True, "tasks": []}

        with patch.object(orchestrator.planner, "execute", side_effect=TimeoutError()):
            # 超时应该导致规划失败
            result = await orchestrator.run("超时测试")

            # 验证返回错误
            assert result.get("success") is False or result.get("error") is not None

    @pytest.mark.asyncio
    async def test_planner_exception(self, orchestrator: Orchestrator) -> None:
        """规划器异常"""
        with patch.object(orchestrator.planner, "execute", side_effect=RuntimeError("规划器内部错误")):
            result = await orchestrator.run("异常测试")

            # 验证错误被正确捕获和记录
            assert result.get("success") is False
            assert "error" in result


class TestExecutionErrors:
    """执行阶段错误测试"""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """创建测试用 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=2,
            worker_pool_size=2,
            enable_auto_commit=False,
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_all_tasks_fail(self, orchestrator: Orchestrator) -> None:
        """所有任务失败"""
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "失败任务1",
                    "description": "会失败的任务",
                    "instruction": "执行失败操作",
                    "target_files": ["fail1.py"],
                },
                {
                    "type": "implement",
                    "title": "失败任务2",
                    "description": "也会失败的任务",
                    "instruction": "执行另一个失败操作",
                    "target_files": ["fail2.py"],
                },
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.ABORT,
            "score": 0,
            "summary": "所有任务都失败了",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            # 模拟所有任务失败
            async def simulate_all_fail(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.fail("模拟执行失败")

            mock_workers.side_effect = simulate_all_fail

            result = await orchestrator.run("全部失败测试")

            # 验证所有任务都被标记为失败
            assert result["total_tasks_failed"] == 2
            assert result["total_tasks_completed"] == 0

    @pytest.mark.asyncio
    async def test_partial_task_failure(self, orchestrator: Orchestrator) -> None:
        """部分任务失败"""
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "成功任务",
                    "description": "会成功",
                    "instruction": "创建 success.py",
                    "target_files": ["success.py"],
                },
                {
                    "type": "implement",
                    "title": "失败任务",
                    "description": "会失败",
                    "instruction": "创建 fail.py",
                    "target_files": ["fail.py"],
                },
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 50,
            "summary": "部分任务完成",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            # 模拟部分成功部分失败
            async def simulate_partial(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for i, task in enumerate(tasks):
                    if i == 0:
                        task.complete({"output": "成功"})
                    else:
                        task.fail("失败原因")

            mock_workers.side_effect = simulate_partial

            result = await orchestrator.run("部分失败测试")

            assert result["total_tasks_completed"] == 1
            assert result["total_tasks_failed"] == 1

    @pytest.mark.asyncio
    async def test_task_timeout(self, orchestrator: Orchestrator) -> None:
        """任务执行超时"""
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "超时任务",
                    "description": "执行超时",
                    "instruction": "执行耗时操作",
                    "target_files": ["timeout.py"],
                },
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.ABORT,
            "score": 0,
            "summary": "任务超时",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            # 模拟任务超时
            async def simulate_timeout(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.fail("执行超时")

            mock_workers.side_effect = simulate_timeout

            result = await orchestrator.run("超时任务测试")

            assert result["total_tasks_failed"] >= 1

    @pytest.mark.asyncio
    async def test_worker_crash_recovery(self, orchestrator: Orchestrator) -> None:
        """Worker 崩溃恢复"""
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "崩溃后恢复任务",
                    "description": "Worker 崩溃后需恢复",
                    "instruction": "执行任务",
                    "target_files": ["recover.py"],
                },
            ],
        }

        crash_count = 0

        async def simulate_crash_then_success(queue: Any, iteration_id: int) -> None:
            nonlocal crash_count
            tasks = queue.get_tasks_by_iteration(iteration_id)
            for task in tasks:
                if crash_count == 0:
                    crash_count += 1
                    # 第一次模拟崩溃
                    raise RuntimeError("Worker 崩溃")
                else:
                    # 后续恢复成功
                    task.complete({"output": "恢复后完成"})

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 80,
            "summary": "崩溃后恢复成功",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            # 第一次崩溃，第二次成功
            mock_workers.side_effect = simulate_crash_then_success

            # 系统应该能处理 Worker 崩溃
            result = await orchestrator.run("崩溃恢复测试")

            # 验证系统正确记录了状态
            assert "iterations_completed" in result


class TestReviewErrors:
    """评审阶段错误测试"""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """创建测试用 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=2,
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_review_timeout(self, orchestrator: Orchestrator) -> None:
        """评审超时"""
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "待评审任务",
                    "description": "任务描述",
                    "instruction": "执行任务",
                    "target_files": ["file.py"],
                },
            ],
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", side_effect=TimeoutError()),
        ):
            mock_planner.return_value = mock_plan_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            result = await orchestrator.run("评审超时测试")

            # 评审超时应该导致整体失败
            assert result.get("success") is False

    @pytest.mark.asyncio
    async def test_invalid_review_decision(self, orchestrator: Orchestrator) -> None:
        """无效评审决策"""
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "任务",
                    "description": "描述",
                    "instruction": "执行",
                    "target_files": ["file.py"],
                },
            ],
        }

        # 返回无效的决策
        mock_review_result = {
            "success": True,
            "decision": "INVALID_DECISION",  # 无效决策
            "score": 50,
            "summary": "评审完成",
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

            # 系统应该能处理无效决策
            result = await orchestrator.run("无效决策测试")

            # 验证系统能容错处理
            assert "iterations_completed" in result

    @pytest.mark.asyncio
    async def test_review_exception(self, orchestrator: Orchestrator) -> None:
        """评审器异常"""
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "任务",
                    "description": "描述",
                    "instruction": "执行",
                    "target_files": ["file.py"],
                },
            ],
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", side_effect=RuntimeError("评审器内部错误")),
        ):
            mock_planner.return_value = mock_plan_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "完成"})

            mock_workers.side_effect = simulate_complete

            result = await orchestrator.run("评审异常测试")

            # 异常应该被正确捕获
            assert result.get("success") is False
            assert "error" in result


class TestCommitErrors:
    """提交阶段错误测试"""

    @pytest.fixture
    def orchestrator_with_commit(self) -> Orchestrator:
        """创建启用自动提交的 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_pool_size=1,
            enable_auto_commit=True,
            commit_on_complete=True,
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_commit_without_changes(self, orchestrator_with_commit: Orchestrator) -> None:
        """无变更提交 - 不应中断主流程，错误应记录到 iteration"""
        orchestrator = orchestrator_with_commit

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "analyze",
                    "title": "分析任务",
                    "description": "只分析不修改",
                    "instruction": "分析代码",
                    "target_files": [],
                },
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 100,
            "summary": "分析完成",
        }

        # 模拟无变更
        mock_commit_result = {
            "success": False,
            "error": "没有需要提交的变更",
            "message": "",
            "files_changed": [],
            "pushed": False,
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
                    task.complete({"output": "分析完成"})

            mock_workers.side_effect = simulate_complete

            # Mock committer
            if orchestrator.committer:
                with patch.object(orchestrator.committer, "commit_iteration", new_callable=AsyncMock) as mock_commit:
                    mock_commit.return_value = mock_commit_result

                    result = await orchestrator.run("无变更提交测试")

                    # 验证提交被调用但返回无变更
                    mock_commit.assert_called_once()

                    # 验证主流程未中断，目标仍然完成
                    assert result["success"] is True

                    # 验证 commit_error 被正确记录到 iteration
                    assert len(result["iterations"]) == 1
                    iteration_result = result["iterations"][0]
                    assert iteration_result["commit_error"] == "没有需要提交的变更"
                    assert iteration_result["commit_hash"] == ""
                    assert iteration_result["commit_pushed"] is False

    @pytest.mark.asyncio
    async def test_commit_conflict(self, orchestrator_with_commit: Orchestrator) -> None:
        """提交冲突处理 - 不应中断主流程，错误应记录到 iteration"""
        orchestrator = orchestrator_with_commit

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "实现任务",
                    "description": "实现功能",
                    "instruction": "创建文件",
                    "target_files": ["conflict.py"],
                },
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 90,
            "summary": "任务完成",
        }

        # 模拟提交冲突
        mock_commit_result = {
            "success": False,
            "error": "合并冲突: 无法自动合并",
            "message": "",
            "files_changed": ["conflict.py"],
            "pushed": False,
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

            if orchestrator.committer:
                with patch.object(orchestrator.committer, "commit_iteration", new_callable=AsyncMock) as mock_commit:
                    mock_commit.return_value = mock_commit_result

                    result = await orchestrator.run("冲突提交测试")

                    # 验证提交被调用
                    mock_commit.assert_called_once()

                    # 验证主流程未中断，目标仍然完成
                    assert result["success"] is True

                    # 验证 commit_error 被正确记录到 iteration
                    assert len(result["iterations"]) == 1
                    iteration_result = result["iterations"][0]
                    assert iteration_result["commit_error"] == "合并冲突: 无法自动合并"
                    assert iteration_result["commit_pushed"] is False

    @pytest.mark.asyncio
    async def test_push_failure(self, orchestrator_with_commit: Orchestrator) -> None:
        """推送失败处理 - 提交成功但 pushed=False，push_error 应记录到 iteration"""
        orchestrator = orchestrator_with_commit
        orchestrator.config.auto_push = True

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "实现任务",
                    "description": "实现功能",
                    "instruction": "创建文件",
                    "target_files": ["push_fail.py"],
                },
            ],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 90,
            "summary": "任务完成",
        }

        # 提交成功但推送失败
        mock_commit_result = {
            "success": True,
            "commit_hash": "abc123def456",
            "message": "feat: 实现功能",
            "files_changed": ["push_fail.py"],
            "pushed": False,
            "push_error": "远程仓库拒绝: 权限不足",
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

            if orchestrator.committer:
                with patch.object(orchestrator.committer, "commit_iteration", new_callable=AsyncMock) as mock_commit:
                    mock_commit.return_value = mock_commit_result

                    result = await orchestrator.run("推送失败测试")

                    # 验证提交被调用
                    mock_commit.assert_called_once()

                    # 验证主流程未中断，目标仍然完成
                    assert result["success"] is True

                    # 验证 iteration 中记录了正确的信息
                    assert len(result["iterations"]) == 1
                    iteration_result = result["iterations"][0]

                    # 提交成功时 commit_error 应为 None
                    assert iteration_result["commit_error"] is None

                    # commit_hash 应被记录
                    assert iteration_result["commit_hash"] == "abc123def456"

                    # pushed 应为 False
                    assert iteration_result["commit_pushed"] is False

                    # push_error 应被记录
                    assert iteration_result["push_error"] == "远程仓库拒绝: 权限不足"

                    # 最终结果的 pushed 字段应为 False（无成功推送的提交）
                    assert result["pushed"] is False


class TestBoundaryConditions:
    """边界条件测试"""

    @pytest.mark.asyncio
    async def test_zero_max_iterations(self) -> None:
        """max_iterations=0"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=0,  # 0 次迭代
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "任务",
                    "description": "描述",
                    "instruction": "执行",
                    "target_files": ["file.py"],
                },
            ],
        }

        with patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner:
            mock_planner.return_value = mock_plan_result

            result = await orchestrator.run("零迭代测试")

            # max_iterations=0 应该不执行任何迭代
            assert result["iterations_completed"] == 0
            # 规划不应该被调用
            mock_planner.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_goal(self) -> None:
        """空目标字符串"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        mock_plan_result = {
            "success": True,
            "tasks": [],
        }

        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 100,
            "summary": "空目标",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            # 测试空字符串目标
            result = await orchestrator.run("")

            # 系统应该能处理空目标
            assert "iterations_completed" in result

    @pytest.mark.asyncio
    async def test_very_long_goal(self) -> None:
        """超长目标字符串"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        # 生成超长目标字符串（10KB）
        very_long_goal = "这是一个超长的目标描述。" * 1000

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "处理超长目标",
                    "description": "任务描述",
                    "instruction": "执行任务",
                    "target_files": ["long.py"],
                },
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

            result = await orchestrator.run(very_long_goal)

            # 系统应该能处理超长目标
            assert result["goal"] == very_long_goal
            assert "iterations_completed" in result

    @pytest.mark.asyncio
    async def test_special_characters_in_goal(self) -> None:
        """目标含特殊字符"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_pool_size=1,
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        # 包含各种特殊字符的目标
        special_goal = "实现功能：包含'引号'、\"双引号\"、`反引号`、\n换行符、\t制表符、🎉表情符号、<html>标签</html>、$变量、%格式化%"

        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "处理特殊字符",
                    "description": "任务描述",
                    "instruction": "执行任务",
                    "target_files": ["special.py"],
                },
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

            result = await orchestrator.run(special_goal)

            # 系统应该能处理特殊字符
            assert result["goal"] == special_goal
            assert "iterations_completed" in result

    @pytest.mark.asyncio
    async def test_concurrent_orchestrator_runs(self) -> None:
        """并发运行多个 Orchestrator"""

        async def create_and_run(index: int) -> dict[str, Any]:
            config = OrchestratorConfig(
                working_directory=".",
                max_iterations=1,
                worker_pool_size=1,
                enable_auto_commit=False,
            )
            orchestrator = Orchestrator(config)

            mock_plan_result = {
                "success": True,
                "tasks": [
                    {
                        "type": "implement",
                        "title": f"并发任务{index}",
                        "description": f"第{index}个并发任务",
                        "instruction": f"执行任务{index}",
                        "target_files": [f"concurrent_{index}.py"],
                    },
                ],
            }

            mock_review_result = {
                "success": True,
                "decision": ReviewDecision.COMPLETE,
                "score": 100,
                "summary": f"并发{index}完成",
            }

            with (
                patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
                patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
                patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
            ):
                mock_planner.return_value = mock_plan_result
                mock_reviewer.return_value = mock_review_result

                async def simulate_complete(queue: Any, iteration_id: int) -> None:
                    # 添加随机延迟模拟真实执行
                    await asyncio.sleep(0.01 * index)
                    tasks = queue.get_tasks_by_iteration(iteration_id)
                    for task in tasks:
                        task.complete({"output": f"完成{index}"})

                mock_workers.side_effect = simulate_complete

                return await orchestrator.run(f"并发测试{index}")

        # 并发运行 5 个 Orchestrator
        tasks = [create_and_run(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 验证所有并发运行都成功完成
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                pytest.fail(f"并发运行 {i} 抛出异常: {result}")
            else:
                assert result["success"] is True, f"并发运行 {i} 失败"
                assert result["iterations_completed"] == 1
