"""测试 MultiProcessOrchestrator 提交阶段

测试 MultiProcessOrchestrator 的提交功能，包括：
- patch `_spawn_agents()` 与 `process_manager.wait_all_ready()` 使其不启动真实子进程
- patch `_planning_phase/_execution_phase/_review_phase` 直接向 TaskQueue 注入任务
- patch `CommitterAgent.commit_iteration` 返回各种结果
- 测试策略组合：`commit_per_iteration=True` 时在非 COMPLETE 决策下的行为

使用 Mock 替代真实子进程和 Cursor CLI 调用
"""
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.committer import CommitterAgent
from agents.reviewer_process import ReviewDecision
from coordinator.orchestrator_mp import (
    MultiProcessOrchestrator,
    MultiProcessOrchestratorConfig,
)
from core.state import IterationStatus
from tasks.task import Task, TaskStatus


class TestMultiProcessOrchestratorCommitPhase:
    """测试 MultiProcessOrchestrator 提交阶段"""

    @pytest.fixture
    def mp_config(self) -> MultiProcessOrchestratorConfig:
        """创建测试用多进程编排器配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            enable_auto_commit=True,
            commit_on_complete=True,
            commit_per_iteration=False,
            auto_push=False,
        )

    @pytest.fixture
    def mp_orchestrator(
        self, mp_config: MultiProcessOrchestratorConfig
    ) -> MultiProcessOrchestrator:
        """创建 MultiProcessOrchestrator 实例（不启动真实进程）"""
        orchestrator = MultiProcessOrchestrator(mp_config)
        return orchestrator

    def _mock_spawn_and_ready(
        self, orchestrator: MultiProcessOrchestrator
    ) -> tuple[MagicMock, MagicMock]:
        """Mock _spawn_agents 和 wait_all_ready，避免启动真实子进程

        Returns:
            (mock_spawn, mock_wait_ready) 元组
        """
        mock_spawn = patch.object(
            orchestrator, "_spawn_agents", return_value=None
        )
        mock_wait_ready = patch.object(
            orchestrator.process_manager, "wait_all_ready", return_value=True
        )
        return mock_spawn, mock_wait_ready

    async def _inject_tasks_to_queue(
        self,
        orchestrator: MultiProcessOrchestrator,
        iteration_id: int,
        tasks_data: list[dict[str, Any]],
        mark_completed: bool = True,
    ) -> list[Task]:
        """向 TaskQueue 注入任务

        Args:
            orchestrator: 编排器实例
            iteration_id: 迭代 ID
            tasks_data: 任务数据列表
            mark_completed: 是否标记为完成

        Returns:
            注入的任务列表
        """
        tasks = []
        for i, data in enumerate(tasks_data):
            task = Task(
                id=f"task-{iteration_id}-{i}",
                iteration_id=iteration_id,
                type=data.get("type", "implement"),
                title=data.get("title", f"测试任务 {i}"),
                description=data.get("description", ""),
                instruction=data.get("instruction", "执行任务"),
                target_files=data.get("target_files", []),
            )
            await orchestrator.task_queue.enqueue(task)
            orchestrator.state.total_tasks_created += 1

            if mark_completed:
                task.status = TaskStatus.COMPLETED
                task.result = data.get("result", {"success": True})
                orchestrator.task_queue.update_task(task)
                orchestrator.state.total_tasks_completed += 1

            tasks.append(task)

        return tasks

    # =========================================================================
    # 测试 _should_commit 决策逻辑
    # =========================================================================

    def test_should_commit_disabled(
        self, mp_config: MultiProcessOrchestratorConfig
    ) -> None:
        """测试禁用自动提交时 _should_commit 返回 False"""
        mp_config.enable_auto_commit = False
        orchestrator = MultiProcessOrchestrator(mp_config)

        assert orchestrator._should_commit(ReviewDecision.COMPLETE) is False
        assert orchestrator._should_commit(ReviewDecision.CONTINUE) is False
        assert orchestrator._should_commit(ReviewDecision.ABORT) is False

    def test_should_commit_on_complete_only(
        self, mp_orchestrator: MultiProcessOrchestrator
    ) -> None:
        """测试仅在 COMPLETE 时提交（commit_on_complete=True, commit_per_iteration=False）"""
        # 默认配置：commit_on_complete=True, commit_per_iteration=False
        assert mp_orchestrator._should_commit(ReviewDecision.COMPLETE) is True
        assert mp_orchestrator._should_commit(ReviewDecision.CONTINUE) is False
        assert mp_orchestrator._should_commit(ReviewDecision.ADJUST) is False
        assert mp_orchestrator._should_commit(ReviewDecision.ABORT) is False

    def test_should_commit_per_iteration(
        self, mp_config: MultiProcessOrchestratorConfig
    ) -> None:
        """测试每次迭代都提交（commit_per_iteration=True）"""
        mp_config.commit_per_iteration = True
        orchestrator = MultiProcessOrchestrator(mp_config)

        # commit_per_iteration=True 时，所有决策都应该触发提交
        assert orchestrator._should_commit(ReviewDecision.COMPLETE) is True
        assert orchestrator._should_commit(ReviewDecision.CONTINUE) is True
        assert orchestrator._should_commit(ReviewDecision.ADJUST) is True
        assert orchestrator._should_commit(ReviewDecision.ABORT) is True

    def test_should_commit_no_committer(
        self, mp_config: MultiProcessOrchestratorConfig
    ) -> None:
        """测试没有 Committer 时 _should_commit 返回 False"""
        mp_config.enable_auto_commit = False
        orchestrator = MultiProcessOrchestrator(mp_config)

        assert orchestrator.committer is None
        assert orchestrator._should_commit(ReviewDecision.COMPLETE) is False

    # =========================================================================
    # 测试 _commit_phase 方法
    # =========================================================================

    @pytest.mark.asyncio
    async def test_commit_phase_success(
        self, mp_orchestrator: MultiProcessOrchestrator
    ) -> None:
        """测试提交阶段成功场景"""
        # 开始新迭代
        iteration = mp_orchestrator.state.start_new_iteration()
        iteration_id = iteration.iteration_id

        # 注入完成的任务
        await self._inject_tasks_to_queue(
            mp_orchestrator,
            iteration_id,
            [
                {"title": "实现功能A", "result": {"files": ["a.py"]}},
                {"title": "实现功能B", "result": {"files": ["b.py"]}},
            ],
        )

        # Mock commit_iteration 返回成功
        mock_commit_result = {
            "success": True,
            "commit_hash": "abc123def456",
            "message": "feat(iter-1): 完成 2 个任务",
            "files_changed": ["a.py", "b.py"],
            "pushed": False,
            "iteration_id": iteration_id,
            "tasks_count": 2,
        }

        with patch.object(
            mp_orchestrator.committer,
            "commit_iteration",
            new_callable=AsyncMock,
            return_value=mock_commit_result,
        ) as mock_commit:
            result = await mp_orchestrator._commit_phase(
                iteration_id, ReviewDecision.COMPLETE
            )

            # 验证 commit_iteration 被调用
            mock_commit.assert_called_once()

            # 验证调用参数
            call_kwargs = mock_commit.call_args.kwargs
            assert call_kwargs["iteration_id"] == iteration_id
            assert call_kwargs["review_decision"] == "complete"
            assert call_kwargs["auto_push"] is False
            assert len(call_kwargs["tasks_completed"]) == 2

            # 验证返回结果
            assert result["success"] is True
            assert result["commit_hash"] == "abc123def456"

            # 验证 IterationState 填充
            assert iteration.commit_hash == "abc123def456"
            assert "feat(iter-1)" in iteration.commit_message
            assert iteration.pushed is False
            assert "a.py" in iteration.commit_files
            assert "b.py" in iteration.commit_files

    @pytest.mark.asyncio
    async def test_commit_phase_no_changes(
        self, mp_orchestrator: MultiProcessOrchestrator
    ) -> None:
        """测试无变更时的提交场景"""
        iteration = mp_orchestrator.state.start_new_iteration()
        iteration_id = iteration.iteration_id

        # 注入完成的任务
        await self._inject_tasks_to_queue(
            mp_orchestrator,
            iteration_id,
            [{"title": "分析任务", "result": {"analysis": "无需修改"}}],
        )

        # Mock commit_iteration 返回无变更
        mock_commit_result = {
            "success": False,
            "error": "没有需要提交的变更",
            "iteration_id": iteration_id,
        }

        with patch.object(
            mp_orchestrator.committer,
            "commit_iteration",
            new_callable=AsyncMock,
            return_value=mock_commit_result,
        ) as mock_commit:
            result = await mp_orchestrator._commit_phase(
                iteration_id, ReviewDecision.COMPLETE
            )

            mock_commit.assert_called_once()
            assert result["success"] is False
            assert "没有需要提交的变更" in result.get("error", "")

            # 验证 IterationState 填充（无变更时应为空）
            assert iteration.commit_hash == ""
            assert iteration.pushed is False

    @pytest.mark.asyncio
    async def test_commit_phase_commit_failed(
        self, mp_orchestrator: MultiProcessOrchestrator
    ) -> None:
        """测试提交失败场景"""
        iteration = mp_orchestrator.state.start_new_iteration()
        iteration_id = iteration.iteration_id

        await self._inject_tasks_to_queue(
            mp_orchestrator,
            iteration_id,
            [{"title": "实现功能", "result": {"success": True}}],
        )

        # Mock commit_iteration 返回提交失败
        mock_commit_result = {
            "success": False,
            "error": "git commit 失败: pre-commit hook 拒绝",
            "message": "feat: 尝试提交",
            "files_changed": ["file.py"],
            "pushed": False,
            "iteration_id": iteration_id,
            "tasks_count": 1,
        }

        with patch.object(
            mp_orchestrator.committer,
            "commit_iteration",
            new_callable=AsyncMock,
            return_value=mock_commit_result,
        ) as mock_commit:
            result = await mp_orchestrator._commit_phase(
                iteration_id, ReviewDecision.COMPLETE
            )

            mock_commit.assert_called_once()
            assert result["success"] is False
            assert "pre-commit hook" in result.get("error", "")

            # 验证 IterationState 填充
            assert iteration.commit_hash == ""
            assert iteration.commit_message == "feat: 尝试提交"
            assert iteration.pushed is False

    @pytest.mark.asyncio
    async def test_commit_phase_push_failed(
        self, mp_config: MultiProcessOrchestratorConfig
    ) -> None:
        """测试推送失败场景（提交成功但推送失败）"""
        mp_config.auto_push = True
        orchestrator = MultiProcessOrchestrator(mp_config)

        iteration = orchestrator.state.start_new_iteration()
        iteration_id = iteration.iteration_id

        await self._inject_tasks_to_queue(
            orchestrator,
            iteration_id,
            [{"title": "实现功能", "result": {"success": True}}],
        )

        # Mock commit_iteration 返回提交成功但推送失败
        mock_commit_result = {
            "success": True,
            "commit_hash": "def789ghi",
            "message": "feat: 提交成功",
            "files_changed": ["feature.py"],
            "pushed": False,
            "push_error": "远程仓库拒绝: 权限不足",
            "iteration_id": iteration_id,
            "tasks_count": 1,
        }

        with patch.object(
            orchestrator.committer,
            "commit_iteration",
            new_callable=AsyncMock,
            return_value=mock_commit_result,
        ) as mock_commit:
            result = await orchestrator._commit_phase(
                iteration_id, ReviewDecision.COMPLETE
            )

            mock_commit.assert_called_once()

            # 验证调用参数中 auto_push=True
            call_kwargs = mock_commit.call_args.kwargs
            assert call_kwargs["auto_push"] is True

            # 提交成功
            assert result["success"] is True
            assert result["commit_hash"] == "def789ghi"
            # 推送失败
            assert result["pushed"] is False
            assert "push_error" in result

            # 验证 IterationState 填充
            assert iteration.commit_hash == "def789ghi"
            assert iteration.pushed is False  # 推送失败

    @pytest.mark.asyncio
    async def test_commit_phase_push_success(
        self, mp_config: MultiProcessOrchestratorConfig
    ) -> None:
        """测试推送成功场景"""
        mp_config.auto_push = True
        orchestrator = MultiProcessOrchestrator(mp_config)

        iteration = orchestrator.state.start_new_iteration()
        iteration_id = iteration.iteration_id

        await self._inject_tasks_to_queue(
            orchestrator,
            iteration_id,
            [{"title": "实现功能", "result": {"success": True}}],
        )

        # Mock commit_iteration 返回提交和推送都成功
        mock_commit_result = {
            "success": True,
            "commit_hash": "abc123pushed",
            "message": "feat: 已推送",
            "files_changed": ["pushed.py"],
            "pushed": True,
            "iteration_id": iteration_id,
            "tasks_count": 1,
        }

        with patch.object(
            orchestrator.committer,
            "commit_iteration",
            new_callable=AsyncMock,
            return_value=mock_commit_result,
        ) as mock_commit:
            result = await orchestrator._commit_phase(
                iteration_id, ReviewDecision.COMPLETE
            )

            mock_commit.assert_called_once()
            assert result["success"] is True
            assert result["pushed"] is True

            # 验证 IterationState 填充
            assert iteration.commit_hash == "abc123pushed"
            assert iteration.pushed is True

    @pytest.mark.asyncio
    async def test_commit_phase_no_committer(
        self, mp_config: MultiProcessOrchestratorConfig
    ) -> None:
        """测试没有 Committer 时的提交阶段"""
        mp_config.enable_auto_commit = False
        orchestrator = MultiProcessOrchestrator(mp_config)

        iteration = orchestrator.state.start_new_iteration()
        iteration_id = iteration.iteration_id

        result = await orchestrator._commit_phase(
            iteration_id, ReviewDecision.COMPLETE
        )

        assert result["success"] is False
        assert "Committer not initialized" in result.get("error", "")

    # =========================================================================
    # 测试完整运行流程中的提交行为
    # =========================================================================

    async def _inject_pending_tasks(
        self,
        orchestrator: MultiProcessOrchestrator,
        iteration_id: int,
        count: int = 1,
    ) -> None:
        """注入待处理任务（PENDING 状态）以通过 pending 检查

        Args:
            orchestrator: 编排器实例
            iteration_id: 迭代 ID
            count: 任务数量
        """
        for i in range(count):
            task = Task(
                id=f"task-{iteration_id}-{i}",
                iteration_id=iteration_id,
                type="implement",
                title=f"任务{i}",
                description="",
                instruction="执行任务",
                target_files=[f"file{i}.py"],
            )
            await orchestrator.task_queue.enqueue(task)
            orchestrator.state.total_tasks_created += 1

    @pytest.mark.asyncio
    async def test_full_run_with_commit_on_complete(
        self, mp_orchestrator: MultiProcessOrchestrator
    ) -> None:
        """测试完整运行流程：COMPLETE 决策时触发提交"""
        mock_spawn, mock_wait_ready = self._mock_spawn_and_ready(mp_orchestrator)

        # Mock 各阶段
        async def mock_planning(goal: str, iteration_id: int) -> None:
            # 注入 PENDING 状态的任务（通过 pending 检查）
            await self._inject_pending_tasks(mp_orchestrator, iteration_id, 2)
            iteration = mp_orchestrator.state.get_current_iteration()
            iteration.tasks_created = 2

        async def mock_execution(iteration_id: int) -> None:
            # 在执行阶段将任务标记为完成
            tasks = mp_orchestrator.task_queue.get_tasks_by_iteration(iteration_id)
            for task in tasks:
                task.status = TaskStatus.COMPLETED
                task.result = {"success": True}
                mp_orchestrator.task_queue.update_task(task)
            iteration = mp_orchestrator.state.get_current_iteration()
            iteration.tasks_completed = len(tasks)
            mp_orchestrator.state.total_tasks_completed += len(tasks)

        async def mock_review(goal: str, iteration_id: int) -> ReviewDecision:
            iteration = mp_orchestrator.state.get_current_iteration()
            iteration.review_passed = True
            return ReviewDecision.COMPLETE

        mock_commit_result = {
            "success": True,
            "commit_hash": "final123",
            "message": "feat: 完成目标",
            "files_changed": ["result.py"],
            "pushed": False,
        }

        with mock_spawn, mock_wait_ready:
            with patch.object(
                mp_orchestrator, "_planning_phase", side_effect=mock_planning
            ):
                with patch.object(
                    mp_orchestrator, "_execution_phase", side_effect=mock_execution
                ):
                    with patch.object(
                        mp_orchestrator, "_review_phase", side_effect=mock_review
                    ):
                        with patch.object(
                            mp_orchestrator.committer,
                            "commit_iteration",
                            new_callable=AsyncMock,
                            return_value=mock_commit_result,
                        ) as mock_commit:
                            with patch.object(
                                mp_orchestrator.committer,
                                "get_commit_summary",
                                return_value={
                                    "total_commits": 1,
                                    "pushed_commits": 0,
                                },
                            ):
                                result = await mp_orchestrator.run("测试目标")

                                # 验证 commit_iteration 被调用
                                mock_commit.assert_called_once()

                                # 验证最终结果包含 commits 信息
                                assert result["success"] is True
                                assert "commits" in result
                                assert result["pushed"] is False

    @pytest.mark.asyncio
    async def test_full_run_continue_decision_no_commit(
        self, mp_orchestrator: MultiProcessOrchestrator
    ) -> None:
        """测试完整运行流程：CONTINUE 决策时不触发提交（commit_on_complete=True）"""
        # 更新 config 和 state 的 max_iterations
        mp_orchestrator.config.max_iterations = 2
        mp_orchestrator.state.max_iterations = 2
        mock_spawn, mock_wait_ready = self._mock_spawn_and_ready(mp_orchestrator)

        call_count = 0

        async def mock_planning(goal: str, iteration_id: int) -> None:
            await self._inject_pending_tasks(mp_orchestrator, iteration_id, 1)
            iteration = mp_orchestrator.state.get_current_iteration()
            iteration.tasks_created = 1

        async def mock_execution(iteration_id: int) -> None:
            tasks = mp_orchestrator.task_queue.get_tasks_by_iteration(iteration_id)
            for task in tasks:
                task.status = TaskStatus.COMPLETED
                task.result = {"success": True}
                mp_orchestrator.task_queue.update_task(task)
            iteration = mp_orchestrator.state.get_current_iteration()
            iteration.tasks_completed = len(tasks)
            mp_orchestrator.state.total_tasks_completed += len(tasks)

        async def mock_review(goal: str, iteration_id: int) -> ReviewDecision:
            nonlocal call_count
            call_count += 1
            # 第一次返回 CONTINUE，第二次返回 COMPLETE
            if call_count == 1:
                return ReviewDecision.CONTINUE
            return ReviewDecision.COMPLETE

        mock_commit_result = {
            "success": True,
            "commit_hash": "only_on_complete",
            "message": "feat: 完成",
            "files_changed": ["done.py"],
            "pushed": False,
        }

        with mock_spawn, mock_wait_ready:
            with patch.object(
                mp_orchestrator, "_planning_phase", side_effect=mock_planning
            ):
                with patch.object(
                    mp_orchestrator, "_execution_phase", side_effect=mock_execution
                ):
                    with patch.object(
                        mp_orchestrator, "_review_phase", side_effect=mock_review
                    ):
                        with patch.object(
                            mp_orchestrator.committer,
                            "commit_iteration",
                            new_callable=AsyncMock,
                            return_value=mock_commit_result,
                        ) as mock_commit:
                            with patch.object(
                                mp_orchestrator.committer,
                                "get_commit_summary",
                                return_value={
                                    "total_commits": 1,
                                    "pushed_commits": 0,
                                },
                            ):
                                result = await mp_orchestrator.run("测试目标")

                                # commit_iteration 应该只被调用一次（COMPLETE 时）
                                assert mock_commit.call_count == 1

                                # 验证是在第二次迭代（COMPLETE）时调用
                                call_args = mock_commit.call_args.kwargs
                                assert call_args["review_decision"] == "complete"

    @pytest.mark.asyncio
    async def test_full_run_commit_per_iteration(
        self, mp_config: MultiProcessOrchestratorConfig
    ) -> None:
        """测试完整运行流程：commit_per_iteration=True 时每次迭代都提交"""
        mp_config.max_iterations = 2
        mp_config.commit_per_iteration = True
        mp_config.commit_on_complete = False  # 显式禁用，验证 per_iteration 优先级
        orchestrator = MultiProcessOrchestrator(mp_config)

        mock_spawn, mock_wait_ready = self._mock_spawn_and_ready(orchestrator)

        iteration_count = 0

        async def mock_planning(goal: str, iteration_id: int) -> None:
            await self._inject_pending_tasks(orchestrator, iteration_id, 1)
            iteration = orchestrator.state.get_current_iteration()
            iteration.tasks_created = 1

        async def mock_execution(iteration_id: int) -> None:
            tasks = orchestrator.task_queue.get_tasks_by_iteration(iteration_id)
            for task in tasks:
                task.status = TaskStatus.COMPLETED
                task.result = {"success": True}
                orchestrator.task_queue.update_task(task)
            iteration = orchestrator.state.get_current_iteration()
            iteration.tasks_completed = len(tasks)
            orchestrator.state.total_tasks_completed += len(tasks)

        async def mock_review(goal: str, iteration_id: int) -> ReviewDecision:
            nonlocal iteration_count
            iteration_count += 1
            # 第一次返回 CONTINUE，第二次返回 COMPLETE
            if iteration_count == 1:
                return ReviewDecision.CONTINUE
            return ReviewDecision.COMPLETE

        def make_commit_result(iteration_id: int) -> dict:
            return {
                "success": True,
                "commit_hash": f"commit_iter_{iteration_id}",
                "message": f"feat(iter-{iteration_id}): 提交",
                "files_changed": [f"iter{iteration_id}.py"],
                "pushed": False,
            }

        commit_calls: list[dict] = []

        async def mock_commit_iteration(**kwargs: Any) -> dict:
            commit_calls.append(kwargs)
            return make_commit_result(kwargs["iteration_id"])

        with mock_spawn, mock_wait_ready:
            with patch.object(
                orchestrator, "_planning_phase", side_effect=mock_planning
            ):
                with patch.object(
                    orchestrator, "_execution_phase", side_effect=mock_execution
                ):
                    with patch.object(
                        orchestrator, "_review_phase", side_effect=mock_review
                    ):
                        with patch.object(
                            orchestrator.committer,
                            "commit_iteration",
                            side_effect=mock_commit_iteration,
                        ):
                            with patch.object(
                                orchestrator.committer,
                                "get_commit_summary",
                                return_value={
                                    "total_commits": 2,
                                    "pushed_commits": 0,
                                },
                            ):
                                result = await orchestrator.run("测试目标")

                                # commit_iteration 应该被调用两次（每次迭代）
                                assert len(commit_calls) == 2

                                # 验证两次调用的迭代 ID
                                assert commit_calls[0]["iteration_id"] == 1
                                assert commit_calls[1]["iteration_id"] == 2

                                # 验证第一次是 CONTINUE，第二次是 COMPLETE
                                assert commit_calls[0]["review_decision"] == "continue"
                                assert commit_calls[1]["review_decision"] == "complete"

    @pytest.mark.asyncio
    async def test_commit_per_iteration_with_abort(
        self, mp_config: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 commit_per_iteration=True 时 ABORT 决策也触发提交"""
        mp_config.max_iterations = 1
        mp_config.commit_per_iteration = True
        orchestrator = MultiProcessOrchestrator(mp_config)

        mock_spawn, mock_wait_ready = self._mock_spawn_and_ready(orchestrator)

        async def mock_planning(goal: str, iteration_id: int) -> None:
            await self._inject_pending_tasks(orchestrator, iteration_id, 1)
            iteration = orchestrator.state.get_current_iteration()
            iteration.tasks_created = 1

        async def mock_execution(iteration_id: int) -> None:
            # 标记任务为失败
            tasks = orchestrator.task_queue.get_tasks_by_iteration(iteration_id)
            for task in tasks:
                task.status = TaskStatus.FAILED
                task.error = "执行失败"
                orchestrator.task_queue.update_task(task)
            iteration = orchestrator.state.get_current_iteration()
            iteration.tasks_failed = len(tasks)
            orchestrator.state.total_tasks_failed += len(tasks)

        async def mock_review(goal: str, iteration_id: int) -> ReviewDecision:
            return ReviewDecision.ABORT

        mock_commit_result = {
            "success": True,
            "commit_hash": "abort_commit",
            "message": "feat: ABORT 时的提交",
            "files_changed": ["partial.py"],
            "pushed": False,
        }

        with mock_spawn, mock_wait_ready:
            with patch.object(
                orchestrator, "_planning_phase", side_effect=mock_planning
            ):
                with patch.object(
                    orchestrator, "_execution_phase", side_effect=mock_execution
                ):
                    with patch.object(
                        orchestrator, "_review_phase", side_effect=mock_review
                    ):
                        with patch.object(
                            orchestrator.committer,
                            "commit_iteration",
                            new_callable=AsyncMock,
                            return_value=mock_commit_result,
                        ) as mock_commit:
                            with patch.object(
                                orchestrator.committer,
                                "get_commit_summary",
                                return_value={
                                    "total_commits": 1,
                                    "pushed_commits": 0,
                                },
                            ):
                                result = await orchestrator.run("测试目标")

                                # commit_per_iteration=True 时，即使 ABORT 也应提交
                                mock_commit.assert_called_once()
                                call_kwargs = mock_commit.call_args.kwargs
                                assert call_kwargs["review_decision"] == "abort"

    # =========================================================================
    # 测试 _generate_final_result 包含提交信息
    # =========================================================================

    @pytest.mark.asyncio
    async def test_final_result_contains_commits(
        self, mp_orchestrator: MultiProcessOrchestrator
    ) -> None:
        """测试最终结果包含 commits 和 pushed 信息"""
        mock_spawn, mock_wait_ready = self._mock_spawn_and_ready(mp_orchestrator)

        async def mock_planning(goal: str, iteration_id: int) -> None:
            await self._inject_pending_tasks(mp_orchestrator, iteration_id, 1)
            iteration = mp_orchestrator.state.get_current_iteration()
            iteration.tasks_created = 1

        async def mock_execution(iteration_id: int) -> None:
            tasks = mp_orchestrator.task_queue.get_tasks_by_iteration(iteration_id)
            for task in tasks:
                task.status = TaskStatus.COMPLETED
                task.result = {"success": True}
                mp_orchestrator.task_queue.update_task(task)
            iteration = mp_orchestrator.state.get_current_iteration()
            iteration.tasks_completed = len(tasks)
            mp_orchestrator.state.total_tasks_completed += len(tasks)

        async def mock_review(goal: str, iteration_id: int) -> ReviewDecision:
            return ReviewDecision.COMPLETE

        mock_commit_result = {
            "success": True,
            "commit_hash": "final_hash",
            "message": "feat: 完成",
            "files_changed": ["done.py"],
            "pushed": True,
        }

        mock_summary = {
            "total_commits": 1,
            "successful_commits": 1,
            "failed_commits": 0,
            "pushed_commits": 1,
            "commit_hashes": ["final_hash"],
            "files_changed": ["done.py"],
        }

        with mock_spawn, mock_wait_ready:
            with patch.object(
                mp_orchestrator, "_planning_phase", side_effect=mock_planning
            ):
                with patch.object(
                    mp_orchestrator, "_execution_phase", side_effect=mock_execution
                ):
                    with patch.object(
                        mp_orchestrator, "_review_phase", side_effect=mock_review
                    ):
                        with patch.object(
                            mp_orchestrator.committer,
                            "commit_iteration",
                            new_callable=AsyncMock,
                            return_value=mock_commit_result,
                        ):
                            with patch.object(
                                mp_orchestrator.committer,
                                "get_commit_summary",
                                return_value=mock_summary,
                            ):
                                result = await mp_orchestrator.run("测试目标")

                                # 验证最终结果结构
                                assert result["success"] is True
                                assert "commits" in result
                                assert result["commits"]["total_commits"] == 1
                                assert result["commits"]["pushed_commits"] == 1
                                assert result["pushed"] is True

                                # 验证迭代信息包含提交详情
                                assert len(result["iterations"]) == 1
                                iter_info = result["iterations"][0]
                                assert iter_info["commit_hash"] == "final_hash"
                                assert iter_info["commit_pushed"] is True

    # =========================================================================
    # 测试迭代状态填充
    # =========================================================================

    @pytest.mark.asyncio
    async def test_iteration_state_population(
        self, mp_orchestrator: MultiProcessOrchestrator
    ) -> None:
        """测试提交后 IterationState 正确填充所有字段"""
        iteration = mp_orchestrator.state.start_new_iteration()
        iteration_id = iteration.iteration_id

        await self._inject_tasks_to_queue(
            mp_orchestrator,
            iteration_id,
            [{"title": "任务1"}, {"title": "任务2"}],
        )

        mock_commit_result = {
            "success": True,
            "commit_hash": "state_test_hash",
            "message": "feat(iter-1): 完成 2 个任务\n\n详细描述",
            "files_changed": ["file1.py", "file2.py", "file3.py"],
            "pushed": True,
        }

        with patch.object(
            mp_orchestrator.committer,
            "commit_iteration",
            new_callable=AsyncMock,
            return_value=mock_commit_result,
        ):
            await mp_orchestrator._commit_phase(iteration_id, ReviewDecision.COMPLETE)

            # 验证所有 IterationState 字段
            assert iteration.status == IterationStatus.COMMITTING
            assert iteration.commit_hash == "state_test_hash"
            assert "feat(iter-1)" in iteration.commit_message
            assert iteration.pushed is True
            assert len(iteration.commit_files) == 3
            assert "file1.py" in iteration.commit_files
            assert "file2.py" in iteration.commit_files
            assert "file3.py" in iteration.commit_files


class TestCommitStrategyEdgeCases:
    """测试提交策略边界情况"""

    @pytest.mark.asyncio
    async def test_both_commit_options_true(self) -> None:
        """测试 commit_per_iteration 和 commit_on_complete 都为 True"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            enable_auto_commit=True,
            commit_per_iteration=True,
            commit_on_complete=True,
        )
        orchestrator = MultiProcessOrchestrator(config)

        # commit_per_iteration 优先级更高，应该对所有决策返回 True
        assert orchestrator._should_commit(ReviewDecision.COMPLETE) is True
        assert orchestrator._should_commit(ReviewDecision.CONTINUE) is True
        assert orchestrator._should_commit(ReviewDecision.ADJUST) is True
        assert orchestrator._should_commit(ReviewDecision.ABORT) is True

    @pytest.mark.asyncio
    async def test_both_commit_options_false(self) -> None:
        """测试 commit_per_iteration 和 commit_on_complete 都为 False"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            enable_auto_commit=True,
            commit_per_iteration=False,
            commit_on_complete=False,
        )
        orchestrator = MultiProcessOrchestrator(config)

        # 两者都为 False 时，不应该触发任何提交
        assert orchestrator._should_commit(ReviewDecision.COMPLETE) is False
        assert orchestrator._should_commit(ReviewDecision.CONTINUE) is False

    @pytest.mark.asyncio
    async def test_commit_phase_status_transition(self) -> None:
        """测试提交阶段的状态转换"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            enable_auto_commit=True,
        )
        orchestrator = MultiProcessOrchestrator(config)

        iteration = orchestrator.state.start_new_iteration()
        iteration_id = iteration.iteration_id

        # 初始状态
        assert iteration.status == IterationStatus.PLANNING

        mock_commit_result = {
            "success": True,
            "commit_hash": "status_test",
            "message": "test",
            "files_changed": [],
            "pushed": False,
        }

        with patch.object(
            orchestrator.committer,
            "commit_iteration",
            new_callable=AsyncMock,
            return_value=mock_commit_result,
        ):
            await orchestrator._commit_phase(iteration_id, ReviewDecision.COMPLETE)

            # 提交阶段后状态应为 COMMITTING
            assert iteration.status == IterationStatus.COMMITTING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
