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


class TestHealthCheckWarningCooldown:
    """测试健康检查告警 cooldown 机制"""

    @pytest.fixture
    def mp_config_for_cooldown(self) -> MultiProcessOrchestratorConfig:
        """创建测试 cooldown 的配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            enable_auto_commit=False,
            health_check_interval=1.0,
            health_check_timeout=2.0,
            skip_busy_workers_in_health_check=True,
        )

    @pytest.mark.asyncio
    async def test_cooldown_prevents_repeated_warnings(
        self, mp_config_for_cooldown: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 cooldown 机制防止重复告警"""
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_for_cooldown)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1"]
        orchestrator.reviewer_id = "reviewer-1"

        # 验证初始状态
        assert orchestrator._health_warning_cooldown == {}
        assert orchestrator._health_warning_cooldown_seconds == 60.0

        # 第一次调用应该返回 True（应该发出警告）
        should_warn_1 = orchestrator._should_emit_health_warning("worker-0")
        assert should_warn_1 is True
        assert "worker-0" in orchestrator._health_warning_cooldown

        # 立即再次调用应该返回 False（在 cooldown 内）
        should_warn_2 = orchestrator._should_emit_health_warning("worker-0")
        assert should_warn_2 is False

        # 不同的 agent 应该独立计算
        should_warn_3 = orchestrator._should_emit_health_warning("worker-1")
        assert should_warn_3 is True

    @pytest.mark.asyncio
    async def test_cooldown_expires_after_interval(
        self, mp_config_for_cooldown: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 cooldown 过期后可以再次告警"""
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_for_cooldown)
        # 设置较短的 cooldown 时间用于测试
        orchestrator._health_warning_cooldown_seconds = 0.1

        # 第一次调用
        should_warn_1 = orchestrator._should_emit_health_warning("worker-0")
        assert should_warn_1 is True

        # 等待 cooldown 过期
        time.sleep(0.15)

        # cooldown 过期后应该可以再次告警
        should_warn_2 = orchestrator._should_emit_health_warning("worker-0")
        assert should_warn_2 is True

    @pytest.mark.asyncio
    async def test_busy_worker_no_heartbeat_uses_debug_log(
        self, mp_config_for_cooldown: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 busy worker 无心跳响应时使用 debug 日志（不告警）"""
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_for_cooldown)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应：worker-0 有响应，worker-1 无响应
        current_time = time.time()
        orchestrator._heartbeat_responses["planner-1"] = current_time + 0.1
        orchestrator._heartbeat_responses["worker-0"] = current_time + 0.1
        orchestrator._heartbeat_responses["reviewer-1"] = current_time + 0.1
        # worker-1 没有响应

        # Mock: worker-1 进程存活且有任务分配（busy）
        def mock_is_alive(agent_id: str) -> bool:
            return True  # 所有进程存活

        def mock_get_tasks_by_agent(agent_id: str) -> list:
            if agent_id == "worker-1":
                return ["task-123"]  # worker-1 有任务
            return []

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
            ):
                with patch.object(
                    orchestrator.process_manager, "get_tasks_by_agent",
                    side_effect=mock_get_tasks_by_agent
                ):
                    result = await orchestrator._perform_health_check()

                    # worker-1 应该被视为健康（assumed_busy）
                    assert "worker-1" in result.healthy
                    assert result.details["worker-1"]["reason"] == "assumed_busy"
                    # 不应该触发 cooldown 机制（因为不输出 warning）
                    assert "worker-1" not in orchestrator._health_warning_cooldown

    @pytest.mark.asyncio
    async def test_non_busy_worker_no_heartbeat_triggers_cooldown(
        self, mp_config_for_cooldown: MultiProcessOrchestratorConfig
    ) -> None:
        """测试非 busy worker 连续多次无心跳响应时触发 cooldown

        新逻辑：只有连续多次未响应（达到 consecutive_unresponsive_threshold）后
        才会触发 WARNING 和 cooldown。单次未响应不触发告警。
        """
        import time

        # 设置较小的阈值以加速测试
        mp_config_for_cooldown.consecutive_unresponsive_threshold = 2
        orchestrator = MultiProcessOrchestrator(mp_config_for_cooldown)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1"]
        orchestrator.reviewer_id = "reviewer-1"

        # Mock: worker-1 进程存活但没有任务分配
        def mock_is_alive(agent_id: str) -> bool:
            return True

        def mock_get_tasks_by_agent(agent_id: str) -> list:
            return []  # 没有任务

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
            ):
                with patch.object(
                    orchestrator.process_manager, "get_tasks_by_agent",
                    side_effect=mock_get_tasks_by_agent
                ):
                    # 执行多次健康检查直到达到阈值
                    threshold = orchestrator.config.consecutive_unresponsive_threshold
                    for i in range(threshold):
                        # 预填充心跳响应：worker-1 无响应
                        current_time = time.time()
                        orchestrator._heartbeat_responses["planner-1"] = current_time + 0.1
                        orchestrator._heartbeat_responses["worker-0"] = current_time + 0.1
                        orchestrator._heartbeat_responses["reviewer-1"] = current_time + 0.1
                        # worker-1 不响应

                        result = await orchestrator._perform_health_check()

                    # worker-1 应该被视为不健康
                    assert "worker-1" in result.unhealthy
                    assert result.details["worker-1"]["reason"] == "no_heartbeat_response"
                    assert result.details["worker-1"]["is_alive"] is True
                    # 连续未响应次数应达到阈值
                    assert result.details["worker-1"]["consecutive_unresponsive"] == threshold
                    # 达到阈值后应该触发 cooldown
                    assert "worker-1" in orchestrator._health_warning_cooldown

    @pytest.mark.asyncio
    async def test_process_dead_always_logs_error(
        self, mp_config_for_cooldown: MultiProcessOrchestratorConfig
    ) -> None:
        """测试进程死亡时始终记录 ERROR（不受 cooldown 限制）"""
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_for_cooldown)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应：worker-1 无响应
        current_time = time.time()
        orchestrator._heartbeat_responses["planner-1"] = current_time + 0.1
        orchestrator._heartbeat_responses["worker-0"] = current_time + 0.1
        orchestrator._heartbeat_responses["reviewer-1"] = current_time + 0.1

        # Mock: worker-1 进程已死亡
        def mock_is_alive(agent_id: str) -> bool:
            if agent_id == "worker-1":
                return False
            return True

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
            ):
                result = await orchestrator._perform_health_check()

                # worker-1 应该被视为不健康且进程死亡
                assert "worker-1" in result.unhealthy
                assert result.details["worker-1"]["reason"] == "process_dead"
                assert result.details["worker-1"]["is_alive"] is False


class TestHealthCheckInOrchestrator:
    """测试 Orchestrator 中的健康检查处理"""

    @pytest.fixture
    def mp_config_with_health_check(self) -> MultiProcessOrchestratorConfig:
        """创建启用健康检查的配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=2,
            worker_count=3,
            enable_auto_commit=False,
            health_check_interval=1.0,  # 1秒间隔（测试用）
            health_check_timeout=2.0,
            max_unhealthy_workers=1,  # 超过1个不健康 Worker 则降级
            requeue_on_worker_death=True,
            fallback_on_critical_failure=True,
        )

    def _mock_spawn_and_ready(
        self, orchestrator: MultiProcessOrchestrator
    ) -> tuple:
        """Mock spawn 和 ready"""
        mock_spawn = patch.object(
            orchestrator, "_spawn_agents", return_value=None
        )
        mock_wait_ready = patch.object(
            orchestrator.process_manager, "wait_all_ready", return_value=True
        )
        return mock_spawn, mock_wait_ready

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(
        self, mp_config_with_health_check: MultiProcessOrchestratorConfig
    ) -> None:
        """测试所有进程健康时的行为（方案 A：心跳收集架构）"""
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_with_health_check)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1", "worker-2"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应以模拟所有进程响应心跳
        all_agents = ["planner-1", "worker-0", "worker-1", "worker-2", "reviewer-1"]
        current_time = time.time()
        for agent_id in all_agents:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1

        # Mock broadcast 和 is_alive
        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", return_value=True
            ):
                result = await orchestrator._perform_health_check()

                assert result.all_healthy is True
                assert orchestrator._degraded is False
                assert orchestrator._unhealthy_worker_count == 0

    @pytest.mark.asyncio
    async def test_health_check_partial_worker_unhealthy_within_threshold(
        self, mp_config_with_health_check: MultiProcessOrchestratorConfig
    ) -> None:
        """测试部分 Worker 不健康但在阈值内（方案 A：心跳收集架构）"""
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_with_health_check)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1", "worker-2"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应：worker-1 不响应（没有心跳记录或过时）
        current_time = time.time()
        healthy_agents = ["planner-1", "worker-0", "worker-2", "reviewer-1"]
        for agent_id in healthy_agents:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1
        # worker-1 没有响应（不添加或使用过时时间戳）

        # Mock broadcast、is_alive 和 get_tasks_by_agent
        def mock_is_alive(agent_id: str) -> bool:
            return True  # 所有进程都存活，worker-1 只是没响应心跳

        def mock_get_tasks_by_agent(agent_id: str) -> list:
            return []  # 没有任务分配给 worker-1（所以不是因为忙碌）

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
            ):
                with patch.object(
                    orchestrator.process_manager, "get_tasks_by_agent",
                    side_effect=mock_get_tasks_by_agent
                ):
                    with patch.object(
                        orchestrator, "_handle_unhealthy_workers", new_callable=AsyncMock
                    ) as mock_handle:
                        result = await orchestrator._perform_health_check()

                        # 新逻辑：进程存活但无心跳响应，标记为 unhealthy
                        assert result.all_healthy is False
                        assert orchestrator._degraded is False  # 未触发降级
                        # 新逻辑：_unhealthy_worker_count 只统计已死亡的 Worker（is_alive=False）
                        # worker-1 进程存活，所以不计入 dead_workers
                        assert orchestrator._unhealthy_worker_count == 0
                        # 不调用 _handle_unhealthy_workers（只处理 dead workers）
                        mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_worker_unhealthy_exceeds_threshold(
        self, mp_config_with_health_check: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 Worker 不健康数量超过阈值触发降级（方案 A：心跳收集架构）

        新逻辑：只有进程真正死亡（is_alive=False）才计入 dead_workers。
        当 dead_workers > max_unhealthy_workers 时触发降级。
        """
        import time

        # 修改配置：max_unhealthy_workers=1，这样 2 个死亡的 Worker 会触发降级
        mp_config_with_health_check.max_unhealthy_workers = 1

        orchestrator = MultiProcessOrchestrator(mp_config_with_health_check)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1", "worker-2"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应：worker-1 和 worker-2 不响应
        current_time = time.time()
        healthy_agents = ["planner-1", "worker-0", "reviewer-1"]
        for agent_id in healthy_agents:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1
        # worker-1, worker-2 没有响应

        # Mock broadcast、is_alive 和 get_tasks_by_agent
        def mock_is_alive(agent_id: str) -> bool:
            if agent_id in ("worker-1", "worker-2"):
                return False  # 两个 Worker 进程都死亡
            return True  # 其他进程存活

        def mock_get_tasks_by_agent(agent_id: str) -> list:
            return []

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
            ):
                with patch.object(
                    orchestrator.process_manager, "get_tasks_by_agent",
                    side_effect=mock_get_tasks_by_agent
                ):
                    result = await orchestrator._perform_health_check()

                    assert orchestrator._degraded is True
                    assert "阈值" in orchestrator._degradation_reason
                    # 只统计已死亡的 Worker
                    assert orchestrator._unhealthy_worker_count == 2

    @pytest.mark.asyncio
    async def test_health_check_planner_unhealthy_triggers_degradation(
        self, mp_config_with_health_check: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 Planner 不健康触发降级（方案 A：心跳收集架构）"""
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_with_health_check)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1", "worker-2"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应：planner-1 不响应
        current_time = time.time()
        healthy_agents = ["worker-0", "worker-1", "worker-2", "reviewer-1"]
        for agent_id in healthy_agents:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1
        # planner-1 没有响应

        # Mock broadcast 和 is_alive
        def mock_is_alive(agent_id: str) -> bool:
            if agent_id == "planner-1":
                return False  # process_dead
            return True

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
            ):
                result = await orchestrator._perform_health_check()

                assert orchestrator._degraded is True
                assert "Planner" in orchestrator._degradation_reason

    @pytest.mark.asyncio
    async def test_health_check_reviewer_unhealthy_triggers_degradation(
        self, mp_config_with_health_check: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 Reviewer 进程死亡触发降级（方案 A：心跳收集架构）

        新逻辑：只有当 Reviewer 进程真正死亡（is_alive=False）时才触发降级。
        """
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_with_health_check)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1", "worker-2"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应：reviewer-1 不响应
        current_time = time.time()
        healthy_agents = ["planner-1", "worker-0", "worker-1", "worker-2"]
        for agent_id in healthy_agents:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1
        # reviewer-1 没有响应

        # Mock broadcast 和 is_alive
        def mock_is_alive(agent_id: str) -> bool:
            if agent_id == "reviewer-1":
                return False  # Reviewer 进程已死亡
            return True

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
            ):
                result = await orchestrator._perform_health_check()

                assert orchestrator._degraded is True
                assert "Reviewer" in orchestrator._degradation_reason

    @pytest.mark.asyncio
    async def test_handle_unhealthy_workers_requeue(
        self, mp_config_with_health_check: MultiProcessOrchestratorConfig
    ) -> None:
        """测试不健康 Worker 的在途任务重新入队"""
        orchestrator = MultiProcessOrchestrator(mp_config_with_health_check)
        orchestrator.state.start_new_iteration()

        # 模拟任务和跟踪
        task = Task(
            id="task-requeue-1",
            iteration_id=1,
            type="implement",
            title="测试任务",
            description="",
            instruction="执行",
            target_files=[],
        )
        task.status = TaskStatus.IN_PROGRESS
        await orchestrator.task_queue.enqueue(task)

        # 跟踪任务分配
        orchestrator.process_manager.track_task_assignment(
            task_id="task-requeue-1",
            agent_id="worker-1",
            message_id="msg-123",
        )

        # 处理不健康 Worker
        await orchestrator._handle_unhealthy_workers(["worker-1"])

        # 验证任务被重新入队（状态变为 QUEUED）
        updated_task = orchestrator.task_queue.get_task("task-requeue-1")
        assert updated_task.status == TaskStatus.QUEUED

    @pytest.mark.asyncio
    async def test_handle_unhealthy_workers_mark_failed(
        self,
    ) -> None:
        """测试不健康 Worker 的在途任务标记失败（requeue_on_worker_death=False）"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            enable_auto_commit=False,
            requeue_on_worker_death=False,  # 不重入队，直接标记失败
        )
        orchestrator = MultiProcessOrchestrator(config)
        orchestrator.state.start_new_iteration()

        # 模拟任务
        task = Task(
            id="task-fail-1",
            iteration_id=1,
            type="implement",
            title="测试任务",
            description="",
            instruction="执行",
            target_files=[],
        )
        task.status = TaskStatus.IN_PROGRESS
        await orchestrator.task_queue.enqueue(task)

        # 跟踪任务分配
        orchestrator.process_manager.track_task_assignment(
            task_id="task-fail-1",
            agent_id="worker-dead",
            message_id="msg-456",
        )

        # 处理不健康 Worker
        await orchestrator._handle_unhealthy_workers(["worker-dead"])

        # 验证任务被标记为 FAILED
        updated_task = orchestrator.task_queue.get_task("task-fail-1")
        assert updated_task.status == TaskStatus.FAILED
        assert "worker-dead" in updated_task.error.lower()

    @pytest.mark.asyncio
    async def test_degraded_state_stops_iteration(
        self, mp_config_with_health_check: MultiProcessOrchestratorConfig
    ) -> None:
        """测试降级状态下 _should_continue_iteration 返回 False"""
        orchestrator = MultiProcessOrchestrator(mp_config_with_health_check)

        assert orchestrator._should_continue_iteration() is True

        orchestrator._trigger_degradation("测试降级")

        assert orchestrator._should_continue_iteration() is False
        assert orchestrator.is_degraded() is True
        assert orchestrator.get_degradation_reason() == "测试降级"

    @pytest.mark.asyncio
    async def test_final_result_includes_degradation_info(
        self, mp_config_with_health_check: MultiProcessOrchestratorConfig
    ) -> None:
        """测试最终结果包含降级信息"""
        from process.manager import HealthCheckResult

        orchestrator = MultiProcessOrchestrator(mp_config_with_health_check)
        mock_spawn, mock_wait_ready = self._mock_spawn_and_ready(orchestrator)

        # 设置降级状态
        orchestrator._degraded = True
        orchestrator._degradation_reason = "Worker 全部死亡"

        with mock_spawn, mock_wait_ready:
            with patch.object(
                orchestrator.process_manager, "shutdown_all", return_value=None
            ):
                result = orchestrator._generate_final_result()

                assert result["degraded"] is True
                assert result["degradation_reason"] == "Worker 全部死亡"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="集成测试需要更复杂的 mock 设置，单元测试已覆盖核心逻辑")
    async def test_full_run_with_degradation_on_health_check(
        self, mp_config_with_health_check: MultiProcessOrchestratorConfig
    ) -> None:
        """测试完整运行流程中健康检查触发降级（方案 A：心跳收集架构）"""
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_with_health_check)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1", "worker-2"]
        orchestrator.reviewer_id = "reviewer-1"

        mock_spawn, mock_wait_ready = self._mock_spawn_and_ready(orchestrator)

        # 预填充心跳响应：planner-1 不响应
        current_time = time.time()
        healthy_agents = ["worker-0", "worker-1", "worker-2", "reviewer-1"]
        for agent_id in healthy_agents:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1
        # planner-1 没有响应

        # Mock broadcast 和 is_alive
        def mock_is_alive(agent_id: str) -> bool:
            if agent_id == "planner-1":
                return False  # process_dead
            return True

        with mock_spawn, mock_wait_ready:
            with patch.object(orchestrator.process_manager, "broadcast"):
                with patch.object(
                    orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
                ):
                    with patch.object(
                        orchestrator.process_manager, "shutdown_all", return_value=None
                    ):
                        # Mock receive_message 返回 None 以避免 _message_loop 挂起
                        with patch.object(
                            orchestrator.process_manager, "receive_message", return_value=None
                        ):
                            # 强制 _should_run_health_check 返回 True
                            orchestrator._last_health_check_time = 0

                            result = await orchestrator.run("测试降级")

                            # 应该因降级而终止
                            assert result["success"] is False
                            assert result["degraded"] is True
                            assert "Planner" in result["degradation_reason"]


class TestDegradationStrategyBOrchestrator:
    """测试降级策略 B 在 MultiProcessOrchestrator 中的实现

    策略 B 契约:
    1. 降级时 success=False（基于 is_completed 状态）
    2. degraded=True 显式标注
    3. degradation_reason 记录降级原因
    4. iterations_completed 记录降级前完成的迭代数
    5. 不触发回退（回退由 SelfIterator 层判断 _fallback_required）

    这是 MultiProcessOrchestrator 层的降级策略实现测试。
    """

    @pytest.fixture
    def mp_config_for_degradation(self) -> MultiProcessOrchestratorConfig:
        """创建用于降级测试的配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=3,
            worker_count=3,
            enable_auto_commit=False,
            health_check_interval=1.0,
            health_check_timeout=2.0,
            max_unhealthy_workers=1,
            fallback_on_critical_failure=True,
        )

    def _mock_spawn_and_ready(
        self, orchestrator: MultiProcessOrchestrator
    ) -> tuple:
        """Mock spawn 和 ready"""
        mock_spawn = patch.object(
            orchestrator, "_spawn_agents", return_value=None
        )
        mock_wait_ready = patch.object(
            orchestrator.process_manager, "wait_all_ready", return_value=True
        )
        return mock_spawn, mock_wait_ready

    @pytest.mark.asyncio
    async def test_degraded_result_success_false(
        self, mp_config_for_degradation: MultiProcessOrchestratorConfig
    ) -> None:
        """测试降级时 success=False（策略 B 核心契约）"""
        orchestrator = MultiProcessOrchestrator(mp_config_for_degradation)

        # 模拟降级状态
        orchestrator._degraded = True
        orchestrator._degradation_reason = "Planner 进程不健康"
        orchestrator.state.is_completed = False  # 确保未完成

        result = orchestrator._generate_final_result()

        # 策略 B 核心断言
        assert result["success"] is False, "降级时 success 必须为 False"
        assert result["degraded"] is True, "降级时 degraded 必须为 True"
        assert result["degradation_reason"] == "Planner 进程不健康"

    @pytest.mark.asyncio
    async def test_degraded_result_iterations_completed_accurate(
        self, mp_config_for_degradation: MultiProcessOrchestratorConfig
    ) -> None:
        """测试降级时 iterations_completed 准确记录"""
        orchestrator = MultiProcessOrchestrator(mp_config_for_degradation)

        # 模拟完成了 2 次迭代后降级
        orchestrator.state.start_new_iteration()  # 迭代 1
        orchestrator.state.start_new_iteration()  # 迭代 2

        orchestrator._degraded = True
        orchestrator._degradation_reason = "Worker 不健康数量超过阈值"

        result = orchestrator._generate_final_result()

        assert result["iterations_completed"] == 2
        assert result["degraded"] is True

    @pytest.mark.asyncio
    async def test_degraded_triggered_by_planner_unhealthy(
        self, mp_config_for_degradation: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 Planner 不健康触发降级"""
        from process.manager import HealthCheckResult

        orchestrator = MultiProcessOrchestrator(mp_config_for_degradation)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1", "worker-2"]
        orchestrator.reviewer_id = "reviewer-1"

        # Planner 不健康
        mock_result = HealthCheckResult(
            healthy=["worker-0", "worker-1", "worker-2", "reviewer-1"],
            unhealthy=["planner-1"],
            all_healthy=False,
            details={"planner-1": {"healthy": False, "reason": "process_dead"}},
        )

        with patch.object(
            orchestrator.process_manager, "health_check", return_value=mock_result
        ):
            await orchestrator._perform_health_check()

            # 验证降级触发
            assert orchestrator._degraded is True
            assert "Planner" in orchestrator._degradation_reason

            # 生成结果验证
            result = orchestrator._generate_final_result()
            assert result["success"] is False
            assert result["degraded"] is True

    @pytest.mark.asyncio
    async def test_degraded_triggered_by_reviewer_unhealthy(
        self, mp_config_for_degradation: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 Reviewer 进程死亡触发降级（方案 A：心跳收集架构）

        新逻辑：只有当 Reviewer 进程真正死亡（is_alive=False）时才触发降级。
        """
        import time

        orchestrator = MultiProcessOrchestrator(mp_config_for_degradation)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1", "worker-2"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应：reviewer-1 不响应
        current_time = time.time()
        healthy_agents = ["planner-1", "worker-0", "worker-1", "worker-2"]
        for agent_id in healthy_agents:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1
        # reviewer-1 没有响应

        # Mock is_alive: reviewer-1 进程死亡
        def mock_is_alive(agent_id: str) -> bool:
            if agent_id == "reviewer-1":
                return False  # Reviewer 进程已死亡
            return True

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
            ):
                await orchestrator._perform_health_check()

                assert orchestrator._degraded is True
                assert "Reviewer" in orchestrator._degradation_reason

    @pytest.mark.asyncio
    async def test_degraded_triggered_by_workers_exceed_threshold(
        self, mp_config_for_degradation: MultiProcessOrchestratorConfig
    ) -> None:
        """测试 Worker 死亡数量超过阈值触发降级（方案 A：心跳收集架构）

        新逻辑：只有进程真正死亡（is_alive=False）才计入死亡 Worker 统计。
        当死亡 Worker 数量 > max_unhealthy_workers 时触发降级。
        """
        import time

        # 修改配置：max_unhealthy_workers=1，这样 2 个死亡的 Worker 会触发降级
        mp_config_for_degradation.max_unhealthy_workers = 1

        orchestrator = MultiProcessOrchestrator(mp_config_for_degradation)
        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1", "worker-2"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应：worker-1 和 worker-2 不响应
        current_time = time.time()
        healthy_agents = ["planner-1", "worker-0", "reviewer-1"]
        for agent_id in healthy_agents:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1
        # worker-1, worker-2 没有响应

        # Mock is_alive: worker-1 和 worker-2 进程都死亡
        def mock_is_alive(agent_id: str) -> bool:
            if agent_id in ("worker-1", "worker-2"):
                return False  # process_dead
            return True

        def mock_get_tasks_by_agent(agent_id: str) -> list:
            return []

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", side_effect=mock_is_alive
            ):
                with patch.object(
                    orchestrator.process_manager, "get_tasks_by_agent",
                    side_effect=mock_get_tasks_by_agent
                ):
                    await orchestrator._perform_health_check()

                    assert orchestrator._degraded is True
                    assert "阈值" in orchestrator._degradation_reason

    @pytest.mark.asyncio
    async def test_degraded_stops_iteration_loop(
        self, mp_config_for_degradation: MultiProcessOrchestratorConfig
    ) -> None:
        """测试降级后 _should_continue_iteration 返回 False

        策略 B 行为：降级后迭代循环应停止
        """
        orchestrator = MultiProcessOrchestrator(mp_config_for_degradation)

        # 初始状态：应继续迭代
        assert orchestrator._should_continue_iteration() is True

        # 触发降级
        orchestrator._trigger_degradation("Planner 进程不健康")

        # 降级后：应停止迭代
        assert orchestrator._should_continue_iteration() is False
        assert orchestrator.is_degraded() is True
        assert orchestrator.get_degradation_reason() == "Planner 进程不健康"

    @pytest.mark.asyncio
    async def test_degraded_result_no_fallback_required_flag(
        self, mp_config_for_degradation: MultiProcessOrchestratorConfig
    ) -> None:
        """测试降级结果不包含 _fallback_required 标志

        这是策略 B 与策略 A 的关键区分点：
        - degraded: 运行时降级，不触发回退
        - _fallback_required: 启动失败，需要回退
        """
        orchestrator = MultiProcessOrchestrator(mp_config_for_degradation)

        orchestrator._degraded = True
        orchestrator._degradation_reason = "测试降级"

        result = orchestrator._generate_final_result()

        # 验证不包含 _fallback_required（这是启动失败的标志）
        assert "_fallback_required" not in result
        assert "_fallback_reason" not in result

        # 验证包含降级标志
        assert result["degraded"] is True
        assert result["degradation_reason"] == "测试降级"

    @pytest.mark.asyncio
    async def test_degraded_preserves_completed_iterations_info(
        self, mp_config_for_degradation: MultiProcessOrchestratorConfig
    ) -> None:
        """测试降级时保留已完成迭代的详细信息"""
        orchestrator = MultiProcessOrchestrator(mp_config_for_degradation)

        # 模拟完成了一次迭代
        iteration = orchestrator.state.start_new_iteration()
        iteration.tasks_created = 3
        iteration.tasks_completed = 2
        iteration.tasks_failed = 1
        iteration.status = IterationStatus.COMPLETED
        iteration.review_passed = True

        # 然后降级
        orchestrator._degraded = True
        orchestrator._degradation_reason = "第二次迭代时 Planner 崩溃"

        result = orchestrator._generate_final_result()

        # 验证已完成迭代的信息被保留
        assert len(result["iterations"]) == 1
        iter_info = result["iterations"][0]
        assert iter_info["tasks_created"] == 3
        assert iter_info["tasks_completed"] == 2
        assert iter_info["tasks_failed"] == 1
        assert iter_info["review_passed"] is True


class TestAutoCommitDefaultDisabled:
    """回归测试：验证 enable_auto_commit 默认禁用时的行为

    确保：
    1. 默认情况下 Committer 不被初始化
    2. 最终结果 commits 为空
    3. 不会触发任何提交操作
    """

    def test_default_config_auto_commit_disabled(self) -> None:
        """测试默认配置 enable_auto_commit=False"""
        config = MultiProcessOrchestratorConfig()
        assert config.enable_auto_commit is False, \
            "默认配置应禁用 auto_commit"

    def test_committer_not_initialized_when_disabled(self) -> None:
        """测试禁用 auto_commit 时 Committer 不被初始化"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            enable_auto_commit=False,
        )
        orchestrator = MultiProcessOrchestrator(config)

        assert orchestrator.committer is None, \
            "禁用 auto_commit 时不应初始化 Committer"

    def test_committer_initialized_when_enabled(self) -> None:
        """测试启用 auto_commit 时 Committer 被初始化"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            enable_auto_commit=True,
        )
        orchestrator = MultiProcessOrchestrator(config)

        assert orchestrator.committer is not None, \
            "启用 auto_commit 时应初始化 Committer"

    @pytest.mark.asyncio
    async def test_should_commit_false_when_disabled(self) -> None:
        """测试禁用 auto_commit 时 _should_commit 始终返回 False"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            enable_auto_commit=False,
        )
        orchestrator = MultiProcessOrchestrator(config)

        # 所有决策都应返回 False
        assert orchestrator._should_commit(ReviewDecision.COMPLETE) is False
        assert orchestrator._should_commit(ReviewDecision.CONTINUE) is False
        assert orchestrator._should_commit(ReviewDecision.ADJUST) is False
        assert orchestrator._should_commit(ReviewDecision.ABORT) is False

    @pytest.mark.asyncio
    async def test_final_result_commits_empty_when_disabled(self) -> None:
        """测试禁用 auto_commit 时最终结果 commits 为空"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            enable_auto_commit=False,
        )
        orchestrator = MultiProcessOrchestrator(config)

        # 模拟运行完成
        orchestrator.state.is_completed = True

        # 生成最终结果
        result = orchestrator._generate_final_result()

        # 验证 commits 为空
        assert result["commits"] == {}, \
            "禁用 auto_commit 时 commits 应为空字典"
        assert result["pushed"] is False, \
            "禁用 auto_commit 时 pushed 应为 False"

    @pytest.mark.asyncio
    async def test_commit_phase_returns_error_when_no_committer(self) -> None:
        """测试没有 Committer 时 _commit_phase 返回错误"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            enable_auto_commit=False,
        )
        orchestrator = MultiProcessOrchestrator(config)
        orchestrator.state.start_new_iteration()

        result = await orchestrator._commit_phase(1, ReviewDecision.COMPLETE)

        assert result["success"] is False
        assert "Committer not initialized" in result.get("error", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
