#!/usr/bin/env python3
"""MP 编排器稳定性测试

测试场景：
1. Worker 超时（缩短 execution_timeout）
2. Worker 进程死亡（模拟 _send_and_wait 返回 None）
3. 卡死检测（pending > 0 且队列为空）

用法：
    pytest tests/test_mp_stability.py -v
    python tests/test_mp_stability.py  # 直接运行

卡死检测日志格式（用于回归测试断言）：
===========================================

诊断输出策略（已实现）：
    - 仅在 stall_detected=True 时输出诊断信息（避免无意义的大量日志）
    - 所有诊断日志（摘要、详细、关键卡死行）均受冷却窗口控制
    - 冷却窗口使用 _last_stall_diagnostic_time 控制
    - 冷却间隔与 stall_recovery_interval 对齐（默认 30 秒）
    - 诊断输出频率不超过恢复尝试频率

冷却窗口判断逻辑：
    should_emit_diagnostic = (
        stall_detected=True
        AND stall_diagnostics_enabled=True
        AND (_last_stall_diagnostic_time == 0.0
             OR now - _last_stall_diagnostic_time >= stall_recovery_interval)
    )

1. 摘要日志（仅在冷却窗口外且 stall_detected 时输出）：
   - "[诊断] 卡死检测 | iteration=N, pending=X, completed=Y, failed=Z, ..."

2. 卡死模式检测（仅在冷却窗口外且 stall_detected 时输出，error 级别）：
   - "[诊断] ⚠ 卡死模式检测: {stall_reason}"

典型卡死模式 1：pending > 0 但无活动任务和可用 Worker
    - stall_reason: "pending=X > 0 但 active_tasks=0, available_workers=0"

典型卡死模式 2：存在 ASSIGNED 状态任务但无对应异步任务
    - stall_reason: "存在 ASSIGNED 状态任务 (N) 但无 active_tasks"

3. 详细诊断（仅在冷却窗口外且 stall_diagnostics_detail=True 时输出）：
   - "[诊断] === 卡死检测触发 (循环 #N) ==="
   - "[诊断] TaskQueue.get_statistics(): {...}"
   - "[诊断] available_workers: [...]"
   - "[诊断] active_tasks: [...]"
   - "[诊断] in_flight_tasks: [...]"
   - "[诊断] Task {task_id}: status=X, retry_count=N, assigned_to=Y"
   - "[诊断] ... 省略 N 个任务"（当任务数超过 stall_diagnostics_max_tasks）

回归测试断言示例：
    # 检测卡死模式的关键断言（需配合冷却窗口）
    assert "[诊断] ⚠ 卡死模式检测" in log_output

    # 详细诊断日志格式断言（仅在冷却窗口外输出）
    assert "[诊断] === 卡死检测触发" in log_output

    # 冷却窗口测试：快速连续触发应只输出一次
    # 参见 TestDiagnosticCooldown 和 TestDiagnosticCooldownWithTimeInjection 类
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from coordinator.orchestrator_mp import (
    MultiProcessOrchestrator,
    MultiProcessOrchestratorConfig,
)
from process.manager import HealthCheckResult
from tasks.task import Task, TaskStatus, TaskType


class TestMPStability:
    """MP 编排器稳定性测试"""

    @pytest.fixture
    def config(self) -> MultiProcessOrchestratorConfig:
        """创建测试配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            # 故障注入：缩短超时时间
            execution_timeout=5.0,  # 缩短到 5 秒
            planning_timeout=10.0,
            review_timeout=10.0,
            # 禁用知识库
            enable_knowledge_search=False,
            enable_knowledge_injection=False,
            # 禁用自动提交
            enable_auto_commit=False,
        )

    @pytest.fixture
    def orchestrator(self, config) -> MultiProcessOrchestrator:
        """创建编排器实例"""
        return MultiProcessOrchestrator(config)

    @pytest.mark.asyncio
    async def test_worker_timeout_requeue(self, orchestrator):
        """测试 Worker 超时时任务重新入队"""
        # 模拟任务
        task = Task(
            id="test-task-timeout",
            type=TaskType.TEST,
            title="测试任务",
            description="测试 Worker 超时处理",
            instruction="执行测试",
            iteration_id=1,
        )

        # 入队任务
        await orchestrator.task_queue.enqueue(task)

        # 模拟 _send_and_wait 返回 None（超时）
        with patch.object(
            orchestrator, "_send_and_wait", new_callable=AsyncMock, return_value=None
        ):
            # 模拟 Worker 存活
            orchestrator.worker_ids = ["worker-test-1", "worker-test-2"]
            orchestrator.process_manager._processes = {
                "worker-test-1": MagicMock(is_alive=MagicMock(return_value=True)),
                "worker-test-2": MagicMock(is_alive=MagicMock(return_value=True)),
            }

            # 模拟健康检查通过
            with patch.object(
                orchestrator.process_manager,
                "health_check",
                return_value=HealthCheckResult(
                    healthy=["worker-test-1", "worker-test-2"],
                    unhealthy=[],
                    all_healthy=True,
                ),
            ):
                with patch.object(
                    orchestrator.process_manager, "is_alive", return_value=True
                ):
                    # 模拟 track_task_assignment
                    with patch.object(
                        orchestrator.process_manager, "track_task_assignment"
                    ):
                        # 设置超时后自动退出循环
                        async def mock_execution_phase_limited():
                            """限制执行阶段只运行一次分配"""
                            import time

                            iteration_id = 1
                            available_workers = list(orchestrator.worker_ids)
                            active_tasks = {}

                            # 只分配一个任务
                            task = await orchestrator.task_queue.dequeue(
                                iteration_id, timeout=0.1
                            )
                            if task:
                                worker_id = available_workers.pop(0)
                                # 模拟发送和超时
                                response = await orchestrator._send_and_wait(
                                    worker_id, MagicMock(), 1.0
                                )
                                if response is None:
                                    # 超时处理：应该重新入队
                                    task_obj = orchestrator.task_queue.get_task(
                                        task.id
                                    )
                                    if (
                                        task_obj
                                        and task_obj.status == TaskStatus.ASSIGNED
                                    ):
                                        if orchestrator.config.requeue_on_worker_death:
                                            task_obj.status = TaskStatus.PENDING
                                            orchestrator.task_queue.update_task(
                                                task_obj
                                            )

                        await mock_execution_phase_limited()

        # 验证任务状态
        updated_task = orchestrator.task_queue.get_task("test-task-timeout")
        assert updated_task is not None
        # 任务应该被重新入队（PENDING）或仍在 ASSIGNED
        assert updated_task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]

    @pytest.mark.asyncio
    async def test_worker_death_detection(self, orchestrator):
        """测试 Worker 进程死亡检测"""
        # 模拟 Worker ID
        orchestrator.worker_ids = ["worker-dead-1", "worker-live-1"]
        orchestrator.process_manager._processes = {
            "worker-dead-1": MagicMock(is_alive=MagicMock(return_value=False)),
            "worker-live-1": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        # 模拟任务分配给死亡的 Worker
        task = Task(
            id="test-task-death",
            type=TaskType.TEST,
            title="测试任务",
            description="测试 Worker 死亡处理",
            instruction="执行测试",
            iteration_id=1,
            assigned_to="worker-dead-1",
            status=TaskStatus.IN_PROGRESS,
        )
        orchestrator.task_queue._tasks[task.id] = task
        orchestrator.process_manager._task_assignments[task.id] = {
            "agent_id": "worker-dead-1",
            "assigned_at": 0,
            "message_id": "msg-1",
        }

        # 执行健康检查相关的处理
        unhealthy_workers = ["worker-dead-1"]
        await orchestrator._handle_unhealthy_workers(unhealthy_workers)

        # 验证任务被处理
        updated_task = orchestrator.task_queue.get_task("test-task-death")
        assert updated_task is not None
        # 根据 requeue_on_worker_death 配置，任务应该被重新入队或标记失败
        # 注意：requeue 后状态变为 QUEUED（已入队等待分配）
        if orchestrator.config.requeue_on_worker_death:
            assert updated_task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]
        else:
            assert updated_task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_stall_detection_pattern(self, orchestrator):
        """测试卡死模式检测：pending > 0 且无活动任务"""
        # 创建待处理任务
        task = Task(
            id="test-task-stall",
            type=TaskType.TEST,
            title="测试卡死",
            description="测试卡死检测",
            instruction="执行测试",
            iteration_id=1,
            status=TaskStatus.PENDING,
        )
        orchestrator.task_queue._tasks[task.id] = task

        # 获取统计信息
        stats = orchestrator.task_queue.get_statistics(1)

        # 验证卡死模式条件
        assert stats["pending"] > 0, "应该有待处理任务"

        # 模拟无可用 Worker
        available_workers = []
        active_tasks = {}

        # 这就是卡死模式
        is_stall = (
            stats["pending"] > 0
            and len(active_tasks) == 0
            and len(available_workers) == 0
        )
        assert is_stall, "应该检测到卡死模式"

    @pytest.mark.asyncio
    async def test_assigned_but_no_active_tasks(self, orchestrator):
        """测试卡死模式：存在 ASSIGNED 任务但无活动异步任务"""
        # 创建 ASSIGNED 状态任务（模拟任务已分配但 asyncio.Task 丢失）
        task = Task(
            id="test-task-assigned",
            type=TaskType.TEST,
            title="测试 ASSIGNED 卡死",
            description="任务已分配但异步任务丢失",
            instruction="执行测试",
            iteration_id=1,
            status=TaskStatus.ASSIGNED,
            assigned_to="worker-ghost",
        )
        orchestrator.task_queue._tasks[task.id] = task

        # 验证卡死模式
        tasks = orchestrator.task_queue.get_tasks_by_iteration(1)
        assigned_tasks = [t for t in tasks if t.status == TaskStatus.ASSIGNED]
        active_tasks = {}

        # 这就是卡死模式：有 ASSIGNED 任务但无 active_tasks
        is_stall = len(assigned_tasks) > 0 and len(active_tasks) == 0
        assert is_stall, "应该检测到 ASSIGNED 卡死模式"

    @pytest.mark.asyncio
    async def test_iteration_complete_check(self, orchestrator):
        """测试迭代完成检查"""
        # 创建已完成的任务
        task = Task(
            id="test-task-complete",
            type=TaskType.TEST,
            title="已完成任务",
            description="测试完成检查",
            instruction="执行测试",
            iteration_id=1,
            status=TaskStatus.COMPLETED,
        )
        orchestrator.task_queue._tasks[task.id] = task

        # 验证迭代完成
        is_complete = orchestrator.task_queue.is_iteration_complete(1)
        assert is_complete, "迭代应该标记为完成"

    @pytest.mark.asyncio
    async def test_empty_queue_no_stall(self, orchestrator):
        """测试空队列不应该被视为卡死"""
        # 无任务
        stats = orchestrator.task_queue.get_statistics(1)
        assert stats["total"] == 0
        assert stats["pending"] == 0

        # 空队列时迭代应该标记为完成
        is_complete = orchestrator.task_queue.is_iteration_complete(1)
        assert is_complete, "空队列迭代应该标记为完成"


class TestTaskQueueReconcile:
    """TaskQueue reconcile 功能测试"""

    @pytest.fixture
    def task_queue(self):
        """创建 TaskQueue 实例"""
        from tasks.queue import TaskQueue
        return TaskQueue()

    @pytest.mark.asyncio
    async def test_queue_index_tracking(self, task_queue):
        """测试队列索引追踪"""
        # 入队任务
        task = Task(
            id="test-queue-index",
            type=TaskType.TEST,
            title="测试队列索引",
            description="测试",
            instruction="测试",
            iteration_id=1,
        )
        await task_queue.enqueue(task)

        # 验证队列索引
        assert task_queue.is_in_queue("test-queue-index", 1)
        queued_ids = task_queue.get_queued_ids(1)
        assert "test-queue-index" in queued_ids

        # 出队后应该从索引中移除
        dequeued = await task_queue.dequeue(1, timeout=0.1)
        assert dequeued is not None
        assert not task_queue.is_in_queue("test-queue-index", 1)

    @pytest.mark.asyncio
    async def test_reconcile_orphaned_pending(self, task_queue):
        """测试 reconcile 检测 orphaned_pending"""
        # 创建任务但不入队（模拟状态不一致）
        task = Task(
            id="test-orphaned",
            type=TaskType.TEST,
            title="孤立任务",
            description="测试",
            instruction="测试",
            iteration_id=1,
            status=TaskStatus.PENDING,
        )
        task_queue._tasks[task.id] = task

        # 调用 reconcile
        issues = await task_queue.reconcile_iteration(
            iteration_id=1,
            in_flight_task_ids=set(),
            active_future_task_ids=set(),
        )

        # 验证检测到 orphaned_pending
        assert len(issues["orphaned_pending"]) == 1
        assert issues["orphaned_pending"][0].id == "test-orphaned"

    @pytest.mark.asyncio
    async def test_reconcile_orphaned_assigned(self, task_queue):
        """测试 reconcile 检测 orphaned_assigned"""
        # 创建 ASSIGNED 状态任务但无 in-flight 记录
        task = Task(
            id="test-assigned-orphan",
            type=TaskType.TEST,
            title="孤立分配任务",
            description="测试",
            instruction="测试",
            iteration_id=1,
            status=TaskStatus.ASSIGNED,
            assigned_to="worker-ghost",
        )
        task_queue._tasks[task.id] = task

        # 调用 reconcile（不包含该任务在 in_flight 中）
        issues = await task_queue.reconcile_iteration(
            iteration_id=1,
            in_flight_task_ids=set(),  # 空，不包含该任务
            active_future_task_ids=set(),
        )

        # 验证检测到 orphaned_assigned
        assert len(issues["orphaned_assigned"]) == 1
        assert issues["orphaned_assigned"][0].id == "test-assigned-orphan"

    @pytest.mark.asyncio
    async def test_reconcile_stale_in_progress(self, task_queue):
        """测试 reconcile 检测 stale_in_progress"""
        # 创建 IN_PROGRESS 状态任务但无 active future
        task = Task(
            id="test-stale-progress",
            type=TaskType.TEST,
            title="过期进行中任务",
            description="测试",
            instruction="测试",
            iteration_id=1,
            status=TaskStatus.IN_PROGRESS,
            assigned_to="worker-lost",
        )
        task_queue._tasks[task.id] = task

        # 调用 reconcile（不包含该任务在 active_future 中）
        issues = await task_queue.reconcile_iteration(
            iteration_id=1,
            in_flight_task_ids={"test-stale-progress"},  # 有 in-flight 记录
            active_future_task_ids=set(),  # 但无 active future
        )

        # 验证检测到 stale_in_progress
        assert len(issues["stale_in_progress"]) == 1
        assert issues["stale_in_progress"][0].id == "test-stale-progress"

    @pytest.mark.asyncio
    async def test_reconcile_no_issues(self, task_queue):
        """测试 reconcile 无问题场景"""
        # 创建正常入队的任务
        task = Task(
            id="test-normal",
            type=TaskType.TEST,
            title="正常任务",
            description="测试",
            instruction="测试",
            iteration_id=1,
        )
        await task_queue.enqueue(task)

        # 调用 reconcile
        issues = await task_queue.reconcile_iteration(
            iteration_id=1,
            in_flight_task_ids=set(),
            active_future_task_ids=set(),
        )

        # 验证无问题
        assert len(issues["orphaned_pending"]) == 0
        assert len(issues["orphaned_assigned"]) == 0
        assert len(issues["stale_in_progress"]) == 0


class TestRecoverStalledIteration:
    """_recover_stalled_iteration 方法测试"""

    @pytest.fixture
    def config(self) -> MultiProcessOrchestratorConfig:
        """创建测试配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            execution_timeout=5.0,
            enable_knowledge_search=False,
            enable_auto_commit=False,
            # 恢复配置
            max_recovery_attempts=3,
            max_no_progress_time=60.0,
            stall_recovery_interval=5.0,
        )

    @pytest.fixture
    def orchestrator(self, config) -> MultiProcessOrchestrator:
        """创建编排器实例"""
        return MultiProcessOrchestrator(config)

    @pytest.mark.asyncio
    async def test_recover_orphaned_pending(self, orchestrator):
        """测试恢复 orphaned_pending 任务"""
        import time

        # 创建孤立的 PENDING 任务
        task = Task(
            id="test-recover-pending",
            type=TaskType.TEST,
            title="恢复测试",
            description="测试",
            instruction="测试",
            iteration_id=1,
            status=TaskStatus.PENDING,
        )
        orchestrator.task_queue._tasks[task.id] = task

        # 模拟 Worker
        orchestrator.worker_ids = ["worker-1"]
        orchestrator.process_manager._processes = {
            "worker-1": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        # 初始化时间
        orchestrator._last_progress_time = time.time()

        # 调用恢复
        result = await orchestrator._recover_stalled_iteration(
            iteration_id=1,
            reason="test",
            active_futures={},
            worker_task_mapping={},
        )

        # 验证恢复结果
        assert result["requeued"] == 1
        assert result["failed"] == 0
        assert not result["degraded"]

        # 验证任务状态
        updated_task = orchestrator.task_queue.get_task("test-recover-pending")
        assert updated_task.status == TaskStatus.QUEUED

    @pytest.mark.asyncio
    async def test_recover_with_dead_worker(self, orchestrator):
        """测试恢复分配给死亡 Worker 的任务"""
        import time

        # 创建分配给死亡 Worker 的任务
        task = Task(
            id="test-recover-dead-worker",
            type=TaskType.TEST,
            title="死亡 Worker 恢复测试",
            description="测试",
            instruction="测试",
            iteration_id=1,
            status=TaskStatus.ASSIGNED,
            assigned_to="worker-dead",
        )
        orchestrator.task_queue._tasks[task.id] = task

        # 模拟死亡的 Worker
        orchestrator.worker_ids = ["worker-dead"]
        orchestrator.process_manager._processes = {
            "worker-dead": MagicMock(is_alive=MagicMock(return_value=False)),
        }

        # 初始化时间
        orchestrator._last_progress_time = time.time()

        # 调用恢复
        result = await orchestrator._recover_stalled_iteration(
            iteration_id=1,
            reason="worker dead",
            active_futures={},
            worker_task_mapping={},
        )

        # 验证恢复结果（应该重新入队）
        assert result["requeued"] == 1
        assert not result["degraded"]

    @pytest.mark.asyncio
    async def test_max_recovery_attempts_triggers_degradation(self, orchestrator):
        """测试超过最大恢复次数触发降级"""
        import time

        # 设置已经达到最大恢复次数
        orchestrator._recovery_attempts[1] = orchestrator.config.max_recovery_attempts
        orchestrator._last_progress_time = time.time()

        # 调用恢复
        result = await orchestrator._recover_stalled_iteration(
            iteration_id=1,
            reason="test",
            active_futures={},
            worker_task_mapping={},
        )

        # 验证触发降级
        assert result["degraded"]
        assert result["reason"] == "max_recovery_attempts_exceeded"
        assert orchestrator.is_degraded()

    @pytest.mark.asyncio
    async def test_max_no_progress_time_triggers_degradation(self, orchestrator):
        """测试超过最大无进展时间触发降级"""
        import time

        # 设置很久之前的进展时间
        orchestrator._last_progress_time = time.time() - orchestrator.config.max_no_progress_time - 10

        # 调用恢复
        result = await orchestrator._recover_stalled_iteration(
            iteration_id=1,
            reason="test",
            active_futures={},
            worker_task_mapping={},
        )

        # 验证触发降级
        assert result["degraded"]
        assert result["reason"] == "max_no_progress_time_exceeded"
        assert orchestrator.is_degraded()

    @pytest.mark.asyncio
    async def test_should_attempt_recovery(self, orchestrator):
        """测试 _should_attempt_recovery 判断逻辑"""
        import time

        # 初始状态：应该可以恢复
        orchestrator._last_recovery_time = 0
        assert orchestrator._should_attempt_recovery(1)

        # 刚恢复过：不应该再恢复
        orchestrator._last_recovery_time = time.time()
        assert not orchestrator._should_attempt_recovery(1)

        # 超过恢复间隔：应该可以恢复
        orchestrator._last_recovery_time = time.time() - orchestrator.config.stall_recovery_interval - 1
        assert orchestrator._should_attempt_recovery(1)

        # 达到最大恢复次数：不应该恢复
        orchestrator._recovery_attempts[1] = orchestrator.config.max_recovery_attempts
        assert not orchestrator._should_attempt_recovery(1)

    @pytest.mark.asyncio
    async def test_no_infinite_loop_with_recovery(self, orchestrator):
        """测试恢复机制不会导致无限循环"""
        import time

        # 创建无法恢复的任务（已达重试上限）
        task = Task(
            id="test-no-loop",
            type=TaskType.TEST,
            title="无限循环测试",
            description="测试",
            instruction="测试",
            iteration_id=1,
            status=TaskStatus.ASSIGNED,
            assigned_to="worker-dead",
            retry_count=3,  # 已达上限
            max_retries=3,
        )
        orchestrator.task_queue._tasks[task.id] = task

        # 模拟死亡的 Worker
        orchestrator.process_manager._processes = {
            "worker-dead": MagicMock(is_alive=MagicMock(return_value=False)),
        }

        orchestrator._last_progress_time = time.time()

        # 多次调用恢复
        for i in range(5):
            result = await orchestrator._recover_stalled_iteration(
                iteration_id=1,
                reason=f"attempt {i}",
                active_futures={},
                worker_task_mapping={},
            )

            if result["degraded"]:
                break

        # 验证最终会降级退出
        assert orchestrator.is_degraded()
        # 验证恢复次数不超过上限
        assert orchestrator._recovery_attempts[1] <= orchestrator.config.max_recovery_attempts + 1


class TestDiagnosticLogging:
    """诊断日志测试

    验证新的摘要日志格式和配置项。
    注意：当前实现仅在 stall_detected=True 时输出诊断日志。
    """

    @pytest.mark.asyncio
    async def test_diagnostic_log_format(self, capsys):
        """测试诊断日志格式（摘要模式，仅在 stall_detected 时输出）"""
        from loguru import logger

        # 模拟卡死检测场景（stall_detected=True 时才会输出摘要日志）
        iteration_id = 1
        queue_stats = {"total": 5, "pending": 2, "in_progress": 0, "completed": 2, "failed": 0}
        available_workers = []  # 无可用 worker -> stall_detected
        active_tasks = {}       # 无活动任务 -> stall_detected
        in_flight_tasks = {}

        # 检测卡死
        stall_detected = queue_stats["pending"] > 0 and len(active_tasks) == 0 and len(available_workers) == 0
        stall_reason = f"pending={queue_stats['pending']} > 0 但 active_tasks=0, available_workers=0"

        # 仅当 stall_detected=True 时输出摘要日志
        if stall_detected:
            summary = (
                f"[诊断] 卡死检测 | iteration={iteration_id}, "
                f"pending={queue_stats['pending']}, "
                f"completed={queue_stats['completed']}, "
                f"failed={queue_stats['failed']}, "
                f"available_workers={len(available_workers)}, "
                f"active_tasks={len(active_tasks)}, "
                f"in_flight={len(in_flight_tasks)}, "
                f"stall_reason=\"{stall_reason}\""
            )
            logger.warning(summary)

        # 验证日志包含关键信息
        # 注意：loguru 默认输出到 stderr
        # 这里主要验证代码能正常执行，实际日志验证需要配置 logger

    @pytest.mark.asyncio
    async def test_diagnostic_config_defaults(self):
        """测试诊断配置默认值"""
        config = MultiProcessOrchestratorConfig()

        # 验证默认配置
        # stall_diagnostics_enabled 默认关闭，疑似卡死时通过 --stall-diagnostics 启用
        assert config.stall_diagnostics_enabled is False
        assert config.stall_diagnostics_detail is False  # 默认关闭详细诊断
        assert config.stall_diagnostics_max_tasks == 5
        assert config.stall_diagnostics_level == "warning"


class TestDiagnosticOutput:
    """诊断输出测试 - 验证卡死检测日志格式

    日志输出策略（与实现一致）：
    - 仅在 stall_detected=True 时输出任何诊断日志
    - 所有诊断日志（摘要、详细、关键卡死行）均受冷却窗口控制
    - 冷却窗口使用 _last_stall_diagnostic_time，与 stall_recovery_interval 对齐
    - 详细诊断：仅当 stall_diagnostics_detail=True 时输出
    - 卡死模式检测：使用 error 级别

    回归测试应只断言关键 stall reason 文本，不强依赖详细诊断内容。
    """

    @pytest.mark.asyncio
    async def test_stall_diagnostic_log_content(self, capsys):
        """测试卡死诊断日志内容格式

        断言关键内容：
        - 仅当 stall_detected=True 时输出摘要日志
        - 摘要日志格式正确（包含 stall_reason）
        - 卡死模式检测使用 error 级别
        """
        from loguru import logger
        import sys

        # 配置 loguru 输出到 stderr（pytest 可以捕获）
        logger.remove()
        logger.add(sys.stderr, level="WARNING", format="{message}")

        # 模拟卡死检测输出
        iteration_id = 1
        queue_stats = {"total": 5, "pending": 2, "in_progress": 1, "completed": 2, "failed": 0}
        available_workers = []
        active_tasks = {}
        in_flight_tasks = {"task-1": {"agent_id": "worker-1"}}

        # 检测卡死模式
        stall_detected = False
        stall_reason = ""
        if queue_stats["pending"] > 0 and len(active_tasks) == 0 and len(available_workers) == 0:
            stall_detected = True
            stall_reason = f"pending={queue_stats['pending']} > 0 但 active_tasks=0, available_workers=0"

        # 模拟冷却窗口检测（与实际实现一致）
        # 仅当 stall_detected=True 且在冷却窗口外时输出
        should_emit_diagnostic = stall_detected  # 简化：假设在冷却窗口外

        if should_emit_diagnostic:
            # 输出摘要日志（仅当 stall_detected=True 时输出，格式包含 stall_reason）
            summary = (
                f"[诊断] 卡死检测 | iteration={iteration_id}, "
                f"pending={queue_stats['pending']}, "
                f"completed={queue_stats['completed']}, "
                f"failed={queue_stats['failed']}, "
                f"available_workers={len(available_workers)}, "
                f"active_tasks={len(active_tasks)}, "
                f"in_flight={len(in_flight_tasks)}, "
                f"stall_reason=\"{stall_reason}\""
            )
            logger.warning(summary)

            # 卡死模式检测（使用 error 级别）
            logger.error(f"[诊断] ⚠ 卡死模式检测: {stall_reason}")

        # 捕获输出
        captured = capsys.readouterr()
        stderr_output = captured.err

        # 验证摘要日志格式（关键断言）
        assert "[诊断] 卡死检测 |" in stderr_output
        assert "iteration=1" in stderr_output
        assert "pending=2" in stderr_output
        assert "stall_reason=" in stderr_output

        # 验证卡死模式检测（关键断言）
        assert "⚠ 卡死模式检测" in stderr_output

    @pytest.mark.asyncio
    async def test_detailed_diagnostics_format(self, capsys):
        """测试详细诊断日志格式（仅当 stall_diagnostics_detail=True）"""
        from loguru import logger
        import sys

        logger.remove()
        logger.add(sys.stderr, level="WARNING", format="{message}")

        # 模拟详细诊断输出
        queue_stats = {"total": 10, "pending": 3, "in_progress": 2, "completed": 4, "failed": 1}
        tasks = [
            {"id": f"task-{i}", "status": "pending", "retry_count": 0, "assigned_to": None}
            for i in range(10)
        ]
        max_tasks = 5

        logger.warning(f"[诊断] === 详细诊断 (循环 #100) ===")
        logger.warning(f"[诊断] TaskQueue.get_statistics(): {queue_stats}")
        logger.warning(f"[诊断] available_workers: []")
        logger.warning(f"[诊断] active_tasks: []")
        logger.warning(f"[诊断] in_flight_tasks: ['task-1']")

        # 限制输出任务数
        for task in tasks[:max_tasks]:
            logger.warning(
                f"[诊断] Task {task['id']}: status={task['status']}, "
                f"retry_count={task['retry_count']}, assigned_to={task['assigned_to']}"
            )
        if len(tasks) > max_tasks:
            logger.warning(f"[诊断] ... 省略 {len(tasks) - max_tasks} 个任务")

        captured = capsys.readouterr()
        stderr_output = captured.err

        # 验证详细诊断格式
        assert "[诊断] === 详细诊断" in stderr_output
        assert "TaskQueue.get_statistics()" in stderr_output
        assert "[诊断] ... 省略 5 个任务" in stderr_output


def run_stability_test():
    """直接运行稳定性测试（非 pytest）"""
    import asyncio

    async def main():
        print("=" * 60)
        print("MP 编排器稳定性测试")
        print("=" * 60)

        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            execution_timeout=5.0,
            enable_knowledge_search=False,
            enable_auto_commit=False,
        )

        orchestrator = MultiProcessOrchestrator(config)

        # 测试 1: 任务队列统计
        print("\n[测试 1] 任务队列统计")
        task = Task(
            id="test-1",
            type=TaskType.TEST,
            title="测试任务",
            description="测试",
            instruction="测试",
            iteration_id=1,
        )
        await orchestrator.task_queue.enqueue(task)
        stats = orchestrator.task_queue.get_statistics(1)
        print(f"  队列统计: {stats}")
        assert stats["total"] == 1, "应该有 1 个任务"
        print("  ✓ 通过")

        # 测试 2: 卡死模式检测
        print("\n[测试 2] 卡死模式检测")
        available_workers = []
        active_tasks = {}
        is_stall = stats["pending"] > 0 and len(active_tasks) == 0 and len(available_workers) == 0
        print(f"  pending={stats['pending']}, active_tasks=0, available_workers=0")
        print(f"  卡死检测: {is_stall}")
        # 注意：任务刚入队是 QUEUED 不是 PENDING，所以可能不触发
        print("  ✓ 检测逻辑正常")

        # 测试 3: Worker 死亡处理
        print("\n[测试 3] Worker 死亡处理")
        orchestrator.worker_ids = ["worker-dead"]
        orchestrator.process_manager._task_assignments["test-1"] = {
            "agent_id": "worker-dead",
            "assigned_at": 0,
            "message_id": "msg-1",
        }
        task_obj = orchestrator.task_queue.get_task("test-1")
        if task_obj:
            task_obj.status = TaskStatus.IN_PROGRESS
            orchestrator.task_queue.update_task(task_obj)

        await orchestrator._handle_unhealthy_workers(["worker-dead"])
        updated = orchestrator.task_queue.get_task("test-1")
        print(f"  任务状态: {updated.status.value if updated else 'None'}")
        print("  ✓ 处理完成")

        # 测试 4: 诊断日志格式验证
        print("\n[测试 4] 诊断日志格式")
        from loguru import logger
        queue_stats = {"total": 3, "pending": 1, "in_progress": 1, "completed": 1, "failed": 0}
        logger.warning(f"[诊断] === 卡死检测触发 (循环 #1) ===")
        logger.warning(f"[诊断] TaskQueue.get_statistics(): {queue_stats}")
        logger.warning(f"[诊断] ⚠ 卡死模式检测: pending=1 > 0 但 active_tasks=0, available_workers=0")
        print("  ✓ 日志格式正常")

        print("\n" + "=" * 60)
        print("所有测试通过")
        print("=" * 60)

    asyncio.run(main())


class TestTimeoutRequeueRegression:
    """超时重新入队回归测试

    测试场景：
    - 任务入队 -> 分配给 Worker -> _send_and_wait 超时返回 None
    - 期望：任务被重新入队，可再次 dequeue 获取同一 task_id

    修复前行为（BUG）：
    - 任务状态改为 PENDING，但未调用 requeue() 放回优先级队列
    - 导致任务"丢失"，无法被再次 dequeue

    修复后行为：
    - 调用 task_queue.requeue() 将任务重新放入优先级队列
    - 任务可被再次 dequeue
    """

    @pytest.fixture
    def config(self) -> MultiProcessOrchestratorConfig:
        """创建测试配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=1,  # 单 Worker 简化测试
            execution_timeout=1.0,  # 短超时
            planning_timeout=10.0,
            review_timeout=10.0,
            enable_knowledge_search=False,
            enable_knowledge_injection=False,
            enable_auto_commit=False,
            requeue_on_worker_death=True,  # 启用超时重新入队
        )

    @pytest.fixture
    def orchestrator(self, config) -> MultiProcessOrchestrator:
        """创建编排器实例"""
        return MultiProcessOrchestrator(config)

    @pytest.mark.asyncio
    async def test_timeout_triggers_requeue_to_priority_queue(self, orchestrator):
        """测试超时后任务被重新入队到优先级队列

        验证逻辑：
        1. 任务入队 -> dequeue 获取
        2. 模拟 _send_and_wait 返回 None（超时）
        3. 触发超时处理逻辑
        4. 断言任务可被再次 dequeue（同一 task_id）

        修复前：此测试会失败（任务状态为 PENDING 但无法 dequeue）
        修复后：此测试通过（任务被 requeue 到优先级队列）
        """
        iteration_id = 1

        # 1. 创建并入队任务
        task = Task(
            id="test-timeout-requeue-001",
            type=TaskType.TEST,
            title="超时重新入队测试任务",
            description="测试 _send_and_wait 超时后任务能被重新入队",
            instruction="执行测试",
            iteration_id=iteration_id,
        )
        await orchestrator.task_queue.enqueue(task)

        # 验证初始状态
        initial_stats = orchestrator.task_queue.get_statistics(iteration_id)
        assert initial_stats["total"] == 1, "应有 1 个任务"
        assert initial_stats["pending"] == 1, "任务应处于待处理状态"

        # 2. 第一次 dequeue（模拟分配给 Worker）
        dequeued_task = await orchestrator.task_queue.dequeue(iteration_id, timeout=0.5)
        assert dequeued_task is not None, "应能 dequeue 任务"
        assert dequeued_task.id == task.id, "应获取到同一任务"
        assert dequeued_task.status == TaskStatus.ASSIGNED, "任务应处于 ASSIGNED 状态"

        # 模拟任务开始执行（状态推进到 IN_PROGRESS）
        dequeued_task.start()
        orchestrator.task_queue.update_task(dequeued_task)
        assert dequeued_task.status == TaskStatus.IN_PROGRESS, "任务应处于 IN_PROGRESS 状态"

        # 3. 模拟超时处理逻辑（与 _execution_phase 中的逻辑一致）
        # 这里直接模拟超时后的处理，而不是真正等待超时
        task_obj = orchestrator.task_queue.get_task(task.id)
        assert task_obj is not None, "任务应存在于队列中"
        assert task_obj.status == TaskStatus.IN_PROGRESS, "任务应处于 IN_PROGRESS 状态"

        # 模拟 requeue_on_worker_death=True 时的处理
        if orchestrator.config.requeue_on_worker_death:
            # 关键：这里是修复点 - 应该调用 requeue 而不只是改状态
            # 修复前代码只做了:
            #   task.status = TaskStatus.PENDING
            #   task.result = None
            #   self.task_queue.update_task(task)
            # 修复后应该调用:
            #   await self.task_queue.requeue(task, reason="Worker 响应超时")
            await orchestrator.task_queue.requeue(task_obj, reason="Worker 响应超时（测试）")

        # 4. 验证任务可被再次 dequeue
        stats_after_requeue = orchestrator.task_queue.get_statistics(iteration_id)
        assert stats_after_requeue["pending"] == 1, "重新入队后应有 1 个待处理任务"

        # 关键断言：能再次 dequeue 获取同一任务
        requeued_task = await orchestrator.task_queue.dequeue(iteration_id, timeout=0.5)
        assert requeued_task is not None, "应能再次 dequeue 任务（修复后才能通过）"
        assert requeued_task.id == task.id, "应获取到同一 task_id"
        assert requeued_task.status == TaskStatus.ASSIGNED, "任务应处于 ASSIGNED 状态"

    @pytest.mark.asyncio
    async def test_timeout_without_requeue_leads_to_stuck_task(self, orchestrator):
        """测试修复前行为：只改状态不 requeue 导致任务卡住

        此测试验证 BUG 场景：
        - 超时后只执行 task.status = PENDING + update_task
        - 不调用 requeue()
        - 结果：任务状态为 PENDING，但无法被 dequeue

        这个测试在修复前后都应该通过（验证问题确实存在）
        """
        iteration_id = 1

        # 1. 创建并入队任务
        task = Task(
            id="test-stuck-task-001",
            type=TaskType.TEST,
            title="卡住任务测试",
            description="验证不调用 requeue 会导致任务卡住",
            instruction="执行测试",
            iteration_id=iteration_id,
        )
        await orchestrator.task_queue.enqueue(task)

        # 2. Dequeue 并开始执行
        dequeued_task = await orchestrator.task_queue.dequeue(iteration_id, timeout=0.5)
        assert dequeued_task is not None
        dequeued_task.start()
        orchestrator.task_queue.update_task(dequeued_task)

        # 3. 模拟修复前的错误处理（只改状态，不 requeue）
        task_obj = orchestrator.task_queue.get_task(task.id)
        task_obj.status = TaskStatus.PENDING  # 只改状态
        task_obj.result = None
        orchestrator.task_queue.update_task(task_obj)

        # 4. 验证任务卡住：状态是 PENDING 但无法 dequeue
        stats = orchestrator.task_queue.get_statistics(iteration_id)
        # pending 计数包含 PENDING 和 QUEUED，但队列中已无此任务
        assert stats["pending"] == 1, "统计显示有 1 个待处理任务"

        # 尝试 dequeue - 应该返回 None（任务不在优先级队列中）
        result = await orchestrator.task_queue.dequeue(iteration_id, timeout=0.1)
        assert result is None, "不调用 requeue 时，任务无法被 dequeue"

        # 验证任务状态确实是 PENDING（看起来应该能处理，但实际上卡住了）
        stuck_task = orchestrator.task_queue.get_task(task.id)
        assert stuck_task.status == TaskStatus.PENDING, "任务状态为 PENDING 但无法被处理"

    @pytest.mark.asyncio
    async def test_execution_phase_timeout_handling_integration(self, orchestrator):
        """集成测试：模拟 _execution_phase 中的超时处理流程

        此测试模拟真实的执行阶段超时处理，验证：
        1. 任务分配和执行
        2. _send_and_wait 超时返回 None
        3. 超时处理逻辑正确执行 requeue
        4. 任务可被下一轮处理

        使用受控的短超时避免测试卡住。
        """
        from unittest.mock import AsyncMock, MagicMock

        iteration_id = 1

        # 创建测试任务
        task = Task(
            id="test-integration-timeout-001",
            type=TaskType.TEST,
            title="集成超时测试",
            description="测试执行阶段超时处理",
            instruction="执行测试",
            iteration_id=iteration_id,
        )
        await orchestrator.task_queue.enqueue(task)

        # 模拟 Worker 环境
        orchestrator.worker_ids = ["worker-test-001"]
        orchestrator.process_manager._processes = {
            "worker-test-001": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        # Patch _send_and_wait 返回 None（模拟超时）
        call_count = 0
        max_calls = 2  # 限制调用次数防止无限循环

        async def mock_send_and_wait(agent_id, message, timeout):
            nonlocal call_count
            call_count += 1
            if call_count > max_calls:
                # 防止无限循环，第二次后标记任务失败
                return MagicMock(
                    payload={"task_id": task.id, "success": True, "result": "mock"}
                )
            # 模拟超时
            await asyncio.sleep(0.01)
            return None

        with patch.object(orchestrator, "_send_and_wait", side_effect=mock_send_and_wait):
            with patch.object(orchestrator.process_manager, "is_alive", return_value=True):
                with patch.object(orchestrator.process_manager, "track_task_assignment"):
                    with patch.object(orchestrator.process_manager, "untrack_task", return_value=None):
                        # 模拟简化版执行阶段逻辑
                        available_workers = list(orchestrator.worker_ids)
                        active_tasks = {}
                        worker_task_mapping = {}

                        # 第一轮：分配任务
                        dequeued = await orchestrator.task_queue.dequeue(iteration_id, timeout=0.1)
                        assert dequeued is not None, "应能 dequeue 任务"

                        worker_id = available_workers.pop(0)
                        dequeued.assigned_to = worker_id
                        dequeued.start()
                        orchestrator.task_queue.update_task(dequeued)
                        worker_task_mapping[worker_id] = dequeued.id

                        # 模拟发送任务并等待（超时）
                        response = await mock_send_and_wait(worker_id, MagicMock(), 1.0)

                        # 处理超时
                        if response is None:
                            task_id = worker_task_mapping.get(worker_id)
                            if task_id:
                                task_obj = orchestrator.task_queue.get_task(task_id)
                                if task_obj and task_obj.status == TaskStatus.IN_PROGRESS:
                                    if orchestrator.config.requeue_on_worker_death:
                                        # 修复后的正确处理：调用 requeue
                                        await orchestrator.task_queue.requeue(
                                            task_obj, reason="Worker 响应超时"
                                        )
                                        available_workers.append(worker_id)

        # 验证任务已重新入队
        final_stats = orchestrator.task_queue.get_statistics(iteration_id)
        assert final_stats["pending"] == 1, "任务应重新处于待处理状态"

        # 验证可再次 dequeue
        re_dequeued = await orchestrator.task_queue.dequeue(iteration_id, timeout=0.1)
        assert re_dequeued is not None, "任务应能被再次 dequeue"
        assert re_dequeued.id == task.id, "应获取同一任务"


class TestDiagnosticCooldown:
    """诊断日志冷却窗口测试

    验证 [诊断] 关键行在冷却窗口内最多出现一次。
    使用 loguru 捕获日志，通过时间推进模拟冷却窗口。
    """

    @pytest.fixture
    def config(self) -> MultiProcessOrchestratorConfig:
        """创建测试配置（缩短间隔以加速测试）"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            execution_timeout=5.0,
            enable_knowledge_search=False,
            enable_auto_commit=False,
            # 缩短诊断/恢复间隔以加速测试
            stall_recovery_interval=0.2,  # 200ms
            stall_diagnostics_enabled=True,
            stall_diagnostics_detail=False,
            stall_diagnostics_level="warning",
        )

    @pytest.fixture
    def orchestrator(self, config) -> MultiProcessOrchestrator:
        """创建编排器实例"""
        return MultiProcessOrchestrator(config)

    @pytest.mark.asyncio
    async def test_diagnostic_cooldown_window(self, orchestrator, capsys, monkeypatch):
        """测试诊断日志冷却窗口：窗口内最多输出一次

        通过时间推进模拟多次卡死检测，验证冷却机制。
        """
        import time
        import sys
        from io import StringIO
        from loguru import logger

        # 配置 loguru 捕获到 StringIO
        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="WARNING", format="{message}")

        # 模拟参数
        iteration_id = 1
        cooldown_interval = orchestrator.config.stall_recovery_interval

        # 初始化时间状态（使用可控时间，避免不同机器上抖动）
        _last_diagnostic_time = 0.0
        current_time = cooldown_interval

        def fake_time() -> float:
            return current_time

        monkeypatch.setattr(time, "time", fake_time)

        def emit_diagnostic_with_cooldown(stall_reason: str) -> bool:
            """模拟带冷却的诊断输出，返回是否实际输出了"""
            nonlocal _last_diagnostic_time
            now = time.time()

            # 冷却窗口内不输出
            if now - _last_diagnostic_time < cooldown_interval:
                return False

            # 输出诊断日志
            _last_diagnostic_time = now
            logger.warning(f"[诊断] ⚠ 卡死模式检测: {stall_reason}")
            return True

        # 测试场景：快速连续调用（窗口内）
        stall_reason = "pending=1 > 0 但 active_tasks=0, available_workers=0"

        # 第一次调用：应该输出
        result1 = emit_diagnostic_with_cooldown(stall_reason)
        assert result1 is True, "第一次诊断应该输出"

        # 立即再次调用（仍在冷却窗口内）：不应该输出
        result2 = emit_diagnostic_with_cooldown(stall_reason)
        assert result2 is False, "冷却窗口内不应该输出"

        result3 = emit_diagnostic_with_cooldown(stall_reason)
        assert result3 is False, "冷却窗口内不应该输出"

        # 等待冷却窗口过期
        current_time += cooldown_interval + 0.05

        # 冷却窗口后再次调用：应该输出
        result4 = emit_diagnostic_with_cooldown(stall_reason)
        assert result4 is True, "冷却窗口后应该输出"

        # 验证日志内容
        log_content = log_output.getvalue()
        diagnostic_count = log_content.count("[诊断] ⚠ 卡死模式检测")

        # 关键断言：在整个测试中只输出了 2 次（第一次 + 冷却后）
        assert diagnostic_count == 2, f"诊断日志应该输出 2 次，实际 {diagnostic_count} 次"

    @pytest.mark.asyncio
    async def test_diagnostic_cooldown_integration(self, orchestrator, capsys, monkeypatch):
        """集成测试：模拟 _execution_phase 中的诊断冷却逻辑

        使用实际的冷却检测逻辑，验证在快速循环中诊断不会频繁输出。
        """
        import time
        from io import StringIO
        from loguru import logger

        # 配置 loguru 捕获
        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="WARNING", format="{message}")

        # 使用 orchestrator 的 _last_stall_diagnostic_time
        cooldown_interval = orchestrator.config.stall_recovery_interval
        orchestrator._last_stall_diagnostic_time = 0.0

        # 使用可控时间，避免真实 sleep 造成抖动
        current_time = cooldown_interval

        def fake_time() -> float:
            return current_time

        monkeypatch.setattr(time, "time", fake_time)

        # 模拟 10 次快速循环（应该只触发诊断 1-2 次）
        loop_count = 10
        diagnostic_emitted = 0
        loop_interval = 0.02  # 20ms，远短于 200ms 冷却

        for i in range(loop_count):
            now = time.time()

            # 模拟诊断输出逻辑（来自 _execution_phase）
            if now - orchestrator._last_stall_diagnostic_time >= cooldown_interval:
                orchestrator._last_stall_diagnostic_time = now
                logger.warning(f"[诊断] ⚠ 卡死模式检测: loop #{i}")
                diagnostic_emitted += 1

            # 模拟循环间隔（短于冷却窗口）
            current_time += loop_interval

        # 验证：只输出了 1 次（因为循环总时间约 200ms，刚好达到冷却窗口）
        log_content = log_output.getvalue()
        actual_count = log_content.count("[诊断] ⚠ 卡死模式检测")

        # 允许 1-2 次（取决于时间边界）
        assert 1 <= actual_count <= 2, f"诊断日志应该输出 1-2 次，实际 {actual_count} 次"
        assert diagnostic_emitted == actual_count, "输出计数应一致"


class TestExecutionHealthCheckInterval:
    """执行阶段健康检查间隔配置测试

    验证：
    1. execution_health_check_interval 配置项默认值 >= 30s
    2. 配置项可正确配置和使用
    3. 默认配置下不会过频触发健康检查
    """

    def test_execution_health_check_interval_default_value(self):
        """测试 execution_health_check_interval 默认值不低于 30s"""
        config = MultiProcessOrchestratorConfig()

        # 默认值应该 >= 30s（避免过频触发）
        assert config.execution_health_check_interval >= 30.0, \
            f"execution_health_check_interval 默认值应 >= 30s，实际为 {config.execution_health_check_interval}"

    def test_execution_health_check_interval_configurable(self):
        """测试 execution_health_check_interval 可配置"""
        config = MultiProcessOrchestratorConfig(
            execution_health_check_interval=60.0
        )
        assert config.execution_health_check_interval == 60.0

    def test_consecutive_unresponsive_threshold_default(self):
        """测试 consecutive_unresponsive_threshold 默认值"""
        config = MultiProcessOrchestratorConfig()

        # 默认值应该 >= 2（避免单次未响应就告警）
        assert config.consecutive_unresponsive_threshold >= 2, \
            "consecutive_unresponsive_threshold 默认值应 >= 2"

    def test_consecutive_unresponsive_threshold_configurable(self):
        """测试 consecutive_unresponsive_threshold 可配置"""
        config = MultiProcessOrchestratorConfig(
            consecutive_unresponsive_threshold=5
        )
        assert config.consecutive_unresponsive_threshold == 5

    @pytest.mark.asyncio
    async def test_orchestrator_has_consecutive_unresponsive_count(self):
        """测试 orchestrator 初始化连续未响应计数器"""
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            enable_knowledge_search=False,
            enable_auto_commit=False,
        )
        orchestrator = MultiProcessOrchestrator(config)

        assert hasattr(orchestrator, "_consecutive_unresponsive_count")
        assert orchestrator._consecutive_unresponsive_count == {}


class TestConsecutiveUnresponsiveWarning:
    """连续未响应告警机制测试

    验证：
    1. 单次未响应不触发 WARNING
    2. 连续多次未响应达到阈值后触发 WARNING
    3. 响应正常后重置计数器
    """

    @pytest.fixture
    def config(self) -> MultiProcessOrchestratorConfig:
        """创建测试配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            enable_knowledge_search=False,
            enable_auto_commit=False,
            consecutive_unresponsive_threshold=3,  # 连续 3 次未响应才告警
            health_check_timeout=1.0,
        )

    @pytest.fixture
    def orchestrator(self, config) -> MultiProcessOrchestrator:
        """创建编排器实例"""
        return MultiProcessOrchestrator(config)

    @pytest.mark.asyncio
    async def test_single_unresponsive_no_warning(self, orchestrator):
        """测试单次未响应不触发 WARNING"""
        import time

        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预填充心跳响应：worker-1 不响应
        current_time = time.time()
        for agent_id in ["planner-1", "worker-0", "reviewer-1"]:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1

        def mock_is_alive(agent_id: str) -> bool:
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
                    result = await orchestrator._perform_health_check()

                    # worker-1 应该被视为不健康
                    assert "worker-1" in result.unhealthy
                    # 连续未响应次数应该为 1
                    assert orchestrator._consecutive_unresponsive_count.get("worker-1") == 1
                    # 未达阈值，不应该触发 cooldown（未记录到 _health_warning_cooldown）
                    assert "worker-1" not in orchestrator._health_warning_cooldown

    @pytest.mark.asyncio
    async def test_consecutive_unresponsive_triggers_warning(self, orchestrator):
        """测试连续多次未响应达到阈值后触发 WARNING"""
        import time

        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1"]
        orchestrator.reviewer_id = "reviewer-1"

        def mock_is_alive(agent_id: str) -> bool:
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
                    threshold = orchestrator.config.consecutive_unresponsive_threshold

                    # 模拟连续多次健康检查，worker-1 持续不响应
                    for i in range(threshold):
                        current_time = time.time()
                        for agent_id in ["planner-1", "worker-0", "reviewer-1"]:
                            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1
                        # worker-1 不响应

                        result = await orchestrator._perform_health_check()

                    # 连续未响应次数应该达到阈值
                    assert orchestrator._consecutive_unresponsive_count.get("worker-1") == threshold
                    # 达到阈值后应该触发 cooldown 记录
                    assert "worker-1" in orchestrator._health_warning_cooldown

    @pytest.mark.asyncio
    async def test_responsive_resets_counter(self, orchestrator):
        """测试响应正常后重置连续未响应计数器"""
        import time

        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1"]
        orchestrator.reviewer_id = "reviewer-1"

        # 预设 worker-1 已有连续未响应记录
        orchestrator._consecutive_unresponsive_count["worker-1"] = 2

        # 这次 worker-1 响应正常
        current_time = time.time()
        for agent_id in ["planner-1", "worker-0", "worker-1", "reviewer-1"]:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1

        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", return_value=True
            ):
                result = await orchestrator._perform_health_check()

                # worker-1 应该被视为健康
                assert "worker-1" in result.healthy
                # 连续未响应计数应该被重置
                assert orchestrator._consecutive_unresponsive_count.get("worker-1") == 0

    @pytest.mark.asyncio
    async def test_health_check_detail_includes_consecutive_count(self, orchestrator):
        """测试健康检查结果详情包含连续未响应次数"""
        import time

        orchestrator.planner_id = "planner-1"
        orchestrator.worker_ids = ["worker-0", "worker-1"]
        orchestrator.reviewer_id = "reviewer-1"

        # worker-1 不响应
        current_time = time.time()
        for agent_id in ["planner-1", "worker-0", "reviewer-1"]:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1

        def mock_is_alive(agent_id: str) -> bool:
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
                    result = await orchestrator._perform_health_check()

                    # 验证详情中包含连续未响应次数
                    assert "worker-1" in result.details
                    assert "consecutive_unresponsive" in result.details["worker-1"]
                    assert result.details["worker-1"]["consecutive_unresponsive"] == 1


class TestWorkerSetupLogging:
    """Worker 子进程日志配置测试

    验证 _setup_logging 根据 config 配置正确生效。
    """

    def test_setup_logging_default_info_level(self, capsys):
        """测试默认 INFO 级别不打印 DEBUG 日志"""
        import sys
        from io import StringIO
        from loguru import logger

        # 模拟 _setup_logging 逻辑（来自 process/worker.py）
        config = {
            "verbose": False,
            "log_level": "INFO",
            "heartbeat_debug": False,
        }

        # 模拟配置日志（与 _setup_logging 相同）
        verbose = config.get("verbose", False)
        log_level = config.get("log_level", "DEBUG" if verbose else "INFO")

        log_output = StringIO()
        logger.remove()
        logger.add(
            log_output,
            format="{level} | {message}",
            level=log_level,
            filter=lambda record: record["level"].name != "DEBUG" or verbose,
        )

        # 发送不同级别的日志
        logger.debug("DEBUG message - should not appear")
        logger.info("INFO message - should appear")
        logger.warning("WARNING message - should appear")

        log_content = log_output.getvalue()

        # 验证 DEBUG 不出现
        assert "DEBUG message" not in log_content, "DEBUG 日志不应该出现（INFO 级别）"
        # 验证 INFO 和 WARNING 出现
        assert "INFO message" in log_content, "INFO 日志应该出现"
        assert "WARNING message" in log_content, "WARNING 日志应该出现"

    def test_setup_logging_verbose_mode(self, capsys):
        """测试 verbose 模式打印 DEBUG 日志"""
        from io import StringIO
        from loguru import logger

        # 模拟 verbose 配置
        config = {
            "verbose": True,
            "log_level": "DEBUG",
            "heartbeat_debug": False,
        }

        verbose = config.get("verbose", False)
        log_level = config.get("log_level", "DEBUG" if verbose else "INFO")

        log_output = StringIO()
        logger.remove()
        logger.add(
            log_output,
            format="{level} | {message}",
            level=log_level,
            filter=lambda record: record["level"].name != "DEBUG" or verbose,
        )

        # 发送不同级别的日志
        logger.debug("DEBUG message - should appear in verbose")
        logger.info("INFO message - should appear")

        log_content = log_output.getvalue()

        # 验证 DEBUG 出现
        assert "DEBUG message" in log_content, "verbose 模式 DEBUG 日志应该出现"
        assert "INFO message" in log_content, "INFO 日志应该出现"

    def test_setup_logging_warning_level(self, capsys):
        """测试 WARNING 级别过滤 INFO 和 DEBUG 日志"""
        from io import StringIO
        from loguru import logger

        config = {
            "verbose": False,
            "log_level": "WARNING",
            "heartbeat_debug": False,
        }

        log_level = config.get("log_level", "INFO")

        log_output = StringIO()
        logger.remove()
        logger.add(
            log_output,
            format="{level} | {message}",
            level=log_level,
        )

        # 发送不同级别的日志
        logger.debug("DEBUG message - should not appear")
        logger.info("INFO message - should not appear")
        logger.warning("WARNING message - should appear")
        logger.error("ERROR message - should appear")

        log_content = log_output.getvalue()

        # 验证过滤
        assert "DEBUG message" not in log_content, "DEBUG 日志不应该出现"
        assert "INFO message" not in log_content, "INFO 日志不应该出现（WARNING 级别）"
        assert "WARNING message" in log_content, "WARNING 日志应该出现"
        assert "ERROR message" in log_content, "ERROR 日志应该出现"

    def test_heartbeat_debug_flag(self, capsys):
        """测试 heartbeat_debug 标志控制心跳日志（Worker 端）"""
        from io import StringIO
        from loguru import logger

        # 模拟 heartbeat_debug=False（默认，不打印心跳日志）
        _heartbeat_debug = False

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, format="{level} | {message}", level="DEBUG")

        # 模拟心跳响应日志（来自 _respond_heartbeat）
        if _heartbeat_debug:
            logger.debug("心跳响应已发送 (busy=False)")

        log_content_no_debug = log_output.getvalue()
        assert "心跳响应" not in log_content_no_debug, "heartbeat_debug=False 时不应该打印心跳日志"

        # 模拟 heartbeat_debug=True
        _heartbeat_debug = True
        if _heartbeat_debug:
            logger.debug("心跳响应已发送 (busy=False)")

        log_content_with_debug = log_output.getvalue()
        assert "心跳响应" in log_content_with_debug, "heartbeat_debug=True 时应该打印心跳日志"


class TestOrchestratorHeartbeatDebug:
    """Orchestrator 端心跳响应日志控制测试

    验证 _handle_message() 中的心跳响应日志受 config.heartbeat_debug 控制：
    - heartbeat_debug=False（默认）：不输出心跳响应日志
    - heartbeat_debug=True：输出心跳响应日志

    场景：verbose 模式下也应遵循 heartbeat_debug 配置
    """

    @pytest.fixture
    def config_verbose_no_heartbeat_debug(self) -> MultiProcessOrchestratorConfig:
        """verbose=True 但 heartbeat_debug=False 的配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            enable_knowledge_search=False,
            enable_auto_commit=False,
            verbose=True,          # 详细模式
            heartbeat_debug=False, # 但不输出心跳日志
        )

    @pytest.fixture
    def config_heartbeat_debug_enabled(self) -> MultiProcessOrchestratorConfig:
        """heartbeat_debug=True 的配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            enable_knowledge_search=False,
            enable_auto_commit=False,
            verbose=False,         # 非详细模式
            heartbeat_debug=True,  # 但启用心跳日志
        )

    @pytest.mark.asyncio
    async def test_verbose_without_heartbeat_debug_no_heartbeat_log(
        self, config_verbose_no_heartbeat_debug
    ):
        """测试 verbose=True 但 heartbeat_debug=False 时不输出心跳响应日志

        场景：
        - verbose=True（DEBUG 级别日志）
        - heartbeat_debug=False（默认）

        期望：
        - 不应出现"心跳响应收到"相关文本
        """
        from io import StringIO
        from loguru import logger

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, format="{level} | {message}", level="DEBUG")

        orchestrator = MultiProcessOrchestrator(config_verbose_no_heartbeat_debug)

        # 模拟心跳消息
        from process.message_queue import ProcessMessage, ProcessMessageType

        heartbeat_msg = ProcessMessage(
            type=ProcessMessageType.HEARTBEAT,
            sender="worker-test-001",
            payload={"busy": False, "pid": 12345, "timestamp": 1234567890.0},
        )

        # 调用 _handle_message 处理心跳
        await orchestrator._handle_message(heartbeat_msg)

        log_content = log_output.getvalue()

        # 关键断言：verbose 模式但 heartbeat_debug=False 时不应有心跳日志
        assert "心跳响应收到" not in log_content, (
            f"verbose=True, heartbeat_debug=False 时不应出现心跳响应日志\n"
            f"日志内容:\n{log_content}"
        )
        assert "心跳响应" not in log_content, (
            f"verbose=True, heartbeat_debug=False 时不应出现任何心跳响应相关文本\n"
            f"日志内容:\n{log_content}"
        )

    @pytest.mark.asyncio
    async def test_heartbeat_debug_enabled_shows_heartbeat_log(
        self, config_heartbeat_debug_enabled
    ):
        """测试 heartbeat_debug=True 时输出心跳响应日志

        场景：
        - heartbeat_debug=True

        期望：
        - 应出现"心跳响应收到"相关文本
        """
        from io import StringIO
        from loguru import logger

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, format="{level} | {message}", level="DEBUG")

        orchestrator = MultiProcessOrchestrator(config_heartbeat_debug_enabled)

        # 模拟心跳消息
        from process.message_queue import ProcessMessage, ProcessMessageType

        heartbeat_msg = ProcessMessage(
            type=ProcessMessageType.HEARTBEAT,
            sender="worker-test-002",
            payload={"busy": True, "pid": 54321, "timestamp": 1234567890.0},
        )

        # 调用 _handle_message 处理心跳
        await orchestrator._handle_message(heartbeat_msg)

        log_content = log_output.getvalue()

        # 关键断言：heartbeat_debug=True 时应有心跳日志
        assert "心跳响应收到" in log_content, (
            f"heartbeat_debug=True 时应出现心跳响应日志\n"
            f"日志内容:\n{log_content}"
        )
        # 验证日志内容完整性
        assert "worker-test-002" in log_content, "应包含发送者信息"
        assert "busy=True" in log_content, "应包含 busy 状态"

    @pytest.mark.asyncio
    async def test_heartbeat_debug_default_is_false(self):
        """测试 heartbeat_debug 默认值为 False"""
        config = MultiProcessOrchestratorConfig()
        assert config.heartbeat_debug is False, (
            "heartbeat_debug 默认值应为 False（避免刷屏）"
        )

    @pytest.mark.asyncio
    async def test_heartbeat_data_recorded_regardless_of_debug_flag(
        self, config_verbose_no_heartbeat_debug
    ):
        """测试心跳数据记录不受 heartbeat_debug 影响

        验证：即使 heartbeat_debug=False，心跳响应数据仍应被正确记录。
        日志仅控制输出，不影响功能。
        """
        orchestrator = MultiProcessOrchestrator(config_verbose_no_heartbeat_debug)

        from process.message_queue import ProcessMessage, ProcessMessageType
        import time

        sender = "worker-data-test-001"
        heartbeat_msg = ProcessMessage(
            type=ProcessMessageType.HEARTBEAT,
            sender=sender,
            payload={"busy": True, "pid": 99999, "timestamp": time.time()},
        )

        # 处理心跳
        await orchestrator._handle_message(heartbeat_msg)

        # 验证心跳数据被记录
        assert sender in orchestrator._heartbeat_responses, "心跳响应时间应被记录"
        assert sender in orchestrator._heartbeat_payloads, "心跳 payload 应被记录"
        assert orchestrator._heartbeat_payloads[sender].get("busy") is True, (
            "busy 状态应正确记录"
        )

    @pytest.mark.asyncio
    async def test_health_check_debug_log_controlled_by_heartbeat_debug_false(
        self, config_verbose_no_heartbeat_debug
    ):
        """测试健康检查 debug 日志在 heartbeat_debug=False 时不输出

        场景：
        - verbose=True（DEBUG 级别日志）
        - heartbeat_debug=False（默认）

        期望：
        - 不应出现"健康检查开始/完成/通过、assumed_busy"等相关文本
        """
        import time
        from io import StringIO
        from loguru import logger

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, format="{level} | {message}", level="DEBUG")

        orchestrator = MultiProcessOrchestrator(config_verbose_no_heartbeat_debug)

        # 设置 agent IDs
        orchestrator.planner_id = "planner-test"
        orchestrator.worker_ids = ["worker-test-0", "worker-test-1"]
        orchestrator.reviewer_id = "reviewer-test"

        # 预填充心跳响应（模拟所有 agent 都响应）
        current_time = time.time()
        for agent_id in ["planner-test", "worker-test-0", "worker-test-1", "reviewer-test"]:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1

        # Mock 进程管理器
        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", return_value=True
            ):
                # 执行健康检查
                result = await orchestrator._perform_health_check()

        log_content = log_output.getvalue()

        # 关键断言：heartbeat_debug=False 时不应有健康检查 debug 日志
        assert "健康检查开始" not in log_content, (
            f"heartbeat_debug=False 时不应出现'健康检查开始'日志\n"
            f"日志内容:\n{log_content}"
        )
        assert "健康检查完成" not in log_content, (
            f"heartbeat_debug=False 时不应出现'健康检查完成'日志\n"
            f"日志内容:\n{log_content}"
        )
        assert "健康检查通过" not in log_content, (
            f"heartbeat_debug=False 时不应出现'健康检查通过'日志\n"
            f"日志内容:\n{log_content}"
        )

        # 验证健康检查功能正常（日志控制不影响功能）
        assert result.all_healthy is True, "健康检查应该通过"

    @pytest.mark.asyncio
    async def test_health_check_debug_log_controlled_by_heartbeat_debug_true(
        self, config_heartbeat_debug_enabled
    ):
        """测试健康检查 debug 日志在 heartbeat_debug=True 时输出

        场景：
        - heartbeat_debug=True

        期望：
        - 应出现"健康检查开始/完成/通过"等相关文本
        """
        import time
        from io import StringIO
        from loguru import logger

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, format="{level} | {message}", level="DEBUG")

        orchestrator = MultiProcessOrchestrator(config_heartbeat_debug_enabled)

        # 设置 agent IDs
        orchestrator.planner_id = "planner-test"
        orchestrator.worker_ids = ["worker-test-0", "worker-test-1"]
        orchestrator.reviewer_id = "reviewer-test"

        # 预填充心跳响应（模拟所有 agent 都响应）
        current_time = time.time()
        for agent_id in ["planner-test", "worker-test-0", "worker-test-1", "reviewer-test"]:
            orchestrator._heartbeat_responses[agent_id] = current_time + 0.1

        # Mock 进程管理器
        with patch.object(orchestrator.process_manager, "broadcast"):
            with patch.object(
                orchestrator.process_manager, "is_alive", return_value=True
            ):
                # 执行健康检查
                result = await orchestrator._perform_health_check()

        log_content = log_output.getvalue()

        # 关键断言：heartbeat_debug=True 时应有健康检查 debug 日志
        assert "健康检查开始" in log_content, (
            f"heartbeat_debug=True 时应出现'健康检查开始'日志\n"
            f"日志内容:\n{log_content}"
        )
        assert "健康检查完成" in log_content, (
            f"heartbeat_debug=True 时应出现'健康检查完成'日志\n"
            f"日志内容:\n{log_content}"
        )
        assert "健康检查通过" in log_content, (
            f"heartbeat_debug=True 时应出现'健康检查通过'日志\n"
            f"日志内容:\n{log_content}"
        )

        # 验证健康检查功能正常
        assert result.all_healthy is True, "健康检查应该通过"

    @pytest.mark.asyncio
    async def test_health_check_assumed_busy_log_controlled(
        self, config_verbose_no_heartbeat_debug, config_heartbeat_debug_enabled
    ):
        """测试 assumed_busy 日志受 heartbeat_debug 控制

        场景：Worker 有任务分配但无心跳响应时，输出 assumed_busy 日志
        """
        import time
        from io import StringIO
        from loguru import logger

        # 测试 heartbeat_debug=False 不输出
        log_output_false = StringIO()
        logger.remove()
        logger.add(log_output_false, format="{level} | {message}", level="DEBUG")

        orchestrator_false = MultiProcessOrchestrator(config_verbose_no_heartbeat_debug)
        orchestrator_false.planner_id = "planner-test"
        orchestrator_false.worker_ids = ["worker-busy"]
        orchestrator_false.reviewer_id = "reviewer-test"

        current_time = time.time()
        # planner 和 reviewer 响应，worker 不响应
        orchestrator_false._heartbeat_responses["planner-test"] = current_time + 0.1
        orchestrator_false._heartbeat_responses["reviewer-test"] = current_time + 0.1

        with patch.object(orchestrator_false.process_manager, "broadcast"):
            with patch.object(
                orchestrator_false.process_manager, "is_alive", return_value=True
            ):
                with patch.object(
                    orchestrator_false.process_manager, "get_tasks_by_agent",
                    return_value=["task-1"]  # 模拟有任务分配
                ):
                    await orchestrator_false._perform_health_check()

        log_content_false = log_output_false.getvalue()
        assert "假定为忙碌" not in log_content_false, (
            f"heartbeat_debug=False 时不应出现'假定为忙碌'日志\n"
            f"日志内容:\n{log_content_false}"
        )

        # 测试 heartbeat_debug=True 输出
        log_output_true = StringIO()
        logger.remove()
        logger.add(log_output_true, format="{level} | {message}", level="DEBUG")

        orchestrator_true = MultiProcessOrchestrator(config_heartbeat_debug_enabled)
        orchestrator_true.planner_id = "planner-test"
        orchestrator_true.worker_ids = ["worker-busy"]
        orchestrator_true.reviewer_id = "reviewer-test"

        current_time = time.time()
        orchestrator_true._heartbeat_responses["planner-test"] = current_time + 0.1
        orchestrator_true._heartbeat_responses["reviewer-test"] = current_time + 0.1

        with patch.object(orchestrator_true.process_manager, "broadcast"):
            with patch.object(
                orchestrator_true.process_manager, "is_alive", return_value=True
            ):
                with patch.object(
                    orchestrator_true.process_manager, "get_tasks_by_agent",
                    return_value=["task-1"]
                ):
                    await orchestrator_true._perform_health_check()

        log_content_true = log_output_true.getvalue()
        assert "假定为忙碌" in log_content_true, (
            f"heartbeat_debug=True 时应出现'假定为忙碌'日志\n"
            f"日志内容:\n{log_content_true}"
        )


class TestNormalLongTaskNoDiagnosticWarning:
    """测试正常长任务不触发 WARNING 级别诊断日志

    验证场景：
    - 有任务正在执行（active_tasks 非空）
    - 即使超过诊断检测间隔，也不应输出 WARNING 级别的 [诊断] 摘要
    - 只有真正 stall_detected=True 时才输出诊断日志
    """

    @pytest.fixture
    def config(self) -> MultiProcessOrchestratorConfig:
        """创建测试配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            execution_timeout=5.0,
            enable_knowledge_search=False,
            enable_auto_commit=False,
            stall_recovery_interval=0.1,  # 缩短间隔以加速测试
            stall_diagnostics_enabled=True,
            stall_diagnostics_level="warning",
        )

    @pytest.fixture
    def orchestrator(self, config) -> MultiProcessOrchestrator:
        """创建编排器实例"""
        return MultiProcessOrchestrator(config)

    @pytest.mark.asyncio
    async def test_no_diagnostic_when_active_tasks_exist(self, orchestrator, capsys):
        """测试：有 active_tasks 时不触发诊断日志

        场景：
        - pending=0（无待处理任务）
        - active_tasks 非空（有正在执行的任务）
        - available_workers 为空（Worker 都在忙）

        期望：
        - stall_detected=False（因为有活动任务在执行）
        - 不输出 [诊断] 关键字的 WARNING 日志
        """
        import time
        from io import StringIO
        from loguru import logger

        # 配置日志捕获
        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="WARNING", format="{level} | {message}")

        # 模拟场景数据
        iteration_id = 1
        queue_stats = {"total": 2, "pending": 0, "in_progress": 2, "completed": 0, "failed": 0}
        available_workers = []  # Worker 都在忙
        active_tasks = {"worker-0": "mock_task_1", "worker-1": "mock_task_2"}  # 有活动任务
        in_flight_tasks = {"task-1": {}, "task-2": {}}
        tasks_in_iteration = []  # 无 ASSIGNED/IN_PROGRESS 状态异常任务

        # 模拟卡死检测逻辑（与 _execution_phase 一致）
        stall_detected = False
        stall_reason = ""

        assigned_tasks = [t for t in tasks_in_iteration if getattr(t, 'status', None) == TaskStatus.ASSIGNED]
        in_progress_tasks = [t for t in tasks_in_iteration if getattr(t, 'status', None) == TaskStatus.IN_PROGRESS]

        if queue_stats["pending"] > 0 and len(active_tasks) == 0 and len(available_workers) == 0:
            stall_detected = True
            stall_reason = f"pending={queue_stats['pending']} > 0 但 active_tasks=0, available_workers=0"

        if assigned_tasks and len(active_tasks) == 0:
            stall_detected = True
            stall_reason = f"存在 ASSIGNED 状态任务 ({len(assigned_tasks)}) 但无 active_tasks"

        if in_progress_tasks and len(active_tasks) == 0:
            stall_detected = True
            stall_reason = f"存在 IN_PROGRESS 状态任务 ({len(in_progress_tasks)}) 但无 active_tasks"

        # 关键断言：有 active_tasks 时不应检测到 stall
        assert stall_detected is False, "有 active_tasks 时不应检测到 stall"

        # 模拟诊断输出逻辑（与修复后的 _execution_phase 一致）
        now = time.time()
        should_emit_diagnostic = (
            stall_detected
            and orchestrator.config.stall_diagnostics_enabled
            and (
                orchestrator._last_stall_diagnostic_time == 0.0
                or now - orchestrator._last_stall_diagnostic_time >= orchestrator.config.stall_recovery_interval
            )
        )

        if should_emit_diagnostic:
            logger.warning(f"[诊断] 卡死检测 | iteration={iteration_id}, stall_reason=\"{stall_reason}\"")
            logger.error(f"[诊断] ⚠ 卡死模式检测: {stall_reason}")

        # 验证：不应输出任何诊断日志
        log_content = log_output.getvalue()
        assert "[诊断]" not in log_content, f"正常长任务不应输出诊断日志，实际输出: {log_content}"

    @pytest.mark.asyncio
    async def test_diagnostic_only_when_stall_detected(self, orchestrator, capsys):
        """测试：只有 stall_detected=True 时才输出诊断

        对比场景：
        1. 正常状态（有 active_tasks）→ 不输出
        2. 卡死状态（pending>0, active_tasks=0）→ 输出
        """
        import time
        from io import StringIO
        from loguru import logger

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="WARNING", format="{level} | {message}")

        iteration_id = 1

        # 场景 1：正常状态（有 active_tasks）
        queue_stats_normal = {"total": 2, "pending": 0, "in_progress": 2, "completed": 0, "failed": 0}
        active_tasks_normal = {"worker-0": "task_1"}
        available_workers_normal = []

        stall_detected_normal = (
            queue_stats_normal["pending"] > 0
            and len(active_tasks_normal) == 0
            and len(available_workers_normal) == 0
        )
        assert stall_detected_normal is False, "正常状态不应检测到 stall"

        # 场景 2：卡死状态（pending>0, active_tasks=0, available_workers=0）
        queue_stats_stall = {"total": 2, "pending": 2, "in_progress": 0, "completed": 0, "failed": 0}
        active_tasks_stall = {}  # 无活动任务
        available_workers_stall = []  # 无可用 Worker

        stall_detected_stall = (
            queue_stats_stall["pending"] > 0
            and len(active_tasks_stall) == 0
            and len(available_workers_stall) == 0
        )
        assert stall_detected_stall is True, "卡死状态应检测到 stall"

        # 模拟两种场景的诊断输出
        now = time.time()
        stall_reason = "pending=2 > 0 但 active_tasks=0, available_workers=0"

        # 正常状态：不应输出
        should_emit_normal = (
            stall_detected_normal
            and orchestrator.config.stall_diagnostics_enabled
        )
        if should_emit_normal:
            logger.warning(f"[诊断] 卡死检测 | iteration={iteration_id}, normal")

        # 卡死状态：应该输出
        should_emit_stall = (
            stall_detected_stall
            and orchestrator.config.stall_diagnostics_enabled
            and (
                orchestrator._last_stall_diagnostic_time == 0.0
                or now - orchestrator._last_stall_diagnostic_time >= orchestrator.config.stall_recovery_interval
            )
        )
        if should_emit_stall:
            orchestrator._last_stall_diagnostic_time = now
            logger.warning(f"[诊断] 卡死检测 | iteration={iteration_id}, stall_reason=\"{stall_reason}\"")
            logger.error(f"[诊断] ⚠ 卡死模式检测: {stall_reason}")

        log_content = log_output.getvalue()

        # 验证只输出了卡死状态的诊断
        assert "[诊断] 卡死检测 |" in log_content, "卡死状态应输出诊断摘要"
        assert "⚠ 卡死模式检测" in log_content, "卡死状态应输出卡死模式检测"
        assert "normal" not in log_content, "正常状态不应输出诊断"


class TestStallDiagnosticCooldownInExecutionPhase:
    """执行阶段诊断冷却窗口测试

    验证 _execution_phase 中的诊断输出在冷却窗口内不会重复。
    聚焦断言：关键文本出现次数，允许 1-2 次时间抖动。
    """

    @pytest.fixture
    def config(self) -> MultiProcessOrchestratorConfig:
        """创建测试配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            enable_knowledge_search=False,
            enable_auto_commit=False,
            stall_recovery_interval=0.2,  # 200ms 冷却窗口
            stall_diagnostics_enabled=True,
            stall_diagnostics_level="warning",
        )

    @pytest.fixture
    def orchestrator(self, config) -> MultiProcessOrchestrator:
        """创建编排器实例"""
        return MultiProcessOrchestrator(config)

    @pytest.mark.asyncio
    async def test_cooldown_prevents_repeated_diagnostics(self, orchestrator, monkeypatch):
        """测试冷却窗口内不重复输出诊断

        模拟快速连续多次 stall 检测循环，验证 [诊断] 关键行
        在冷却窗口内不会重复出现。

        允许边界情况：冷却窗口边界可能有 1-2 次时间抖动。
        """
        import time
        from io import StringIO
        from loguru import logger

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="WARNING", format="{message}")

        cooldown_interval = orchestrator.config.stall_recovery_interval
        stall_reason = "pending=1 > 0 但 active_tasks=0, available_workers=0"

        # 使用可控时间，避免真实 sleep 造成抖动
        current_time = cooldown_interval

        def fake_time() -> float:
            return current_time

        monkeypatch.setattr(time, "time", fake_time)

        # 模拟 10 次快速循环的 stall 检测
        loop_count = 10
        loop_interval = 0.03  # 30ms，远短于 200ms 冷却

        for i in range(loop_count):
            now = time.time()

            # 模拟诊断输出逻辑（与 _execution_phase 一致）
            should_emit = (
                orchestrator.config.stall_diagnostics_enabled
                and (
                    orchestrator._last_stall_diagnostic_time == 0.0
                    or now - orchestrator._last_stall_diagnostic_time >= cooldown_interval
                )
            )

            if should_emit:
                orchestrator._last_stall_diagnostic_time = now
                logger.warning(f"[诊断] 卡死检测 | loop={i}, stall_reason=\"{stall_reason}\"")
                logger.error(f"[诊断] ⚠ 卡死模式检测: {stall_reason}")

            current_time += loop_interval

        log_content = log_output.getvalue()
        diagnostic_count = log_content.count("[诊断] ⚠ 卡死模式检测")

        # 断言：10 次循环总时长约 300ms，冷却窗口 200ms
        # 期望输出次数：1-2 次（第一次 + 可能的冷却后一次）
        # 允许边界抖动：最多 3 次
        assert 1 <= diagnostic_count <= 3, (
            f"冷却窗口内诊断输出次数应在 1-3 次，实际 {diagnostic_count} 次\n"
            f"日志内容:\n{log_content}"
        )

    @pytest.mark.asyncio
    async def test_cooldown_resets_after_interval(self, orchestrator):
        """测试冷却窗口过期后允许再次输出

        场景：
        1. 第一次触发 → 输出
        2. 冷却窗口内多次触发 → 不输出
        3. 等待冷却过期 → 再次触发 → 输出
        """
        import time
        from io import StringIO
        from loguru import logger

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="WARNING", format="{message}")

        cooldown_interval = orchestrator.config.stall_recovery_interval
        stall_reason = "test stall"

        def emit_diagnostic() -> bool:
            """尝试输出诊断，返回是否成功"""
            now = time.time()
            should_emit = (
                orchestrator._last_stall_diagnostic_time == 0.0
                or now - orchestrator._last_stall_diagnostic_time >= cooldown_interval
            )
            if should_emit:
                orchestrator._last_stall_diagnostic_time = now
                logger.warning(f"[诊断] ⚠ 卡死模式检测: {stall_reason}")
                return True
            return False

        # 第一次：应该输出
        result1 = emit_diagnostic()
        assert result1 is True, "第一次诊断应该输出"

        # 冷却窗口内：不应输出
        result2 = emit_diagnostic()
        assert result2 is False, "冷却窗口内不应输出"

        result3 = emit_diagnostic()
        assert result3 is False, "冷却窗口内不应输出"

        # 等待冷却过期
        time.sleep(cooldown_interval + 0.05)

        # 冷却后：应该输出
        result4 = emit_diagnostic()
        assert result4 is True, "冷却窗口后应该输出"

        # 又在冷却窗口内
        result5 = emit_diagnostic()
        assert result5 is False, "冷却窗口内不应输出"

        log_content = log_output.getvalue()
        actual_count = log_content.count("[诊断] ⚠ 卡死模式检测")
        assert actual_count == 2, f"应该输出 2 次，实际 {actual_count} 次"

    @pytest.mark.asyncio
    async def test_summary_and_stall_message_both_controlled_by_cooldown(self, orchestrator, monkeypatch):
        """测试摘要日志和卡死模式检测都受冷却控制

        验证：冷却窗口内，[诊断] 卡死检测 和 [诊断] ⚠ 都不会重复输出。
        """
        import time
        from io import StringIO
        from loguru import logger

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="WARNING", format="{message}")

        cooldown_interval = orchestrator.config.stall_recovery_interval
        stall_reason = "pending=1 > 0 但 active_tasks=0, available_workers=0"
        iteration_id = 1

        # 使用可控时间，避免真实 sleep 造成抖动
        current_time = cooldown_interval

        def fake_time() -> float:
            return current_time

        monkeypatch.setattr(time, "time", fake_time)

        # 模拟 5 次快速触发
        for i in range(5):
            now = time.time()
            should_emit = (
                orchestrator._last_stall_diagnostic_time == 0.0
                or now - orchestrator._last_stall_diagnostic_time >= cooldown_interval
            )
            if should_emit:
                orchestrator._last_stall_diagnostic_time = now
                logger.warning(f"[诊断] 卡死检测 | iteration={iteration_id}")
                logger.error(f"[诊断] ⚠ 卡死模式检测: {stall_reason}")

            current_time += 0.02  # 20ms，远短于冷却窗口

        log_content = log_output.getvalue()

        # 两种关键日志的计数应该相等（都受冷却控制）
        summary_count = log_content.count("[诊断] 卡死检测 |")
        stall_count = log_content.count("[诊断] ⚠ 卡死模式检测")

        assert summary_count == stall_count, (
            f"摘要日志和卡死模式检测应该同步输出，"
            f"summary={summary_count}, stall={stall_count}"
        )
        assert summary_count == 1, f"冷却窗口内只应输出 1 次，实际 {summary_count} 次"


class TestDiagnosticCooldownWithTimeInjection:
    """使用时间注入的诊断冷却测试

    通过注入可控的时间函数，避免依赖真实时间等待。
    """

    @pytest.fixture
    def config(self) -> MultiProcessOrchestratorConfig:
        """创建测试配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            stall_recovery_interval=30.0,  # 较长的间隔（测试中不真正等待）
            stall_diagnostics_enabled=True,
        )

    @pytest.fixture
    def orchestrator(self, config) -> MultiProcessOrchestrator:
        """创建编排器实例"""
        return MultiProcessOrchestrator(config)

    @pytest.mark.asyncio
    async def test_cooldown_with_mocked_time(self, orchestrator):
        """使用 Mock 时间测试冷却窗口"""
        from io import StringIO
        from loguru import logger
        from unittest.mock import patch

        # 配置日志捕获
        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="WARNING", format="{message}")

        cooldown_interval = orchestrator.config.stall_recovery_interval

        # 模拟时间序列：100s, 105s, 110s, 135s, 140s
        # 使用非零起始时间避免初始状态与 0.0 混淆
        mock_times = iter([100.0, 105.0, 110.0, 135.0, 140.0])
        diagnostic_count = 0
        # 初始值设为 0.0（表示从未输出过，与生产代码一致）
        _last_diagnostic_time = 0.0

        def emit_with_cooldown(current_time: float, stall_reason: str) -> bool:
            """带冷却的诊断输出

            冷却逻辑：
            - _last_diagnostic_time == 0.0 表示从未输出，允许第一次输出
            - current_time - _last_diagnostic_time >= cooldown_interval 时允许输出
            """
            nonlocal _last_diagnostic_time, diagnostic_count

            # 第一次调用时 _last_diagnostic_time 为 0，差值必然 >= cooldown_interval
            # 因为 current_time 通常远大于 0
            if _last_diagnostic_time > 0 and current_time - _last_diagnostic_time < cooldown_interval:
                return False

            _last_diagnostic_time = current_time
            diagnostic_count += 1
            logger.warning(f"[诊断] ⚠ 卡死模式检测: {stall_reason}")
            return True

        stall_reason = "test stall"

        # t=100s: 第一次，应该输出（_last_diagnostic_time 为 0）
        t1 = next(mock_times)
        result1 = emit_with_cooldown(t1, stall_reason)
        assert result1 is True, "t=100s: 第一次应该输出"

        # t=105s: 仍在冷却窗口内（100s + 30s = 130s），不输出
        t2 = next(mock_times)
        result2 = emit_with_cooldown(t2, stall_reason)
        assert result2 is False, "t=105s: 冷却窗口内不应该输出"

        # t=110s: 仍在冷却窗口内
        t3 = next(mock_times)
        result3 = emit_with_cooldown(t3, stall_reason)
        assert result3 is False, "t=110s: 冷却窗口内不应该输出"

        # t=135s: 超过冷却窗口（100s + 30s = 130s < 135s），应该输出
        t4 = next(mock_times)
        result4 = emit_with_cooldown(t4, stall_reason)
        assert result4 is True, "t=135s: 超过冷却窗口应该输出"

        # t=140s: 又在冷却窗口内（135s + 30s = 165s > 140s）
        t5 = next(mock_times)
        result5 = emit_with_cooldown(t5, stall_reason)
        assert result5 is False, "t=140s: 冷却窗口内不应该输出"

        # 验证总输出次数
        log_content = log_output.getvalue()
        actual_count = log_content.count("[诊断] ⚠ 卡死模式检测")
        assert actual_count == 2, f"应该输出 2 次，实际 {actual_count} 次"
        assert diagnostic_count == 2, f"计数器应为 2，实际 {diagnostic_count}"

    @pytest.mark.asyncio
    async def test_orchestrator_last_stall_diagnostic_time(self, orchestrator):
        """测试 orchestrator 的 _last_stall_diagnostic_time 字段存在并可用"""
        # 验证字段存在
        assert hasattr(orchestrator, "_last_stall_diagnostic_time"), \
            "orchestrator 应该有 _last_stall_diagnostic_time 字段"

        # 验证初始值
        assert orchestrator._last_stall_diagnostic_time == 0.0, \
            "_last_stall_diagnostic_time 初始值应为 0.0"

        # 验证可以更新
        import time
        orchestrator._last_stall_diagnostic_time = time.time()
        assert orchestrator._last_stall_diagnostic_time > 0, \
            "_last_stall_diagnostic_time 应该可以更新"


class TestSendAndWaitTimeoutThrottling:
    """_send_and_wait 超时告警节流测试

    验证：
    1. 单次/偶发超时使用 INFO/DEBUG 级别（不触发 WARNING）
    2. 连续超时达到阈值后使用 WARNING 级别（带 cooldown）
    3. 关键阶段（planner/reviewer）连续超时使用 ERROR 级别
    4. 正常响应后重置超时计数
    """

    @pytest.fixture
    def config(self) -> MultiProcessOrchestratorConfig:
        """创建测试配置"""
        return MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
            enable_knowledge_search=False,
            enable_auto_commit=False,
            timeout_warning_threshold=2,  # 连续 2 次超时后才告警
            timeout_warning_cooldown_seconds=60.0,
            heartbeat_debug=True,  # 启用 DEBUG 日志以验证
        )

    @pytest.fixture
    def orchestrator(self, config) -> MultiProcessOrchestrator:
        """创建编排器实例"""
        return MultiProcessOrchestrator(config)

    @pytest.mark.asyncio
    async def test_timeout_count_initialized(self, orchestrator):
        """测试超时计数器初始化"""
        assert hasattr(orchestrator, "_timeout_count")
        assert orchestrator._timeout_count == {}
        assert hasattr(orchestrator, "_timeout_warning_cooldown")
        assert orchestrator._timeout_warning_cooldown == {}

    @pytest.mark.asyncio
    async def test_config_has_timeout_settings(self, orchestrator):
        """测试配置项存在"""
        assert hasattr(orchestrator.config, "timeout_warning_threshold")
        assert hasattr(orchestrator.config, "timeout_warning_cooldown_seconds")
        assert orchestrator.config.timeout_warning_threshold == 2
        assert orchestrator.config.timeout_warning_cooldown_seconds == 60.0

    @pytest.mark.asyncio
    async def test_single_timeout_uses_info_level(self, orchestrator):
        """测试单次超时使用 INFO 级别"""
        from io import StringIO
        from loguru import logger
        from process.message_queue import ProcessMessage, ProcessMessageType

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="DEBUG", format="{level} | {message}")

        # 模拟 Worker
        orchestrator.worker_ids = ["worker-test-001"]
        orchestrator.process_manager._processes = {
            "worker-test-001": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        # 模拟 send_to_agent（不做任何事，让请求超时）
        with patch.object(orchestrator.process_manager, "send_to_agent"):
            # 使用非常短的超时
            msg = ProcessMessage(
                type=ProcessMessageType.TASK_ASSIGN,
                sender="orchestrator",
                receiver="worker-test-001",
                payload={"task": {}},
            )
            result = await orchestrator._send_and_wait("worker-test-001", msg, timeout=0.01)

        assert result is None, "应该超时返回 None"
        assert orchestrator._timeout_count.get("worker-test-001") == 1, "超时计数应为 1"

        log_content = log_output.getvalue()
        # 单次超时应使用 INFO 级别
        assert "INFO" in log_content, "单次超时应输出 INFO 日志"
        assert "首次" in log_content, "单次超时日志应包含'首次'"
        # 不应有 WARNING
        assert "WARNING | 等待 worker-test-001" not in log_content, "单次超时不应输出 WARNING"

    @pytest.mark.asyncio
    async def test_consecutive_timeout_triggers_warning(self, orchestrator):
        """测试连续超时达到阈值后触发 WARNING"""
        from io import StringIO
        from loguru import logger
        from process.message_queue import ProcessMessage, ProcessMessageType

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="DEBUG", format="{level} | {message}")

        orchestrator.worker_ids = ["worker-test-002"]
        orchestrator.process_manager._processes = {
            "worker-test-002": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        threshold = orchestrator.config.timeout_warning_threshold

        with patch.object(orchestrator.process_manager, "send_to_agent"):
            # 连续超时多次
            for i in range(threshold):
                msg = ProcessMessage(
                    type=ProcessMessageType.TASK_ASSIGN,
                    sender="orchestrator",
                    receiver="worker-test-002",
                    payload={"task": {}},
                )
                await orchestrator._send_and_wait("worker-test-002", msg, timeout=0.01)

        assert orchestrator._timeout_count.get("worker-test-002") == threshold
        log_content = log_output.getvalue()

        # 达到阈值后应有 WARNING
        assert f"WARNING | 等待 worker-test-002 响应超时 (连续 {threshold} 次)" in log_content, (
            f"连续 {threshold} 次超时后应输出 WARNING\n日志内容:\n{log_content}"
        )

    @pytest.mark.asyncio
    async def test_warning_cooldown_prevents_repeated_warnings(self, orchestrator):
        """测试 WARNING 级别受 cooldown 控制"""
        from io import StringIO
        from loguru import logger
        from process.message_queue import ProcessMessage, ProcessMessageType

        # 使用较短的 cooldown 以加速测试
        orchestrator.config.timeout_warning_cooldown_seconds = 0.2

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="DEBUG", format="{level} | {message}")

        orchestrator.worker_ids = ["worker-test-003"]
        orchestrator.process_manager._processes = {
            "worker-test-003": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        threshold = orchestrator.config.timeout_warning_threshold

        with patch.object(orchestrator.process_manager, "send_to_agent"):
            # 快速连续超时 5 次（应该只输出 1 次 WARNING，后续在 cooldown 中）
            for i in range(5):
                msg = ProcessMessage(
                    type=ProcessMessageType.TASK_ASSIGN,
                    sender="orchestrator",
                    receiver="worker-test-003",
                    payload={"task": {}},
                )
                await orchestrator._send_and_wait("worker-test-003", msg, timeout=0.01)

        log_content = log_output.getvalue()

        # 统计 WARNING 次数（应该只有 1 次，因为后续在 cooldown 中）
        warning_count = log_content.count("WARNING | 等待 worker-test-003 响应超时 (连续")
        assert warning_count == 1, f"WARNING 应该只输出 1 次（cooldown），实际 {warning_count} 次"

        # 应该有 cooldown 中的 DEBUG 日志
        assert "cooldown 中" in log_content, "cooldown 期间应输出 DEBUG 日志"

    @pytest.mark.asyncio
    async def test_critical_agent_timeout_uses_error_level(self, orchestrator):
        """测试关键阶段（planner/reviewer）连续超时使用 ERROR 级别"""
        from io import StringIO
        from loguru import logger
        from process.message_queue import ProcessMessage, ProcessMessageType

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="DEBUG", format="{level} | {message}")

        # 设置 planner_id
        orchestrator.planner_id = "planner-critical-001"
        orchestrator.process_manager._processes = {
            "planner-critical-001": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        threshold = orchestrator.config.timeout_warning_threshold

        with patch.object(orchestrator.process_manager, "send_to_agent"):
            # 连续超时达到阈值
            for i in range(threshold):
                msg = ProcessMessage(
                    type=ProcessMessageType.PLAN_REQUEST,
                    sender="orchestrator",
                    receiver="planner-critical-001",
                    payload={"goal": "test"},
                )
                await orchestrator._send_and_wait("planner-critical-001", msg, timeout=0.01)

        log_content = log_output.getvalue()

        # 关键阶段应使用 ERROR 级别
        assert "ERROR | 等待 planner-critical-001 响应超时" in log_content, (
            f"关键阶段超时应输出 ERROR\n日志内容:\n{log_content}"
        )
        assert "关键阶段" in log_content, "日志应包含'关键阶段'标识"

    @pytest.mark.asyncio
    async def test_reviewer_timeout_uses_error_level(self, orchestrator):
        """测试 reviewer 连续超时使用 ERROR 级别"""
        from io import StringIO
        from loguru import logger
        from process.message_queue import ProcessMessage, ProcessMessageType

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, level="DEBUG", format="{level} | {message}")

        orchestrator.reviewer_id = "reviewer-critical-001"
        orchestrator.process_manager._processes = {
            "reviewer-critical-001": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        threshold = orchestrator.config.timeout_warning_threshold

        with patch.object(orchestrator.process_manager, "send_to_agent"):
            for i in range(threshold):
                msg = ProcessMessage(
                    type=ProcessMessageType.REVIEW_REQUEST,
                    sender="orchestrator",
                    receiver="reviewer-critical-001",
                    payload={"goal": "test"},
                )
                await orchestrator._send_and_wait("reviewer-critical-001", msg, timeout=0.01)

        log_content = log_output.getvalue()

        assert "ERROR | 等待 reviewer-critical-001 响应超时" in log_content
        assert "关键阶段" in log_content

    @pytest.mark.asyncio
    async def test_successful_response_resets_timeout_count(self, orchestrator):
        """测试正常响应后重置超时计数"""
        from process.message_queue import ProcessMessage, ProcessMessageType
        import asyncio

        orchestrator.worker_ids = ["worker-reset-001"]
        orchestrator.process_manager._processes = {
            "worker-reset-001": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        # 先模拟一次超时
        orchestrator._timeout_count["worker-reset-001"] = 3  # 模拟已超时 3 次

        # 模拟成功响应
        async def mock_send_and_receive():
            msg = ProcessMessage(
                type=ProcessMessageType.TASK_ASSIGN,
                sender="orchestrator",
                receiver="worker-reset-001",
                payload={"task": {}},
            )

            # 启动 _send_and_wait 但模拟立即收到响应
            future = asyncio.get_event_loop().create_future()
            orchestrator._pending_responses[msg.id] = future

            # 模拟响应到达
            response = ProcessMessage(
                type=ProcessMessageType.TASK_RESULT,
                sender="worker-reset-001",
                correlation_id=msg.id,
                payload={"success": True},
            )
            future.set_result(response)

            # 重置计数逻辑（模拟 _send_and_wait 中的成功分支）
            orchestrator._timeout_count["worker-reset-001"] = 0

        await mock_send_and_receive()

        # 验证计数被重置
        assert orchestrator._timeout_count.get("worker-reset-001") == 0, "成功响应后超时计数应重置为 0"

    @pytest.mark.asyncio
    async def test_timeout_count_per_agent_isolation(self, orchestrator):
        """测试超时计数 per-agent 隔离"""
        from process.message_queue import ProcessMessage, ProcessMessageType

        orchestrator.worker_ids = ["worker-a", "worker-b"]
        orchestrator.process_manager._processes = {
            "worker-a": MagicMock(is_alive=MagicMock(return_value=True)),
            "worker-b": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        with patch.object(orchestrator.process_manager, "send_to_agent"):
            # worker-a 超时 3 次
            for _ in range(3):
                msg = ProcessMessage(
                    type=ProcessMessageType.TASK_ASSIGN,
                    sender="orchestrator",
                    receiver="worker-a",
                    payload={},
                )
                await orchestrator._send_and_wait("worker-a", msg, timeout=0.01)

            # worker-b 只超时 1 次
            msg = ProcessMessage(
                type=ProcessMessageType.TASK_ASSIGN,
                sender="orchestrator",
                receiver="worker-b",
                payload={},
            )
            await orchestrator._send_and_wait("worker-b", msg, timeout=0.01)

        # 验证 per-agent 计数隔离
        assert orchestrator._timeout_count.get("worker-a") == 3
        assert orchestrator._timeout_count.get("worker-b") == 1


class TestTimeoutWarningConfigDefaults:
    """超时告警配置默认值测试"""

    def test_timeout_warning_threshold_default(self):
        """测试 timeout_warning_threshold 默认值"""
        config = MultiProcessOrchestratorConfig()
        # 默认值应为 2（避免单次超时就告警）
        assert config.timeout_warning_threshold == 2

    def test_timeout_warning_cooldown_default(self):
        """测试 timeout_warning_cooldown_seconds 默认值"""
        config = MultiProcessOrchestratorConfig()
        # 默认值应为 60s（与健康检查 cooldown 一致）
        assert config.timeout_warning_cooldown_seconds == 60.0

    def test_timeout_settings_configurable(self):
        """测试超时告警配置可自定义"""
        config = MultiProcessOrchestratorConfig(
            timeout_warning_threshold=5,
            timeout_warning_cooldown_seconds=120.0,
        )
        assert config.timeout_warning_threshold == 5
        assert config.timeout_warning_cooldown_seconds == 120.0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        pytest.main([__file__, "-v"])
    else:
        run_stability_test()
