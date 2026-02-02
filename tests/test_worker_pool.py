"""WorkerPool 测试

测试 WorkerPool 的初始化、任务分配、并发执行、worker 状态管理等功能。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.worker import WorkerConfig
from coordinator.worker_pool import WorkerPool
from tasks.queue import TaskQueue
from tasks.task import Task, TaskStatus, TaskType
from tests.conftest_e2e import (
    MockAgentExecutor,
    MockKnowledgeManager,
    assert_all_executions_completed,
    assert_execution_status_transitions,
    assert_execution_success_rate,
    assert_execution_trace_count,
    create_test_task,
)

# ==================== 测试辅助函数 ====================


def create_mock_worker(worker_id: str, execute_side_effect=None):
    """创建一个配置完整的 mock worker

    Args:
        worker_id: worker 标识符
        execute_side_effect: execute_task 的 side_effect 函数

    Returns:
        配置好的 MagicMock worker
    """
    mock_worker = MagicMock()
    mock_worker.id = worker_id
    if execute_side_effect is None:
        execute_side_effect = lambda task: _complete_task(task)
    mock_worker.execute_task = AsyncMock(side_effect=execute_side_effect)
    mock_worker.reset = AsyncMock()
    mock_worker.completed_tasks = []
    mock_worker.get_statistics = MagicMock(
        return_value={
            "worker_id": worker_id,
            "status": "idle",
            "completed_tasks_count": 0,
        }
    )
    return mock_worker


async def wait_for_task_status(
    task_queue: TaskQueue,
    task_id: str,
    expected_status: TaskStatus,
    timeout: float = 2.0,
) -> bool:
    """等待任务达到预期状态

    Args:
        task_queue: 任务队列
        task_id: 任务 ID
        expected_status: 预期状态
        timeout: 超时时间

    Returns:
        是否在超时前达到预期状态
    """
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        task = task_queue.get_task(task_id)
        if task and task.status == expected_status:
            return True
        await asyncio.sleep(0.02)
    return False


async def safely_stop_pool(pool: WorkerPool, pool_task: asyncio.Task, timeout: float = 2.0):
    """安全停止 WorkerPool 并等待其任务完成

    Args:
        pool: WorkerPool 实例
        pool_task: pool.start() 创建的任务
        timeout: 等待超时时间
    """
    await pool.stop()
    try:
        await asyncio.wait_for(asyncio.shield(pool_task), timeout=timeout)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        # 取消或超时都是预期的行为
        pass
    except Exception:
        # 忽略其他异常，确保测试清理完成
        pass


# ==================== Fixtures ====================


@pytest.fixture
def worker_config():
    """创建测试用的 WorkerConfig"""
    return WorkerConfig(
        name="test-worker",
        working_directory="/tmp/test",
        task_timeout=60,
    )


@pytest.fixture
def mock_knowledge_manager_instance():
    """创建 MockKnowledgeManager 实例"""
    return MockKnowledgeManager()


@pytest.fixture
def task_queue():
    """创建任务队列实例"""
    return TaskQueue()


# ==================== 初始化测试 ====================


class TestWorkerPoolInit:
    """WorkerPool 初始化测试"""

    def test_init_with_default_config(self):
        """测试默认配置初始化"""
        pool = WorkerPool()

        assert pool.size == 3
        assert pool.workers == []
        assert pool._running is False
        assert pool._worker_tasks == []
        assert pool._knowledge_manager is None

    def test_init_with_custom_size(self):
        """测试自定义大小初始化"""
        pool = WorkerPool(size=5)

        assert pool.size == 5
        assert pool.workers == []

    def test_init_with_worker_config(self, worker_config):
        """测试使用 WorkerConfig 初始化"""
        pool = WorkerPool(size=2, worker_config=worker_config)

        assert pool.size == 2
        assert pool.worker_config == worker_config
        assert pool.worker_config.name == "test-worker"

    def test_init_with_knowledge_manager(self, mock_knowledge_manager_instance):
        """测试使用知识库管理器初始化"""
        pool = WorkerPool(
            size=2,
            knowledge_manager=mock_knowledge_manager_instance,
        )

        assert pool._knowledge_manager == mock_knowledge_manager_instance

    @patch("coordinator.worker_pool.WorkerAgent")
    def test_initialize_creates_workers(self, mock_worker_class, worker_config):
        """测试 initialize 方法创建 worker"""
        pool = WorkerPool(size=3, worker_config=worker_config)
        pool.initialize()

        assert len(pool.workers) == 3
        assert mock_worker_class.call_count == 3

    @patch("coordinator.worker_pool.WorkerAgent")
    def test_initialize_worker_naming(self, mock_worker_class, worker_config):
        """测试 worker 命名规则"""
        pool = WorkerPool(size=3, worker_config=worker_config)
        pool.initialize()

        # 验证每个 worker 被创建时使用了正确的配置
        call_args_list = mock_worker_class.call_args_list
        for i, call_args in enumerate(call_args_list):
            config = call_args[0][0]  # 第一个位置参数
            assert config.name == f"worker-{i}"


class TestWorkerPoolForceWriteConfig:
    """WorkerPool force_write 配置测试

    验证：
    1. WorkerPool 初始化后每个 worker 的 cursor_config.force_write=True
    2. 每个 worker 的 cursor_config.stream_agent_id 唯一
    """

    def test_workers_force_write_enabled(self):
        """测试 WorkerPool 中所有 Worker 的 force_write=True"""
        from cursor.client import CursorAgentConfig

        # 创建 WorkerConfig，确保 force_write=True
        cursor_config = CursorAgentConfig(force_write=True)
        worker_config = WorkerConfig(
            name="test-worker",
            working_directory="/tmp/test",
            cursor_config=cursor_config,
        )

        pool = WorkerPool(size=3, worker_config=worker_config)
        pool.initialize()

        # 验证每个 worker 的 cursor_config.force_write=True
        for worker in pool.workers:
            assert worker.worker_config.cursor_config.force_write is True, (
                f"Worker {worker.id} 的 force_write 应为 True"
            )

    def test_workers_stream_agent_id_unique(self):
        """测试 WorkerPool 中每个 Worker 的 stream_agent_id 唯一"""
        from cursor.client import CursorAgentConfig

        cursor_config = CursorAgentConfig(force_write=True)
        worker_config = WorkerConfig(
            name="test-worker",
            working_directory="/tmp/test",
            cursor_config=cursor_config,
        )

        pool = WorkerPool(size=5, worker_config=worker_config)
        pool.initialize()

        # 收集所有 worker 的 stream_agent_id
        stream_agent_ids = []
        for worker in pool.workers:
            stream_agent_id = worker.worker_config.cursor_config.stream_agent_id
            assert stream_agent_id is not None, f"Worker {worker.id} 的 stream_agent_id 不应为 None"
            stream_agent_ids.append(stream_agent_id)

        # 验证唯一性
        assert len(stream_agent_ids) == len(set(stream_agent_ids)), f"Worker stream_agent_id 不唯一: {stream_agent_ids}"

    def test_workers_inherit_cursor_config(self):
        """测试 WorkerPool 正确传递 cursor_config 到每个 Worker"""
        from cursor.client import CursorAgentConfig

        # 创建带有自定义配置的 cursor_config
        cursor_config = CursorAgentConfig(
            force_write=True,
            model="custom-model",
            timeout=600,
        )
        worker_config = WorkerConfig(
            name="test-worker",
            working_directory="/custom/path",
            cursor_config=cursor_config,
        )

        pool = WorkerPool(size=3, worker_config=worker_config)
        pool.initialize()

        # 验证每个 worker 继承了正确的配置
        for worker in pool.workers:
            assert worker.worker_config.cursor_config.model == "custom-model", f"Worker {worker.id} 的 model 配置不正确"
            assert worker.worker_config.cursor_config.timeout == 600, f"Worker {worker.id} 的 timeout 配置不正确"
            assert worker.worker_config.working_directory == "/custom/path", (
                f"Worker {worker.id} 的 working_directory 配置不正确"
            )

    def test_initialize_workers_have_independent_cursor_configs(self, worker_config):
        """测试每个 Worker 拥有独立的 CursorAgentConfig（stream_agent_id 不相同）"""
        pool = WorkerPool(size=3, worker_config=worker_config)
        pool.initialize()

        # 验证每个 worker 拥有独立的 stream_agent_id
        stream_agent_ids = [w.worker_config.cursor_config.stream_agent_id for w in pool.workers]

        # 所有 stream_agent_id 都不为空
        assert all(agent_id is not None for agent_id in stream_agent_ids), "所有 Worker 的 stream_agent_id 都应该被设置"

        # 所有 stream_agent_id 互不相同
        assert len(set(stream_agent_ids)) == len(stream_agent_ids), (
            f"每个 Worker 的 stream_agent_id 应该独立且不相同，实际: {stream_agent_ids}"
        )

        # 验证 cursor_config 对象本身是独立的（不是同一个引用）
        cursor_configs = [w.worker_config.cursor_config for w in pool.workers]
        for i in range(len(cursor_configs)):
            for j in range(i + 1, len(cursor_configs)):
                assert cursor_configs[i] is not cursor_configs[j], (
                    f"Worker {i} 和 Worker {j} 不应共享同一个 cursor_config 实例"
                )


# ==================== 任务分配测试 ====================


class TestWorkerPoolTaskAssignment:
    """WorkerPool 任务分配测试"""

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_start_and_process_single_task(self, mock_worker_class, task_queue, worker_config):
        """测试启动并处理单个任务"""
        # 创建 mock worker
        mock_worker = create_mock_worker("worker-0")
        mock_worker_class.return_value = mock_worker

        # 创建 pool 并初始化
        pool = WorkerPool(size=1, worker_config=worker_config)
        pool.initialize()

        # 创建任务并入队
        task = create_test_task(
            title="测试任务",
            iteration_id=1,
            task_type=TaskType.IMPLEMENT,
        )
        await task_queue.enqueue(task)

        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))

        # 等待任务处理完成
        await wait_for_task_status(task_queue, task.id, TaskStatus.COMPLETED, timeout=2.0)

        # 安全停止 pool
        await safely_stop_pool(pool, pool_task)

        # 验证任务已被处理
        assert mock_worker.execute_task.called

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_multiple_tasks_distributed(self, mock_worker_class, task_queue, worker_config):
        """测试多个任务分配给多个 worker"""
        # 创建 mock workers 列表（使用索引跟踪，避免竞态条件）
        mock_workers = [create_mock_worker(f"worker-{i}") for i in range(2)]
        worker_index = {"current": 0}

        def get_next_worker(config, **kwargs):
            idx = worker_index["current"]
            worker_index["current"] += 1
            if idx < len(mock_workers):
                return mock_workers[idx]
            # 如果超出范围，返回一个新的 mock worker
            return create_mock_worker(f"worker-{idx}")

        mock_worker_class.side_effect = get_next_worker

        # 创建 pool 并初始化
        pool = WorkerPool(size=2, worker_config=worker_config)
        pool.initialize()

        # 创建多个任务并入队
        tasks = []
        for i in range(3):
            task = create_test_task(
                title=f"测试任务 {i}",
                iteration_id=1,
            )
            tasks.append(task)
            await task_queue.enqueue(task)

        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))

        # 等待所有任务完成
        for task in tasks:
            await wait_for_task_status(task_queue, task.id, TaskStatus.COMPLETED, timeout=2.0)

        # 安全停止 pool
        await safely_stop_pool(pool, pool_task)


# ==================== 并发执行测试 ====================


class TestWorkerPoolConcurrency:
    """WorkerPool 并发执行测试"""

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_concurrent_task_execution(self, mock_worker_class, task_queue, worker_config):
        """测试任务并发执行"""
        # 使用线程安全的方式记录执行时间
        execution_times: list[datetime] = []
        execution_lock = asyncio.Lock()

        async def slow_execute(task):
            """模拟慢速执行，记录开始时间"""
            async with execution_lock:
                execution_times.append(datetime.now())
            await asyncio.sleep(0.05)
            task.complete({"output": "done"})
            return task

        # 创建 mock workers（使用索引跟踪）
        mock_workers = [create_mock_worker(f"worker-{i}", slow_execute) for i in range(3)]
        worker_index = {"current": 0}

        def get_next_worker(config, **kwargs):
            idx = worker_index["current"]
            worker_index["current"] += 1
            if idx < len(mock_workers):
                return mock_workers[idx]
            return create_mock_worker(f"worker-{idx}", slow_execute)

        mock_worker_class.side_effect = get_next_worker

        # 创建 pool
        pool = WorkerPool(size=3, worker_config=worker_config)
        pool.initialize()

        # 创建多个任务
        tasks = []
        for i in range(3):
            task = create_test_task(title=f"并发任务 {i}", iteration_id=1)
            tasks.append(task)
            await task_queue.enqueue(task)

        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))

        # 等待所有任务完成
        for task in tasks:
            await wait_for_task_status(task_queue, task.id, TaskStatus.COMPLETED, timeout=2.0)

        # 安全停止 pool
        await safely_stop_pool(pool, pool_task)

        # 验证至少有一些任务被执行
        assert len(execution_times) > 0, "应该有任务被执行"

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_running_state_during_execution(self, mock_worker_class, task_queue, worker_config):
        """测试执行期间的运行状态"""
        pool = WorkerPool(size=1, worker_config=worker_config)

        assert pool._running is False

        # 使用事件来控制执行流程
        execute_started = asyncio.Event()
        execute_continue = asyncio.Event()

        async def controlled_execute(task):
            """可控制的执行，用于测试运行状态"""
            execute_started.set()
            # 等待信号继续，或者 0.5 秒超时
            try:
                await asyncio.wait_for(execute_continue.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
            task.complete({"output": "done"})
            return task

        mock_worker = create_mock_worker("worker-0", controlled_execute)
        mock_worker_class.return_value = mock_worker

        pool.initialize()

        # 创建一个任务
        task = create_test_task(title="状态测试任务", iteration_id=1)
        await task_queue.enqueue(task)

        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))

        # 等待任务开始执行
        try:
            await asyncio.wait_for(execute_started.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass

        # 验证运行状态
        assert pool._running is True

        # 允许任务继续
        execute_continue.set()

        # 安全停止 pool
        await safely_stop_pool(pool, pool_task)

        assert pool._running is False


# ==================== Worker 状态管理测试 ====================


class TestWorkerPoolStateManagement:
    """WorkerPool 状态管理测试"""

    @pytest.mark.asyncio
    async def test_stop_cancels_worker_tasks(self, worker_config):
        """测试 stop 方法取消 worker 任务"""
        pool = WorkerPool(size=2, worker_config=worker_config)

        # 模拟 worker 任务
        async def long_running_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # 正常的取消行为
                raise

        pool._running = True
        pool._worker_tasks = [
            asyncio.create_task(long_running_task()),
            asyncio.create_task(long_running_task()),
        ]

        await pool.stop()

        assert pool._running is False
        assert pool._worker_tasks == []

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_reset_stops_and_resets_workers(self, mock_worker_class, worker_config):
        """测试 reset 方法"""
        mock_worker = create_mock_worker("worker-0")
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=1, worker_config=worker_config)
        pool.initialize()
        pool._running = True

        await pool.reset()

        assert pool._running is False
        mock_worker.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_with_no_tasks(self, worker_config):
        """测试没有任务时的 stop"""
        pool = WorkerPool(size=1, worker_config=worker_config)

        # 应该不抛异常
        await pool.stop()

        assert pool._running is False
        assert pool._worker_tasks == []


# ==================== 知识库管理器测试 ====================


class TestWorkerPoolKnowledgeManager:
    """WorkerPool 知识库管理器测试"""

    @patch("coordinator.worker_pool.WorkerAgent")
    def test_set_knowledge_manager(self, mock_worker_class, worker_config, mock_knowledge_manager_instance):
        """测试设置知识库管理器"""
        mock_worker = MagicMock()
        mock_worker.set_knowledge_manager = MagicMock()
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=2, worker_config=worker_config)
        pool.initialize()

        pool.set_knowledge_manager(mock_knowledge_manager_instance)

        assert pool._knowledge_manager == mock_knowledge_manager_instance
        # 验证每个 worker 都设置了 knowledge_manager
        assert mock_worker.set_knowledge_manager.call_count == 2

    @patch("coordinator.worker_pool.WorkerAgent")
    def test_knowledge_manager_passed_to_workers_on_init(
        self, mock_worker_class, worker_config, mock_knowledge_manager_instance
    ):
        """测试初始化时知识库管理器传递给 worker"""
        pool = WorkerPool(
            size=2,
            worker_config=worker_config,
            knowledge_manager=mock_knowledge_manager_instance,
        )
        pool.initialize()

        # 验证 WorkerAgent 创建时传入了 knowledge_manager
        for call_args in mock_worker_class.call_args_list:
            assert call_args.kwargs.get("knowledge_manager") == mock_knowledge_manager_instance


# ==================== 统计信息测试 ====================


class TestWorkerPoolStatistics:
    """WorkerPool 统计信息测试"""

    @patch("coordinator.worker_pool.WorkerAgent")
    def test_get_statistics_empty_pool(self, mock_worker_class, worker_config):
        """测试空池的统计信息"""
        pool = WorkerPool(size=2, worker_config=worker_config)

        stats = pool.get_statistics()

        assert stats["pool_size"] == 2
        assert stats["running"] is False
        assert stats["workers"] == []
        assert stats["total_completed"] == 0

    @patch("coordinator.worker_pool.WorkerAgent")
    def test_get_statistics_with_workers(self, mock_worker_class, worker_config):
        """测试有 worker 的统计信息"""
        mock_worker = MagicMock()
        mock_worker.get_statistics.return_value = {
            "worker_id": "worker-0",
            "status": "idle",
            "completed_tasks_count": 5,
        }
        mock_worker.completed_tasks = ["task-1", "task-2", "task-3", "task-4", "task-5"]
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=2, worker_config=worker_config)
        pool.initialize()
        pool._running = True

        stats = pool.get_statistics()

        assert stats["pool_size"] == 2
        assert stats["running"] is True
        assert len(stats["workers"]) == 2
        assert stats["total_completed"] == 10  # 2 workers * 5 tasks each


# ==================== Worker 循环测试 ====================


class TestWorkerLoop:
    """Worker 循环测试"""

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_worker_loop_exits_when_iteration_complete(self, mock_worker_class, task_queue, worker_config):
        """测试当迭代完成时 worker 循环退出"""
        mock_worker = create_mock_worker("worker-0")
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=1, worker_config=worker_config)
        pool.initialize()

        # 创建一个任务
        task = create_test_task(title="单一任务", iteration_id=1)
        await task_queue.enqueue(task)

        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))

        # 等待任务处理完成
        completed = await wait_for_task_status(task_queue, task.id, TaskStatus.COMPLETED, timeout=2.0)

        # 验证任务已被处理
        assert completed, "任务应该在超时前完成"
        assert task_queue.get_task(task.id).status == TaskStatus.COMPLETED

        # 等待 pool 完成（worker loop 会在下次 dequeue 超时后检查 is_iteration_complete）
        # 由于 dequeue 超时为 2s，需要等待足够时间或手动停止
        try:
            await asyncio.wait_for(pool_task, timeout=3.0)
        except asyncio.TimeoutError:
            # 手动停止
            await safely_stop_pool(pool, pool_task)

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_worker_loop_handles_task_exception(self, mock_worker_class, task_queue, worker_config):
        """测试 worker 循环处理任务异常"""

        async def failing_execute(task):
            raise Exception("任务执行失败")

        mock_worker = create_mock_worker("worker-0", failing_execute)
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=1, worker_config=worker_config)
        pool.initialize()

        # 创建任务
        task = create_test_task(title="失败任务", iteration_id=1)
        await task_queue.enqueue(task)

        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))

        # 等待任务被处理（成功或失败）
        failed = await wait_for_task_status(task_queue, task.id, TaskStatus.FAILED, timeout=2.0)

        # 安全停止 pool
        await safely_stop_pool(pool, pool_task)

        # 验证任务被标记为失败
        updated_task = task_queue.get_task(task.id)
        assert failed, "任务应该被标记为失败"
        assert updated_task.status == TaskStatus.FAILED
        assert updated_task.error == "任务执行失败"


# ==================== 边界情况测试 ====================


class TestWorkerPoolEdgeCases:
    """WorkerPool 边界情况测试"""

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_start_with_empty_queue(self, mock_worker_class, task_queue, worker_config):
        """测试空队列时的启动"""
        mock_worker = create_mock_worker("worker-0")
        mock_worker_class.return_value = mock_worker

        pool = WorkerPool(size=1, worker_config=worker_config)
        pool.initialize()

        # 启动空队列
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))

        # 应该快速完成（空队列会立即检测到迭代完成）
        try:
            await asyncio.wait_for(pool_task, timeout=1.0)
        except asyncio.TimeoutError:
            await safely_stop_pool(pool, pool_task)

    def test_initialize_multiple_times(self, worker_config):
        """测试多次初始化"""
        with patch("coordinator.worker_pool.WorkerAgent") as mock_worker_class:
            mock_worker = MagicMock()
            mock_worker_class.return_value = mock_worker

            pool = WorkerPool(size=2, worker_config=worker_config)

            # 第一次初始化
            pool.initialize()
            assert len(pool.workers) == 2

            # 第二次初始化 - 应该重新创建 workers
            pool.initialize()
            assert len(pool.workers) == 2
            assert mock_worker_class.call_count == 4  # 2 + 2

    def test_pool_with_zero_size(self):
        """测试大小为0的池"""
        pool = WorkerPool(size=0)

        with patch("coordinator.worker_pool.WorkerAgent"):
            pool.initialize()

        assert len(pool.workers) == 0

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, worker_config):
        """测试多次调用 stop 是幂等的"""
        pool = WorkerPool(size=1, worker_config=worker_config)

        # 多次调用 stop 不应该出错
        await pool.stop()
        await pool.stop()
        await pool.stop()

        assert pool._running is False


# ==================== 执行追踪测试 ====================


class TestMockAgentExecutorTracing:
    """MockAgentExecutor 执行追踪测试"""

    @pytest.mark.asyncio
    async def test_execute_creates_trace(self):
        """测试 execute 创建执行追踪记录"""
        executor = MockAgentExecutor()

        # 执行任务
        result = await executor.execute(
            prompt="测试任务",
            context={"key": "value"},
            working_directory="/tmp/test",
            timeout=30,
        )

        # 验证追踪记录
        assert len(executor.execution_traces) == 1
        trace = executor.execution_traces[0]

        assert trace.execution_id == "exec-0001"
        assert trace.prompt == "测试任务"
        assert trace.context == {"key": "value"}
        assert trace.working_directory == "/tmp/test"
        assert trace.timeout == 30
        assert trace.status == "completed"
        assert trace.success is True
        assert trace.started_at is not None
        assert trace.completed_at is not None
        assert trace.duration >= 0

    @pytest.mark.asyncio
    async def test_execute_trace_status_transitions(self):
        """测试执行状态变更正确"""
        executor = MockAgentExecutor()

        # 配置成功和失败响应
        executor.configure_responses(
            [
                {"success": True, "output": "成功"},
                {"success": False, "output": "", "error": "失败"},
            ]
        )

        # 执行两个任务
        await executor.execute(prompt="任务1")
        await executor.execute(prompt="任务2")

        # 使用断言助手验证
        assert_execution_status_transitions(
            executor,
            expected_final_statuses=["completed", "failed"],
        )

    @pytest.mark.asyncio
    async def test_execute_trace_count(self):
        """测试执行追踪记录数量"""
        executor = MockAgentExecutor()

        # 执行多个任务
        for i in range(5):
            await executor.execute(prompt=f"任务 {i}")

        # 验证追踪记录数量
        assert_execution_trace_count(executor, 5)

    @pytest.mark.asyncio
    async def test_execute_all_completed(self):
        """测试所有执行都已完成"""
        executor = MockAgentExecutor()

        # 执行多个任务
        await executor.execute(prompt="任务1")
        await executor.execute(prompt="任务2")
        await executor.execute(prompt="任务3")

        # 验证所有执行都已完成
        assert_all_executions_completed(executor)

    @pytest.mark.asyncio
    async def test_execute_success_rate(self):
        """测试执行成功率"""
        executor = MockAgentExecutor()

        # 配置 3 个成功，1 个失败
        executor.configure_responses(
            [
                {"success": True, "output": "成功1"},
                {"success": True, "output": "成功2"},
                {"success": True, "output": "成功3"},
                {"success": False, "output": "", "error": "失败"},
            ]
        )

        for i in range(4):
            await executor.execute(prompt=f"任务 {i}")

        # 验证成功率 >= 75%
        assert_execution_success_rate(executor, min_success_rate=0.75)

    @pytest.mark.asyncio
    async def test_get_trace_by_id(self):
        """测试根据 ID 获取追踪记录"""
        executor = MockAgentExecutor()

        await executor.execute(prompt="任务1")
        await executor.execute(prompt="任务2")

        trace = executor.get_trace_by_id("exec-0002")
        assert trace is not None
        assert trace.prompt == "任务2"

        # 不存在的 ID
        assert executor.get_trace_by_id("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_traces_by_status(self):
        """测试根据状态获取追踪记录"""
        executor = MockAgentExecutor()

        executor.configure_responses(
            [
                {"success": True, "output": "成功"},
                {"success": False, "output": "", "error": "失败"},
                {"success": True, "output": "成功"},
            ]
        )

        for i in range(3):
            await executor.execute(prompt=f"任务 {i}")

        completed = executor.get_traces_by_status("completed")
        failed = executor.get_traces_by_status("failed")

        assert len(completed) == 2
        assert len(failed) == 1

    @pytest.mark.asyncio
    async def test_get_successful_and_failed_traces(self):
        """测试获取成功和失败的追踪记录"""
        executor = MockAgentExecutor()

        executor.configure_responses(
            [
                {"success": True, "output": "成功"},
                {"success": False, "output": "", "error": "失败1"},
                {"success": False, "output": "", "error": "失败2"},
            ]
        )

        for i in range(3):
            await executor.execute(prompt=f"任务 {i}")

        successful = executor.get_successful_traces()
        failed = executor.get_failed_traces()

        assert len(successful) == 1
        assert len(failed) == 2

    @pytest.mark.asyncio
    async def test_reset_clears_traces(self):
        """测试 reset 清除追踪记录"""
        executor = MockAgentExecutor()

        await executor.execute(prompt="任务1")
        await executor.execute(prompt="任务2")

        assert len(executor.execution_traces) == 2

        executor.reset()

        assert len(executor.execution_traces) == 0
        assert executor._execution_counter == 0

    @pytest.mark.asyncio
    async def test_execute_with_delay_records_duration(self):
        """测试执行延迟正确记录时长"""
        executor = MockAgentExecutor(default_delay=0.05)

        await executor.execute(prompt="延迟任务")

        trace = executor.execution_traces[0]
        assert trace.duration >= 0.05

    @pytest.mark.asyncio
    async def test_execution_history_backward_compatible(self):
        """测试执行历史保持向后兼容"""
        executor = MockAgentExecutor()

        await executor.execute(prompt="任务")

        # 验证向后兼容的执行历史
        assert len(executor.execution_history) == 1
        history = executor.execution_history[0]

        assert history["prompt"] == "任务"
        assert "execution_id" in history  # 新增字段
        assert "timestamp" in history


# ==================== 辅助函数 ====================


def _complete_task(task: Task) -> Task:
    """辅助函数：完成任务"""
    task.complete({"output": "任务已完成"})
    return task
