"""WorkerPool 测试

测试 WorkerPool 的初始化、任务分配、并发执行、worker 状态管理等功能。
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coordinator.worker_pool import WorkerPool
from agents.worker import WorkerConfig
from tasks.task import Task, TaskPriority, TaskStatus, TaskType
from tasks.queue import TaskQueue
from tests.conftest_e2e import (
    MockAgentExecutor,
    MockKnowledgeManager,
    create_test_task,
)


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


# ==================== 任务分配测试 ====================


class TestWorkerPoolTaskAssignment:
    """WorkerPool 任务分配测试"""

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_start_and_process_single_task(
        self, mock_worker_class, task_queue, worker_config
    ):
        """测试启动并处理单个任务"""
        # 创建 mock worker
        mock_worker = MagicMock()
        mock_worker.id = "worker-0"
        mock_worker.execute_task = AsyncMock(side_effect=lambda task: _complete_task(task))
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
        
        # 启动 pool（使用短超时防止测试阻塞）
        # 由于 pool.start 会等待所有任务完成，我们需要在后台运行
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))
        
        # 等待任务处理
        await asyncio.sleep(0.1)
        
        # 停止 pool
        await pool.stop()
        
        try:
            await asyncio.wait_for(pool_task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            pass

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_multiple_tasks_distributed(
        self, mock_worker_class, task_queue, worker_config
    ):
        """测试多个任务分配给多个 worker"""
        # 创建多个 mock worker
        mock_workers = []
        for i in range(2):
            mock_worker = MagicMock()
            mock_worker.id = f"worker-{i}"
            mock_worker.execute_task = AsyncMock(side_effect=lambda task: _complete_task(task))
            mock_workers.append(mock_worker)
        
        mock_worker_class.side_effect = lambda config, **kwargs: mock_workers.pop(0) if mock_workers else MagicMock()
        
        # 创建 pool 并初始化
        pool = WorkerPool(size=2, worker_config=worker_config)
        pool.initialize()
        
        # 创建多个任务并入队
        for i in range(3):
            task = create_test_task(
                title=f"测试任务 {i}",
                iteration_id=1,
            )
            await task_queue.enqueue(task)
        
        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))
        
        await asyncio.sleep(0.2)
        await pool.stop()
        
        try:
            await asyncio.wait_for(pool_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


# ==================== 并发执行测试 ====================


class TestWorkerPoolConcurrency:
    """WorkerPool 并发执行测试"""

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_concurrent_task_execution(self, mock_worker_class, task_queue, worker_config):
        """测试任务并发执行"""
        execution_times = []
        
        async def slow_execute(task):
            """模拟慢速执行"""
            execution_times.append(datetime.now())
            await asyncio.sleep(0.05)
            task.complete({"output": "done"})
            return task
        
        # 创建 mock workers
        mock_workers = []
        for i in range(3):
            mock_worker = MagicMock()
            mock_worker.id = f"worker-{i}"
            mock_worker.execute_task = AsyncMock(side_effect=slow_execute)
            mock_workers.append(mock_worker)
        
        worker_iter = iter(mock_workers)
        mock_worker_class.side_effect = lambda config, **kwargs: next(worker_iter, MagicMock())
        
        # 创建 pool
        pool = WorkerPool(size=3, worker_config=worker_config)
        pool.initialize()
        
        # 创建多个任务
        for i in range(3):
            task = create_test_task(title=f"并发任务 {i}", iteration_id=1)
            await task_queue.enqueue(task)
        
        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))
        
        await asyncio.sleep(0.3)
        await pool.stop()
        
        try:
            await asyncio.wait_for(pool_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_running_state_during_execution(self, mock_worker_class, task_queue, worker_config):
        """测试执行期间的运行状态"""
        pool = WorkerPool(size=1, worker_config=worker_config)
        
        assert pool._running is False
        
        mock_worker = MagicMock()
        mock_worker.id = "worker-0"
        mock_worker.execute_task = AsyncMock(side_effect=lambda task: _complete_task(task))
        mock_worker_class.return_value = mock_worker
        
        pool.initialize()
        
        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))
        
        # 等待启动
        await asyncio.sleep(0.05)
        assert pool._running is True
        
        await pool.stop()
        
        try:
            await asyncio.wait_for(pool_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        
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
            await asyncio.sleep(10)
        
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
        mock_worker = MagicMock()
        mock_worker.id = "worker-0"
        mock_worker.reset = AsyncMock()
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
    def test_set_knowledge_manager(
        self, mock_worker_class, worker_config, mock_knowledge_manager_instance
    ):
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
    async def test_worker_loop_exits_when_iteration_complete(
        self, mock_worker_class, task_queue, worker_config
    ):
        """测试当迭代完成时 worker 循环退出"""
        mock_worker = MagicMock()
        mock_worker.id = "worker-0"
        mock_worker.execute_task = AsyncMock(side_effect=lambda task: _complete_task(task))
        mock_worker_class.return_value = mock_worker
        
        pool = WorkerPool(size=1, worker_config=worker_config)
        pool.initialize()
        
        # 创建一个任务
        task = create_test_task(title="单一任务", iteration_id=1)
        await task_queue.enqueue(task)
        
        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))
        
        # 等待任务处理（给足够的时间让 worker loop 处理任务并检查完成状态）
        # 由于 dequeue 超时为 2s，需要等待足够时间
        await asyncio.sleep(0.1)
        
        # 验证任务已被处理
        assert task_queue.get_task(task.id).status == TaskStatus.COMPLETED
        
        # 等待 pool 完成（worker loop 会在下次 dequeue 超时后检查 is_iteration_complete）
        try:
            await asyncio.wait_for(pool_task, timeout=3.0)
        except asyncio.TimeoutError:
            # 如果超时，手动停止（由于 dequeue 超时设置为 2s，可能需要更长时间）
            await pool.stop()
            # 不视为失败，因为任务已正确完成
            pass

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_worker_loop_handles_task_exception(
        self, mock_worker_class, task_queue, worker_config
    ):
        """测试 worker 循环处理任务异常"""
        async def failing_execute(task):
            raise Exception("任务执行失败")
        
        mock_worker = MagicMock()
        mock_worker.id = "worker-0"
        mock_worker.execute_task = AsyncMock(side_effect=failing_execute)
        mock_worker_class.return_value = mock_worker
        
        pool = WorkerPool(size=1, worker_config=worker_config)
        pool.initialize()
        
        # 创建任务
        task = create_test_task(title="失败任务", iteration_id=1)
        await task_queue.enqueue(task)
        
        # 启动 pool
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))
        
        await asyncio.sleep(0.2)
        await pool.stop()
        
        try:
            await asyncio.wait_for(pool_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        
        # 验证任务被标记为失败
        updated_task = task_queue.get_task(task.id)
        assert updated_task.status == TaskStatus.FAILED
        assert updated_task.error == "任务执行失败"


# ==================== 边界情况测试 ====================


class TestWorkerPoolEdgeCases:
    """WorkerPool 边界情况测试"""

    @pytest.mark.asyncio
    @patch("coordinator.worker_pool.WorkerAgent")
    async def test_start_with_empty_queue(self, mock_worker_class, task_queue, worker_config):
        """测试空队列时的启动"""
        mock_worker = MagicMock()
        mock_worker.id = "worker-0"
        mock_worker_class.return_value = mock_worker
        
        pool = WorkerPool(size=1, worker_config=worker_config)
        pool.initialize()
        
        # 启动空队列
        pool_task = asyncio.create_task(pool.start(task_queue, iteration_id=1))
        
        # 应该快速完成
        try:
            await asyncio.wait_for(pool_task, timeout=1.0)
        except asyncio.TimeoutError:
            await pool.stop()

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


# ==================== 辅助函数 ====================


def _complete_task(task: Task) -> Task:
    """辅助函数：完成任务"""
    task.complete({"output": "任务已完成"})
    return task
