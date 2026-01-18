"""TaskQueue 测试

测试 TaskQueue 的 enqueue/dequeue 方法，
验证 get_pending_count 和 get_statistics 返回正确的统计信息
"""
import asyncio
import pytest
from datetime import datetime

from tasks.task import Task, TaskStatus, TaskPriority, TaskType
from tasks.queue import TaskQueue


class TestTaskQueue:
    """TaskQueue 测试类"""
    
    @pytest.fixture
    def queue(self):
        """创建测试用队列"""
        return TaskQueue()
    
    @pytest.fixture
    def sample_task(self):
        """创建示例任务"""
        return Task(
            title="测试任务",
            description="这是一个测试任务",
            instruction="执行测试操作",
            type=TaskType.TEST,
            iteration_id=1,
        )
    
    def create_task(self, iteration_id: int = 1, priority: TaskPriority = TaskPriority.NORMAL) -> Task:
        """创建任务辅助方法"""
        return Task(
            title=f"任务-{priority.name}",
            description="测试任务描述",
            instruction="测试指令",
            type=TaskType.TEST,
            iteration_id=iteration_id,
            priority=priority,
        )
    
    # ========== enqueue 测试 ==========
    
    @pytest.mark.asyncio
    async def test_enqueue_single_task(self, queue, sample_task):
        """测试单个任务入队"""
        await queue.enqueue(sample_task)
        
        # 验证任务被添加到索引
        assert sample_task.id in queue._tasks
        # 验证状态变为 QUEUED
        assert sample_task.status == TaskStatus.QUEUED
        # 验证任务可以通过 get_task 获取
        assert queue.get_task(sample_task.id) == sample_task
    
    @pytest.mark.asyncio
    async def test_enqueue_multiple_tasks(self, queue):
        """测试多个任务入队"""
        tasks = [self.create_task() for _ in range(5)]
        
        for task in tasks:
            await queue.enqueue(task)
        
        # 验证所有任务都被添加
        assert len(queue._tasks) == 5
        for task in tasks:
            assert task.status == TaskStatus.QUEUED
    
    @pytest.mark.asyncio
    async def test_enqueue_with_different_priorities(self, queue):
        """测试不同优先级任务入队"""
        low_task = self.create_task(priority=TaskPriority.LOW)
        normal_task = self.create_task(priority=TaskPriority.NORMAL)
        high_task = self.create_task(priority=TaskPriority.HIGH)
        critical_task = self.create_task(priority=TaskPriority.CRITICAL)
        
        # 按低到高顺序入队
        await queue.enqueue(low_task)
        await queue.enqueue(normal_task)
        await queue.enqueue(high_task)
        await queue.enqueue(critical_task)
        
        # 验证所有任务都入队了
        assert len(queue._tasks) == 4
    
    @pytest.mark.asyncio
    async def test_enqueue_different_iterations(self, queue):
        """测试不同迭代的任务入队"""
        task_iter1 = self.create_task(iteration_id=1)
        task_iter2 = self.create_task(iteration_id=2)
        
        await queue.enqueue(task_iter1)
        await queue.enqueue(task_iter2)
        
        # 验证任务被分配到不同的队列
        assert 1 in queue._queues
        assert 2 in queue._queues
    
    # ========== dequeue 测试 ==========
    
    @pytest.mark.asyncio
    async def test_dequeue_single_task(self, queue, sample_task):
        """测试单个任务出队"""
        await queue.enqueue(sample_task)
        
        dequeued = await queue.dequeue(iteration_id=1)
        
        assert dequeued is not None
        assert dequeued.id == sample_task.id
        # 验证状态变为 ASSIGNED
        assert dequeued.status == TaskStatus.ASSIGNED
    
    @pytest.mark.asyncio
    async def test_dequeue_priority_order(self, queue):
        """测试按优先级顺序出队（高优先级先出）"""
        low_task = self.create_task(priority=TaskPriority.LOW)
        high_task = self.create_task(priority=TaskPriority.HIGH)
        normal_task = self.create_task(priority=TaskPriority.NORMAL)
        
        # 按混乱顺序入队
        await queue.enqueue(low_task)
        await queue.enqueue(high_task)
        await queue.enqueue(normal_task)
        
        # 出队应该按优先级顺序
        first = await queue.dequeue(iteration_id=1)
        second = await queue.dequeue(iteration_id=1)
        third = await queue.dequeue(iteration_id=1)
        
        assert first.priority == TaskPriority.HIGH
        assert second.priority == TaskPriority.NORMAL
        assert third.priority == TaskPriority.LOW
    
    @pytest.mark.asyncio
    async def test_dequeue_empty_queue_with_timeout(self, queue):
        """测试从空队列出队（带超时）"""
        result = await queue.dequeue(iteration_id=1, timeout=0.1)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_dequeue_from_specific_iteration(self, queue):
        """测试从指定迭代出队"""
        task_iter1 = self.create_task(iteration_id=1)
        task_iter2 = self.create_task(iteration_id=2)
        
        await queue.enqueue(task_iter1)
        await queue.enqueue(task_iter2)
        
        # 从迭代2出队
        dequeued = await queue.dequeue(iteration_id=2)
        assert dequeued.id == task_iter2.id
        
        # 从迭代1出队
        dequeued = await queue.dequeue(iteration_id=1)
        assert dequeued.id == task_iter1.id
    
    # ========== get_pending_count 测试 ==========
    
    @pytest.mark.asyncio
    async def test_get_pending_count_empty(self, queue):
        """测试空队列的待处理数量"""
        count = queue.get_pending_count(iteration_id=1)
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_pending_count_after_enqueue(self, queue):
        """测试入队后的待处理数量"""
        for _ in range(3):
            await queue.enqueue(self.create_task())
        
        count = queue.get_pending_count(iteration_id=1)
        assert count == 3
    
    @pytest.mark.asyncio
    async def test_get_pending_count_after_dequeue(self, queue):
        """测试出队后待处理数量减少"""
        for _ in range(3):
            await queue.enqueue(self.create_task())
        
        # 出队一个
        await queue.dequeue(iteration_id=1)
        
        # 出队后变为 ASSIGNED，不再计入 pending
        count = queue.get_pending_count(iteration_id=1)
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_get_pending_count_by_iteration(self, queue):
        """测试按迭代获取待处理数量"""
        # 迭代1添加3个任务
        for _ in range(3):
            await queue.enqueue(self.create_task(iteration_id=1))
        
        # 迭代2添加2个任务
        for _ in range(2):
            await queue.enqueue(self.create_task(iteration_id=2))
        
        assert queue.get_pending_count(iteration_id=1) == 3
        assert queue.get_pending_count(iteration_id=2) == 2
        assert queue.get_pending_count(iteration_id=3) == 0  # 不存在的迭代
    
    # ========== get_statistics 测试 ==========
    
    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, queue):
        """测试空队列的统计信息"""
        stats = queue.get_statistics(iteration_id=1)
        
        assert stats == {
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
        }
    
    @pytest.mark.asyncio
    async def test_get_statistics_with_queued_tasks(self, queue):
        """测试有入队任务的统计信息"""
        for _ in range(3):
            await queue.enqueue(self.create_task())
        
        stats = queue.get_statistics(iteration_id=1)
        
        assert stats["total"] == 3
        assert stats["pending"] == 3  # QUEUED 计入 pending
        assert stats["in_progress"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0
    
    @pytest.mark.asyncio
    async def test_get_statistics_with_mixed_status(self, queue):
        """测试混合状态任务的统计信息"""
        # 创建并入队任务
        task1 = self.create_task()
        task2 = self.create_task()
        task3 = self.create_task()
        task4 = self.create_task()
        
        await queue.enqueue(task1)
        await queue.enqueue(task2)
        await queue.enqueue(task3)
        await queue.enqueue(task4)
        
        # 出队两个（变为 ASSIGNED）
        dequeued1 = await queue.dequeue(iteration_id=1)
        dequeued2 = await queue.dequeue(iteration_id=1)
        
        # 其中一个开始执行（变为 IN_PROGRESS）
        dequeued1.start()
        queue.update_task(dequeued1)
        
        # 另一个完成（变为 COMPLETED）
        dequeued2.complete({"result": "success"})
        queue.update_task(dequeued2)
        
        stats = queue.get_statistics(iteration_id=1)
        
        assert stats["total"] == 4
        assert stats["pending"] == 2      # 2个还在队列中
        assert stats["in_progress"] == 1  # 1个执行中
        assert stats["completed"] == 1    # 1个完成
        assert stats["failed"] == 0
    
    @pytest.mark.asyncio
    async def test_get_statistics_with_failed_task(self, queue):
        """测试包含失败任务的统计信息"""
        task = self.create_task()
        await queue.enqueue(task)
        
        dequeued = await queue.dequeue(iteration_id=1)
        dequeued.fail("测试错误")
        queue.update_task(dequeued)
        
        stats = queue.get_statistics(iteration_id=1)
        
        assert stats["total"] == 1
        assert stats["pending"] == 0
        assert stats["in_progress"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 1
    
    @pytest.mark.asyncio
    async def test_get_statistics_by_iteration(self, queue):
        """测试按迭代获取统计信息"""
        # 迭代1
        for _ in range(2):
            await queue.enqueue(self.create_task(iteration_id=1))
        
        # 迭代2
        for _ in range(3):
            await queue.enqueue(self.create_task(iteration_id=2))
        
        stats1 = queue.get_statistics(iteration_id=1)
        stats2 = queue.get_statistics(iteration_id=2)
        
        assert stats1["total"] == 2
        assert stats2["total"] == 3
    
    # ========== 综合集成测试 ==========
    
    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self, queue):
        """测试完整任务生命周期"""
        task = self.create_task()
        
        # 初始状态
        assert task.status == TaskStatus.PENDING
        
        # 入队
        await queue.enqueue(task)
        assert task.status == TaskStatus.QUEUED
        assert queue.get_pending_count(iteration_id=1) == 1
        
        # 出队
        dequeued = await queue.dequeue(iteration_id=1)
        assert dequeued.status == TaskStatus.ASSIGNED
        assert queue.get_pending_count(iteration_id=1) == 0
        
        stats = queue.get_statistics(iteration_id=1)
        assert stats["in_progress"] == 1  # ASSIGNED 计入 in_progress
        
        # 开始执行
        dequeued.start()
        queue.update_task(dequeued)
        assert dequeued.status == TaskStatus.IN_PROGRESS
        
        # 完成
        dequeued.complete({"files_modified": ["test.py"]})
        queue.update_task(dequeued)
        assert dequeued.status == TaskStatus.COMPLETED
        
        final_stats = queue.get_statistics(iteration_id=1)
        assert final_stats["completed"] == 1
        assert final_stats["in_progress"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
