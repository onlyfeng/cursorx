"""任务队列"""
import asyncio
from typing import Optional
from collections import defaultdict
from loguru import logger

from .task import Task, TaskStatus


class TaskQueue:
    """任务队列
    
    管理任务的入队、出队、优先级排序
    支持按迭代隔离任务
    """
    
    def __init__(self):
        self._queues: dict[int, asyncio.PriorityQueue] = defaultdict(asyncio.PriorityQueue)
        self._tasks: dict[str, Task] = {}  # 所有任务的索引
        self._lock = asyncio.Lock()
    
    async def enqueue(self, task: Task) -> None:
        """入队任务"""
        async with self._lock:
            iteration_id = task.iteration_id
            # 优先级队列：负优先级值使高优先级任务排在前面
            priority = -task.priority.value
            await self._queues[iteration_id].put((priority, task.created_at.timestamp(), task.id))
            self._tasks[task.id] = task
            task.status = TaskStatus.QUEUED
            logger.debug(f"任务入队: {task.id} (迭代 {iteration_id}, 优先级 {task.priority.name})")
    
    async def dequeue(self, iteration_id: int, timeout: Optional[float] = None) -> Optional[Task]:
        """出队任务
        
        Args:
            iteration_id: 迭代 ID
            timeout: 超时时间（秒）
            
        Returns:
            任务，如果队列为空或超时则返回 None
        """
        queue = self._queues[iteration_id]
        try:
            if timeout:
                priority, timestamp, task_id = await asyncio.wait_for(
                    queue.get(), timeout=timeout
                )
            else:
                priority, timestamp, task_id = await queue.get()
            
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.QUEUED:
                task.status = TaskStatus.ASSIGNED
                logger.debug(f"任务出队: {task_id}")
                return task
            return None
            
        except asyncio.TimeoutError:
            return None
        except asyncio.QueueEmpty:
            return None
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self._tasks.get(task_id)
    
    def update_task(self, task: Task) -> None:
        """更新任务"""
        self._tasks[task.id] = task
    
    def get_tasks_by_iteration(self, iteration_id: int) -> list[Task]:
        """获取指定迭代的所有任务"""
        return [t for t in self._tasks.values() if t.iteration_id == iteration_id]
    
    def get_pending_count(self, iteration_id: int) -> int:
        """获取待处理任务数量"""
        return sum(
            1 for t in self._tasks.values()
            if t.iteration_id == iteration_id and t.status in (TaskStatus.PENDING, TaskStatus.QUEUED)
        )
    
    def get_in_progress_count(self, iteration_id: int) -> int:
        """获取执行中任务数量"""
        return sum(
            1 for t in self._tasks.values()
            if t.iteration_id == iteration_id and t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        )
    
    def get_completed_count(self, iteration_id: int) -> int:
        """获取已完成任务数量"""
        return sum(
            1 for t in self._tasks.values()
            if t.iteration_id == iteration_id and t.status == TaskStatus.COMPLETED
        )
    
    def get_failed_count(self, iteration_id: int) -> int:
        """获取失败任务数量"""
        return sum(
            1 for t in self._tasks.values()
            if t.iteration_id == iteration_id and t.status == TaskStatus.FAILED
        )
    
    def is_iteration_complete(self, iteration_id: int) -> bool:
        """检查迭代是否完成（所有任务都处于终态）"""
        tasks = self.get_tasks_by_iteration(iteration_id)
        if not tasks:
            return True
        return all(t.is_terminal() for t in tasks)
    
    def get_statistics(self, iteration_id: int) -> dict:
        """获取迭代统计信息"""
        tasks = self.get_tasks_by_iteration(iteration_id)
        return {
            "total": len(tasks),
            "pending": self.get_pending_count(iteration_id),
            "in_progress": self.get_in_progress_count(iteration_id),
            "completed": self.get_completed_count(iteration_id),
            "failed": self.get_failed_count(iteration_id),
        }
    
    async def clear_iteration(self, iteration_id: int) -> None:
        """清除指定迭代的队列（保留任务记录）"""
        async with self._lock:
            if iteration_id in self._queues:
                # 清空队列
                while not self._queues[iteration_id].empty():
                    try:
                        self._queues[iteration_id].get_nowait()
                    except asyncio.QueueEmpty:
                        break
                logger.info(f"已清除迭代 {iteration_id} 的任务队列")
