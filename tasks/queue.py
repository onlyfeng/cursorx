"""任务队列"""
import asyncio
from collections import defaultdict
from typing import Optional

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
        self._lock: Optional[asyncio.Lock] = None
        # 队列索引：追踪已入队的任务 ID（用于 reconcile 检测不一致）
        self._queued_ids: dict[int, set[str]] = defaultdict(set)

    async def _get_lock(self) -> asyncio.Lock:
        """延迟创建锁，避免无事件循环时报错。"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def enqueue(self, task: Task) -> None:
        """入队任务"""
        lock = await self._get_lock()
        async with lock:
            iteration_id = task.iteration_id
            # 优先级队列：负优先级值使高优先级任务排在前面
            priority = -task.priority.value
            await self._queues[iteration_id].put((priority, task.created_at.timestamp(), task.id))
            self._tasks[task.id] = task
            task.status = TaskStatus.QUEUED
            # 更新队列索引
            self._queued_ids[iteration_id].add(task.id)
            logger.debug(f"任务入队: {task.id} (迭代 {iteration_id}, 优先级 {task.priority.name})")

    async def requeue(self, task: Task, reason: str = "") -> None:
        """重新入队任务（用于超时/Worker 死亡后的重试）

        与 enqueue 的区别:
            - 重置任务状态为 PENDING（然后入队时变为 QUEUED）
            - 清除 assigned_to、result、error 等执行状态
            - 不增加 retry_count（由调用方决定是否增加）

        Args:
            task: 要重新入队的任务
            reason: 重新入队原因（用于日志）
        """
        # 重置任务执行状态
        task.status = TaskStatus.PENDING
        task.assigned_to = None
        task.result = None
        task.error = None
        task.started_at = None

        lock = await self._get_lock()
        async with lock:
            iteration_id = task.iteration_id
            # 优先级队列：负优先级值使高优先级任务排在前面
            priority = -task.priority.value
            await self._queues[iteration_id].put((priority, task.created_at.timestamp(), task.id))
            self._tasks[task.id] = task
            task.status = TaskStatus.QUEUED
            # 更新队列索引
            self._queued_ids[iteration_id].add(task.id)
            log_msg = f"任务重新入队: {task.id} (迭代 {iteration_id}, 重试次数 {task.retry_count}/{task.max_retries})"
            if reason:
                log_msg += f" - 原因: {reason}"
            logger.info(log_msg)

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

            # 从队列索引中移除
            self._queued_ids[iteration_id].discard(task_id)

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
        lock = await self._get_lock()
        async with lock:
            if iteration_id in self._queues:
                # 清空队列
                while not self._queues[iteration_id].empty():
                    try:
                        self._queues[iteration_id].get_nowait()
                    except asyncio.QueueEmpty:
                        break
                # 清空队列索引
                self._queued_ids[iteration_id].clear()
                logger.info(f"已清除迭代 {iteration_id} 的任务队列")

    def is_in_queue(self, task_id: str, iteration_id: int) -> bool:
        """检查任务是否在队列中

        Args:
            task_id: 任务 ID
            iteration_id: 迭代 ID

        Returns:
            任务是否在队列索引中
        """
        return task_id in self._queued_ids.get(iteration_id, set())

    def get_queued_ids(self, iteration_id: int) -> set[str]:
        """获取指定迭代的队列中的任务 ID 集合

        Args:
            iteration_id: 迭代 ID

        Returns:
            任务 ID 集合的副本
        """
        return self._queued_ids.get(iteration_id, set()).copy()

    async def reconcile_iteration(
        self,
        iteration_id: int,
        in_flight_task_ids: set[str],
        active_future_task_ids: set[str],
    ) -> dict[str, list[Task]]:
        """协调迭代中的任务状态不一致

        检测并返回需要恢复的任务，按照问题类型分类：
        1. orphaned_pending: status=PENDING/QUEUED 但不在队列索引中的任务
        2. orphaned_assigned: status=ASSIGNED/IN_PROGRESS 但无 in-flight 记录的任务
        3. stale_in_progress: status=IN_PROGRESS 但无对应 active future 的任务

        Args:
            iteration_id: 迭代 ID
            in_flight_task_ids: 当前在途任务 ID 集合（来自 process_manager）
            active_future_task_ids: 当前有活跃 asyncio.Future 的任务 ID 集合

        Returns:
            分类后的问题任务字典:
            {
                "orphaned_pending": [Task, ...],     # 需要重新入队
                "orphaned_assigned": [Task, ...],   # 需要检查 worker 状态
                "stale_in_progress": [Task, ...],   # 需要超时处理或重新入队
            }
        """
        result: dict[str, list[Task]] = {
            "orphaned_pending": [],
            "orphaned_assigned": [],
            "stale_in_progress": [],
        }

        tasks = self.get_tasks_by_iteration(iteration_id)
        queued_ids = self._queued_ids.get(iteration_id, set())

        for task in tasks:
            # 跳过已终态的任务
            if task.is_terminal():
                continue

            # 检测 orphaned_pending: PENDING/QUEUED 但不在队列中
            if task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
                if task.id not in queued_ids:
                    result["orphaned_pending"].append(task)
                    logger.debug(
                        f"[reconcile] 发现 orphaned_pending: {task.id} "
                        f"(status={task.status.value}, not in queue)"
                    )

            # 检测 orphaned_assigned: ASSIGNED 但无 in-flight 记录
            elif task.status == TaskStatus.ASSIGNED:
                if task.id not in in_flight_task_ids:
                    result["orphaned_assigned"].append(task)
                    logger.debug(
                        f"[reconcile] 发现 orphaned_assigned: {task.id} "
                        f"(status=ASSIGNED, no in-flight record)"
                    )

            # 检测 stale_in_progress: IN_PROGRESS 但无对应 active future
            elif task.status == TaskStatus.IN_PROGRESS:
                if task.id not in active_future_task_ids:
                    result["stale_in_progress"].append(task)
                    logger.debug(
                        f"[reconcile] 发现 stale_in_progress: {task.id} "
                        f"(status=IN_PROGRESS, no active future)"
                    )

        # 记录 reconcile 结果
        total_issues = sum(len(v) for v in result.values())
        if total_issues > 0:
            logger.warning(
                f"[reconcile] 迭代 {iteration_id} 发现 {total_issues} 个状态不一致任务: "
                f"orphaned_pending={len(result['orphaned_pending'])}, "
                f"orphaned_assigned={len(result['orphaned_assigned'])}, "
                f"stale_in_progress={len(result['stale_in_progress'])}"
            )

        return result
