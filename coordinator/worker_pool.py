"""Worker 池管理"""
import asyncio
from typing import Optional
from loguru import logger

from agents.worker import WorkerAgent, WorkerConfig
from tasks.task import Task
from tasks.queue import TaskQueue


class WorkerPool:
    """Worker 池
    
    管理一组 Worker Agent，并行处理任务
    """
    
    def __init__(
        self,
        size: int = 3,
        worker_config: Optional[WorkerConfig] = None,
    ):
        self.size = size
        self.worker_config = worker_config or WorkerConfig()
        self.workers: list[WorkerAgent] = []
        self._running = False
        self._worker_tasks: list[asyncio.Task] = []
    
    def initialize(self) -> None:
        """初始化 Worker 池"""
        self.workers = []
        for i in range(self.size):
            config = WorkerConfig(
                name=f"worker-{i}",
                working_directory=self.worker_config.working_directory,
                task_timeout=self.worker_config.task_timeout,
                cursor_config=self.worker_config.cursor_config,
            )
            worker = WorkerAgent(config)
            self.workers.append(worker)
        logger.info(f"Worker 池已初始化: {self.size} 个 Worker")
    
    async def start(self, task_queue: TaskQueue, iteration_id: int) -> None:
        """启动 Worker 池处理任务
        
        Args:
            task_queue: 任务队列
            iteration_id: 当前迭代 ID
        """
        self._running = True
        logger.info(f"Worker 池启动，迭代 {iteration_id}")
        
        # 为每个 Worker 创建工作协程
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(worker, task_queue, iteration_id))
            for worker in self.workers
        ]
        
        # 等待所有 Worker 完成
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        logger.info(f"Worker 池完成，迭代 {iteration_id}")
    
    async def _worker_loop(
        self,
        worker: WorkerAgent,
        task_queue: TaskQueue,
        iteration_id: int,
    ) -> None:
        """单个 Worker 的工作循环"""
        logger.debug(f"[{worker.id}] Worker 循环启动")
        
        while self._running:
            # 尝试获取任务
            task = await task_queue.dequeue(iteration_id, timeout=2.0)
            
            if task is None:
                # 检查是否所有任务都已完成
                if task_queue.is_iteration_complete(iteration_id):
                    logger.debug(f"[{worker.id}] 迭代 {iteration_id} 所有任务已完成")
                    break
                continue
            
            try:
                # 执行任务
                updated_task = await worker.execute_task(task)
                task_queue.update_task(updated_task)
                
            except Exception as e:
                logger.error(f"[{worker.id}] 任务执行异常: {e}")
                task.fail(str(e))
                task_queue.update_task(task)
        
        logger.debug(f"[{worker.id}] Worker 循环结束")
    
    async def stop(self) -> None:
        """停止 Worker 池"""
        self._running = False
        
        # 取消所有 Worker 任务
        for task in self._worker_tasks:
            task.cancel()
        
        # 等待任务取消完成
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        self._worker_tasks.clear()
        logger.info("Worker 池已停止")
    
    async def reset(self) -> None:
        """重置 Worker 池"""
        await self.stop()
        for worker in self.workers:
            await worker.reset()
        logger.info("Worker 池已重置")
    
    def get_statistics(self) -> dict:
        """获取 Worker 池统计"""
        return {
            "pool_size": self.size,
            "running": self._running,
            "workers": [w.get_statistics() for w in self.workers],
            "total_completed": sum(len(w.completed_tasks) for w in self.workers),
        }
