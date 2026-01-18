"""多进程编排器

基于多进程架构的编排器，每个 Agent 作为独立进程运行
"""
import asyncio
import uuid
from typing import Any, Optional
from pydantic import BaseModel
from loguru import logger

from core.state import SystemState, IterationStatus
from tasks.task import Task, TaskStatus
from tasks.queue import TaskQueue
from process.manager import AgentProcessManager
from process.message_queue import ProcessMessage, ProcessMessageType
from agents.planner_process import PlannerAgentProcess
from agents.worker_process import WorkerAgentProcess
from agents.reviewer_process import ReviewerAgentProcess, ReviewDecision


class MultiProcessOrchestratorConfig(BaseModel):
    """多进程编排器配置"""
    working_directory: str = "."
    max_iterations: int = 10
    worker_count: int = 3              # Worker 进程数量
    enable_sub_planners: bool = True
    strict_review: bool = False
    
    # 超时设置
    planning_timeout: float = 120.0    # 规划超时
    execution_timeout: float = 300.0   # 单任务执行超时
    review_timeout: float = 60.0       # 评审超时
    
    # 模型配置 - 不同 Agent 使用不同模型
    # 可用模型: gpt-5.2-high, opus-4.5-thinking, gpt-5.2-codex, sonnet-4.5-thinking 等
    # 使用 `agent models` 查看完整列表
    planner_model: str = "gpt-5.2-high"           # 规划者使用 GPT 5.2 High
    worker_model: str = "opus-4.5-thinking"       # 执行者使用 Claude 4.5 Opus (Thinking)
    reviewer_model: str = "opus-4.5-thinking"     # 评审者使用 Claude 4.5 Opus (Thinking)
    stream_events_enabled: bool = False
    stream_log_console: bool = True
    stream_log_detail_dir: str = "logs/stream_json/detail/"
    stream_log_raw_dir: str = "logs/stream_json/raw/"


class MultiProcessOrchestrator:
    """多进程编排器
    
    使用多进程架构，每个 Agent 作为独立进程运行：
    - 1 个 Planner 进程
    - N 个 Worker 进程（并行）
    - 1 个 Reviewer 进程
    
    通过消息队列进行进程间通信
    """
    
    def __init__(self, config: MultiProcessOrchestratorConfig):
        self.config = config
        
        # 系统状态
        self.state = SystemState(
            working_directory=config.working_directory,
            max_iterations=config.max_iterations,
        )
        
        # 任务队列
        self.task_queue = TaskQueue()
        
        # 进程管理器
        self.process_manager = AgentProcessManager()
        
        # Agent ID
        self.planner_id: Optional[str] = None
        self.worker_ids: list[str] = []
        self.reviewer_id: Optional[str] = None
        
        # 待处理的消息响应
        self._pending_responses: dict[str, asyncio.Future] = {}
    
    def _spawn_agents(self) -> None:
        """创建并启动所有 Agent 进程"""
        # 创建 Planner - 使用 GPT 5.2-high
        planner_config = {
            "working_directory": self.config.working_directory,
            "timeout": int(self.config.planning_timeout),
            "model": self.config.planner_model,
            "stream_events_enabled": self.config.stream_events_enabled,
            "stream_log_console": self.config.stream_log_console,
            "stream_log_detail_dir": self.config.stream_log_detail_dir,
            "stream_log_raw_dir": self.config.stream_log_raw_dir,
            "agent_name": "planner",
        }
        self.planner_id = f"planner-{uuid.uuid4().hex[:8]}"
        self.process_manager.spawn_agent(
            agent_class=PlannerAgentProcess,
            agent_id=self.planner_id,
            agent_type="planner",
            config=planner_config,
        )
        logger.info(f"Planner 进程已创建: {self.planner_id} (模型: {self.config.planner_model})")
        
        # 创建 Workers - 使用 opus-4.5-thinking
        worker_config = {
            "working_directory": self.config.working_directory,
            "task_timeout": int(self.config.execution_timeout),
            "model": self.config.worker_model,
            "stream_events_enabled": self.config.stream_events_enabled,
            "stream_log_console": self.config.stream_log_console,
            "stream_log_detail_dir": self.config.stream_log_detail_dir,
            "stream_log_raw_dir": self.config.stream_log_raw_dir,
        }
        for i in range(self.config.worker_count):
            worker_id = f"worker-{i}-{uuid.uuid4().hex[:8]}"
            self.process_manager.spawn_agent(
                agent_class=WorkerAgentProcess,
                agent_id=worker_id,
                agent_type="worker",
                config=worker_config,
            )
            self.worker_ids.append(worker_id)
            logger.info(f"Worker 进程已创建: {worker_id} (模型: {self.config.worker_model})")
        
        # 创建 Reviewer - 使用 opus-4.5-thinking
        reviewer_config = {
            "working_directory": self.config.working_directory,
            "timeout": int(self.config.review_timeout),
            "model": self.config.reviewer_model,
            "strict_mode": self.config.strict_review,
            "stream_events_enabled": self.config.stream_events_enabled,
            "stream_log_console": self.config.stream_log_console,
            "stream_log_detail_dir": self.config.stream_log_detail_dir,
            "stream_log_raw_dir": self.config.stream_log_raw_dir,
            "agent_name": "reviewer",
        }
        self.reviewer_id = f"reviewer-{uuid.uuid4().hex[:8]}"
        self.process_manager.spawn_agent(
            agent_class=ReviewerAgentProcess,
            agent_id=self.reviewer_id,
            agent_type="reviewer",
            config=reviewer_config,
        )
        logger.info(f"Reviewer 进程已创建: {self.reviewer_id} (模型: {self.config.reviewer_model})")
    
    async def run(self, goal: str) -> dict[str, Any]:
        """运行编排器
        
        Args:
            goal: 用户目标
            
        Returns:
            执行结果
        """
        self.state.goal = goal
        self.state.is_running = True
        
        logger.info("=" * 60)
        logger.info("多进程编排器启动")
        logger.info(f"目标: {goal}")
        logger.info(f"Worker 数量: {self.config.worker_count}")
        logger.info("=" * 60)
        
        try:
            # 1. 创建 Agent 进程
            self._spawn_agents()
            
            # 2. 等待所有进程就绪
            if not self.process_manager.wait_all_ready(timeout=30.0):
                raise RuntimeError("Agent 进程启动失败")
            
            # 3. 启动消息处理循环
            message_task = asyncio.create_task(self._message_loop())
            
            # 4. 执行主循环
            try:
                while self.state.current_iteration < self.state.max_iterations:
                    # 开始新迭代
                    iteration = self.state.start_new_iteration()
                    logger.info(f"\n{'='*50}")
                    logger.info(f"迭代 {iteration.iteration_id} 开始")
                    
                    # 规划阶段
                    await self._planning_phase(goal, iteration.iteration_id)
                    
                    # 检查任务
                    pending = self.task_queue.get_pending_count(iteration.iteration_id)
                    if pending == 0:
                        logger.warning("规划阶段未产生任务")
                        continue
                    
                    # 执行阶段
                    await self._execution_phase(iteration.iteration_id)
                    
                    # 评审阶段
                    decision = await self._review_phase(goal, iteration.iteration_id)
                    
                    # 处理决策
                    if decision == ReviewDecision.COMPLETE:
                        logger.info("目标已完成")
                        self.state.is_completed = True
                        break
                    elif decision == ReviewDecision.ABORT:
                        logger.error("评审决定终止")
                        break
                    
                    # 重置状态准备下一轮
                    self.state.reset_for_new_iteration()
                    
            finally:
                message_task.cancel()
                try:
                    await message_task
                except asyncio.CancelledError:
                    pass
            
            return self._generate_final_result()
            
        except Exception as e:
            logger.exception(f"编排器异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "iterations_completed": self.state.current_iteration,
            }
        finally:
            # 关闭所有进程
            self.process_manager.shutdown_all(graceful=True)
            self.state.is_running = False
    
    async def _message_loop(self) -> None:
        """消息处理循环"""
        while True:
            try:
                # 非阻塞接收消息
                message = self.process_manager.receive_message(timeout=0.1)
                
                if message:
                    await self._handle_message(message)
                else:
                    await asyncio.sleep(0.05)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"消息处理异常: {e}")
    
    async def _handle_message(self, message: ProcessMessage) -> None:
        """处理来自 Agent 的消息"""
        # 检查是否有等待此消息的 Future
        if message.correlation_id and message.correlation_id in self._pending_responses:
            future = self._pending_responses.pop(message.correlation_id)
            if not future.done():
                future.set_result(message)
            return
        
        # 处理任务进度消息
        if message.type == ProcessMessageType.TASK_PROGRESS:
            task_id = message.payload.get("task_id")
            progress = message.payload.get("progress", 0)
            logger.debug(f"任务进度: {task_id} - {progress}%")
    
    async def _send_and_wait(
        self,
        agent_id: str,
        message: ProcessMessage,
        timeout: float,
    ) -> Optional[ProcessMessage]:
        """发送消息并等待响应"""
        # 创建 Future 等待响应
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[message.id] = future
        
        # 发送消息
        self.process_manager.send_to_agent(agent_id, message)
        
        try:
            # 等待响应
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_responses.pop(message.id, None)
            logger.warning(f"等待 {agent_id} 响应超时")
            return None
    
    async def _planning_phase(self, goal: str, iteration_id: int) -> None:
        """规划阶段"""
        iteration = self.state.get_current_iteration()
        iteration.status = IterationStatus.PLANNING
        
        logger.info(f"[迭代 {iteration_id}] 规划阶段")
        
        # 构建上下文
        context = {
            "iteration_id": iteration_id,
            "working_directory": self.config.working_directory,
        }
        
        # 发送规划请求
        request = ProcessMessage(
            type=ProcessMessageType.PLAN_REQUEST,
            sender="orchestrator",
            receiver=self.planner_id,
            payload={
                "goal": goal,
                "context": context,
                "iteration_id": iteration_id,
            },
        )
        
        response = await self._send_and_wait(
            self.planner_id,
            request,
            timeout=self.config.planning_timeout,
        )
        
        if not response or not response.payload.get("success"):
            error = response.payload.get("error") if response else "规划超时"
            logger.error(f"规划失败: {error}")
            return
        
        # 处理规划结果
        tasks_data = response.payload.get("tasks", [])
        for task_data in tasks_data:
            task = Task(**task_data)
            await self.task_queue.enqueue(task)
            self.state.total_tasks_created += 1
            iteration.tasks_created += 1
        
        logger.info(f"[迭代 {iteration_id}] 规划完成，创建 {len(tasks_data)} 个任务")
    
    async def _execution_phase(self, iteration_id: int) -> None:
        """执行阶段"""
        iteration = self.state.get_current_iteration()
        iteration.status = IterationStatus.EXECUTING
        
        pending = self.task_queue.get_pending_count(iteration_id)
        logger.info(f"[迭代 {iteration_id}] 执行阶段，{pending} 个任务")
        
        # 并行分发任务给 Workers
        active_tasks: dict[str, asyncio.Task] = {}
        available_workers = list(self.worker_ids)
        
        while True:
            # 分配任务给空闲 Worker
            while available_workers:
                task = await self.task_queue.dequeue(iteration_id, timeout=0.1)
                if not task:
                    break
                
                worker_id = available_workers.pop(0)
                
                # 发送任务
                request = ProcessMessage(
                    type=ProcessMessageType.TASK_ASSIGN,
                    sender="orchestrator",
                    receiver=worker_id,
                    payload={"task": task.model_dump()},
                )
                
                # 创建异步等待任务
                async_task = asyncio.create_task(
                    self._send_and_wait(worker_id, request, self.config.execution_timeout)
                )
                active_tasks[worker_id] = async_task
                
                logger.debug(f"任务 {task.id} 分配给 {worker_id}")
            
            # 如果没有活动任务，检查是否完成
            if not active_tasks:
                if self.task_queue.is_iteration_complete(iteration_id):
                    break
                await asyncio.sleep(0.1)
                continue
            
            # 等待任意任务完成
            done, pending_tasks = await asyncio.wait(
                active_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0,
            )
            
            # 处理完成的任务
            for completed in done:
                # 找到对应的 worker
                worker_id = None
                for wid, task in active_tasks.items():
                    if task == completed:
                        worker_id = wid
                        break
                
                if worker_id:
                    del active_tasks[worker_id]
                    available_workers.append(worker_id)
                    
                    # 处理结果
                    try:
                        response = completed.result()
                        if response:
                            task_id = response.payload.get("task_id")
                            success = response.payload.get("success", False)
                            
                            # 更新任务状态
                            task = self.task_queue.get_task(task_id)
                            if task:
                                if success:
                                    task.complete(response.payload)
                                    iteration.tasks_completed += 1
                                    self.state.total_tasks_completed += 1
                                else:
                                    task.fail(response.payload.get("error", "未知错误"))
                                    iteration.tasks_failed += 1
                                    self.state.total_tasks_failed += 1
                                self.task_queue.update_task(task)
                    except Exception as e:
                        logger.error(f"处理任务结果异常: {e}")
        
        stats = self.task_queue.get_statistics(iteration_id)
        logger.info(f"[迭代 {iteration_id}] 执行完成: {stats['completed']} 成功, {stats['failed']} 失败")
    
    async def _review_phase(self, goal: str, iteration_id: int) -> ReviewDecision:
        """评审阶段"""
        iteration = self.state.get_current_iteration()
        iteration.status = IterationStatus.REVIEWING
        
        logger.info(f"[迭代 {iteration_id}] 评审阶段")
        
        # 收集任务信息
        tasks = self.task_queue.get_tasks_by_iteration(iteration_id)
        completed = [
            {"id": t.id, "title": t.title, "result": t.result}
            for t in tasks if t.status == TaskStatus.COMPLETED
        ]
        failed = [
            {"id": t.id, "title": t.title, "error": t.error}
            for t in tasks if t.status == TaskStatus.FAILED
        ]
        
        # 发送评审请求
        request = ProcessMessage(
            type=ProcessMessageType.REVIEW_REQUEST,
            sender="orchestrator",
            receiver=self.reviewer_id,
            payload={
                "goal": goal,
                "iteration_id": iteration_id,
                "tasks_completed": completed,
                "tasks_failed": failed,
            },
        )
        
        response = await self._send_and_wait(
            self.reviewer_id,
            request,
            timeout=self.config.review_timeout,
        )
        
        if not response:
            logger.warning("评审超时，默认继续")
            return ReviewDecision.CONTINUE
        
        decision_str = response.payload.get("decision", "continue")
        decision = ReviewDecision(decision_str)
        
        iteration.review_passed = decision == ReviewDecision.COMPLETE
        iteration.review_feedback = response.payload.get("summary", "")
        iteration.status = IterationStatus.COMPLETED
        
        logger.info(f"[迭代 {iteration_id}] 评审决策: {decision.value}")
        logger.info(f"[迭代 {iteration_id}] 评审得分: {response.payload.get('score', 'N/A')}")
        
        return decision
    
    def _generate_final_result(self) -> dict[str, Any]:
        """生成最终结果"""
        process_info = self.process_manager.get_all_process_info()
        
        return {
            "success": self.state.is_completed,
            "goal": self.state.goal,
            "iterations_completed": self.state.current_iteration,
            "total_tasks_created": self.state.total_tasks_created,
            "total_tasks_completed": self.state.total_tasks_completed,
            "total_tasks_failed": self.state.total_tasks_failed,
            "process_info": process_info,
            "iterations": [
                {
                    "id": it.iteration_id,
                    "status": it.status.value,
                    "tasks_created": it.tasks_created,
                    "tasks_completed": it.tasks_completed,
                    "tasks_failed": it.tasks_failed,
                    "review_passed": it.review_passed,
                }
                for it in self.state.iterations
            ],
        }
