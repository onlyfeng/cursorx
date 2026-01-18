"""多进程编排器

基于多进程架构的编排器，每个 Agent 作为独立进程运行
"""
import asyncio
import uuid
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from agents.committer import CommitterAgent, CommitterConfig
from agents.planner_process import PlannerAgentProcess
from agents.reviewer_process import ReviewDecision, ReviewerAgentProcess
from agents.worker_process import WorkerAgentProcess
from core.base import AgentRole
from core.knowledge import (
    CURSOR_KEYWORDS,
    MAX_CHARS_PER_DOC,
    MAX_KNOWLEDGE_DOCS,
    MAX_TOTAL_KNOWLEDGE_CHARS,
    is_cursor_related,
    truncate_knowledge_docs,
)
from core.state import CommitContext, CommitPolicy, IterationStatus, SystemState
from cursor.client import CursorAgentConfig
from process.manager import AgentProcessManager
from process.message_queue import ProcessMessage, ProcessMessageType
from tasks.queue import TaskQueue
from tasks.task import Task, TaskStatus

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
    stream_events_enabled: bool = True   # 默认启用流式日志
    stream_log_console: bool = True
    stream_log_detail_dir: str = "logs/stream_json/detail/"
    stream_log_raw_dir: str = "logs/stream_json/raw/"

    # 自动提交配置（与 OrchestratorConfig 对齐）
    enable_auto_commit: bool = True    # 默认启用自动提交
    auto_push: bool = False            # 是否自动推送
    commit_on_complete: bool = True    # 仅在完成时提交
    commit_per_iteration: bool = False # 每次迭代都提交

    # 知识库配置（Cursor 相关问题自动搜索）
    # 注意: 实际的 payload 上限常量定义在 core/knowledge.py 中
    # - MAX_KNOWLEDGE_DOCS: 最大文档数 (默认 3)
    # - MAX_CHARS_PER_DOC: 单文档最大字符 (默认 1200)
    # - MAX_TOTAL_KNOWLEDGE_CHARS: 总字符上限 (默认 3000)
    # - 降级策略: truncate_knowledge_docs() 函数
    enable_knowledge_search: bool = True                      # 是否启用知识库搜索
    knowledge_search_top_k: int = MAX_KNOWLEDGE_DOCS + 2      # 搜索返回数（略多于展示数，供降级选择）
    knowledge_doc_max_chars: int = MAX_CHARS_PER_DOC          # 单文档截断字符数（与 worker 对齐）

    # 知识库注入配置（任务级知识库上下文增强）
    # 当任务触发关键词命中时，自动搜索并注入知识库文档
    enable_knowledge_injection: bool = True                   # 是否启用知识库注入
    knowledge_top_k: int = MAX_KNOWLEDGE_DOCS                 # 注入时使用的最大文档数
    knowledge_max_chars_per_doc: int = MAX_CHARS_PER_DOC      # 注入时单文档最大字符数
    knowledge_max_total_chars: int = MAX_TOTAL_KNOWLEDGE_CHARS  # 注入时总字符上限
    knowledge_trigger_keywords: list[str] = CURSOR_KEYWORDS   # 触发知识库注入的关键词列表


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

        # 初始化 CommitterAgent（运行在 orchestrator 主进程中，避免新增子进程复杂度）
        self.committer: Optional[CommitterAgent] = None
        if config.enable_auto_commit:
            committer_cursor_config = CursorAgentConfig(
                stream_events_enabled=config.stream_events_enabled,
                stream_log_console=config.stream_log_console,
                stream_log_detail_dir=config.stream_log_detail_dir,
                stream_log_raw_dir=config.stream_log_raw_dir,
            )
            self.committer = CommitterAgent(CommitterConfig(
                working_directory=config.working_directory,
                auto_push=config.auto_push,
                cursor_config=committer_cursor_config,
            ))
            # 注册 Committer 到系统状态
            self.state.register_agent("committer", AgentRole.COMMITTER)
            logger.info(f"CommitterAgent 已初始化 (auto_push={config.auto_push})")

        # 初始化知识库存储（延迟初始化，用于 Cursor 相关任务的上下文增强）
        self._knowledge_storage = None
        self._knowledge_initialized = False

    def _should_continue_iteration(self) -> bool:
        """判断是否应该继续迭代

        Returns:
            True 表示继续迭代，False 表示停止
        """
        # 无限迭代模式（max_iterations == -1）
        if self.config.max_iterations == -1:
            return True

        # 正常模式：检查是否达到最大迭代次数
        return self.state.current_iteration < self.state.max_iterations

    async def _init_knowledge_storage(self) -> bool:
        """延迟初始化知识库存储

        Returns:
            是否初始化成功
        """
        if self._knowledge_initialized:
            return self._knowledge_storage is not None

        self._knowledge_initialized = True

        if not self.config.enable_knowledge_search:
            return False

        try:
            from knowledge.storage import KnowledgeStorage, StorageConfig

            storage_config = StorageConfig(
                enable_vector_index=False,  # 多进程场景下禁用向量索引，使用关键词搜索
            )
            self._knowledge_storage = KnowledgeStorage(
                config=storage_config,
                workspace_root=self.config.working_directory,
            )
            await self._knowledge_storage.initialize()
            logger.info("知识库存储已初始化")
            return True
        except Exception as e:
            logger.warning(f"知识库存储初始化失败，将跳过知识库检索: {e}")
            self._knowledge_storage = None
            return False

    def _is_knowledge_trigger(self, text: str) -> bool:
        """检查文本是否触发知识库注入

        使用配置的关键词列表进行匹配。

        Args:
            text: 要检测的文本

        Returns:
            是否触发知识库注入
        """
        text_lower = text.lower()
        keywords = self.config.knowledge_trigger_keywords
        return any(kw.lower() in text_lower for kw in keywords)

    async def _search_knowledge_for_task(self, task: Task) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """为任务搜索知识库文档

        当任务触发知识库关键词时，自动搜索知识库补充上下文。

        Args:
            task: 任务对象

        Returns:
            tuple: (知识库文档列表, 注入统计指标)
                - knowledge_docs: 用于注入 task context 的文档列表
                - metrics: 包含 triggered, matched, injected, truncated 等指标
        """
        # 初始化统计指标
        metrics: dict[str, Any] = {
            "triggered": False,      # 是否触发了关键词检测
            "matched": 0,            # 搜索命中的原始文档数
            "injected": 0,           # 实际注入的文档数
            "truncated": False,      # 是否发生了截断
            "total_chars": 0,        # 注入的总字符数
            "keywords_matched": [],  # 命中的关键词列表
        }

        # 检查是否启用知识库搜索和注入
        if not self.config.enable_knowledge_search or not self.config.enable_knowledge_injection:
            return [], metrics

        # 确保知识库已初始化
        if not await self._init_knowledge_storage():
            return [], metrics

        if self._knowledge_storage is None:
            return [], metrics

        # 构建搜索查询：组合 title + description + instruction
        query_parts = [task.title]
        if task.description:
            query_parts.append(task.description)
        if task.instruction:
            query_parts.append(task.instruction)
        query = ". ".join(query_parts)

        # 检查是否触发知识库注入（使用配置的关键词列表）
        if not self._is_knowledge_trigger(query):
            return [], metrics

        # 记录触发状态和命中的关键词
        metrics["triggered"] = True
        query_lower = query.lower()
        metrics["keywords_matched"] = [
            kw for kw in self.config.knowledge_trigger_keywords
            if kw.lower() in query_lower
        ]

        try:
            logger.debug(f"[任务 {task.id}] 知识库注入触发，关键词: {metrics['keywords_matched']}")

            # 使用关键词搜索（多进程环境下更稳定）
            results = await self._knowledge_storage.search(
                query=query,
                limit=self.config.knowledge_search_top_k,
                search_content=True,
                mode="keyword",
            )

            if not results:
                logger.debug(f"[任务 {task.id}] 知识库搜索无结果")
                return [], metrics

            metrics["matched"] = len(results)

            # 加载文档内容并构建 knowledge_docs
            raw_docs: list[dict[str, Any]] = []

            for result in results:
                try:
                    doc = await self._knowledge_storage.load_document(result.doc_id)
                    if doc:
                        raw_docs.append({
                            "title": doc.title or result.title,
                            "url": doc.url or result.url,
                            "content": doc.content,
                            "score": result.score,
                            "source": "cursor-docs",
                        })
                except Exception as e:
                    logger.debug(f"加载文档 {result.doc_id} 失败: {e}")
                    continue

            if not raw_docs:
                return [], metrics

            # 使用配置的参数进行截断处理
            truncated_docs, total_chars = truncate_knowledge_docs(
                docs=raw_docs,
                max_docs=self.config.knowledge_top_k,
                max_chars_per_doc=self.config.knowledge_max_chars_per_doc,
                max_total_chars=self.config.knowledge_max_total_chars,
            )

            # 更新统计指标
            metrics["injected"] = len(truncated_docs)
            metrics["total_chars"] = total_chars
            metrics["truncated"] = any(d.get("truncated", False) for d in truncated_docs)

            # 记录详细的注入日志
            if truncated_docs:
                logger.info(
                    f"[任务 {task.id}] 知识库注入: "
                    f"触发={metrics['triggered']}, "
                    f"命中={metrics['matched']}, "
                    f"注入={metrics['injected']}, "
                    f"截断={metrics['truncated']}, "
                    f"字符={metrics['total_chars']}"
                )

            return truncated_docs, metrics

        except Exception as e:
            # 静默降级：记录日志但不影响任务执行
            logger.warning(f"[任务 {task.id}] 知识库搜索失败，跳过: {e}")
            return [], metrics

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
        # 注册 Planner 到系统状态
        self.state.register_agent(self.planner_id, AgentRole.PLANNER)
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
            # 注册 Worker 到系统状态
            self.state.register_agent(worker_id, AgentRole.WORKER)
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
        # 注册 Reviewer 到系统状态
        self.state.register_agent(self.reviewer_id, AgentRole.REVIEWER)
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
        if self.config.max_iterations == -1:
            logger.info("最大迭代: 无限制（直到完成或用户中断）")
        else:
            logger.info(f"最大迭代: {self.config.max_iterations}")
        logger.info("=" * 60)

        try:
            # 1. 创建 Agent 进程
            self._spawn_agents()

            # 2. 等待所有进程就绪
            if not self.process_manager.wait_all_ready(timeout=30.0):
                raise RuntimeError("Agent 进程启动失败")

            # 3. 启动消息处理循环
            message_task = asyncio.create_task(self._message_loop())

            # 4. 执行主循环（max_iterations == -1 表示无限迭代）
            try:
                while self._should_continue_iteration():
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

                    # 提交阶段（根据配置和评审决策判断是否执行）
                    if self._should_commit(decision):
                        await self._commit_phase(iteration.iteration_id, decision)

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

                # 知识库检索：检测触发关键词并注入知识库上下文
                task_data = task.model_dump()
                if self.config.enable_knowledge_search and self.config.enable_knowledge_injection:
                    try:
                        knowledge_docs, injection_metrics = await self._search_knowledge_for_task(task)
                        if knowledge_docs:
                            # 将知识库文档注入 task_data 的 context 字段
                            if "context" not in task_data or task_data["context"] is None:
                                task_data["context"] = {}
                            task_data["context"]["knowledge_docs"] = knowledge_docs
                            # 将注入指标也添加到 context 中供调试
                            task_data["context"]["knowledge_injection_metrics"] = injection_metrics
                    except Exception as e:
                        # 静默降级：记录日志但不影响任务分发
                        logger.debug(f"[任务 {task.id}] 知识库检索异常，跳过: {e}")

                # 发送任务
                request = ProcessMessage(
                    type=ProcessMessageType.TASK_ASSIGN,
                    sender="orchestrator",
                    receiver=worker_id,
                    payload={"task": task_data},
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

        # 收集任务信息（使用统一的 to_commit_entry 格式）
        tasks = self.task_queue.get_tasks_by_iteration(iteration_id)
        completed = [
            t.to_commit_entry()
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

    def _should_commit(self, decision: ReviewDecision) -> bool:
        """判断是否应该执行提交

        提交触发策略遵循以下优先级规则（参见 CommitPolicy 类型定义）：

        触发优先级（从高到低）:
            1. enable_auto_commit=False 或 committer 未初始化 → 禁用所有自动提交
            2. commit_per_iteration=True → 每次迭代完成后都提交
            3. commit_on_complete=True + decision==COMPLETE → 仅在目标完成时提交

        评审决策对提交的影响:
            - COMPLETE: 允许提交（如果 commit_on_complete=True 或 commit_per_iteration=True）
            - CONTINUE: 仅当 commit_per_iteration=True 时允许提交
            - ADJUST: 仅当 commit_per_iteration=True 时允许提交
            - ABORT: 仅当 commit_per_iteration=True 时允许提交（记录中间进度）

        Args:
            decision: 评审决策 (ReviewDecision 枚举)

        Returns:
            True 表示应该提交，False 表示跳过

        See Also:
            - CommitPolicy: 提交策略配置类型
            - _commit_phase: 执行实际提交的方法
        """
        # 未启用自动提交或 committer 未初始化
        if not self.config.enable_auto_commit or not self.committer:
            return False

        # 使用 CommitPolicy 进行判断（优先级: commit_per_iteration > commit_on_complete）
        policy = CommitPolicy(
            enable_auto_commit=self.config.enable_auto_commit,
            commit_per_iteration=self.config.commit_per_iteration,
            commit_on_complete=self.config.commit_on_complete,
            auto_push=self.config.auto_push,
        )
        return policy.should_commit(decision.value)

    async def _commit_phase(self, iteration_id: int, decision: ReviewDecision) -> dict[str, Any]:
        """提交阶段

        从 TaskQueue 收集已完成任务，构建 CommitContext，执行 Git 提交。

        提交输入（CommitContext）:
            从 TaskQueue.get_tasks_by_iteration() 获取已完成任务，
            使用 Task.to_commit_entry() 方法获取标准格式，必须包含：
            - id: 任务唯一标识符
            - title: 任务标题（用于生成 commit message）
            - result: 任务执行结果（包含变更详情）

            可选字段:
            - description: 任务描述（增强 commit message 可读性）

        错误处理:
            - commit 失败: result.success=False, error 记录到 commit_result["error"]
            - 无变更: result.success=True, commit_hash=None, message="No changes to commit"
            - push 失败: result.success 取决于 commit 是否成功，push_error 记录错误信息
            - 提交失败不影响主流程 result.success，仅记录到 iteration 状态

        IterationState 字段填充:
            - iteration.commit_hash: Git 提交哈希（无变更时为空字符串）
            - iteration.commit_message: 提交信息
            - iteration.pushed: 是否已推送到远程
            - iteration.commit_files: 变更的文件列表

        Args:
            iteration_id: 迭代 ID
            decision: 评审决策

        Returns:
            提交结果字典:
            {
                "success": bool,           # 提交是否成功
                "commit_hash": str | None, # Git 提交哈希
                "message": str,            # 提交信息或错误描述
                "files_changed": list[str],# 变更的文件列表
                "pushed": bool,            # 是否已推送
                "push_error": str | None,  # 推送错误信息（如有）
            }

        See Also:
            - CommitContext: 提交上下文数据类型
            - CommitPolicy: 提交策略配置类型
            - Task.to_commit_entry: 任务转提交条目的标准方法
        """
        if not self.committer:
            return {"success": False, "error": "Committer not initialized"}

        iteration = self.state.get_current_iteration()
        iteration.status = IterationStatus.COMMITTING

        logger.info(f"[迭代 {iteration_id}] 提交阶段开始")

        # 收集已完成的任务（使用统一的 to_commit_entry 格式）
        # to_commit_entry() 返回: {"id", "title", "description", "result"}
        tasks = self.task_queue.get_tasks_by_iteration(iteration_id)
        completed_tasks = [
            t.to_commit_entry()
            for t in tasks if t.status == TaskStatus.COMPLETED
        ]

        # 构建 CommitContext（用于文档化和验证）
        commit_context = CommitContext(
            iteration_id=iteration_id,
            tasks_completed=completed_tasks,
            review_decision=decision.value,
            auto_push=self.config.auto_push,
        )

        # 验证任务数据完整性
        validation_errors = commit_context.validate_tasks()
        if validation_errors:
            logger.warning(f"[迭代 {iteration_id}] 任务数据验证警告: {validation_errors}")

        # 执行提交
        commit_result = await self.committer.commit_iteration(
            iteration_id=commit_context.iteration_id,
            tasks_completed=commit_context.tasks_completed,
            review_decision=commit_context.review_decision,
            auto_push=commit_context.auto_push,
        )

        # 记录提交结果到 iteration（填充 IterationState 字段）
        iteration.commit_hash = commit_result.get('commit_hash', '')
        iteration.commit_message = commit_result.get('message', '')
        iteration.pushed = commit_result.get('pushed', False)
        iteration.commit_files = commit_result.get('files_changed', [])

        # 记录错误信息到 iteration（commit 失败或 push 失败均不中断主流程）
        success = commit_result.get("success", False)
        if not success:
            iteration.commit_error = commit_result.get('error', 'Unknown commit error')
        if commit_result.get("push_error"):
            iteration.push_error = commit_result.get('push_error')

        result_status = "success" if success else "failed"

        logger.info(f"[迭代 {iteration_id}] 提交完成: {result_status}")
        if commit_result.get("commit_hash"):
            logger.info(f"[迭代 {iteration_id}] 提交哈希: {commit_result.get('commit_hash')}")
        if commit_result.get("pushed"):
            logger.info(f"[迭代 {iteration_id}] 已推送到远程仓库")
        if iteration.commit_error:
            logger.warning(f"[迭代 {iteration_id}] 提交失败: {iteration.commit_error}")
        if iteration.push_error:
            logger.warning(f"[迭代 {iteration_id}] 推送失败: {iteration.push_error}")

        return commit_result

    def _generate_final_result(self) -> dict[str, Any]:
        """生成最终结果

        输出结构（用于测试断言）:
            {
                "success": bool,                    # 目标是否完成
                "goal": str,                        # 用户目标
                "iterations_completed": int,        # 完成的迭代数
                "total_tasks_created": int,         # 创建的任务总数
                "total_tasks_completed": int,       # 完成的任务总数
                "total_tasks_failed": int,          # 失败的任务总数
                "process_info": dict,               # 进程信息（MP 特有）
                "commits": dict,                    # 提交摘要（来自 CommitterAgent）
                "pushed": bool,                     # 是否有提交被推送到远程
                "iterations": [                     # 各迭代详情
                    {
                        "id": int,                  # 迭代 ID
                        "status": str,              # 迭代状态
                        "tasks_created": int,       # 本轮创建任务数
                        "tasks_completed": int,     # 本轮完成任务数
                        "tasks_failed": int,        # 本轮失败任务数
                        "review_passed": bool,      # 评审是否通过
                        "commit_hash": str | None,  # Git 提交哈希
                        "commit_message": str | None, # 提交信息
                        "commit_pushed": bool,      # 是否已推送
                    },
                    ...
                ],
            }

        Returns:
            包含完整执行结果的字典
        """
        process_info = self.process_manager.get_all_process_info()

        # 获取提交信息
        commit_summary = {}
        pushed = False
        if self.committer:
            commit_summary = self.committer.get_commit_summary()
            pushed = commit_summary.get("pushed_commits", 0) > 0

        return {
            "success": self.state.is_completed,
            "goal": self.state.goal,
            "iterations_completed": self.state.current_iteration,
            "total_tasks_created": self.state.total_tasks_created,
            "total_tasks_completed": self.state.total_tasks_completed,
            "total_tasks_failed": self.state.total_tasks_failed,
            "process_info": process_info,
            "commits": commit_summary,
            "pushed": pushed,
            "iterations": [
                {
                    "id": it.iteration_id,
                    "status": it.status.value,
                    "tasks_created": it.tasks_created,
                    "tasks_completed": it.tasks_completed,
                    "tasks_failed": it.tasks_failed,
                    "review_passed": it.review_passed,
                    "commit_hash": it.commit_hash,
                    "commit_message": it.commit_message,
                    "commit_pushed": it.pushed,
                    "commit_error": it.commit_error,
                    "push_error": it.push_error,
                }
                for it in self.state.iterations
            ],
        }
