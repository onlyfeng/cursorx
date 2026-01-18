"""编排器 - 系统核心协调组件

协调规划者、执行者、评审者的工作流程
支持知识库集成
"""
import asyncio
from typing import Any, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field
from loguru import logger

from core.state import SystemState, IterationStatus
from core.base import AgentRole
from tasks.queue import TaskQueue
from agents.planner import PlannerAgent, PlannerConfig
from agents.reviewer import ReviewerAgent, ReviewerConfig, ReviewDecision
from agents.committer import CommitterAgent, CommitterConfig
from cursor.client import CursorAgentConfig
from cursor.executor import ExecutionMode
from cursor.cloud_client import CloudAuthConfig
from .worker_pool import WorkerPool

if TYPE_CHECKING:
    from knowledge import KnowledgeManager


class OrchestratorConfig(BaseModel):
    """编排器配置"""
    working_directory: str = "."
    max_iterations: int = 10           # 最大迭代次数
    worker_pool_size: int = 3          # Worker 池大小
    enable_sub_planners: bool = True   # 是否启用子规划者
    strict_review: bool = False        # 严格评审模式
    cursor_config: CursorAgentConfig = Field(default_factory=CursorAgentConfig)
    stream_events_enabled: bool = True   # 默认启用流式日志
    stream_log_console: bool = True
    stream_log_detail_dir: str = "logs/stream_json/detail/"
    stream_log_raw_dir: str = "logs/stream_json/raw/"
    # 自动提交配置
    enable_auto_commit: bool = True    # 默认启用自动提交
    auto_push: bool = False            # 是否自动推送
    commit_on_complete: bool = True    # 仅在完成时提交
    commit_per_iteration: bool = False # 每次迭代都提交
    # Cloud Agent 配置
    execution_mode: ExecutionMode = ExecutionMode.CLI  # 执行模式: cli, cloud, auto
    cloud_auth_config: Optional[CloudAuthConfig] = None  # Cloud 认证配置


class Orchestrator:
    """编排器
    
    整个系统的核心协调组件，负责:
    1. 管理系统状态
    2. 协调规划-执行-评审循环
    3. 控制迭代流程
    4. 聚合执行结果
    """
    
    def __init__(
        self,
        config: OrchestratorConfig,
        knowledge_manager: Optional["KnowledgeManager"] = None,
    ):
        self.config = config
        self._apply_stream_config()
        
        # 系统状态
        self.state = SystemState(
            working_directory=config.working_directory,
            max_iterations=config.max_iterations,
        )
        
        # 任务队列
        self.task_queue = TaskQueue()
        
        # 知识库管理器（用于 Cursor 相关问题自动搜索）
        self._knowledge_manager: Optional["KnowledgeManager"] = knowledge_manager
        
        # 初始化 Agents
        planner_cursor_config = config.cursor_config.model_copy(deep=True)
        reviewer_cursor_config = config.cursor_config.model_copy(deep=True)
        worker_cursor_config = config.cursor_config.model_copy(deep=True)
        
        # 记录执行模式
        logger.info(f"编排器使用执行模式: {config.execution_mode.value}")

        self.planner = PlannerAgent(PlannerConfig(
            working_directory=config.working_directory,
            cursor_config=planner_cursor_config,
            execution_mode=config.execution_mode,
            cloud_auth_config=config.cloud_auth_config,
        ))
        
        self.reviewer = ReviewerAgent(ReviewerConfig(
            working_directory=config.working_directory,
            strict_mode=config.strict_review,
            cursor_config=reviewer_cursor_config,
            execution_mode=config.execution_mode,
            cloud_auth_config=config.cloud_auth_config,
        ))
        
        # Worker 池（传递知识库管理器和执行模式配置）
        from agents.worker import WorkerConfig
        self.worker_pool = WorkerPool(
            size=config.worker_pool_size,
            worker_config=WorkerConfig(
                working_directory=config.working_directory,
                cursor_config=worker_cursor_config,
                execution_mode=config.execution_mode,
                cloud_auth_config=config.cloud_auth_config,
            ),
            knowledge_manager=knowledge_manager,
        )
        self.worker_pool.initialize()
        
        # 初始化 CommitterAgent（当 enable_auto_commit=True 时）
        self.committer: Optional[CommitterAgent] = None
        if config.enable_auto_commit:
            committer_cursor_config = config.cursor_config.model_copy(deep=True)
            self.committer = CommitterAgent(CommitterConfig(
                working_directory=config.working_directory,
                auto_push=config.auto_push,
                cursor_config=committer_cursor_config,
            ))
        
        # 注册 Agents
        self.state.register_agent(self.planner.id, AgentRole.PLANNER)
        self.state.register_agent(self.reviewer.id, AgentRole.REVIEWER)
        for worker in self.worker_pool.workers:
            self.state.register_agent(worker.id, AgentRole.WORKER)
        if self.committer:
            self.state.register_agent(self.committer.id, AgentRole.COMMITTER)

    def _apply_stream_config(self) -> None:
        """将流式日志配置注入 CursorAgentConfig"""
        cursor_config = self.config.cursor_config
        cursor_config.stream_events_enabled = self.config.stream_events_enabled
        cursor_config.stream_log_console = self.config.stream_log_console
        cursor_config.stream_log_detail_dir = self.config.stream_log_detail_dir
        cursor_config.stream_log_raw_dir = self.config.stream_log_raw_dir
    
    def set_knowledge_manager(self, manager: "KnowledgeManager") -> None:
        """设置知识库管理器（延迟初始化）
        
        Args:
            manager: KnowledgeManager 实例
        """
        self._knowledge_manager = manager
        self.worker_pool.set_knowledge_manager(manager)
        logger.info("Orchestrator 已绑定知识库管理器")
    
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
    
    async def run(self, goal: str) -> dict[str, Any]:
        """运行编排器完成目标
        
        Args:
            goal: 用户目标
            
        Returns:
            执行结果
        """
        self.state.goal = goal
        self.state.is_running = True
        
        logger.info("=== 开始执行目标 ===")
        logger.info(f"目标: {goal}")
        logger.info(f"工作目录: {self.config.working_directory}")
        if self.config.max_iterations == -1:
            logger.info("最大迭代: 无限制（直到完成或用户中断）")
        else:
            logger.info(f"最大迭代: {self.config.max_iterations}")
        
        try:
            # max_iterations == -1 表示无限迭代
            while self._should_continue_iteration():
                # 开始新迭代
                iteration = self.state.start_new_iteration()
                logger.info(f"\n{'='*50}")
                logger.info(f"=== 迭代 {iteration.iteration_id} 开始 ===")
                
                # 1. 规划阶段
                await self._planning_phase(goal, iteration.iteration_id)
                
                # 检查是否有任务
                if self.task_queue.get_pending_count(iteration.iteration_id) == 0:
                    logger.warning("规划阶段未产生任务，跳过执行阶段")
                    iteration.status = IterationStatus.COMPLETED
                    continue
                
                # 2. 执行阶段
                await self._execution_phase(iteration.iteration_id)
                
                # 3. 评审阶段
                decision = await self._review_phase(goal, iteration.iteration_id)
                
                # 4. 提交阶段（根据配置和评审决策判断是否执行）
                if self._should_commit(decision):
                    await self._commit_phase(iteration.iteration_id, decision)
                
                # 根据评审决策处理
                if decision == ReviewDecision.COMPLETE:
                    logger.info("=== 目标已完成 ===")
                    self.state.is_completed = True
                    break
                elif decision == ReviewDecision.ABORT:
                    logger.error("=== 评审决定终止 ===")
                    break
                elif decision == ReviewDecision.ADJUST:
                    logger.info("=== 需要调整方向 ===")
                    # 调整会在下一轮规划中体现
                
                # 为下一轮迭代重置状态
                self.state.reset_for_new_iteration()
                await self._reset_for_next_iteration()
            
            # 生成最终结果
            return self._generate_final_result()
            
        except Exception as e:
            logger.exception(f"编排器执行异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "iterations_completed": self.state.current_iteration,
            }
        finally:
            self.state.is_running = False
    
    async def _planning_phase(self, goal: str, iteration_id: int) -> None:
        """规划阶段"""
        iteration = self.state.get_current_iteration()
        iteration.status = IterationStatus.PLANNING
        
        logger.info(f"[迭代 {iteration_id}] 规划阶段开始")
        
        # 构建规划上下文
        context = {
            "iteration_id": iteration_id,
            "working_directory": self.config.working_directory,
        }
        
        # 如果有之前的评审，添加反馈
        if self.reviewer.review_history:
            last_review = self.reviewer.review_history[-1]
            context["previous_review"] = {
                "score": last_review.get("score"),
                "suggestions": last_review.get("suggestions", []),
                "next_focus": last_review.get("next_iteration_focus"),
            }
        
        # 执行规划
        plan_result = await self.planner.execute(goal, context)
        
        if not plan_result.get("success"):
            logger.error(f"规划失败: {plan_result.get('error')}")
            return
        
        # 处理规划结果中的任务
        tasks_data = plan_result.get("tasks", [])
        for task_data in tasks_data:
            task = self.planner.create_task_from_plan(task_data, iteration_id)
            await self.task_queue.enqueue(task)
            self.state.total_tasks_created += 1
            iteration.tasks_created += 1
        
        logger.info(f"[迭代 {iteration_id}] 规划完成，创建 {len(tasks_data)} 个任务")
        
        # 处理子规划者需求
        if self.config.enable_sub_planners:
            sub_planners_needed = plan_result.get("sub_planners_needed", [])
            await self._handle_sub_planners(sub_planners_needed, iteration_id)
    
    async def _handle_sub_planners(
        self,
        sub_planners_needed: list[dict],
        iteration_id: int,
    ) -> None:
        """处理子规划者"""
        if not sub_planners_needed:
            return
        
        logger.info(f"需要 {len(sub_planners_needed)} 个子规划者")
        
        # 并行执行子规划
        sub_tasks = []
        for sp_info in sub_planners_needed:
            area = sp_info.get("area", "")
            reason = sp_info.get("reason", "")
            
            try:
                sub_planner = await self.planner.spawn_sub_planner(
                    area=area,
                    context={"reason": reason},
                )
                self.state.register_agent(sub_planner.id, AgentRole.SUB_PLANNER)
                
                # 创建子规划任务
                sub_task = asyncio.create_task(
                    sub_planner.execute(
                        f"深入分析和规划: {area}",
                        {"parent_analysis": reason},
                    )
                )
                sub_tasks.append((sub_planner, sub_task))
                
            except ValueError as e:
                logger.warning(f"无法创建子规划者: {e}")
        
        # 等待所有子规划完成
        for sub_planner, sub_task in sub_tasks:
            try:
                result = await sub_task
                if result.get("success"):
                    # 将子规划的任务加入队列
                    for task_data in result.get("tasks", []):
                        task = sub_planner.create_task_from_plan(task_data, iteration_id)
                        await self.task_queue.enqueue(task)
                        self.state.total_tasks_created += 1
            except Exception as e:
                logger.error(f"子规划者执行失败: {e}")
    
    async def _execution_phase(self, iteration_id: int) -> None:
        """执行阶段"""
        iteration = self.state.get_current_iteration()
        iteration.status = IterationStatus.EXECUTING
        
        pending = self.task_queue.get_pending_count(iteration_id)
        logger.info(f"[迭代 {iteration_id}] 执行阶段开始，{pending} 个任务待处理")
        
        # 启动 Worker 池处理任务
        await self.worker_pool.start(self.task_queue, iteration_id)
        
        # 更新统计
        stats = self.task_queue.get_statistics(iteration_id)
        iteration.tasks_completed = stats["completed"]
        iteration.tasks_failed = stats["failed"]
        self.state.total_tasks_completed += stats["completed"]
        self.state.total_tasks_failed += stats["failed"]
        
        logger.info(f"[迭代 {iteration_id}] 执行完成: {stats['completed']} 成功, {stats['failed']} 失败")
    
    async def _review_phase(self, goal: str, iteration_id: int) -> ReviewDecision:
        """评审阶段"""
        iteration = self.state.get_current_iteration()
        iteration.status = IterationStatus.REVIEWING
        
        logger.info(f"[迭代 {iteration_id}] 评审阶段开始")
        
        # 收集已完成和失败的任务
        tasks = self.task_queue.get_tasks_by_iteration(iteration_id)
        completed_tasks = [
            {"id": t.id, "title": t.title, "result": t.result}
            for t in tasks if t.status.value == "completed"
        ]
        failed_tasks = [
            {"id": t.id, "title": t.title, "error": t.error}
            for t in tasks if t.status.value == "failed"
        ]
        
        # 执行评审
        review_result = await self.reviewer.review_iteration(
            goal=goal,
            iteration_id=iteration_id,
            tasks_completed=completed_tasks,
            tasks_failed=failed_tasks,
        )
        
        decision = review_result.get("decision", ReviewDecision.CONTINUE)
        iteration.review_passed = decision == ReviewDecision.COMPLETE
        iteration.review_feedback = review_result.get("summary", "")
        iteration.status = IterationStatus.COMPLETED
        iteration.completed_at = asyncio.get_event_loop().time()
        
        logger.info(f"[迭代 {iteration_id}] 评审决策: {decision.value}")
        logger.info(f"[迭代 {iteration_id}] 评审得分: {review_result.get('score', 'N/A')}")
        
        return decision
    
    def _should_commit(self, decision: ReviewDecision) -> bool:
        """判断是否应该执行提交
        
        Args:
            decision: 评审决策
            
        Returns:
            True 表示应该提交，False 表示跳过
        """
        # 未启用自动提交
        if not self.config.enable_auto_commit or not self.committer:
            return False
        
        # 每次迭代都提交
        if self.config.commit_per_iteration:
            return True
        
        # 仅在完成时提交
        if self.config.commit_on_complete and decision == ReviewDecision.COMPLETE:
            return True
        
        return False
    
    async def _commit_phase(self, iteration_id: int, decision: ReviewDecision) -> dict[str, Any]:
        """提交阶段
        
        Args:
            iteration_id: 迭代 ID
            decision: 评审决策
            
        Returns:
            提交结果
        """
        if not self.committer:
            return {"success": False, "error": "Committer not initialized"}
        
        iteration = self.state.get_current_iteration()
        iteration.status = IterationStatus.COMMITTING
        
        logger.info(f"[迭代 {iteration_id}] 提交阶段开始")
        
        # 收集已完成的任务
        tasks = self.task_queue.get_tasks_by_iteration(iteration_id)
        completed_tasks = [
            {"id": t.id, "title": t.title, "result": t.result}
            for t in tasks if t.status.value == "completed"
        ]
        
        # 执行提交
        commit_result = await self.committer.commit_iteration(
            iteration_id=iteration_id,
            tasks_completed=completed_tasks,
            review_decision=decision.value,
            auto_push=self.config.auto_push,
        )
        
        # 记录提交结果到 iteration
        iteration.commit_hash = commit_result.get('commit_hash', '')
        iteration.commit_message = commit_result.get('message', '')
        iteration.pushed = commit_result.get('pushed', False)
        iteration.commit_files = commit_result.get('files_changed', [])
        
        success = commit_result.get("success", False)
        result_status = "success" if success else "failed"
        
        logger.info(f"[迭代 {iteration_id}] 提交完成: {result_status}")
        if commit_result.get("commit_hash"):
            logger.info(f"[迭代 {iteration_id}] 提交哈希: {commit_result.get('commit_hash')}")
        if commit_result.get("pushed"):
            logger.info(f"[迭代 {iteration_id}] 已推送到远程仓库")
        
        return commit_result
    
    async def _reset_for_next_iteration(self) -> None:
        """为下一轮迭代重置"""
        await self.planner.reset()
        await self.worker_pool.reset()
        # Reviewer 不重置，保留评审历史
        logger.debug("已重置，准备下一轮迭代")
    
    def _generate_final_result(self) -> dict[str, Any]:
        """生成最终结果"""
        review_summary = self.reviewer.get_review_summary()
        worker_stats = self.worker_pool.get_statistics()
        
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
            "final_score": review_summary.get("average_score", 0),
            "review_summary": review_summary,
            "worker_stats": worker_stats,
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
                }
                for it in self.state.iterations
            ],
        }
    
    def get_status(self) -> dict[str, Any]:
        """获取当前状态"""
        return {
            "is_running": self.state.is_running,
            "is_completed": self.state.is_completed,
            "current_iteration": self.state.current_iteration,
            "goal": self.state.goal,
            "task_queue_stats": self.task_queue.get_statistics(self.state.current_iteration),
            "worker_pool_stats": self.worker_pool.get_statistics(),
        }
