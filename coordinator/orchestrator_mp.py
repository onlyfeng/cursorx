"""多进程编排器

基于多进程架构的编排器，每个 Agent 作为独立进程运行
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from knowledge.storage import KnowledgeStorage
from pydantic import BaseModel

from agents.committer import CommitterAgent, CommitterConfig
from agents.planner_process import PlannerAgentProcess
from agents.reviewer_process import ReviewDecision, ReviewerAgentProcess
from agents.worker_process import WorkerAgentProcess
from core.base import AgentRole
from core.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_PLANNING_TIMEOUT,
    DEFAULT_REVIEW_TIMEOUT,
    DEFAULT_REVIEWER_MODEL,
    DEFAULT_WORKER_MODEL,
    DEFAULT_WORKER_POOL_SIZE,
    DEFAULT_WORKER_TIMEOUT,
    get_config,
)
from core.knowledge import (
    CURSOR_KEYWORDS,
    MAX_CHARS_PER_DOC,
    MAX_KNOWLEDGE_DOCS,
    MAX_TOTAL_KNOWLEDGE_CHARS,
    truncate_knowledge_docs,
)
from core.state import CommitContext, CommitPolicy, IterationStatus, SystemState
from cursor.client import CursorAgentConfig
from process.manager import AgentProcessManager, HealthCheckResult
from process.message_queue import ProcessMessage, ProcessMessageType
from tasks.queue import TaskQueue
from tasks.task import Task, TaskStatus


class MultiProcessOrchestratorConfig(BaseModel):
    """多进程编排器配置

    默认值从 config.yaml 加载，通过 core.config 模块统一管理。
    """

    working_directory: str = "."
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    worker_count: int = DEFAULT_WORKER_POOL_SIZE  # Worker 进程数量
    enable_sub_planners: bool = True
    strict_review: bool = False
    # 迭代助手上下文（.iteration + Engram + 规则摘要）
    iteration_context: dict[str, Any] | None = None

    # 超时设置 - 默认值从 core.config 获取
    planning_timeout: float = DEFAULT_PLANNING_TIMEOUT  # 规划超时
    execution_timeout: float = DEFAULT_WORKER_TIMEOUT  # 单任务执行超时
    review_timeout: float = DEFAULT_REVIEW_TIMEOUT  # 评审超时

    # 模型配置 - 默认值从 core.config 获取
    # 可用模型: gpt-5.2-high, gpt-5.2-codex-high, gpt-5.2-codex-xhigh, gpt-5.2-codex 等
    # 使用 `agent models` 查看完整列表
    planner_model: str = DEFAULT_PLANNER_MODEL  # 规划者模型
    worker_model: str = DEFAULT_WORKER_MODEL  # 执行者模型
    reviewer_model: str = DEFAULT_REVIEWER_MODEL  # 评审者模型

    # 执行模式配置（MP 模式主要使用 CLI，但支持配置以便与 basic 编排器保持一致）
    # 注意：MP 编排器内部子进程始终使用 CLI 执行，这些配置主要用于透传和兼容性
    execution_mode: str = "cli"  # 全局执行模式: cli, cloud, auto
    # 角色级执行模式配置（默认继承全局 execution_mode）
    # 若为 None，则使用全局 execution_mode
    # 注意：MP 子进程实际使用 CLI 执行，这些配置用于记录和透传
    planner_execution_mode: str | None = None  # 规划者执行模式
    worker_execution_mode: str | None = None  # 执行者执行模式
    reviewer_execution_mode: str | None = None  # 评审者执行模式

    # 流式日志配置 - tri-state 设计
    # None 表示使用 config.yaml 中的值（通过 get_config().logging.stream_json 获取）
    # 显式传入的值优先（最高优先级）
    # 解析在 MultiProcessOrchestrator._resolve_stream_config() 中进行
    stream_events_enabled: bool | None = None  # 是否启用流式日志
    stream_log_console: bool | None = None  # 是否输出到控制台
    stream_log_detail_dir: str | None = None  # 详细日志目录
    stream_log_raw_dir: str | None = None  # 原始日志目录
    # 流式控制台渲染配置（默认关闭，避免噪声）
    stream_console_renderer: bool = False  # 启用流式控制台渲染器
    stream_advanced_renderer: bool = False  # 使用高级终端渲染器
    stream_typing_effect: bool = False  # 启用打字机效果
    stream_typing_delay: float = 0.02  # 打字延迟（秒）
    stream_word_mode: bool = True  # 逐词输出模式
    stream_color_enabled: bool = True  # 启用颜色输出
    stream_show_word_diff: bool = False  # 显示逐词差异

    # 自动提交配置（与 OrchestratorConfig 对齐）
    enable_auto_commit: bool = False  # 默认禁用自动提交（需显式开启）
    auto_push: bool = False  # 是否自动推送
    commit_on_complete: bool = True  # 仅在完成时提交
    commit_per_iteration: bool = False  # 每次迭代都提交

    # 知识库配置（Cursor 相关问题自动搜索）
    # 注意: 实际的 payload 上限常量定义在 core/knowledge.py 中
    # - MAX_KNOWLEDGE_DOCS: 最大文档数 (默认 3)
    # - MAX_CHARS_PER_DOC: 单文档最大字符 (默认 1200)
    # - MAX_TOTAL_KNOWLEDGE_CHARS: 总字符上限 (默认 3000)
    # - 降级策略: truncate_knowledge_docs() 函数
    enable_knowledge_search: bool = True  # 是否启用知识库搜索
    knowledge_search_top_k: int = MAX_KNOWLEDGE_DOCS + 2  # 搜索返回数（略多于展示数，供降级选择）
    knowledge_doc_max_chars: int = MAX_CHARS_PER_DOC  # 单文档截断字符数（与 worker 对齐）

    # 知识库注入配置（任务级知识库上下文增强）
    # 当任务触发关键词命中时，自动搜索并注入知识库文档
    enable_knowledge_injection: bool = True  # 是否启用知识库注入
    knowledge_top_k: int = MAX_KNOWLEDGE_DOCS  # 注入时使用的最大文档数
    knowledge_max_chars_per_doc: int = MAX_CHARS_PER_DOC  # 注入时单文档最大字符数
    knowledge_max_total_chars: int = MAX_TOTAL_KNOWLEDGE_CHARS  # 注入时总字符上限
    knowledge_trigger_keywords: list[str] = CURSOR_KEYWORDS  # 触发知识库注入的关键词列表

    # 健康检查配置
    # 注意：Worker 在执行任务期间可能无法立即响应心跳，需要合理配置容忍度
    health_check_interval: float = 45.0  # 健康检查间隔（秒）- 增加间隔减少干扰
    health_check_timeout: float = 10.0  # 健康检查超时（秒）- 增加超时容忍任务执行
    execution_health_check_interval: float = 30.0  # 执行阶段健康检查间隔（秒）- 不低于 30s 避免频繁检查
    max_unhealthy_workers: int = -1  # 允许的最大不健康 Worker 数量（-1 表示动态计算为 worker_count-1）
    requeue_on_worker_death: bool = True  # Worker 死亡时是否重新入队任务（False 则标记失败）
    fallback_on_critical_failure: bool = True  # 关键进程（planner/reviewer）不健康时是否降级终止
    skip_busy_workers_in_health_check: bool = True  # 跳过正在执行任务的 Worker（避免误判）
    consecutive_unresponsive_threshold: int = 3  # 连续未响应次数阈值，超过才告警（避免短暂抖动）

    # 卡死恢复配置
    stall_recovery_interval: float = 30.0  # 卡死检测/恢复间隔（秒）
    max_recovery_attempts: int = 3  # 单次迭代最大恢复尝试次数
    max_no_progress_time: float = 120.0  # 最大无进展时间（秒），超过触发降级

    # 卡死诊断日志配置
    # 控制卡死检测时输出的诊断信息详细程度
    # 设计目的：默认关闭避免刷屏，疑似卡死时通过 --stall-diagnostics 启用排查
    stall_diagnostics_enabled: bool = False  # 是否启用卡死诊断日志（默认关闭）
    stall_diagnostics_detail: bool = False  # 是否输出详细诊断（逐任务状态）
    stall_diagnostics_max_tasks: int = 5  # 详细诊断时最大输出任务条数
    stall_diagnostics_level: str = "warning"  # 诊断日志级别: debug, info, warning, error

    # 健康检查告警冷却时间（秒）
    # 防止同一 agent 的警告刷屏，在此时间内不会重复告警
    health_warning_cooldown_seconds: float = 60.0

    # _send_and_wait 超时告警配置
    # 连续超时阈值：达到此阈值后才输出 WARNING（之前为 DEBUG/INFO）
    timeout_warning_threshold: int = 2
    # 超时告警冷却时间（秒）：同一 agent 在此时间内不会重复输出 WARNING
    timeout_warning_cooldown_seconds: float = 60.0

    # 日志配置（透传到子进程）
    # 这些配置会传递给 planner/worker/reviewer 子进程，控制其日志输出行为
    verbose: bool = False  # 是否启用详细模式（等价于 log_level=DEBUG）
    log_level: str = "INFO"  # 日志级别: DEBUG, INFO, WARNING, ERROR
    heartbeat_debug: bool = False  # 是否输出心跳调试日志（默认 False，避免刷屏）


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

        # 解析流式日志配置（优先级: 显式传入 > config.yaml > DEFAULT_STREAM_* 常量）
        self._resolved_stream_config = self._resolve_stream_config(config)

        # 系统状态
        self.state = SystemState(
            working_directory=config.working_directory,
            max_iterations=config.max_iterations,
        )

        # 任务队列
        self.task_queue = TaskQueue()

        # 迭代助手上下文
        self._iteration_context: dict[str, Any] = config.iteration_context or {}

        # 进程管理器
        self.process_manager = AgentProcessManager()

        # Agent ID
        self.planner_id: str | None = None
        self.worker_ids: list[str] = []
        self.reviewer_id: str | None = None

        # 待处理的消息响应
        self._pending_responses: dict[str, asyncio.Future] = {}

        # 初始化 CommitterAgent（运行在 orchestrator 主进程中，避免新增子进程复杂度）
        self.committer: CommitterAgent | None = None
        if config.enable_auto_commit:
            # 使用解析后的流式日志配置
            committer_cursor_config = CursorAgentConfig(
                stream_events_enabled=self._resolved_stream_config["stream_events_enabled"],
                stream_log_console=self._resolved_stream_config["stream_log_console"],
                stream_log_detail_dir=self._resolved_stream_config["stream_log_detail_dir"],
                stream_log_raw_dir=self._resolved_stream_config["stream_log_raw_dir"],
            )
            self.committer = CommitterAgent(
                CommitterConfig(
                    working_directory=config.working_directory,
                    auto_push=config.auto_push,
                    cursor_config=committer_cursor_config,
                )
            )
            # 注册 Committer 到系统状态
            self.state.register_agent("committer", AgentRole.COMMITTER)
            logger.info(f"CommitterAgent 已初始化 (auto_push={config.auto_push})")

        # 初始化知识库存储（延迟初始化，用于 Cursor 相关任务的上下文增强）
        self._knowledge_storage: KnowledgeStorage | None = None
        self._knowledge_initialized = False

        # 健康检查状态
        self._last_health_check_time: float = 0.0
        self._unhealthy_worker_count: int = 0
        self._degraded: bool = False  # 是否已降级
        self._degradation_reason: str | None = None

        # 心跳收集机制（方案 A：_message_loop 作为唯一消费者）
        # {agent_id: last_heartbeat_timestamp}
        self._heartbeat_responses: dict[str, float] = {}
        # {agent_id: heartbeat_payload} 存储心跳响应的详细信息（如 busy 状态）
        self._heartbeat_payloads: dict[str, dict] = {}
        self._heartbeat_request_id: str | None = None  # 当前心跳请求的 ID
        self._heartbeat_pending: set[str] = set()  # 等待心跳响应的 agent_id

        # 健康检查告警 cooldown 机制
        # {agent_id: last_warning_timestamp} 防止同一 agent 的警告刷屏
        self._health_warning_cooldown: dict[str, float] = {}
        # 告警冷却时间（秒），同一 agent 在此时间内不会重复告警（从配置读取）
        self._health_warning_cooldown_seconds: float = config.health_warning_cooldown_seconds

        # 连续未响应计数器
        # {agent_id: consecutive_unresponsive_count} 跟踪每个 agent 连续未响应心跳的次数
        self._consecutive_unresponsive_count: dict[str, int] = {}

        # _send_and_wait 超时计数与告警节流
        # {agent_id: consecutive_timeout_count} 跟踪每个 agent 连续超时的次数
        self._timeout_count: dict[str, int] = {}
        # {agent_id: last_timeout_warning_timestamp} 超时告警冷却
        self._timeout_warning_cooldown: dict[str, float] = {}

        # 消息处理统计（用于验证消息不丢失）
        self._message_stats: dict[str, int] = {
            "total_received": 0,
            "plan_result": 0,
            "task_result": 0,
            "heartbeat": 0,
            "status_response": 0,
            "task_progress": 0,
            "other": 0,
        }

        # 卡死恢复状态
        self._recovery_attempts: dict[int, int] = {}  # {iteration_id: attempt_count}
        self._last_progress_time: float = 0.0  # 最后有进展的时间戳（用于 max_no_progress_time 计算）
        self._last_recovery_time: float = 0.0  # 最后恢复尝试的时间戳
        self._last_stall_diagnostic_time: float = 0.0  # 最后输出卡死诊断的时间戳
        self._last_stall_check_time: float = 0.0  # 最后卡死检测的时间戳（用于检测间隔冷却）

    @staticmethod
    def _resolve_stream_config(config: MultiProcessOrchestratorConfig) -> dict[str, Any]:
        """解析流式日志配置（优先级: 显式传入 > config.yaml > DEFAULT_STREAM_* 常量）

        tri-state 设计: None 表示使用 config.yaml 值，显式传入优先。

        Args:
            config: MultiProcessOrchestratorConfig 实例

        Returns:
            解析后的流式日志配置字典，包含:
            - stream_events_enabled: 是否启用流式日志
            - stream_log_console: 是否输出到控制台
            - stream_log_detail_dir: 详细日志目录
            - stream_log_raw_dir: 原始日志目录
        """
        yaml_config = get_config()
        stream_json = yaml_config.logging.stream_json

        stream_events_enabled = (
            config.stream_events_enabled if config.stream_events_enabled is not None else stream_json.enabled
        )
        stream_log_console = config.stream_log_console if config.stream_log_console is not None else stream_json.console
        stream_log_detail_dir = (
            config.stream_log_detail_dir if config.stream_log_detail_dir is not None else stream_json.detail_dir
        )
        stream_log_raw_dir = config.stream_log_raw_dir if config.stream_log_raw_dir is not None else stream_json.raw_dir

        return {
            "stream_events_enabled": stream_events_enabled,
            "stream_log_console": stream_log_console,
            "stream_log_detail_dir": stream_log_detail_dir,
            "stream_log_raw_dir": stream_log_raw_dir,
        }

    def _should_continue_iteration(self) -> bool:
        """判断是否应该继续迭代

        Returns:
            True 表示继续迭代，False 表示停止
        """
        # 已降级，停止迭代
        if self._degraded:
            return False

        # 无限迭代模式（max_iterations == -1）
        if self.config.max_iterations == -1:
            return True

        # 正常模式：检查是否达到最大迭代次数
        return self.state.current_iteration < self.state.max_iterations

    def _should_run_health_check(self) -> bool:
        """判断是否应该执行健康检查

        Returns:
            True 表示应该执行健康检查
        """
        import time

        now = time.time()
        return now - self._last_health_check_time >= self.config.health_check_interval

    def _should_emit_health_warning(self, agent_id: str) -> bool:
        """检查是否应该发出健康检查警告（cooldown 机制）

        防止同一 agent 的警告刷屏。在 cooldown 时间内不会重复告警。

        Args:
            agent_id: Agent 标识符

        Returns:
            True 表示应该发出警告，False 表示在 cooldown 内
        """
        import time

        now = time.time()
        last_warning = self._health_warning_cooldown.get(agent_id, 0)
        if now - last_warning >= self._health_warning_cooldown_seconds:
            self._health_warning_cooldown[agent_id] = now
            return True
        return False

    async def _perform_health_check(self) -> HealthCheckResult:
        """执行健康检查（方案 A：通过 _message_loop 收集心跳）

        检测所有 Agent 进程的健康状态，处理不健康的进程。

        实现机制：
        1. 广播 HEARTBEAT 消息给所有 Agent
        2. 在 _message_loop 的 _handle_message 中收集心跳响应
        3. 等待超时后，检查哪些 Agent 未响应

        容忍策略：
        - 正在执行任务的 Worker（busy=True）不视为不健康
        - 进程存活但无响应的 Worker 会被记录但不立即降级
        - 只有进程真正死亡时才触发任务重入队

        告警节流策略：
        - 使用 per-agent cooldown 机制防止同一 agent 的警告刷屏
        - busy worker 的未响应仅输出 debug 日志（不告警）
        - 进程死亡保留 ERROR 级别日志
        - 进程存活但无响应的非 busy worker 使用 WARNING（带 cooldown）

        Returns:
            健康检查结果
        """
        import time
        import uuid

        self._last_health_check_time = time.time()

        # 初始化心跳收集状态
        self._heartbeat_request_id = uuid.uuid4().hex
        self._heartbeat_pending = set(self._get_all_agent_ids())
        heartbeat_start_time = time.time()

        # 记录健康检查开始（用于验证消息不丢失）
        if self.config.heartbeat_debug:
            logger.debug(
                f"健康检查开始: request_id={self._heartbeat_request_id}, "
                f"pending_agents={len(self._heartbeat_pending)}, "
                f"message_stats={self._message_stats}"
            )

        # 广播心跳请求
        heartbeat_msg = ProcessMessage(
            type=ProcessMessageType.HEARTBEAT,
            sender="orchestrator",
            payload={"request_id": self._heartbeat_request_id},
        )
        self.process_manager.broadcast(heartbeat_msg)

        # 等待心跳响应（通过 _message_loop 收集）
        deadline = time.time() + self.config.health_check_timeout
        while self._heartbeat_pending and time.time() < deadline:
            await asyncio.sleep(0.1)

        # 构建健康检查结果
        result = HealthCheckResult()

        # 跟踪忙碌的 Worker
        busy_workers: list[str] = []

        for agent_id in self._get_all_agent_ids():
            last_heartbeat = self._heartbeat_responses.get(agent_id, 0)
            # 检查心跳时间是否在本次请求之后
            is_responsive = last_heartbeat >= heartbeat_start_time

            if is_responsive:
                result.healthy.append(agent_id)
                # 检查是否为忙碌状态（从 _heartbeat_payloads 获取）
                heartbeat_payload = self._heartbeat_payloads.get(agent_id, {})
                is_busy = heartbeat_payload.get("busy", False)
                result.details[agent_id] = {
                    "healthy": True,
                    "reason": "heartbeat_ok",
                    "response_time": last_heartbeat - heartbeat_start_time,
                    "busy": is_busy,
                }
                if is_busy:
                    busy_workers.append(agent_id)
                # 响应正常，重置连续未响应计数
                self._consecutive_unresponsive_count[agent_id] = 0
            else:
                # 检查进程是否还存活
                is_alive = self.process_manager.is_alive(agent_id)

                if is_alive:
                    # 进程存活但无响应 - 可能正在执行长任务
                    # 根据配置决定是否视为不健康
                    if self.config.skip_busy_workers_in_health_check:
                        # 检查是否有分配给该 Worker 的任务（表示正在执行）
                        has_assigned_task = len(self.process_manager.get_tasks_by_agent(agent_id)) > 0
                        if has_assigned_task:
                            # 有任务分配，视为忙碌而非不健康
                            result.healthy.append(agent_id)
                            result.details[agent_id] = {
                                "healthy": True,
                                "reason": "assumed_busy",
                                "is_alive": True,
                                "has_assigned_task": True,
                            }
                            busy_workers.append(agent_id)
                            # busy worker 的未响应仅输出 debug 日志，不告警
                            if self.config.heartbeat_debug:
                                logger.debug(f"健康检查: {agent_id} 无心跳响应但有任务分配，假定为忙碌")
                            continue

                    # 无任务分配但也无响应 - 可能有问题
                    # 递增连续未响应计数
                    self._consecutive_unresponsive_count[agent_id] = (
                        self._consecutive_unresponsive_count.get(agent_id, 0) + 1
                    )
                    consecutive_count = self._consecutive_unresponsive_count[agent_id]
                    threshold = self.config.consecutive_unresponsive_threshold

                    result.unhealthy.append(agent_id)
                    result.details[agent_id] = {
                        "healthy": False,
                        "reason": "no_heartbeat_response",
                        "is_alive": True,
                        "consecutive_unresponsive": consecutive_count,
                    }
                    # 只有连续多次未响应后才输出 WARNING（避免短暂抖动导致告警）
                    if consecutive_count >= threshold:
                        # 使用 cooldown 机制防止刷屏（仅 warning 级别）
                        if self._should_emit_health_warning(agent_id):
                            logger.warning(
                                f"健康检查警告: {agent_id} (无心跳响应，进程存活，连续 {consecutive_count} 次)"
                            )
                        else:
                            if self.config.heartbeat_debug:
                                logger.debug(f"健康检查: {agent_id} 无心跳响应（cooldown 中，跳过告警）")
                    else:
                        if self.config.heartbeat_debug:
                            logger.debug(
                                f"健康检查: {agent_id} 无心跳响应 ({consecutive_count}/{threshold}，未达阈值，暂不告警)"
                            )
                else:
                    # 进程已死亡 - 始终使用 ERROR 级别，不受 cooldown 限制
                    result.unhealthy.append(agent_id)
                    result.details[agent_id] = {
                        "healthy": False,
                        "reason": "process_dead",
                        "is_alive": False,
                    }
                    logger.error(f"健康检查失败: {agent_id} (进程已死亡)")

        result.all_healthy = len(result.unhealthy) == 0

        # 清理心跳状态
        self._heartbeat_request_id = None
        self._heartbeat_pending.clear()

        # 记录健康检查完成
        if self.config.heartbeat_debug:
            if busy_workers:
                logger.debug(
                    f"健康检查完成: healthy={len(result.healthy)}, "
                    f"unhealthy={len(result.unhealthy)}, "
                    f"busy_workers={len(busy_workers)}"
                )
            else:
                logger.debug(f"健康检查完成: healthy={len(result.healthy)}, unhealthy={len(result.unhealthy)}")

        if result.all_healthy:
            if self.config.heartbeat_debug:
                logger.debug("健康检查通过: 所有进程正常")
            self._unhealthy_worker_count = 0
            return result

        # 检查关键进程（planner/reviewer）- 只在进程真正死亡时降级
        if self.config.fallback_on_critical_failure:
            if self.planner_id:
                planner_detail = result.details.get(self.planner_id, {})
                if self.planner_id in result.unhealthy and not planner_detail.get("is_alive", True):
                    self._trigger_degradation("Planner 进程已死亡")
                    return result

            if self.reviewer_id:
                reviewer_detail = result.details.get(self.reviewer_id, {})
                if self.reviewer_id in result.unhealthy and not reviewer_detail.get("is_alive", True):
                    self._trigger_degradation("Reviewer 进程已死亡")
                    return result

        # 统计真正不健康的 Worker 数量（排除进程存活的）
        unhealthy_workers = result.get_unhealthy_workers()
        dead_workers = [wid for wid in unhealthy_workers if not result.details.get(wid, {}).get("is_alive", True)]
        self._unhealthy_worker_count = len(dead_workers)

        # 动态计算 max_unhealthy_workers（如果配置为 -1）
        max_unhealthy = self.config.max_unhealthy_workers
        if max_unhealthy < 0:
            # 默认允许 worker_count - 1 个不健康（至少保留 1 个正常）
            max_unhealthy = max(self.config.worker_count - 1, 1)

        # 检查是否超过允许的不健康 Worker 数量（仅统计已死亡的）
        if self._unhealthy_worker_count > max_unhealthy:
            self._trigger_degradation(f"已死亡 Worker 数量超过阈值: {self._unhealthy_worker_count} > {max_unhealthy}")
            return result

        # 只处理真正死亡的 Worker 的在途任务
        if dead_workers:
            logger.warning(f"检测到 {len(dead_workers)} 个已死亡的 Worker: {dead_workers}")
            await self._handle_unhealthy_workers(dead_workers)

        return result

    def _get_all_agent_ids(self) -> list[str]:
        """获取所有 Agent ID 列表"""
        agent_ids = []
        if self.planner_id:
            agent_ids.append(self.planner_id)
        agent_ids.extend(self.worker_ids)
        if self.reviewer_id:
            agent_ids.append(self.reviewer_id)
        return agent_ids

    def _trigger_degradation(self, reason: str) -> None:
        """触发降级

        标记系统进入降级状态，后续迭代将终止。

        Args:
            reason: 降级原因
        """
        if self._degraded:
            return  # 已经降级，忽略

        self._degraded = True
        self._degradation_reason = reason
        logger.error(f"系统降级: {reason}")
        logger.warning("将在当前迭代结束后终止（或回退到 basic 编排器）")

    async def _handle_unhealthy_workers(self, unhealthy_worker_ids: list[str]) -> None:
        """处理不健康的 Worker 进程

        根据配置决定将在途任务重新入队或标记为失败。

        Args:
            unhealthy_worker_ids: 不健康的 Worker ID 列表
        """
        if not unhealthy_worker_ids:
            return

        for worker_id in unhealthy_worker_ids:
            # 获取该 Worker 的在途任务
            task_ids = self.process_manager.get_tasks_by_agent(worker_id)

            if not task_ids:
                continue

            logger.warning(f"Worker {worker_id} 不健康，处理 {len(task_ids)} 个在途任务")

            for task_id in task_ids:
                # 取消跟踪
                self.process_manager.untrack_task(task_id)

                # 获取任务
                task = self.task_queue.get_task(task_id)
                if not task:
                    logger.warning(f"任务 {task_id} 未在队列中找到")
                    continue

                # 递增重试计数
                task.retry_count += 1

                if self.config.requeue_on_worker_death and task.can_retry():
                    # 重新入队到优先级队列
                    logger.info(
                        f"任务 {task_id} 重新入队 (Worker {worker_id} 死亡) "
                        f"(重试 {task.retry_count}/{task.max_retries})"
                    )
                    await self.task_queue.requeue(task, reason=f"Worker {worker_id} 死亡")
                else:
                    # 超过重试上限或配置不允许重试，标记失败
                    if task.retry_count > task.max_retries:
                        fail_reason = f"Worker {worker_id} 进程不健康，已达最大重试次数 ({task.max_retries})"
                    else:
                        fail_reason = f"Worker {worker_id} 进程不健康，任务丢失（配置禁止重试）"
                    logger.info(f"任务 {task_id} 标记为失败: {fail_reason}")
                    task.fail(fail_reason)
                    self.task_queue.update_task(task)

                    # 更新统计
                    iteration = self.state.get_current_iteration()
                    if iteration:
                        iteration.tasks_failed += 1
                    self.state.total_tasks_failed += 1

            # 从可用 Worker 列表中移除（如果有的话）
            if worker_id in self.worker_ids:
                logger.warning(f"从 Worker 池中移除不健康进程: {worker_id}")
                # 不移除，保留在列表中但后续分配时会跳过不健康的

    async def _recover_stalled_iteration(
        self,
        iteration_id: int,
        reason: str,
        active_futures: dict[str, asyncio.Task],
        worker_task_mapping: dict[str, str],
    ) -> dict[str, Any]:
        """恢复卡死的迭代

        扫描迭代中的任务，找出状态不一致的任务并进行恢复：
        1. PENDING/QUEUED 但不在队列中的任务 → 重新入队
        2. ASSIGNED/IN_PROGRESS 但无 in-flight 记录的任务 → 检查 worker 状态后决定
        3. 超过最大恢复次数或无进展时间 → 触发降级

        Args:
            iteration_id: 迭代 ID
            reason: 触发恢复的原因
            active_futures: 当前活跃的 asyncio.Task 字典 {worker_id: asyncio.Task}
            worker_task_mapping: worker 到任务的映射 {worker_id: task_id}

        Returns:
            恢复结果:
            {
                "recovered": int,        # 恢复的任务数
                "failed": int,           # 标记失败的任务数
                "requeued": int,         # 重新入队的任务数
                "degraded": bool,        # 是否触发降级
                "reason": str,           # 恢复或降级原因
            }
        """
        import time

        result: dict[str, Any] = {
            "recovered": 0,
            "failed": 0,
            "requeued": 0,
            "degraded": False,
            "reason": reason,
        }

        # 检查恢复次数限制
        self._recovery_attempts.setdefault(iteration_id, 0)
        self._recovery_attempts[iteration_id] += 1

        if self._recovery_attempts[iteration_id] > self.config.max_recovery_attempts:
            self._trigger_degradation(f"迭代 {iteration_id} 恢复尝试超过上限 ({self.config.max_recovery_attempts})")
            result["degraded"] = True
            result["reason"] = "max_recovery_attempts_exceeded"
            return result

        # 检查无进展时间
        now = time.time()
        if self._last_progress_time > 0:
            no_progress_duration = now - self._last_progress_time
            if no_progress_duration > self.config.max_no_progress_time:
                self._trigger_degradation(
                    f"迭代 {iteration_id} 无进展时间超过 {self.config.max_no_progress_time}s "
                    f"(已 {no_progress_duration:.1f}s)"
                )
                result["degraded"] = True
                result["reason"] = "max_no_progress_time_exceeded"
                return result

        self._last_recovery_time = now

        # 恢复开始日志：使用与卡死诊断相同的日志级别，并复用节流机制
        # 检查是否应该输出恢复日志（节流控制：复用 _last_stall_diagnostic_time）
        should_emit_recovery_log = (
            self._last_stall_diagnostic_time == 0.0
            or now - self._last_stall_diagnostic_time >= self.config.stall_recovery_interval
        )

        if should_emit_recovery_log:
            # 选择日志级别（复用 stall_diagnostics_level 配置）
            log_level = self.config.stall_diagnostics_level.lower()
            recovery_log_fn = {
                "debug": logger.debug,
                "info": logger.info,
                "warning": logger.warning,
                "error": logger.error,
            }.get(log_level, logger.info)

            recovery_log_fn(
                f"[恢复] 迭代 {iteration_id} 开始恢复 (原因: {reason}, "
                f"尝试 {self._recovery_attempts[iteration_id]}/{self.config.max_recovery_attempts})"
            )
        else:
            # 节流中，仅输出 debug 日志
            logger.debug(f"[恢复] 迭代 {iteration_id} 开始恢复 (节流中，跳过详细日志)")

        # 构建活跃 future 对应的任务 ID 集合
        active_future_task_ids = {worker_task_mapping.get(wid, "") for wid in active_futures}
        active_future_task_ids.discard("")  # 移除空字符串

        # 获取 in-flight 任务 ID 集合
        in_flight_tasks = self.process_manager.get_all_in_flight_tasks()
        in_flight_task_ids = set(in_flight_tasks.keys())

        # 调用 TaskQueue.reconcile_iteration 检测不一致
        issues = await self.task_queue.reconcile_iteration(
            iteration_id=iteration_id,
            in_flight_task_ids=in_flight_task_ids,
            active_future_task_ids=active_future_task_ids,
        )

        # 处理 orphaned_pending: 重新入队
        for task in issues["orphaned_pending"]:
            try:
                await self.task_queue.enqueue(task)
                result["requeued"] += 1
                logger.info(f"[恢复] orphaned_pending 任务 {task.id} 已重新入队")
            except Exception as e:
                logger.error(f"[恢复] 任务 {task.id} 重新入队失败: {e}")
                task.fail(f"恢复失败: {e}")
                self.task_queue.update_task(task)
                result["failed"] += 1

        # 处理 orphaned_assigned: 检查 worker 状态
        for task in issues["orphaned_assigned"]:
            worker_id = task.assigned_to
            if worker_id and self.process_manager.is_alive(worker_id):
                # Worker 存活，可能是 in-flight 记录丢失，重新入队
                task.retry_count += 1
                if task.can_retry():
                    await self.task_queue.requeue(task, reason=f"orphaned_assigned (worker {worker_id} alive)")
                    result["requeued"] += 1
                    logger.info(f"[恢复] orphaned_assigned 任务 {task.id} 重新入队 (worker 存活)")
                else:
                    task.fail(f"orphaned_assigned 且超过重试上限 (worker {worker_id} alive)")
                    self.task_queue.update_task(task)
                    result["failed"] += 1
                    logger.warning(f"[恢复] orphaned_assigned 任务 {task.id} 标记失败 (超过重试上限)")
            else:
                # Worker 已死亡
                task.retry_count += 1
                if self.config.requeue_on_worker_death and task.can_retry():
                    await self.task_queue.requeue(task, reason=f"worker {worker_id} dead")
                    result["requeued"] += 1
                    logger.info(f"[恢复] orphaned_assigned 任务 {task.id} 重新入队 (worker 死亡)")
                else:
                    task.fail(f"Worker {worker_id} 死亡，任务丢失")
                    self.task_queue.update_task(task)
                    result["failed"] += 1
                    logger.warning(f"[恢复] orphaned_assigned 任务 {task.id} 标记失败 (worker 死亡)")

        # 处理 stale_in_progress: 检查 worker 状态后决定
        for task in issues["stale_in_progress"]:
            worker_id = task.assigned_to
            if worker_id and self.process_manager.is_alive(worker_id):
                # Worker 存活但无 active future，可能是异步任务丢失
                task.retry_count += 1
                if task.can_retry():
                    await self.task_queue.requeue(
                        task, reason=f"stale_in_progress (worker {worker_id} alive, no future)"
                    )
                    result["requeued"] += 1
                    logger.info(f"[恢复] stale_in_progress 任务 {task.id} 重新入队 (worker 存活)")
                else:
                    task.fail("stale_in_progress 且超过重试上限")
                    self.task_queue.update_task(task)
                    result["failed"] += 1
                    logger.warning(f"[恢复] stale_in_progress 任务 {task.id} 标记失败 (超过重试上限)")
            else:
                # Worker 已死亡
                task.retry_count += 1
                if self.config.requeue_on_worker_death and task.can_retry():
                    await self.task_queue.requeue(task, reason=f"worker {worker_id} dead (stale)")
                    result["requeued"] += 1
                    logger.info(f"[恢复] stale_in_progress 任务 {task.id} 重新入队 (worker 死亡)")
                else:
                    task.fail(f"Worker {worker_id} 死亡，任务执行丢失")
                    self.task_queue.update_task(task)
                    result["failed"] += 1
                    logger.warning(f"[恢复] stale_in_progress 任务 {task.id} 标记失败 (worker 死亡)")

        result["recovered"] = result["requeued"] + result["failed"]

        logger.info(f"[恢复] 迭代 {iteration_id} 恢复完成: requeued={result['requeued']}, failed={result['failed']}")

        return result

    def _should_attempt_recovery(self, iteration_id: int) -> bool:
        """判断是否应该尝试恢复

        基于恢复间隔和恢复次数限制判断。

        Args:
            iteration_id: 迭代 ID

        Returns:
            是否应该尝试恢复
        """
        import time

        # 检查恢复间隔
        now = time.time()
        if now - self._last_recovery_time < self.config.stall_recovery_interval:
            return False

        # 检查恢复次数
        attempts = self._recovery_attempts.get(iteration_id, 0)
        return not attempts >= self.config.max_recovery_attempts

    def is_degraded(self) -> bool:
        """检查系统是否已降级

        Returns:
            True 表示已降级
        """
        return self._degraded

    def get_degradation_reason(self) -> str | None:
        """获取降级原因

        Returns:
            降级原因，如果未降级则返回 None
        """
        return self._degradation_reason

    async def _init_knowledge_storage(self) -> bool:
        """延迟初始化知识库存储（只读模式）

        多进程编排器中的知识库仅用于搜索/检索，不进行写入操作。
        使用只读模式可以：
        1. 避免并发写入冲突
        2. 符合 minimal 策略要求
        3. 提高安全性

        Returns:
            是否初始化成功
        """
        if self._knowledge_initialized:
            return self._knowledge_storage is not None

        self._knowledge_initialized = True

        if not self.config.enable_knowledge_search:
            return False

        try:
            from knowledge.storage import KnowledgeStorage

            # 使用只读模式：多进程编排器仅搜索，不写入
            self._knowledge_storage = KnowledgeStorage.create_read_only(
                workspace_root=self.config.working_directory,
                enable_vector_index=False,  # 多进程场景下禁用向量索引，使用关键词搜索
            )
            assert self._knowledge_storage is not None
            await self._knowledge_storage.initialize()
            logger.info("知识库存储已初始化 (只读模式)")
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
            "triggered": False,  # 是否触发了关键词检测
            "matched": 0,  # 搜索命中的原始文档数
            "injected": 0,  # 实际注入的文档数
            "truncated": False,  # 是否发生了截断
            "total_chars": 0,  # 注入的总字符数
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
        metrics["keywords_matched"] = [kw for kw in self.config.knowledge_trigger_keywords if kw.lower() in query_lower]

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
                        raw_docs.append(
                            {
                                "title": doc.title or result.title,
                                "url": doc.url or result.url,
                                "content": doc.content,
                                "score": result.score,
                                "source": "cursor-docs",
                            }
                        )
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
        """创建并启动所有 Agent 进程

        配置注入策略：
        - 从 config.yaml 读取 agent_cli/cloud_agent 配置
        - 注入 agent_path/timeout/max_retries 等字段
        - 确保子进程使用统一的配置来源
        """
        # 从 config.yaml 获取配置
        yaml_config = get_config()
        agent_cli = yaml_config.agent_cli
        cloud_agent = yaml_config.cloud_agent

        # 构建通用日志配置（透传到所有子进程）
        log_config = {
            "verbose": self.config.verbose,
            "log_level": self.config.log_level if not self.config.verbose else "DEBUG",
            "heartbeat_debug": self.config.heartbeat_debug,
        }

        # 构建通用 agent_cli 配置（从 config.yaml 注入）
        agent_cli_config = {
            "agent_path": agent_cli.path,
            "max_retries": agent_cli.max_retries,
        }

        # 构建通用 cloud_agent 配置（从 config.yaml 注入）
        cloud_config = {
            "cloud_api_base": cloud_agent.api_base_url,
            "cloud_timeout": cloud_agent.timeout,
            "cloud_enabled": cloud_agent.enabled,
        }

        # 解析角色级执行模式（默认继承全局 execution_mode）
        # 注意：MP 子进程实际使用 CLI 执行，这些配置主要用于记录和透传
        planner_exec_mode = self.config.planner_execution_mode or self.config.execution_mode
        worker_exec_mode = self.config.worker_execution_mode or self.config.execution_mode
        reviewer_exec_mode = self.config.reviewer_execution_mode or self.config.execution_mode

        # 记录角色级执行模式（如果有配置差异）
        if (
            self.config.planner_execution_mode
            or self.config.worker_execution_mode
            or self.config.reviewer_execution_mode
        ):
            logger.info(
                f"角色级执行模式 - Planner: {planner_exec_mode}, "
                f"Worker: {worker_exec_mode}, Reviewer: {reviewer_exec_mode}"
            )

        # 创建 Planner - 使用 GPT 5.2-high
        # 注意：Planner 强制使用 mode='plan' 和 force_write=False（只读语义）
        planner_config = {
            "working_directory": self.config.working_directory,
            "timeout": int(self.config.planning_timeout),
            "model": self.config.planner_model,
            "mode": "plan",  # 规划模式（只读）
            "force_write": False,  # 确保不会修改文件
            "execution_mode": planner_exec_mode,  # 角色执行模式
            "stream_events_enabled": self._resolved_stream_config["stream_events_enabled"],
            "stream_log_console": self._resolved_stream_config["stream_log_console"],
            "stream_log_detail_dir": self._resolved_stream_config["stream_log_detail_dir"],
            "stream_log_raw_dir": self._resolved_stream_config["stream_log_raw_dir"],
            "agent_name": "planner",
            **log_config,  # 日志配置透传
            **agent_cli_config,  # agent_cli 配置透传
            **cloud_config,  # cloud_agent 配置透传
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

        # 创建 Workers - 使用执行者模型
        # 注意：Worker 使用 mode='agent' 和 force_write=True（允许修改文件）
        worker_config = {
            "working_directory": self.config.working_directory,
            "task_timeout": int(self.config.execution_timeout),
            "model": self.config.worker_model,
            "mode": "agent",  # 完整代理模式
            "force_write": True,  # 允许修改文件
            "execution_mode": worker_exec_mode,  # 角色执行模式
            "stream_events_enabled": self._resolved_stream_config["stream_events_enabled"],
            "stream_log_console": self._resolved_stream_config["stream_log_console"],
            "stream_log_detail_dir": self._resolved_stream_config["stream_log_detail_dir"],
            "stream_log_raw_dir": self._resolved_stream_config["stream_log_raw_dir"],
            **log_config,  # 日志配置透传
            **agent_cli_config,  # agent_cli 配置透传
            **cloud_config,  # cloud_agent 配置透传
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

        # 创建 Reviewer - 使用评审者模型
        # 注意：Reviewer 强制使用 mode='ask' 和 force_write=False（只读语义）
        reviewer_config = {
            "working_directory": self.config.working_directory,
            "timeout": int(self.config.review_timeout),
            "model": self.config.reviewer_model,
            "mode": "ask",  # 问答模式（只读）
            "force_write": False,  # 确保不会修改文件
            "execution_mode": reviewer_exec_mode,  # 角色执行模式
            "strict_mode": self.config.strict_review,
            "stream_events_enabled": self._resolved_stream_config["stream_events_enabled"],
            "stream_log_console": self._resolved_stream_config["stream_log_console"],
            "stream_log_detail_dir": self._resolved_stream_config["stream_log_detail_dir"],
            "stream_log_raw_dir": self._resolved_stream_config["stream_log_raw_dir"],
            "agent_name": "reviewer",
            **log_config,  # 日志配置透传
            **agent_cli_config,  # agent_cli 配置透传
            **cloud_config,  # cloud_agent 配置透传
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
                import time

                self._last_health_check_time = time.time()  # 初始化健康检查时间

                while self._should_continue_iteration():
                    # 周期性健康检查
                    if self._should_run_health_check():
                        await self._perform_health_check()
                        if self._degraded:
                            logger.error(f"系统降级，终止迭代: {self._degradation_reason}")
                            break

                    # 开始新迭代
                    iteration = self.state.start_new_iteration()
                    logger.info(f"\n{'=' * 50}")
                    logger.info(f"迭代 {iteration.iteration_id} 开始")

                    # 规划阶段
                    await self._planning_phase(goal, iteration.iteration_id)

                    # 检查任务
                    pending = self.task_queue.get_pending_count(iteration.iteration_id)
                    if pending == 0:
                        logger.warning("规划阶段未产生任务")
                        continue

                    # 执行阶段（内部会进行健康检查）
                    await self._execution_phase(iteration.iteration_id)

                    # 检查是否在执行阶段降级
                    if self._degraded:
                        logger.error(f"执行阶段降级，终止迭代: {self._degradation_reason}")
                        break

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
                with contextlib.suppress(asyncio.CancelledError):
                    await message_task

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
        """处理来自 Agent 的消息（唯一消费者）

        此方法是消息队列的唯一消费者，确保消息不会在健康检查时丢失。
        所有消息类型都在此处理，包括心跳响应。

        Late Result 处理:
            当收到 PLAN_RESULT/TASK_RESULT 但 correlation_id 不在 _pending_responses 时，
            可能是以下情况：
            1. 任务已超时，_send_and_wait 已移除 Future 但响应仍然到达
            2. 任务已被重试/重新入队
            3. Worker 处理较慢导致响应延迟

            处理策略：
            - 通过 message_id -> task_id 反向索引查找任务
            - 如果任务已被重试/重新入队（状态为 PENDING），记录 "late result ignored"
            - 如果任务仍处于 IN_PROGRESS 且 assigned_to 匹配，则完成/失败并更新统计
        """
        import time

        # 更新消息统计（用于验证消息不丢失）
        self._message_stats["total_received"] += 1
        msg_type = message.type.value if hasattr(message.type, "value") else str(message.type)
        if msg_type in self._message_stats:
            self._message_stats[msg_type] += 1
        elif message.type == ProcessMessageType.PLAN_RESULT:
            self._message_stats["plan_result"] += 1
        elif message.type == ProcessMessageType.TASK_RESULT:
            self._message_stats["task_result"] += 1
        elif message.type == ProcessMessageType.HEARTBEAT:
            self._message_stats["heartbeat"] += 1
        elif message.type == ProcessMessageType.STATUS_RESPONSE:
            self._message_stats["status_response"] += 1
        elif message.type == ProcessMessageType.TASK_PROGRESS:
            self._message_stats["task_progress"] += 1
        else:
            self._message_stats["other"] += 1

        # 处理心跳响应（方案 A 核心：在消息循环中收集心跳）
        if message.type == ProcessMessageType.HEARTBEAT:
            sender = message.sender
            self._heartbeat_responses[sender] = time.time()
            # 存储心跳 payload（包含 busy 状态等信息）
            self._heartbeat_payloads[sender] = message.payload or {}
            # 从待响应集合中移除
            self._heartbeat_pending.discard(sender)
            # 心跳日志仅在 heartbeat_debug 模式下输出（避免高频刷屏）
            if self.config.heartbeat_debug:
                is_busy = message.payload.get("busy", False) if message.payload else False
                logger.debug(
                    f"心跳响应收到: sender={sender}, busy={is_busy}, remaining_pending={len(self._heartbeat_pending)}"
                )
            return

        # 检查是否有等待此消息的 Future
        if message.correlation_id and message.correlation_id in self._pending_responses:
            future = self._pending_responses.pop(message.correlation_id)
            if not future.done():
                future.set_result(message)

            # 记录关键消息（验证不丢失）
            if message.type in (ProcessMessageType.PLAN_RESULT, ProcessMessageType.TASK_RESULT):
                logger.debug(
                    f"关键消息已处理: type={message.type.value}, "
                    f"sender={message.sender}, correlation_id={message.correlation_id}"
                )
            return

        # ============ Late Result 兜底处理 ============
        # 当 correlation_id 不在 _pending_responses 时，尝试处理 late result
        if message.type in (ProcessMessageType.PLAN_RESULT, ProcessMessageType.TASK_RESULT):
            await self._handle_late_result(message)
            return

        # 处理任务进度消息
        if message.type == ProcessMessageType.TASK_PROGRESS:
            task_id = message.payload.get("task_id")
            progress = message.payload.get("progress", 0)
            logger.debug(f"任务进度: {task_id} - {progress}%")

    async def _handle_late_result(self, message: ProcessMessage) -> None:
        """处理 late result（延迟到达的任务/规划结果）

        当 PLAN_RESULT/TASK_RESULT 的 correlation_id 不在 _pending_responses 时调用。
        这通常发生在：
        1. _send_and_wait 超时后，响应仍然到达
        2. 任务已被重试/重新入队，旧响应延迟到达

        处理策略：
        - 通过 process_manager 的反向索引查找任务
        - 检查任务状态决定是忽略还是应用结果
        """
        # 初始化 late result 统计（首次访问时）
        if "late_result_ignored" not in self._message_stats:
            self._message_stats["late_result_ignored"] = 0
        if "late_result_applied" not in self._message_stats:
            self._message_stats["late_result_applied"] = 0

        # 解析 payload
        payload = message.payload or {}
        task_id = payload.get("task_id")
        success = payload.get("success", False)
        error = payload.get("error")
        sender = message.sender
        correlation_id = message.correlation_id

        # 如果 payload 中没有 task_id，尝试通过 correlation_id 反查
        if not task_id and correlation_id:
            task_info = self.process_manager.get_task_by_message_id(correlation_id)
            if task_info:
                task_id, assignment_info = task_info
                logger.debug(f"Late result: 通过 correlation_id={correlation_id} 反查到 task_id={task_id}")

        # 如果仍然无法确定 task_id，记录并忽略
        if not task_id:
            logger.warning(
                f"Late result ignored: 无法确定 task_id, "
                f"type={message.type.value}, sender={sender}, "
                f"correlation_id={correlation_id}"
            )
            self._message_stats["late_result_ignored"] += 1
            return

        # 获取任务对象
        task = self.task_queue.get_task(task_id)
        if not task:
            logger.debug(
                f"Late result ignored: task_id={task_id} 不在 TaskQueue 中, type={message.type.value}, sender={sender}"
            )
            self._message_stats["late_result_ignored"] += 1
            return

        # 检查任务状态
        current_status = task.status

        # 情况1: 任务已被重试/重新入队（状态为 PENDING）
        if current_status == TaskStatus.PENDING:
            logger.info(
                f"Late result ignored: task_id={task_id} 已重新入队 (status=PENDING), "
                f"success={success}, sender={sender}"
            )
            self._message_stats["late_result_ignored"] += 1
            return

        # 情况2: 任务已完成/失败（终态）
        if current_status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            logger.debug(
                f"Late result ignored: task_id={task_id} 已处于终态 (status={current_status.value}), sender={sender}"
            )
            self._message_stats["late_result_ignored"] += 1
            return

        # 情况3: 任务仍处于 IN_PROGRESS，检查 assigned_to 是否匹配
        if current_status == TaskStatus.IN_PROGRESS:
            # 获取分配信息验证 sender
            assignment_info = self.process_manager.get_task_assignment(task_id) or {}
            assigned_to = task.assigned_to or (assignment_info.get("agent_id") if assignment_info else None)

            if assigned_to and assigned_to != sender:
                # 发送者不匹配，可能是旧 worker 的响应
                logger.info(
                    f"Late result ignored: task_id={task_id} assigned_to={assigned_to} 但 sender={sender} 不匹配"
                )
                self._message_stats["late_result_ignored"] += 1
                return

            # 应用 late result
            logger.info(
                f"Late result applied: task_id={task_id}, success={success}, "
                f"sender={sender}, correlation_id={correlation_id}"
            )

            # 更新任务状态
            if success:
                task.complete(payload)
                # 更新统计
                iteration = self.state.get_current_iteration()
                if iteration:
                    iteration.tasks_completed += 1
                self.state.total_tasks_completed += 1
            else:
                task.fail(error or "未知错误")
                # 更新统计
                iteration = self.state.get_current_iteration()
                if iteration:
                    iteration.tasks_failed += 1
                self.state.total_tasks_failed += 1

            self.task_queue.update_task(task)

            # 取消任务跟踪
            self.process_manager.untrack_task(task_id)

            self._message_stats["late_result_applied"] += 1
            return

        # 其他状态（ASSIGNED 等），记录警告
        logger.warning(
            f"Late result: task_id={task_id} 处于非预期状态 (status={current_status.value}), "
            f"success={success}, sender={sender}"
        )
        self._message_stats["late_result_ignored"] += 1

    async def _send_and_wait(
        self,
        agent_id: str,
        message: ProcessMessage,
        timeout: float,
    ) -> ProcessMessage | None:
        """发送消息并等待响应

        超时告警策略：
        - 单次/偶发超时 → DEBUG/INFO 级别（避免告警刷屏）
        - 连续超时达到 timeout_warning_threshold 后 → WARNING 级别（带 cooldown）
        - 关键阶段（planner/reviewer）连续超时 → ERROR 级别
        - 触发降级时 → ERROR 级别

        超时计数管理：
        - 正常响应时重置该 agent 的超时计数
        - 超时时递增计数
        """
        import time

        # 创建 Future 等待响应
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[message.id] = future

        # 发送消息
        self.process_manager.send_to_agent(agent_id, message)

        try:
            # 等待响应
            response = await asyncio.wait_for(future, timeout=timeout)
            # 正常响应：重置超时计数
            self._timeout_count[agent_id] = 0
            return response
        except asyncio.TimeoutError:
            self._pending_responses.pop(message.id, None)
            # 递增超时计数
            self._timeout_count[agent_id] = self._timeout_count.get(agent_id, 0) + 1
            consecutive_count = self._timeout_count[agent_id]
            threshold = self.config.timeout_warning_threshold

            # 判断是否为关键阶段（planner/reviewer）
            is_critical_agent = agent_id in (self.planner_id, self.reviewer_id)

            # 日志级别策略
            now = time.time()
            if is_critical_agent and consecutive_count >= threshold:
                # 关键阶段连续超时 → ERROR 级别
                logger.error(f"等待 {agent_id} 响应超时 (连续 {consecutive_count} 次, 关键阶段)")
            elif consecutive_count >= threshold:
                # 达到阈值后 → WARNING 级别（带 cooldown）
                last_warning = self._timeout_warning_cooldown.get(agent_id, 0)
                if now - last_warning >= self.config.timeout_warning_cooldown_seconds:
                    self._timeout_warning_cooldown[agent_id] = now
                    logger.warning(f"等待 {agent_id} 响应超时 (连续 {consecutive_count} 次)")
                else:
                    # cooldown 中，降级为 DEBUG
                    if self.config.heartbeat_debug:
                        logger.debug(f"等待 {agent_id} 响应超时 (连续 {consecutive_count} 次, cooldown 中)")
            elif consecutive_count == 1:
                # 单次超时 → INFO 级别
                logger.info(f"等待 {agent_id} 响应超时 (首次)")
            else:
                # 偶发超时（未达阈值）→ DEBUG 级别
                if self.config.heartbeat_debug:
                    logger.debug(f"等待 {agent_id} 响应超时 ({consecutive_count}/{threshold})")

            return None

    async def _planning_phase(self, goal: str, iteration_id: int) -> None:
        """规划阶段"""
        iteration = self.state.get_current_iteration()
        if iteration is None:
            raise RuntimeError("当前迭代状态为空，无法进入规划阶段")
        iteration.status = IterationStatus.PLANNING

        logger.info(f"[迭代 {iteration_id}] 规划阶段")

        if self.planner_id is None:
            raise RuntimeError("Planner 进程未初始化")

        # 构建上下文
        context = {
            "iteration_id": iteration_id,
            "working_directory": self.config.working_directory,
        }
        if self._iteration_context:
            context["iteration_assistant"] = self._iteration_context

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
            # 记录消息统计（验证消息不丢失）
            logger.debug(f"[迭代 {iteration_id}] 规划失败时消息统计: plan_result={self._message_stats['plan_result']}")
            return

        # 记录成功收到 PLAN_RESULT（验证消息不丢失）
        logger.debug(
            f"[迭代 {iteration_id}] PLAN_RESULT 已收到并处理: total_plan_results={self._message_stats['plan_result']}"
        )

        # 处理规划结果
        tasks_data = response.payload.get("tasks", [])
        for task_data in tasks_data:
            if self._iteration_context:
                task_context = task_data.get("context") or {}
                task_context.setdefault("iteration_assistant", self._iteration_context)
                task_data["context"] = task_context
            task = Task(**task_data)
            await self.task_queue.enqueue(task)
            self.state.total_tasks_created += 1
            iteration.tasks_created += 1

        logger.info(f"[迭代 {iteration_id}] 规划完成，创建 {len(tasks_data)} 个任务")

    async def _execution_phase(self, iteration_id: int) -> None:
        """执行阶段

        包含周期性健康检查，处理不健康 Worker 的在途任务。
        """
        import time

        iteration = self.state.get_current_iteration()
        if iteration is None:
            raise RuntimeError("当前迭代状态为空，无法进入执行阶段")
        iteration.status = IterationStatus.EXECUTING

        pending = self.task_queue.get_pending_count(iteration_id)
        logger.info(f"[迭代 {iteration_id}] 执行阶段，{pending} 个任务")

        # 并行分发任务给 Workers
        active_tasks: dict[str, asyncio.Task] = {}
        # 跟踪 worker -> task_id 的映射
        worker_task_mapping: dict[str, str] = {}
        available_workers = list(self.worker_ids)
        last_health_check = time.time()

        # 诊断计数器：用于检测卡死
        _loop_count = 0
        self._last_progress_time = time.time()  # 用于 max_no_progress_time 计算（仅在任务完成时更新）
        self._last_stall_check_time = time.time()  # 用于卡死检测间隔冷却（进入检测分支时更新）
        _STALL_DETECTION_INTERVAL = self.config.stall_recovery_interval  # 使用配置的间隔

        while True:
            _loop_count += 1

            # ========== 诊断日志与卡死恢复 ==========
            now = time.time()
            if now - self._last_stall_check_time >= _STALL_DETECTION_INTERVAL:
                # 更新卡死检测时间戳（冷却窗口起点）
                self._last_stall_check_time = now

                # 收集诊断信息
                queue_stats = self.task_queue.get_statistics(iteration_id)
                in_flight_tasks = self.process_manager.get_all_in_flight_tasks()
                tasks_in_iteration = self.task_queue.get_tasks_by_iteration(iteration_id)

                # 检测典型卡死模式
                stall_detected = False
                stall_reason = ""

                assigned_tasks = [t for t in tasks_in_iteration if t.status == TaskStatus.ASSIGNED]
                in_progress_tasks = [t for t in tasks_in_iteration if t.status == TaskStatus.IN_PROGRESS]

                if queue_stats["pending"] > 0 and len(active_tasks) == 0 and len(available_workers) == 0:
                    stall_detected = True
                    stall_reason = f"pending={queue_stats['pending']} > 0 但 active_tasks=0, available_workers=0"

                if assigned_tasks and len(active_tasks) == 0:
                    stall_detected = True
                    stall_reason = f"存在 ASSIGNED 状态任务 ({len(assigned_tasks)}) 但无 active_tasks"

                if in_progress_tasks and len(active_tasks) == 0:
                    stall_detected = True
                    stall_reason = f"存在 IN_PROGRESS 状态任务 ({len(in_progress_tasks)}) 但无 active_tasks"

                # 诊断输出策略：
                # - 仅在 stall_detected=True 时输出诊断信息（避免正常长任务时刷屏）
                # - 所有诊断日志（摘要、详细、关键卡死行）均受冷却窗口控制
                # - 冷却窗口使用 _last_stall_diagnostic_time，与 stall_recovery_interval 对齐
                should_emit_diagnostic = (
                    stall_detected
                    and self.config.stall_diagnostics_enabled
                    and (
                        self._last_stall_diagnostic_time == 0.0
                        or now - self._last_stall_diagnostic_time >= self.config.stall_recovery_interval
                    )
                )

                if should_emit_diagnostic:
                    # 更新诊断输出时间戳（冷却窗口起点）
                    self._last_stall_diagnostic_time = now

                    # 选择日志级别
                    log_level = self.config.stall_diagnostics_level.lower()
                    log_fn = {
                        "debug": logger.debug,
                        "info": logger.info,
                        "warning": logger.warning,
                        "error": logger.error,
                    }.get(log_level, logger.warning)

                    # 摘要日志（仅在 stall_detected=True 且冷却窗口外时输出）
                    summary = (
                        f"[诊断] 卡死检测 | iteration={iteration_id}, "
                        f"pending={queue_stats['pending']}, "
                        f"completed={queue_stats['completed']}, "
                        f"failed={queue_stats['failed']}, "
                        f"available_workers={len(available_workers)}, "
                        f"active_tasks={len(active_tasks)}, "
                        f"in_flight={len(in_flight_tasks)}, "
                        f'stall_reason="{stall_reason}"'
                    )
                    log_fn(summary)

                    # 计算无进展时间，决定卡死模式日志级别
                    # 仅在接近 max_no_progress_time 阈值时使用 ERROR 级别
                    no_progress_duration = now - self._last_progress_time
                    near_degradation_threshold = self.config.max_no_progress_time * 0.8
                    if no_progress_duration >= near_degradation_threshold:
                        # 接近降级阈值，使用 ERROR 级别提醒
                        logger.error(
                            f"[诊断] ⚠ 卡死模式检测: {stall_reason} "
                            f"(无进展 {no_progress_duration:.1f}s，临近降级阈值 {self.config.max_no_progress_time}s)"
                        )
                    else:
                        # 正常卡死检测，使用配置的日志级别
                        log_fn(f"[诊断] ⚠ 卡死模式检测: {stall_reason}")

                    # 详细诊断（仅当 stall_diagnostics_detail=True 时输出）
                    if self.config.stall_diagnostics_detail:
                        log_fn(f"[诊断] === 详细诊断 (循环 #{_loop_count}) ===")
                        log_fn(f"[诊断] TaskQueue.get_statistics(): {queue_stats}")
                        log_fn(f"[诊断] available_workers: {available_workers}")
                        log_fn(f"[诊断] active_tasks: {list(active_tasks.keys())}")
                        log_fn(f"[诊断] in_flight_tasks: {list(in_flight_tasks.keys())}")

                        # 限制输出的任务数量
                        max_tasks = self.config.stall_diagnostics_max_tasks
                        task_count = len(tasks_in_iteration)
                        tasks_to_show = tasks_in_iteration[:max_tasks]

                        for diagnostic_task in tasks_to_show:
                            log_fn(
                                f"[诊断] Task {diagnostic_task.id}: status={diagnostic_task.status.value}, "
                                f"retry_count={diagnostic_task.retry_count}, assigned_to={diagnostic_task.assigned_to}"
                            )

                        if task_count > max_tasks:
                            log_fn(f"[诊断] ... 省略 {task_count - max_tasks} 个任务")

                # 尝试恢复
                if stall_detected and self._should_attempt_recovery(iteration_id):
                    recovery_result = await self._recover_stalled_iteration(
                        iteration_id=iteration_id,
                        reason=stall_reason,
                        active_futures=active_tasks,
                        worker_task_mapping=worker_task_mapping,
                    )

                    if recovery_result["degraded"]:
                        # 恢复触发了降级
                        logger.error(f"[迭代 {iteration_id}] 恢复触发降级: {recovery_result['reason']}")
                        break

                    if recovery_result["requeued"] > 0:
                        # 有任务被重新入队，刷新可用 worker 列表
                        alive_workers = [wid for wid in self.worker_ids if self.process_manager.is_alive(wid)]
                        # 只添加不在 active_tasks 中的 worker
                        for wid in alive_workers:
                            if wid not in active_tasks and wid not in available_workers:
                                available_workers.append(wid)
                        logger.info(f"[恢复后] 刷新可用 workers: {available_workers}")

                # 只有在检测到卡死时才重置计时器（避免频繁触发）
                # 正常情况下计时器在任务完成时重置

            # 执行过程中的周期性健康检查（使用配置的间隔）
            if time.time() - last_health_check >= self.config.execution_health_check_interval:
                last_health_check = time.time()
                health_result = await self._perform_health_check()

                if self._degraded:
                    logger.warning(f"[迭代 {iteration_id}] 执行阶段降级，中止剩余任务")
                    break

                # 从可用 Worker 中移除不健康的
                unhealthy_workers = health_result.get_unhealthy_workers()
                for unhealthy_worker_id in unhealthy_workers:
                    if unhealthy_worker_id in available_workers:
                        available_workers.remove(unhealthy_worker_id)
                        logger.warning(f"从可用 Worker 列表中移除: {unhealthy_worker_id}")

            # 分配任务给空闲 Worker（跳过不健康的）
            while available_workers:
                # 检查 Worker 是否健康（使用进程存活状态快速检查）
                worker_id = available_workers[0]
                if not self.process_manager.is_alive(worker_id):
                    available_workers.pop(0)
                    logger.warning(f"Worker {worker_id} 进程已死亡，跳过")
                    continue

                task: Task | None = await self.task_queue.dequeue(iteration_id, timeout=0.1)
                if task is None:
                    break

                available_workers.pop(0)

                # 设置任务分配状态并启动
                task.assigned_to = worker_id
                task.start()  # 将状态推进到 IN_PROGRESS
                self.task_queue.update_task(task)

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

                # 跟踪任务分配
                self.process_manager.track_task_assignment(
                    task_id=task.id,
                    agent_id=worker_id,
                    message_id=request.id,
                )
                worker_task_mapping[worker_id] = task.id

                # 创建异步等待任务
                async_task = asyncio.create_task(self._send_and_wait(worker_id, request, self.config.execution_timeout))
                active_tasks[worker_id] = async_task

                logger.debug(f"任务 {task.id} 分配给 {worker_id}")

            # 如果没有活动任务，检查是否完成
            if not active_tasks:
                if self.task_queue.is_iteration_complete(iteration_id):
                    break

                # 兜底退出条件：无可用 worker 且仍有未终态任务
                # 检查是否还有待处理或进行中的任务
                pending_count = self.task_queue.get_pending_count(iteration_id)
                in_progress_count = self.task_queue.get_in_progress_count(iteration_id)
                has_unfinished_tasks = pending_count > 0 or in_progress_count > 0

                if not available_workers and has_unfinished_tasks:
                    # 检查是否所有 worker 都已死亡
                    alive_workers = [wid for wid in self.worker_ids if self.process_manager.is_alive(wid)]
                    if not alive_workers:
                        # 无可用 worker，触发降级退出
                        degradation_msg = (
                            f"无可用 Worker 且仍有 {pending_count} 个待处理 + {in_progress_count} 个进行中任务"
                        )
                        self._trigger_degradation(degradation_msg)
                        logger.error(f"[迭代 {iteration_id}] 兜底退出: {degradation_msg}")
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
                completed_worker_id: str | None = None
                for wid, atask in active_tasks.items():
                    if atask == completed:
                        completed_worker_id = wid
                        break

                if completed_worker_id:
                    del active_tasks[completed_worker_id]

                    # 取消任务跟踪
                    task_id = worker_task_mapping.pop(completed_worker_id, None)
                    if task_id:
                        self.process_manager.untrack_task(task_id)

                    # 只有健康的 Worker 才重新加入可用列表
                    if self.process_manager.is_alive(completed_worker_id):
                        available_workers.append(completed_worker_id)

                    # 重置进度计时器（有任务完成说明有进展）
                    self._last_progress_time = time.time()

                    # 处理结果
                    try:
                        response = completed.result()
                        if response:
                            resp_task_id = response.payload.get("task_id")
                            success = response.payload.get("success", False)

                            # 更新任务状态
                            result_task = self.task_queue.get_task(resp_task_id)
                            if result_task is None:
                                continue
                            if success:
                                result_task.complete(response.payload)
                                iteration.tasks_completed += 1
                                self.state.total_tasks_completed += 1
                            else:
                                result_task.fail(response.payload.get("error", "未知错误"))
                                iteration.tasks_failed += 1
                                self.state.total_tasks_failed += 1
                                self.task_queue.update_task(result_task)
                        else:
                            # 响应为 None（超时），检查任务是否需要重新入队
                            if task_id:
                                timeout_task = self.task_queue.get_task(task_id)
                                if timeout_task and timeout_task.status == TaskStatus.IN_PROGRESS:
                                    if self.config.requeue_on_worker_death:
                                        # 关键修复：必须调用 requeue 而不只是改状态
                                        # 只改状态会导致任务卡在 PENDING 但不在优先级队列中
                                        logger.warning(f"任务 {task_id} 响应超时，重新入队")
                                        await self.task_queue.requeue(timeout_task, reason="Worker 响应超时")
                                    else:
                                        timeout_task.fail("Worker 响应超时")
                                        iteration.tasks_failed += 1
                                        self.state.total_tasks_failed += 1
                                        self.task_queue.update_task(timeout_task)
                    except Exception as e:
                        logger.error(f"处理任务结果异常: {e}")

        stats = self.task_queue.get_statistics(iteration_id)
        logger.info(f"[迭代 {iteration_id}] 执行完成: {stats['completed']} 成功, {stats['failed']} 失败")

        # 记录消息统计（验证 TASK_RESULT 不丢失）
        logger.debug(
            f"[迭代 {iteration_id}] 执行阶段消息统计: "
            f"task_result={self._message_stats['task_result']}, "
            f"heartbeat={self._message_stats['heartbeat']}, "
            f"total={self._message_stats['total_received']}"
        )

    async def _review_phase(self, goal: str, iteration_id: int) -> ReviewDecision:
        """评审阶段"""
        iteration = self.state.get_current_iteration()
        if iteration is None:
            raise RuntimeError("当前迭代状态为空，无法进入评审阶段")
        iteration.status = IterationStatus.REVIEWING

        logger.info(f"[迭代 {iteration_id}] 评审阶段")

        if self.reviewer_id is None:
            raise RuntimeError("Reviewer 进程未初始化")

        # 收集任务信息（使用统一的 to_commit_entry 格式）
        tasks = self.task_queue.get_tasks_by_iteration(iteration_id)
        completed = [t.to_commit_entry() for t in tasks if t.status == TaskStatus.COMPLETED]
        failed = [{"id": t.id, "title": t.title, "error": t.error} for t in tasks if t.status == TaskStatus.FAILED]

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
                "context": {"iteration_assistant": self._iteration_context} if self._iteration_context else {},
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
        if iteration is None:
            raise RuntimeError("当前迭代状态为空，无法进入提交阶段")
        iteration.status = IterationStatus.COMMITTING

        logger.info(f"[迭代 {iteration_id}] 提交阶段开始")

        # 收集已完成的任务（使用统一的 to_commit_entry 格式）
        # to_commit_entry() 返回: {"id", "title", "description", "result"}
        tasks = self.task_queue.get_tasks_by_iteration(iteration_id)
        completed_tasks = [t.to_commit_entry() for t in tasks if t.status == TaskStatus.COMPLETED]

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
        iteration.commit_hash = commit_result.get("commit_hash", "")
        iteration.commit_message = commit_result.get("message", "")
        iteration.pushed = commit_result.get("pushed", False)
        iteration.commit_files = commit_result.get("files_changed", [])

        # 记录错误信息到 iteration（commit 失败或 push 失败均不中断主流程）
        success = commit_result.get("success", False)
        if not success:
            iteration.commit_error = commit_result.get("error", "Unknown commit error")
        if commit_result.get("push_error"):
            iteration.push_error = commit_result.get("push_error")

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
            "degraded": self._degraded,
            "degradation_reason": self._degradation_reason,
            "message_stats": self._message_stats.copy(),  # 消息处理统计（验证消息不丢失）
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
