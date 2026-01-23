"""Agent 工作进程

每个 Agent 作为独立进程运行
"""
import concurrent.futures
import os
import signal
import sys
import threading
import time
from abc import abstractmethod
from multiprocessing import Process, Queue
from queue import Empty
from typing import Optional

from loguru import logger

from core.platform import register_signal_handler
from .message_queue import ProcessMessage, ProcessMessageType


class AgentWorkerProcess(Process):
    """Agent 工作进程基类

    每个 Agent 实例作为独立进程运行。

    线程架构说明：
        为避免任务执行期间阻塞心跳响应，采用多线程架构：
        1. 主线程：消息循环，快速处理控制消息（心跳、状态查询等）
        2. 工作线程：执行耗时的业务任务（通过 ThreadPoolExecutor）

        这确保了即使任务执行时间很长，主线程也能及时响应心跳请求。
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        inbox: Queue,
        outbox: Queue,
        config: dict,
    ):
        super().__init__(name=f"Agent-{agent_type}-{agent_id[:8]}", daemon=True)
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.inbox = inbox      # 接收消息的队列
        self.outbox = outbox    # 发送消息的队列（到协调器）
        self.config = config
        self._running = False

        # 工作线程池（用于执行耗时任务）
        # 注意：这些对象在 run() 方法中初始化，以支持 macOS 的 spawn 启动方式
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._current_task_future: Optional[concurrent.futures.Future] = None
        self._last_heartbeat_time: float = 0.0           # 最后心跳响应时间
        self._is_busy: bool = False                      # 是否正在执行任务
        # 锁对象在 run() 中初始化，避免 pickle 序列化问题
        # macOS 默认使用 spawn 方式启动子进程，需要序列化所有对象
        # threading.Lock 无法被 pickle 序列化
        self._task_lock: Optional[threading.Lock] = None

    def run(self) -> None:
        """进程主循环"""
        # 设置信号处理（跨平台兼容）
        # Windows 不完全支持 SIGTERM，使用 register_signal_handler 处理
        register_signal_handler(
            signal.SIGTERM,
            self._handle_shutdown,
            fallback_sig=signal.SIGINT,  # Windows 回退到 SIGINT
        )
        # SIGINT 在所有平台都支持
        register_signal_handler(signal.SIGINT, self._handle_shutdown)

        # 配置进程内日志
        self._setup_logging()

        logger.info(f"[{self.agent_id}] 进程启动 (PID: {os.getpid()})")

        self._running = True

        # 初始化锁对象（在子进程中创建，避免 pickle 序列化问题）
        # macOS 使用 spawn 启动方式，需要在子进程中初始化不可序列化的对象
        self._task_lock = threading.Lock()

        # 初始化工作线程池（单线程，确保任务串行执行）
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"worker-{self.agent_id[:8]}"
        )

        try:
            # 初始化
            self.on_start()

            # 发送就绪消息
            self._send_message(ProcessMessageType.STATUS_RESPONSE, {
                "status": "ready",
                "pid": os.getpid(),
            })

            # 主循环 - 快速处理消息，保持响应性
            while self._running:
                try:
                    # 使用短超时，保持响应性
                    message = self.inbox.get(timeout=0.5)
                    self._handle_message(message)
                except Empty:
                    # 队列超时是正常的，继续循环
                    # 检查工作线程是否完成（清理状态）
                    self._check_task_completion()
                except Exception as e:
                    if self._running:
                        if "Empty" not in str(type(e)):
                            logger.error(f"[{self.agent_id}] 消息处理异常: {e}")

        except Exception as e:
            logger.exception(f"[{self.agent_id}] 进程异常: {e}")
        finally:
            self._running = False
            # 关闭工作线程池
            if self._executor:
                self._executor.shutdown(wait=False)
            self.on_stop()
            logger.info(f"[{self.agent_id}] 进程退出")

    def _check_task_completion(self) -> None:
        """检查工作线程是否完成任务"""
        with self._task_lock:
            if self._current_task_future and self._current_task_future.done():
                try:
                    # 获取结果（如果有异常会抛出）
                    self._current_task_future.result()
                except Exception as e:
                    logger.error(f"[{self.agent_id}] 任务执行异常: {e}")
                finally:
                    self._current_task_future = None
                    self._is_busy = False

    def _setup_logging(self) -> None:
        """配置进程内日志

        根据 config 配置日志级别：
        - log_level: 日志级别（DEBUG/INFO/WARNING/ERROR），默认 INFO
        - verbose: 是否启用详细模式（等价于 log_level=DEBUG）
        - heartbeat_debug: 是否输出心跳调试日志（默认 False）

        日志过滤策略：
        - 非 verbose 模式下过滤 DEBUG 级别日志
        - 心跳相关日志仅在 heartbeat_debug=True 时输出
        """
        # 读取配置
        verbose = self.config.get("verbose", False)
        log_level = self.config.get("log_level", "DEBUG" if verbose else "INFO")
        self._heartbeat_debug = self.config.get("heartbeat_debug", False)

        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[agent_id]}</cyan> - <level>{message}</level>",
            level=log_level,
            filter=lambda record: record["level"].name != "DEBUG" or verbose,
        )
        logger.configure(extra={"agent_id": self.agent_id[:12]})

    def _handle_shutdown(self, signum, frame) -> None:
        """处理关闭信号"""
        logger.info(f"[{self.agent_id}] 收到关闭信号")
        self._running = False

    def _respond_heartbeat(self, message: ProcessMessage) -> None:
        """响应心跳请求（在主线程中快速执行）

        Args:
            message: 心跳请求消息
        """
        self._last_heartbeat_time = time.time()
        payload = {
            "alive": True,
            "busy": self._is_busy,
            "pid": os.getpid(),
            "timestamp": self._last_heartbeat_time,
        }
        # 如果有 request_id，添加到响应中
        if message.payload.get("request_id"):
            payload["request_id"] = message.payload.get("request_id")

        self._send_message(ProcessMessageType.HEARTBEAT, payload)
        # 心跳日志仅在 heartbeat_debug 模式下输出（避免高频刷屏）
        if getattr(self, "_heartbeat_debug", False):
            logger.debug(f"[{self.agent_id}] 心跳响应已发送 (busy={self._is_busy})")

    def _handle_message(self, message: ProcessMessage) -> None:
        """处理接收到的消息

        控制消息（心跳、状态查询、关闭）在主线程立即处理。
        业务消息（任务等）提交到工作线程异步执行。
        """
        # 心跳消息仅在 heartbeat_debug 模式下输出日志（避免高频刷屏）
        if message.type == ProcessMessageType.HEARTBEAT:
            if getattr(self, "_heartbeat_debug", False):
                logger.debug(f"[{self.agent_id}] 收到消息: {message.type.value}")
        else:
            logger.debug(f"[{self.agent_id}] 收到消息: {message.type.value}")

        if message.type == ProcessMessageType.SHUTDOWN:
            self._running = False
            return

        if message.type == ProcessMessageType.HEARTBEAT:
            # 心跳在主线程立即响应（不阻塞）
            self._respond_heartbeat(message)
            return

        if message.type == ProcessMessageType.STATUS_REQUEST:
            self._send_message(ProcessMessageType.STATUS_RESPONSE, self.get_status())
            return

        # 业务消息提交到工作线程执行
        self._submit_to_worker(message)

    def _submit_to_worker(self, message: ProcessMessage) -> None:
        """将业务消息提交到工作线程执行

        Args:
            message: 业务消息
        """
        with self._task_lock:
            if self._is_busy:
                # 已有任务在执行，记录警告
                logger.warning(
                    f"[{self.agent_id}] 收到新任务但当前正忙，消息将排队: {message.type.value}"
                )

            self._is_busy = True

            def execute_task():
                try:
                    self.handle_message(message)
                except Exception as e:
                    logger.exception(f"[{self.agent_id}] 任务执行异常: {e}")
                finally:
                    with self._task_lock:
                        self._is_busy = False
                        self._current_task_future = None

            self._current_task_future = self._executor.submit(execute_task)

    def _send_message(
        self,
        msg_type: ProcessMessageType,
        payload: dict,
        correlation_id: Optional[str] = None,
    ) -> None:
        """发送消息到协调器"""
        message = ProcessMessage(
            type=msg_type,
            sender=self.agent_id,
            payload=payload,
            correlation_id=correlation_id,
        )
        self.outbox.put(message)

    @abstractmethod
    def on_start(self) -> None:
        """进程启动时的初始化"""
        pass

    @abstractmethod
    def on_stop(self) -> None:
        """进程停止时的清理"""
        pass

    @abstractmethod
    def handle_message(self, message: ProcessMessage) -> None:
        """处理业务消息"""
        pass

    @abstractmethod
    def get_status(self) -> dict:
        """获取当前状态"""
        pass
