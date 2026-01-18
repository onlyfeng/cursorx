"""Agent 工作进程

每个 Agent 作为独立进程运行
"""
import os
import signal
import sys
from abc import abstractmethod
from multiprocessing import Process, Queue
from typing import Optional

from loguru import logger

from .message_queue import ProcessMessage, ProcessMessageType


class AgentWorkerProcess(Process):
    """Agent 工作进程基类

    每个 Agent 实例作为独立进程运行
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

    def run(self) -> None:
        """进程主循环"""
        # 设置信号处理
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # 配置进程内日志
        self._setup_logging()

        logger.info(f"[{self.agent_id}] 进程启动 (PID: {os.getpid()})")

        self._running = True

        try:
            # 初始化
            self.on_start()

            # 发送就绪消息
            self._send_message(ProcessMessageType.STATUS_RESPONSE, {
                "status": "ready",
                "pid": os.getpid(),
            })

            # 主循环
            while self._running:
                try:
                    # 等待消息
                    message = self.inbox.get(timeout=1.0)
                    self._handle_message(message)
                except Exception as e:
                    if self._running:
                        # 队列超时是正常的，继续循环
                        if "Empty" not in str(type(e)):
                            logger.error(f"[{self.agent_id}] 消息处理异常: {e}")

        except Exception as e:
            logger.exception(f"[{self.agent_id}] 进程异常: {e}")
        finally:
            self.on_stop()
            logger.info(f"[{self.agent_id}] 进程退出")

    def _setup_logging(self) -> None:
        """配置进程内日志"""
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[agent_id]}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )
        logger.configure(extra={"agent_id": self.agent_id[:12]})

    def _handle_shutdown(self, signum, frame) -> None:
        """处理关闭信号"""
        logger.info(f"[{self.agent_id}] 收到关闭信号")
        self._running = False

    def _handle_message(self, message: ProcessMessage) -> None:
        """处理接收到的消息"""
        logger.debug(f"[{self.agent_id}] 收到消息: {message.type.value}")

        if message.type == ProcessMessageType.SHUTDOWN:
            self._running = False
            return

        if message.type == ProcessMessageType.HEARTBEAT:
            self._send_message(ProcessMessageType.HEARTBEAT, {"alive": True})
            return

        if message.type == ProcessMessageType.STATUS_REQUEST:
            self._send_message(ProcessMessageType.STATUS_RESPONSE, self.get_status())
            return

        # 分发给具体处理方法
        self.handle_message(message)

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
