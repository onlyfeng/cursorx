"""Agent 进程管理器

管理所有 Agent 进程的生命周期
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from .message_queue import MessageQueue, ProcessMessage, ProcessMessageType
from .worker import AgentWorkerProcess


@dataclass
class HealthCheckResult:
    """健康检查结果

    Attributes:
        healthy: 健康的 agent_id 列表
        unhealthy: 不健康的 agent_id 列表
        all_healthy: 是否所有进程都健康
        details: 每个进程的详细状态 {agent_id: {"healthy": bool, "reason": str}}
    """

    healthy: list[str] = field(default_factory=list)
    unhealthy: list[str] = field(default_factory=list)
    all_healthy: bool = True
    details: dict[str, dict] = field(default_factory=dict)

    def get_unhealthy_workers(self) -> list[str]:
        """获取不健康的 worker 进程 ID 列表"""
        return [aid for aid in self.unhealthy if "worker" in aid.lower()]


class AgentProcessManager:
    """Agent 进程管理器

    负责:
    1. 创建和管理 Agent 进程
    2. 进程间消息路由
    3. 进程健康监控
    4. 优雅关闭
    5. 跟踪任务分配状态（用于健康检查后的任务恢复）
    """

    def __init__(self):
        self.message_queue = MessageQueue()
        self._processes: dict[str, AgentWorkerProcess] = {}
        self._process_info: dict[str, dict] = {}
        self._running = False
        # 跟踪任务分配：{task_id: {"agent_id": str, "assigned_at": float, "message_id": str}}
        self._task_assignments: dict[str, dict] = {}
        # 反向索引：{message_id: task_id}，用于通过消息 ID 快速查找任务 ID
        self._message_to_task: dict[str, str] = {}

    def spawn_agent(
        self,
        agent_class: type[AgentWorkerProcess],
        agent_id: str,
        agent_type: str,
        config: dict,
    ) -> AgentWorkerProcess:
        """创建并启动 Agent 进程

        Args:
            agent_class: Agent 进程类
            agent_id: Agent ID
            agent_type: Agent 类型
            config: 配置

        Returns:
            Agent 进程实例
        """
        # 创建该 Agent 的专用队列
        inbox = self.message_queue.create_agent_queue(agent_id)
        outbox = self.message_queue.to_coordinator

        # 创建进程
        process = agent_class(
            agent_id=agent_id,
            agent_type=agent_type,
            inbox=inbox,
            outbox=outbox,
            config=config,
        )

        # 启动进程
        process.start()

        self._processes[agent_id] = process
        self._process_info[agent_id] = {
            "type": agent_type,
            "pid": process.pid,
            "started_at": time.time(),
            "status": "starting",
        }

        logger.info(f"Agent 进程已创建: {agent_id} (类型: {agent_type}, PID: {process.pid})")

        return process

    def send_to_agent(self, agent_id: str, message: ProcessMessage) -> bool:
        """发送消息给指定 Agent"""
        return self.message_queue.send_to_agent(agent_id, message)

    def broadcast(self, message: ProcessMessage) -> None:
        """广播消息给所有 Agent"""
        self.message_queue.broadcast_to_agents(message)

    def receive_message(self, timeout: Optional[float] = None) -> Optional[ProcessMessage]:
        """接收来自 Agent 的消息"""
        return self.message_queue.receive_from_coordinator(timeout)

    def wait_for_ready(self, agent_id: str, timeout: float = 30.0) -> bool:
        """等待 Agent 就绪

        Args:
            agent_id: Agent ID
            timeout: 超时时间

        Returns:
            是否就绪
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            message = self.receive_message(timeout=1.0)

            if (
                message
                and message.sender == agent_id
                and message.type == ProcessMessageType.STATUS_RESPONSE
                and message.payload.get("status") == "ready"
            ):
                self._process_info[agent_id]["status"] = "ready"
                self._process_info[agent_id]["pid"] = message.payload.get("pid")
                logger.info(f"Agent 就绪: {agent_id}")
                return True

        logger.warning(f"Agent 就绪超时: {agent_id}")
        return False

    def wait_all_ready(self, timeout: float = 60.0) -> bool:
        """等待所有 Agent 就绪"""
        pending = set(self._processes.keys())
        start_time = time.time()

        while pending and time.time() - start_time < timeout:
            message = self.receive_message(timeout=1.0)

            if (
                message
                and message.sender in pending
                and message.type == ProcessMessageType.STATUS_RESPONSE
                and message.payload.get("status") == "ready"
            ):
                self._process_info[message.sender]["status"] = "ready"
                self._process_info[message.sender]["pid"] = message.payload.get("pid")
                pending.remove(message.sender)
                logger.info(f"Agent 就绪: {message.sender} (剩余 {len(pending)})")

        if pending:
            logger.warning(f"以下 Agent 未就绪: {pending}")
            return False

        logger.info("所有 Agent 已就绪")
        return True

    def is_alive(self, agent_id: str) -> bool:
        """检查 Agent 进程是否存活"""
        process = self._processes.get(agent_id)
        return process is not None and process.is_alive()

    def get_process_info(self, agent_id: str) -> Optional[dict]:
        """获取进程信息"""
        return self._process_info.get(agent_id)

    def get_all_process_info(self) -> dict[str, dict]:
        """获取所有进程信息"""
        # 更新存活状态
        for agent_id in self._processes:
            self._process_info[agent_id]["alive"] = self.is_alive(agent_id)
        return self._process_info.copy()

    def terminate_agent(self, agent_id: str, graceful: bool = True) -> None:
        """终止 Agent 进程

        Args:
            agent_id: Agent ID
            graceful: 是否优雅关闭
        """
        process = self._processes.get(agent_id)
        if not process:
            return

        if graceful:
            # 发送关闭消息
            self.send_to_agent(
                agent_id,
                ProcessMessage(
                    type=ProcessMessageType.SHUTDOWN,
                    sender="manager",
                ),
            )
            # 等待进程退出
            process.join(timeout=10.0)

        # 如果还在运行，强制终止
        if process.is_alive():
            logger.warning(f"强制终止 Agent: {agent_id}")
            process.terminate()
            process.join(timeout=5.0)

            if process.is_alive():
                process.kill()

        self._process_info[agent_id]["status"] = "terminated"
        logger.info(f"Agent 已终止: {agent_id}")

    def shutdown_all(self, graceful: bool = True) -> None:
        """关闭所有 Agent 进程"""
        logger.info("正在关闭所有 Agent 进程...")

        if graceful:
            # 广播关闭消息
            self.broadcast(
                ProcessMessage(
                    type=ProcessMessageType.SHUTDOWN,
                    sender="manager",
                )
            )

            # 等待所有进程退出
            for agent_id, process in self._processes.items():
                process.join(timeout=10.0)
                if process.is_alive():
                    logger.warning(f"Agent {agent_id} 未响应关闭，强制终止")
                    process.terminate()
        else:
            # 直接终止所有进程
            for process in self._processes.values():
                process.terminate()

        # 确保所有进程都已退出
        for process in self._processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()

        # 清理
        self.message_queue.cleanup()
        self._processes.clear()

        logger.info("所有 Agent 进程已关闭")

    def track_task_assignment(
        self,
        task_id: str,
        agent_id: str,
        message_id: str,
    ) -> None:
        """跟踪任务分配

        Args:
            task_id: 任务 ID
            agent_id: 分配给的 Agent ID
            message_id: 发送的消息 ID（用于关联响应）
        """
        self._task_assignments[task_id] = {
            "agent_id": agent_id,
            "assigned_at": time.time(),
            "message_id": message_id,
        }
        # 维护反向索引
        self._message_to_task[message_id] = task_id

    def untrack_task(self, task_id: str) -> Optional[dict]:
        """取消跟踪任务

        Args:
            task_id: 任务 ID

        Returns:
            被移除的分配信息，如果不存在则返回 None
        """
        info = self._task_assignments.pop(task_id, None)
        # 同时清理反向索引
        if info and info.get("message_id"):
            self._message_to_task.pop(info["message_id"], None)
        return info

    def get_tasks_by_agent(self, agent_id: str) -> list[str]:
        """获取分配给指定 Agent 的所有任务 ID

        Args:
            agent_id: Agent ID

        Returns:
            任务 ID 列表
        """
        return [task_id for task_id, info in self._task_assignments.items() if info["agent_id"] == agent_id]

    def get_all_in_flight_tasks(self) -> dict[str, dict]:
        """获取所有在途任务的分配信息

        Returns:
            {task_id: {"agent_id": str, "assigned_at": float, "message_id": str}}
        """
        return self._task_assignments.copy()

    def get_task_by_message_id(self, message_id: str) -> Optional[tuple[str, dict]]:
        """通过消息 ID 查找任务信息

        用于处理 late result 场景：当收到 TASK_RESULT 但 correlation_id
        不在 _pending_responses 时，通过此方法反查任务 ID。

        Args:
            message_id: 发送任务时的消息 ID（即 correlation_id）

        Returns:
            (task_id, assignment_info) 元组，如果不存在则返回 None
        """
        task_id = self._message_to_task.get(message_id)
        if task_id is None:
            return None
        assignment_info = self._task_assignments.get(task_id)
        if assignment_info is None:
            return None
        return (task_id, assignment_info)

    def get_task_assignment(self, task_id: str) -> Optional[dict]:
        """获取任务的分配信息

        Args:
            task_id: 任务 ID

        Returns:
            分配信息 {"agent_id": str, "assigned_at": float, "message_id": str}
            如果不存在则返回 None
        """
        return self._task_assignments.get(task_id)

    def health_check(self, timeout: float = 5.0) -> HealthCheckResult:
        """健康检查（仅检查进程存活状态）

        注意：此方法已重构，不再读取消息队列以避免与 orchestrator 的
        _message_loop 竞争消息。心跳响应的收集由 orchestrator 的
        _perform_health_check() 通过 _message_loop 完成。

        此方法仅检查进程的存活状态（is_alive()），用于快速判断
        进程是否已死亡，不涉及心跳响应的等待。

        Args:
            timeout: 保留参数（用于向后兼容，实际不使用）

        Returns:
            HealthCheckResult 对象，包含健康/不健康的 agent 列表和详细信息
        """
        result = HealthCheckResult()

        if not self._processes:
            return result

        # 仅检查进程存活状态，不读取消息队列
        for agent_id, process in self._processes.items():
            is_alive = process.is_alive() if process else False

            if is_alive:
                result.healthy.append(agent_id)
                result.details[agent_id] = {
                    "healthy": True,
                    "reason": "process_alive",
                    "pid": process.pid if process else None,
                }
            else:
                result.unhealthy.append(agent_id)
                result.details[agent_id] = {
                    "healthy": False,
                    "reason": "process_dead",
                    "is_alive": False,
                }
                logger.warning(f"进程已死亡: {agent_id}")

        result.all_healthy = len(result.unhealthy) == 0
        return result

    def check_processes_alive(self) -> dict[str, bool]:
        """快速检查所有进程的存活状态

        轻量级方法，仅调用 is_alive()，不涉及消息队列操作。

        Returns:
            {agent_id: is_alive} 字典
        """
        return {agent_id: (process.is_alive() if process else False) for agent_id, process in self._processes.items()}

    def health_check_simple(self) -> dict[str, bool]:
        """简单健康检查（向后兼容）

        直接调用 check_processes_alive()，仅检查进程存活状态。

        Returns:
            {agent_id: is_healthy} 字典
        """
        return self.check_processes_alive()
