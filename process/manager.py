"""Agent 进程管理器

管理所有 Agent 进程的生命周期
"""
import multiprocessing as mp
from multiprocessing import Queue
from typing import Dict, Optional, Type
import time
import os
import signal

from loguru import logger

from .worker import AgentWorkerProcess
from .message_queue import MessageQueue, ProcessMessage, ProcessMessageType


class AgentProcessManager:
    """Agent 进程管理器
    
    负责:
    1. 创建和管理 Agent 进程
    2. 进程间消息路由
    3. 进程健康监控
    4. 优雅关闭
    """
    
    def __init__(self):
        self.message_queue = MessageQueue()
        self._processes: Dict[str, AgentWorkerProcess] = {}
        self._process_info: Dict[str, dict] = {}
        self._running = False
    
    def spawn_agent(
        self,
        agent_class: Type[AgentWorkerProcess],
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
            
            if message and message.sender == agent_id:
                if message.type == ProcessMessageType.STATUS_RESPONSE:
                    if message.payload.get("status") == "ready":
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
            
            if message and message.sender in pending:
                if message.type == ProcessMessageType.STATUS_RESPONSE:
                    if message.payload.get("status") == "ready":
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
    
    def get_all_process_info(self) -> Dict[str, dict]:
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
            self.send_to_agent(agent_id, ProcessMessage(
                type=ProcessMessageType.SHUTDOWN,
                sender="manager",
            ))
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
            self.broadcast(ProcessMessage(
                type=ProcessMessageType.SHUTDOWN,
                sender="manager",
            ))
            
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
    
    def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        results = {}
        
        # 发送心跳请求
        self.broadcast(ProcessMessage(
            type=ProcessMessageType.HEARTBEAT,
            sender="manager",
        ))
        
        # 收集响应
        pending = set(self._processes.keys())
        deadline = time.time() + 5.0
        
        while pending and time.time() < deadline:
            message = self.receive_message(timeout=1.0)
            if message and message.type == ProcessMessageType.HEARTBEAT:
                if message.sender in pending:
                    results[message.sender] = True
                    pending.remove(message.sender)
        
        # 未响应的标记为不健康
        for agent_id in pending:
            results[agent_id] = False
        
        return results
