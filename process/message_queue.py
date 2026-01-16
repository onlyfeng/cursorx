"""进程间消息队列

基于 multiprocessing.Queue 实现进程间通信
"""
import multiprocessing as mp
from multiprocessing import Queue
from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import pickle


class ProcessMessageType(str, Enum):
    """进程消息类型"""
    # 任务相关
    TASK_ASSIGN = "task_assign"          # 分配任务
    TASK_RESULT = "task_result"          # 任务结果
    TASK_PROGRESS = "task_progress"      # 任务进度
    
    # 控制相关
    SHUTDOWN = "shutdown"                # 关闭进程
    HEARTBEAT = "heartbeat"              # 心跳
    STATUS_REQUEST = "status_request"    # 请求状态
    STATUS_RESPONSE = "status_response"  # 状态响应
    
    # 规划相关
    PLAN_REQUEST = "plan_request"        # 请求规划
    PLAN_RESULT = "plan_result"          # 规划结果
    
    # 评审相关
    REVIEW_REQUEST = "review_request"    # 请求评审
    REVIEW_RESULT = "review_result"      # 评审结果


@dataclass
class ProcessMessage:
    """进程间消息"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    type: ProcessMessageType = ProcessMessageType.HEARTBEAT
    sender: str = ""                     # 发送者进程 ID
    receiver: str = ""                   # 接收者进程 ID（空表示广播）
    payload: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # 关联消息 ID
    
    def to_bytes(self) -> bytes:
        """序列化为字节"""
        return pickle.dumps(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "ProcessMessage":
        """从字节反序列化"""
        return pickle.loads(data)
    
    def create_reply(self, msg_type: ProcessMessageType, payload: dict) -> "ProcessMessage":
        """创建回复消息"""
        return ProcessMessage(
            type=msg_type,
            sender=self.receiver,
            receiver=self.sender,
            payload=payload,
            correlation_id=self.id,
        )


class MessageQueue:
    """消息队列封装
    
    提供进程间通信的消息队列
    """
    
    def __init__(self):
        # 主队列：Agent -> Coordinator
        self.to_coordinator: Queue = mp.Queue()
        # 各 Agent 的接收队列
        self._agent_queues: dict[str, Queue] = {}
        
    def create_agent_queue(self, agent_id: str) -> Queue:
        """为 Agent 创建专用队列"""
        queue = mp.Queue()
        self._agent_queues[agent_id] = queue
        return queue
    
    def get_agent_queue(self, agent_id: str) -> Optional[Queue]:
        """获取 Agent 队列"""
        return self._agent_queues.get(agent_id)
    
    def send_to_coordinator(self, message: ProcessMessage) -> None:
        """发送消息给协调器"""
        self.to_coordinator.put(message)
    
    def send_to_agent(self, agent_id: str, message: ProcessMessage) -> bool:
        """发送消息给指定 Agent"""
        queue = self._agent_queues.get(agent_id)
        if queue:
            queue.put(message)
            return True
        return False
    
    def broadcast_to_agents(self, message: ProcessMessage) -> None:
        """广播消息给所有 Agent"""
        for queue in self._agent_queues.values():
            queue.put(message)
    
    def receive_from_coordinator(self, timeout: Optional[float] = None) -> Optional[ProcessMessage]:
        """从协调器队列接收消息"""
        try:
            if timeout:
                return self.to_coordinator.get(timeout=timeout)
            return self.to_coordinator.get_nowait()
        except:
            return None
    
    def cleanup(self) -> None:
        """清理所有队列"""
        # 关闭主队列
        try:
            self.to_coordinator.close()
            self.to_coordinator.join_thread()
        except:
            pass
        
        # 关闭所有 Agent 队列
        for queue in self._agent_queues.values():
            try:
                queue.close()
                queue.join_thread()
            except:
                pass
        
        self._agent_queues.clear()
