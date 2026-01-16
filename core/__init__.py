"""核心抽象层"""
from .base import AgentRole, AgentStatus, BaseAgent
from .message import Message, MessageType
from .state import AgentState, SystemState

__all__ = [
    "AgentRole",
    "AgentStatus", 
    "BaseAgent",
    "Message",
    "MessageType",
    "AgentState",
    "SystemState",
]
