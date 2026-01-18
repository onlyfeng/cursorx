"""协调层"""
from .orchestrator import Orchestrator, OrchestratorConfig
from .orchestrator_mp import MultiProcessOrchestrator, MultiProcessOrchestratorConfig
from .worker_pool import WorkerPool

__all__ = [
    # 原版（协程）
    "Orchestrator",
    "OrchestratorConfig",
    "WorkerPool",
    # 多进程版本
    "MultiProcessOrchestrator",
    "MultiProcessOrchestratorConfig",
]
