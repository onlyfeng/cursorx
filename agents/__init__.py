"""Agent 实现"""
from .committer import CommitResult, CommitterAgent, CommitterConfig
from .planner import PlannerAgent, PlannerConfig

# 多进程版本
from .planner_process import PlannerAgentProcess
from .reviewer import ReviewDecision, ReviewerAgent, ReviewerConfig
from .reviewer_process import ReviewerAgentProcess
from .worker import WorkerAgent, WorkerConfig
from .worker_process import WorkerAgentProcess

# 简短别名（兼容性）
Planner = PlannerAgent
Worker = WorkerAgent
Reviewer = ReviewerAgent

__all__ = [
    # 原版（协程）
    "PlannerAgent",
    "PlannerConfig",
    "WorkerAgent",
    "WorkerConfig",
    "ReviewerAgent",
    "ReviewerConfig",
    "ReviewDecision",
    "CommitterAgent",
    "CommitterConfig",
    "CommitResult",
    # 多进程版本
    "PlannerAgentProcess",
    "WorkerAgentProcess",
    "ReviewerAgentProcess",
    # 简短别名
    "Planner",
    "Worker",
    "Reviewer",
]
